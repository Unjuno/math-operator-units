from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from opfusion import fusion_oracle_sequential_composition as seq
from opfusion import fusion_stateful_dual_timescale as dual_timescale
from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


ENV_CONTROLLER_STRENGTH = "OPFUSION_CONTROLLER_OPERATOR_STRENGTH"
ENV_CONTROLLER_FEATURE_MODE = "OPFUSION_CONTROLLER_FEATURE_MODE"
FUNCTIONAL_OPERATORS = seq.FUNCTIONAL_OPERATORS
DEFAULT_CONTROLLER_SEED = 734_000


class PromptOperatorController(nn.Module):
    """Small prompt-only classifier for the current functional operator.

    The controller never sees model logits, generated answers, gold intermediate
    values, or an operator label at inference time.  It embeds the stage prompt,
    concatenates the first post-BOS token embedding with a masked mean prompt
    embedding, and predicts one of ADD/SUM/MIN/MAX.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        embedding_size: int = 16,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(FUNCTIONAL_OPERATORS)),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("ids and mask must have matching [batch, time] shape")
        embedded = self.embedding(ids)
        weights = mask.to(dtype=embedded.dtype).unsqueeze(-1)
        pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        anchor_index = 1 if ids.shape[1] > 1 else 0
        anchor = embedded[:, anchor_index, :]
        return self.network(torch.cat([anchor, pooled], dim=-1))


def _feature_mode() -> str:
    mode = os.environ.get(ENV_CONTROLLER_FEATURE_MODE, "full").strip().lower()
    if mode not in {"full", "masked"}:
        raise ValueError(f"unsupported controller feature mode: {mode}")
    return mode


def apply_feature_mode_to_prompt(
    prompt: Sequence[int], *, bos_id: int, mode: str
) -> list[int]:
    ids = list(prompt)
    if mode == "full":
        return ids
    if mode != "masked":
        raise ValueError(mode)
    if len(ids) > 1:
        # Prompts are encoded as BOS, OPERATOR_TOKEN, state..., RESPONSE.
        # Replacing the operator token by BOS removes direct operator identity
        # while leaving length and state-syntax cues available as a control.
        ids[1] = int(bos_id)
    return ids


def _pad_prompts(
    prompts: Sequence[Sequence[int]], *, pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompts:
        raise ValueError("at least one prompt is required")
    width = max(len(prompt) for prompt in prompts)
    ids = torch.full(
        (len(prompts), width), int(pad_id), dtype=torch.long, device=device
    )
    mask = torch.zeros((len(prompts), width), dtype=torch.bool, device=device)
    for row, prompt in enumerate(prompts):
        length = len(prompt)
        ids[row, :length] = torch.tensor(prompt, dtype=torch.long, device=device)
        mask[row, :length] = True
    return ids, mask


def _single_operator_values(operator: str, *, seed: int, sample_index: int) -> tuple[int, ...]:
    rng = random.Random(seq._stable_seed("learned-controller-single-v1", seed, operator, sample_index))
    count = 2 if operator == "scalar.add" else 3
    return tuple(rng.randint(-16, 16) for _ in range(count))


def _controller_dataset(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    seed: int,
    examples_per_operator: int,
    feature_mode: str,
) -> tuple[list[list[int]], torch.Tensor]:
    prompts: list[list[int]] = []
    labels: list[int] = []
    for label, operator in enumerate(FUNCTIONAL_OPERATORS):
        for sample_index in range(examples_per_operator):
            values = _single_operator_values(operator, seed=seed, sample_index=sample_index)
            prompt = seq.prompt_ids_for_values(
                factory=factory,
                tokenizer=tokenizer,
                operator=operator,
                values=values,
            )
            prompts.append(
                apply_feature_mode_to_prompt(
                    prompt, bos_id=tokenizer.bos_id, mode=feature_mode
                )
            )
            labels.append(label)
    return prompts, torch.tensor(labels, dtype=torch.long)


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean().item())
    confidence = float(probabilities.max(dim=-1).values.mean().item())
    per_operator: dict[str, float] = {}
    for index, operator in enumerate(FUNCTIONAL_OPERATORS):
        mask = labels == index
        per_operator[operator] = float(
            (predictions[mask] == labels[mask]).float().mean().item()
        )
    return {
        "accuracy": accuracy,
        "mean_confidence": confidence,
        "per_operator_accuracy": per_operator,
    }


def fit_prompt_controller(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    device: torch.device,
    seed: int,
    examples_per_operator: int,
    validation_examples_per_operator: int,
    steps: int,
    learning_rate: float,
    feature_mode: str,
) -> tuple[PromptOperatorController, dict[str, Any]]:
    train_prompts, train_labels_cpu = _controller_dataset(
        factory=factory,
        tokenizer=tokenizer,
        seed=seed,
        examples_per_operator=examples_per_operator,
        feature_mode=feature_mode,
    )
    validation_prompts, validation_labels_cpu = _controller_dataset(
        factory=factory,
        tokenizer=tokenizer,
        seed=seed + 1,
        examples_per_operator=validation_examples_per_operator,
        feature_mode=feature_mode,
    )
    train_ids, train_mask = _pad_prompts(
        train_prompts, pad_id=tokenizer.eos_id, device=device
    )
    validation_ids, validation_mask = _pad_prompts(
        validation_prompts, pad_id=tokenizer.eos_id, device=device
    )
    train_labels = train_labels_cpu.to(device)
    validation_labels = validation_labels_cpu.to(device)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        controller = PromptOperatorController(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    controller.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = controller(train_ids, train_mask)
        loss = torch.nn.functional.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 4) == 0:
            with torch.no_grad():
                accuracy = float((logits.argmax(dim=-1) == train_labels).float().mean())
            history.append({"step": float(step + 1), "loss": float(loss), "accuracy": accuracy})

    controller.eval()
    with torch.no_grad():
        train_logits = controller(train_ids, train_mask)
        validation_logits = controller(validation_ids, validation_mask)
    return controller, {
        "seed": seed,
        "feature_mode": feature_mode,
        "examples_per_operator": examples_per_operator,
        "validation_examples_per_operator": validation_examples_per_operator,
        "steps": steps,
        "learning_rate": learning_rate,
        "history": history,
        "train": _classification_metrics(train_logits, train_labels),
        "validation": _classification_metrics(validation_logits, validation_labels),
    }


def controller_operator_prior(
    probabilities: torch.Tensor,
    *,
    source_count: int,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map a soft operator posterior to a finite source log-weight prior."""
    if probabilities.ndim != 1 or len(probabilities) != len(FUNCTIONAL_OPERATORS):
        raise ValueError("probabilities must contain one value per functional operator")
    if source_count != len(EXPERIMENT_OPERATORS) + 1:
        raise ValueError(
            f"expected {len(EXPERIMENT_OPERATORS) + 1} sources, got {source_count}"
        )
    prior = torch.zeros(source_count, device=device, dtype=dtype)
    probs = probabilities.to(device=device, dtype=dtype)
    for class_index, operator in enumerate(FUNCTIONAL_OPERATORS):
        source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
        prior[source_index] = float(strength) * probs[class_index]
    return prior


def _controller_posterior(
    controller: PromptOperatorController,
    *,
    prompt: Sequence[int],
    tokenizer: FixedVocabTokenizer,
    feature_mode: str,
    device: torch.device,
) -> torch.Tensor:
    transformed = apply_feature_mode_to_prompt(
        prompt, bos_id=tokenizer.bos_id, mode=feature_mode
    )
    ids, mask = _pad_prompts([transformed], pad_id=tokenizer.eos_id, device=device)
    with torch.no_grad():
        return torch.softmax(controller(ids, mask)[0].float(), dim=-1)


def _generate_controller_conditioned(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    controller: PromptOperatorController,
    candidate: dual_timescale.DualTimescaleCandidate,
    prompt: Sequence[int],
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, Any]]:
    strength = float(os.environ.get(ENV_CONTROLLER_STRENGTH, "0.0"))
    feature_mode = _feature_mode()
    posterior = _controller_posterior(
        controller,
        prompt=prompt,
        tokenizer=tokenizer,
        feature_mode=feature_mode,
        device=device,
    )
    predicted_class = int(posterior.argmax().item())
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    fast_state: torch.Tensor | None = None
    slow_state: torch.Tensor | None = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    timescale_gap_sum = 0.0
    source_weight_sum: torch.Tensor | None = None
    positions = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = implementation._source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()

            if fast_state is None or slow_state is None:
                fast_state = instant_state
                slow_state = instant_state
                state_change = instant_state.new_tensor(0.0)
            else:
                previous_combined = (
                    (1.0 - float(candidate.slow_mix)) * fast_state
                    + float(candidate.slow_mix) * slow_state
                )
                fast_state = (
                    float(candidate.fast_memory) * fast_state
                    + (1.0 - float(candidate.fast_memory)) * instant_state
                )
                slow_state = (
                    float(candidate.slow_memory) * slow_state
                    + (1.0 - float(candidate.slow_memory)) * instant_state
                )
                current_combined = (
                    (1.0 - float(candidate.slow_mix)) * fast_state
                    + float(candidate.slow_mix) * slow_state
                )
                state_change = (current_combined - previous_combined).abs().mean()

            combined = (
                (1.0 - float(candidate.slow_mix)) * fast_state
                + float(candidate.slow_mix) * slow_state
            ) / float(candidate.temperature)
            prior = controller_operator_prior(
                posterior,
                source_count=int(combined.shape[-1]),
                strength=strength,
                device=combined.device,
                dtype=combined.dtype,
            )
            weights = torch.softmax(combined + prior, dim=-1)
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0.0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                token_support = token_support - token_support.mean()
                fast_state = fast_state + float(candidate.feedback) * token_support
                slow_state = slow_state + 0.25 * float(candidate.feedback) * token_support

            entropy_sum += float(
                (-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu()
            )
            max_weight_sum += float(weights.max().detach().cpu())
            state_change_sum += float(state_change.detach().cpu())
            timescale_gap_sum += float((fast_state - slow_state).abs().mean().detach().cpu())
            source_weight_sum = weights.detach().clone() if source_weight_sum is None else source_weight_sum + weights.detach()
            positions += 1

            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == tokenizer.eos_id:
                break

    mean_source_weights = (
        source_weight_sum / max(1, positions)
        if source_weight_sum is not None
        else torch.zeros(len(EXPERIMENT_OPERATORS) + 1, device=device)
    )
    return output, {
        "controller_posterior": [float(value) for value in posterior.detach().cpu()],
        "controller_predicted_operator": FUNCTIONAL_OPERATORS[predicted_class],
        "controller_confidence": float(posterior.max().detach().cpu()),
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
        "mean_timescale_gap": timescale_gap_sum / max(1, positions),
        "mean_source_weights": [float(value) for value in mean_source_weights.detach().cpu()],
    }


def _generate_value(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    controller: PromptOperatorController,
    candidate,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[int | None, dict[str, Any]]:
    prompt = seq.prompt_ids_for_values(
        factory=factory,
        tokenizer=tokenizer,
        operator=operator,
        values=values,
    )
    generated, diagnostics = _generate_controller_conditioned(
        base=base,
        units=units,
        mixer=mixer,
        controller=controller,
        candidate=candidate,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    # Gold operator is used only after generation for evaluation diagnostics.
    source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
    diagnostics["mean_matching_source_weight"] = diagnostics["mean_source_weights"][source_index]
    diagnostics["controller_correct"] = float(
        diagnostics["controller_predicted_operator"] == operator
    )
    return seq.parse_final_numeric_token(generated, tokenizer), diagnostics


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    mixer: torch.nn.Module,
    controller: PromptOperatorController,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    candidate = confirmatory.candidate_grid()[1]
    aggregate = seq._empty_counter()
    pair_rows: dict[str, dict[str, float | int | None]] = {}
    matching_weight_sum = 0.0
    controller_correct_sum = 0.0
    controller_confidence_sum = 0.0
    diagnostic_calls = 0

    for inner_operator in FUNCTIONAL_OPERATORS:
        for outer_operator in FUNCTIONAL_OPERATORS:
            pair_id = f"{inner_operator}->{outer_operator}"
            counter = seq._empty_counter()
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = seq.composition_operands(
                    inner_operator=inner_operator,
                    outer_operator=outer_operator,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = seq.apply_operator(inner_operator, inner_values)
                true_outer_values = (true_inner, *outer_extras)
                true_final = seq.apply_operator(outer_operator, true_outer_values)

                generated_inner, inner_diag = _generate_value(
                    base=base,
                    units=units,
                    mixer=mixer,
                    controller=controller,
                    candidate=candidate,
                    operator=inner_operator,
                    values=inner_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                oracle_outer, oracle_outer_diag = _generate_value(
                    base=base,
                    units=units,
                    mixer=mixer,
                    controller=controller,
                    candidate=candidate,
                    operator=outer_operator,
                    values=true_outer_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )

                chained_outer: int | None = None
                chained_diag: dict[str, Any] | None = None
                if generated_inner is not None:
                    chained_outer, chained_diag = _generate_value(
                        base=base,
                        units=units,
                        mixer=mixer,
                        controller=controller,
                        candidate=candidate,
                        operator=outer_operator,
                        values=(generated_inner, *outer_extras),
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )

                inner_correct = generated_inner == true_inner
                counter["cases"] += 1
                counter["inner_parse"] += int(generated_inner is not None)
                counter["inner_correct"] += int(inner_correct)
                counter["oracle_intermediate_outer_parse"] += int(oracle_outer is not None)
                counter["oracle_intermediate_outer_correct"] += int(oracle_outer == true_final)
                counter["end_to_end_outer_parse"] += int(chained_outer is not None)
                counter["end_to_end_correct"] += int(chained_outer == true_final)
                if inner_correct:
                    counter["inner_correct_cases"] += 1
                    counter["end_to_end_correct_given_inner_correct"] += int(chained_outer == true_final)

                for row in (inner_diag, oracle_outer_diag, chained_diag):
                    if row is not None:
                        matching_weight_sum += float(row["mean_matching_source_weight"])
                        controller_correct_sum += float(row["controller_correct"])
                        controller_confidence_sum += float(row["controller_confidence"])
                        diagnostic_calls += 1

            seq._merge_counter(aggregate, counter)
            pair_rows[pair_id] = seq._finalize_counter(counter)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "candidate": candidate.__dict__,
        "aggregate": seq._finalize_counter(aggregate),
        "pairs": pair_rows,
        "mean_matching_source_weight": matching_weight_sum / max(1, diagnostic_calls),
        "controller_stage_accuracy": controller_correct_sum / max(1, diagnostic_calls),
        "controller_mean_confidence": controller_confidence_sum / max(1, diagnostic_calls),
    }


def run_experiment(
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    calibration_seed: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    hidden_size: int,
    sketch_size: int,
    controller_seed: int,
    controller_examples_per_operator: int,
    controller_validation_examples_per_operator: int,
    controller_steps: int,
    controller_learning_rate: float,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    mixer, fit_report = seq.fit_ensemble(
        cohorts,
        root=root,
        calibration_examples_per_operator=calibration_examples_per_operator,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions_per_cohort=max_positions_per_cohort,
        calibration_seed=calibration_seed,
        fit_steps=fit_steps,
        fit_batch_positions=fit_batch_positions,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        sketch_size=sketch_size,
        device=device,
    )

    controller_run = load_run_config(cohorts[0].config_path)
    controller_tokenizer = FixedVocabTokenizer.from_config(root / controller_run.tokenizer_config)
    controller_factory = SyntheticTraceFactory(controller_tokenizer, controller_run.data)
    feature_mode = _feature_mode()
    controller, controller_fit = fit_prompt_controller(
        factory=controller_factory,
        tokenizer=controller_tokenizer,
        device=device,
        seed=controller_seed,
        examples_per_operator=controller_examples_per_operator,
        validation_examples_per_operator=controller_validation_examples_per_operator,
        steps=controller_steps,
        learning_rate=controller_learning_rate,
        feature_mode=feature_mode,
    )

    cohort_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            mixer=mixer,
            controller=controller,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]

    aggregate = seq._empty_counter()
    pair_aggregate = {
        f"{inner}->{outer}": seq._empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    mean_weights: list[float] = []
    controller_stage_accuracies: list[float] = []
    controller_confidences: list[float] = []
    for report in cohort_reports:
        raw_aggregate = {key: int(report["aggregate"][key]) for key in seq._empty_counter()}
        seq._merge_counter(aggregate, raw_aggregate)
        mean_weights.append(float(report["mean_matching_source_weight"]))
        controller_stage_accuracies.append(float(report["controller_stage_accuracy"]))
        controller_confidences.append(float(report["controller_mean_confidence"]))
        for pair_id, row in report["pairs"].items():
            raw_pair = {key: int(row[key]) for key in seq._empty_counter()}
            seq._merge_counter(pair_aggregate[pair_id], raw_pair)

    strength = float(os.environ.get(ENV_CONTROLLER_STRENGTH, "0.0"))
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_learned_operator_controller_sequential_composition",
        "claim_boundary": (
            "the operator prior is predicted only from each stage prompt by a controller trained "
            "on isolated ADD/SUM/MIN/MAX prompts; stage boundaries are still externally supplied; "
            "gold operator identity is used only for post-generation scoring; no NEG, nested "
            "single-pass execution, final IID test, or OOD split is tested"
        ),
        "operators": list(FUNCTIONAL_OPERATORS),
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "controller_operator_strength": strength,
        "controller_feature_mode": feature_mode,
        "controller_seed": controller_seed,
        "data_seed": data_seed,
        "calibration_seed": calibration_seed,
        "fixed_candidate": confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": fit_report,
        "controller_fit": controller_fit,
        "cohorts": cohort_reports,
        "aggregate": seq._finalize_counter(aggregate),
        "pairs": {pair_id: seq._finalize_counter(counter) for pair_id, counter in pair_aggregate.items()},
        "mean_matching_source_weight": sum(mean_weights) / max(1, len(mean_weights)),
        "controller_stage_accuracy": sum(controller_stage_accuracies) / max(1, len(controller_stage_accuracies)),
        "controller_mean_confidence": sum(controller_confidences) / max(1, len(controller_confidences)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples-per-pair", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=seq.DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--calibration-examples-per-operator", type=int, default=8)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1536)
    parser.add_argument("--calibration-seed", type=int, default=seq.DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sketch-size", type=int, default=8)
    parser.add_argument("--controller-seed", type=int, default=DEFAULT_CONTROLLER_SEED)
    parser.add_argument("--controller-examples-per-operator", type=int, default=64)
    parser.add_argument("--controller-validation-examples-per-operator", type=int, default=32)
    parser.add_argument("--controller-steps", type=int, default=300)
    parser.add_argument("--controller-learning-rate", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_experiment(
        root=args.root.resolve(),
        examples_per_pair=args.examples_per_pair,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        max_prefixes_per_example=args.max_prefixes_per_example,
        max_positions_per_cohort=args.max_positions_per_cohort,
        calibration_seed=args.calibration_seed,
        fit_steps=args.fit_steps,
        fit_batch_positions=args.fit_batch_positions,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        sketch_size=args.sketch_size,
        controller_seed=args.controller_seed,
        controller_examples_per_operator=args.controller_examples_per_operator,
        controller_validation_examples_per_operator=args.controller_validation_examples_per_operator,
        controller_steps=args.controller_steps,
        controller_learning_rate=args.controller_learning_rate,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(json.dumps({
        "aggregate": report["aggregate"],
        "controller_fit_validation": report["controller_fit"]["validation"],
        "controller_stage_accuracy": report["controller_stage_accuracy"],
        "controller_mean_confidence": report["controller_mean_confidence"],
        "mean_matching_source_weight": report["mean_matching_source_weight"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
