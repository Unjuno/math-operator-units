from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
DEFAULT_CONTROLLER_SEED = 737_000
DEFAULT_CONTROLLER_HOLDOUT_SEED = 737_500
ENV_CONTROLLER_STRENGTH = "OPFUSION_LEARNED_CONTROLLER_STRENGTH"
_ACTIVE_CONTROLLER: "PromptOperatorController | None" = None


class PromptOperatorController(nn.Module):
    """Predict soft operator state from pooled prompt-token embeddings."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_size: int = 16,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 1 or embedding_size <= 0 or hidden_size <= 0:
            raise ValueError("invalid controller dimensions")
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(FUNCTIONAL_OPERATORS)),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("controller inputs must have matching [batch, time] shape")
        embedded = self.embedding(input_ids)
        mask = attention_mask.to(embedded.dtype).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)

    def probabilities(self, prompt: Sequence[int], *, device: torch.device) -> torch.Tensor:
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return torch.softmax(self(ids, mask).squeeze(0), dim=-1)


def _pad_examples(
    examples: Sequence[tuple[Sequence[int], int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("controller examples must not be empty")
    width = max(len(prompt) for prompt, _ in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.empty(len(examples), dtype=torch.long, device=device)
    for row, (prompt, label) in enumerate(examples):
        values = torch.tensor(list(prompt), dtype=torch.long, device=device)
        ids[row, : values.numel()] = values
        mask[row, : values.numel()] = True
        labels[row] = int(label)
    return ids, mask, labels


def _standalone_examples(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    examples_per_operator: int,
    seed: int,
    split: str,
) -> list[tuple[list[int], int]]:
    rows: list[tuple[list[int], int]] = []
    for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
        for sample_index in range(examples_per_operator):
            example = factory.training_example(
                operator,
                seed=seed,
                split=split,
                step=operator_index,
                sample_index=sample_index,
            )
            rows.append(
                (
                    tokenizer.encode_tokens(
                        example.prompt_tokens, add_bos=True, add_eos=False
                    ),
                    operator_index,
                )
            )
    return rows


def controller_metrics(
    controller: PromptOperatorController,
    examples: Sequence[tuple[Sequence[int], int]],
    *,
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    ids, mask, labels = _pad_examples(examples, pad_id=pad_id, device=device)
    controller.eval()
    with torch.no_grad():
        probabilities = torch.softmax(controller(ids, mask), dim=-1)
        prediction = probabilities.argmax(dim=-1)
    per_operator = {}
    for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
        selected = labels == operator_index
        per_operator[operator] = float(
            (prediction[selected] == operator_index).float().mean().detach().cpu()
        )
    return {
        "examples": len(examples),
        "accuracy": float((prediction == labels).float().mean().detach().cpu()),
        "mean_confidence": float(probabilities.amax(dim=-1).mean().detach().cpu()),
        "per_operator_accuracy": per_operator,
    }


def fit_prompt_controller(
    *,
    examples: Sequence[tuple[Sequence[int], int]],
    holdout_examples: Sequence[tuple[Sequence[int], int]],
    vocabulary_size: int,
    pad_id: int,
    embedding_size: int,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[PromptOperatorController, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    controller = PromptOperatorController(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    ).to(device)
    ids, mask, labels = _pad_examples(examples, pad_id=pad_id, device=device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    losses: list[float] = []
    controller.train()
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(controller(ids, mask), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    controller.eval()
    return controller, {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "optimization_first": losses[0] if losses else None,
        "optimization_last": losses[-1] if losses else None,
        "train_metrics": controller_metrics(
            controller, examples, pad_id=pad_id, device=device
        ),
        "holdout_metrics": controller_metrics(
            controller, holdout_examples, pad_id=pad_id, device=device
        ),
    }


def continuous_operator_prior(
    probabilities: torch.Tensor,
    *,
    source_count: int,
    strength: float,
) -> torch.Tensor:
    if probabilities.shape != (len(FUNCTIONAL_OPERATORS),):
        raise ValueError("expected one probability per functional operator")
    if source_count != len(EXPERIMENT_OPERATORS) + 1:
        raise ValueError("expected Base plus all five specialist sources")
    prior = probabilities.new_zeros(source_count)
    for index, operator in enumerate(FUNCTIONAL_OPERATORS):
        prior[1 + EXPERIMENT_OPERATORS.index(operator)] = (
            float(strength) * probabilities[index]
        )
    return prior


def combine_learned_operator_states(
    fast_state: torch.Tensor,
    slow_state: torch.Tensor,
    *,
    candidate,
    controller_probabilities: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if fast_state.shape != slow_state.shape:
        raise ValueError("fast_state and slow_state must have the same shape")
    combined = (
        (1.0 - float(candidate.slow_mix)) * fast_state
        + float(candidate.slow_mix) * slow_state
    ) / float(candidate.temperature)
    prior = continuous_operator_prior(
        controller_probabilities,
        source_count=int(combined.shape[-1]),
        strength=strength,
    )
    return torch.softmax(combined + prior, dim=-1)


def _generate_learned_operator(
    *,
    base: torch.nn.Module,
    units,
    mixer: torch.nn.Module,
    candidate,
    operator: str,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
):
    """Oracle-compatible generator; operator is used only for post-hoc diagnostics."""
    if _ACTIVE_CONTROLLER is None:
        raise RuntimeError("learned controller is not fitted")
    strength = float(os.environ.get(ENV_CONTROLLER_STRENGTH, "0.0"))
    controller_probabilities = _ACTIVE_CONTROLLER.probabilities(prompt, device=device)
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    fast_state = None
    slow_state = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    timescale_gap_sum = 0.0
    matching_weight_sum = 0.0
    positions = 0
    matching_source_index = 1 + EXPERIMENT_OPERATORS.index(operator)

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

            weights = combine_learned_operator_states(
                fast_state,
                slow_state,
                candidate=candidate,
                controller_probabilities=controller_probabilities,
                strength=strength,
            )
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
            timescale_gap_sum += float(
                (fast_state - slow_state).abs().mean().detach().cpu()
            )
            matching_weight_sum += float(weights[matching_source_index].detach().cpu())
            positions += 1
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == eos_id:
                break

    matching_probability = float(
        controller_probabilities[FUNCTIONAL_OPERATORS.index(operator)].detach().cpu()
    )
    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
        "mean_timescale_gap": timescale_gap_sum / max(1, positions),
        "mean_oracle_source_weight": matching_weight_sum / max(1, positions),
        "controller_matching_probability": matching_probability,
    }


def fit_controller_for_cohorts(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    train_examples_per_operator: int,
    holdout_examples_per_operator: int,
    controller_seed: int,
    holdout_seed: int,
    embedding_size: int,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    device: torch.device,
):
    tokenizers = []
    factories = []
    for cohort in cohorts[:3]:
        run = load_run_config(cohort.config_path)
        tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
        tokenizers.append(tokenizer)
        factories.append(SyntheticTraceFactory(tokenizer, run.data))
    if len({tokenizer.vocab_hash for tokenizer in tokenizers}) != 1:
        raise RuntimeError("controller experiment requires identical tokenizers")
    tokenizer = tokenizers[0]
    factory = factories[0]
    train_examples = _standalone_examples(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=train_examples_per_operator,
        seed=controller_seed,
        split="train",
    )
    holdout_examples = _standalone_examples(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=holdout_examples_per_operator,
        seed=holdout_seed,
        split="validation",
    )
    controller, report = fit_prompt_controller(
        examples=train_examples,
        holdout_examples=holdout_examples,
        vocabulary_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        steps=steps,
        seed=controller_seed,
        device=device,
    )
    report.update(
        {
            "training_split": "train",
            "holdout_split": "validation",
            "train_examples_per_operator": train_examples_per_operator,
            "holdout_examples_per_operator": holdout_examples_per_operator,
            "vocab_hash": tokenizer.vocab_hash,
        }
    )
    return controller, report


def composition_prompt_controller_metrics(
    controller: PromptOperatorController,
    cohort: Cohort,
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    correct = 0
    count = 0
    matching_probability = 0.0
    for inner in FUNCTIONAL_OPERATORS:
        for outer in FUNCTIONAL_OPERATORS:
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = sequential.composition_operands(
                    inner_operator=inner,
                    outer_operator=outer,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = sequential.apply_operator(inner, inner_values)
                for operator, values in (
                    (inner, inner_values),
                    (outer, (true_inner, *outer_extras)),
                ):
                    prompt = sequential.prompt_ids_for_values(
                        factory=factory,
                        tokenizer=tokenizer,
                        operator=operator,
                        values=values,
                    )
                    probabilities = controller.probabilities(prompt, device=device)
                    expected = FUNCTIONAL_OPERATORS.index(operator)
                    correct += int(int(probabilities.argmax().item()) == expected)
                    matching_probability += float(probabilities[expected].detach().cpu())
                    count += 1
    return {
        "examples": count,
        "accuracy": correct / max(1, count),
        "mean_matching_probability": matching_probability / max(1, count),
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
    controller_train_examples_per_operator: int,
    controller_holdout_examples_per_operator: int,
    controller_seed: int,
    controller_holdout_seed: int,
    controller_embedding_size: int,
    controller_hidden_size: int,
    controller_learning_rate: float,
    controller_steps: int,
    device_name: str,
) -> dict[str, Any]:
    global _ACTIVE_CONTROLLER
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    mixer, mixer_fit = sequential.fit_ensemble(
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
    controller, controller_fit = fit_controller_for_cohorts(
        cohorts,
        root=root,
        train_examples_per_operator=controller_train_examples_per_operator,
        holdout_examples_per_operator=controller_holdout_examples_per_operator,
        controller_seed=controller_seed,
        holdout_seed=controller_holdout_seed,
        embedding_size=controller_embedding_size,
        hidden_size=controller_hidden_size,
        learning_rate=controller_learning_rate,
        steps=controller_steps,
        device=device,
    )
    _ACTIVE_CONTROLLER = controller
    original_generator = oracle._generate_oracle_operator
    oracle._generate_oracle_operator = _generate_learned_operator
    try:
        cohort_reports = [
            sequential.evaluate_cohort(
                cohort,
                root=root,
                mixer=mixer,
                examples_per_pair=examples_per_pair,
                data_seed=data_seed,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            for cohort in cohorts[:3]
        ]
    finally:
        oracle._generate_oracle_operator = original_generator

    aggregate = sequential._empty_counter()
    pair_aggregate = {
        f"{inner}->{outer}": sequential._empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    mean_weights = []
    for report in cohort_reports:
        sequential._merge_counter(
            aggregate,
            {key: int(report["aggregate"][key]) for key in sequential._empty_counter()},
        )
        mean_weights.append(float(report["mean_matching_source_weight"]))
        for pair_id, row in report["pairs"].items():
            sequential._merge_counter(
                pair_aggregate[pair_id],
                {key: int(row[key]) for key in sequential._empty_counter()},
            )

    strength = float(os.environ.get(ENV_CONTROLLER_STRENGTH, "0.0"))
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_learned_operator_state_sequential_composition",
        "claim_boundary": (
            "controller training uses standalone ADD/SUM/MIN/MAX prompts only; "
            "composition operator labels are not used to construct the learned prior; "
            "stage boundaries remain external and generation resets at each stage; "
            "NEG, learned boundaries, single-pass nesting, final IID test, and OOD are untested"
        ),
        "operators": list(FUNCTIONAL_OPERATORS),
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "controller_strength": strength,
        "data_seed": data_seed,
        "calibration_seed": calibration_seed,
        "controller_seed": controller_seed,
        "controller_holdout_seed": controller_holdout_seed,
        "fixed_candidate": sequential.confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": mixer_fit,
        "controller_fit": controller_fit,
        "composition_prompt_controller": composition_prompt_controller_metrics(
            controller,
            cohorts[0],
            root=root,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            device=device,
        ),
        "cohort_reports": cohort_reports,
        "aggregate": sequential._finalize_counter(aggregate),
        "pairs": {
            pair_id: sequential._finalize_counter(counter)
            for pair_id, counter in pair_aggregate.items()
        },
        "mean_matching_source_weight": sum(mean_weights) / max(1, len(mean_weights)),
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test a standalone-trained prompt operator controller on two-stage composition"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--examples-per-pair", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=sequential.DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--calibration-examples-per-operator", type=int, default=8)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1536)
    parser.add_argument("--calibration-seed", type=int, default=sequential.DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sketch-size", type=int, default=8)
    parser.add_argument("--controller-train-examples-per-operator", type=int, default=64)
    parser.add_argument("--controller-holdout-examples-per-operator", type=int, default=32)
    parser.add_argument("--controller-seed", type=int, default=DEFAULT_CONTROLLER_SEED)
    parser.add_argument("--controller-holdout-seed", type=int, default=DEFAULT_CONTROLLER_HOLDOUT_SEED)
    parser.add_argument("--controller-embedding-size", type=int, default=16)
    parser.add_argument("--controller-hidden-size", type=int, default=16)
    parser.add_argument("--controller-learning-rate", type=float, default=0.03)
    parser.add_argument("--controller-steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/learned_operator_controller/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        report = run_experiment(
            root=args.root,
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
            controller_train_examples_per_operator=args.controller_train_examples_per_operator,
            controller_holdout_examples_per_operator=args.controller_holdout_examples_per_operator,
            controller_seed=args.controller_seed,
            controller_holdout_seed=args.controller_holdout_seed,
            controller_embedding_size=args.controller_embedding_size,
            controller_hidden_size=args.controller_hidden_size,
            controller_learning_rate=args.controller_learning_rate,
            controller_steps=args.controller_steps,
            device_name=args.device,
        )
    finally:
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output.resolve())
    print(json.dumps(report["controller_fit"], sort_keys=True))
    print(json.dumps(report["composition_prompt_controller"], sort_keys=True))
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
