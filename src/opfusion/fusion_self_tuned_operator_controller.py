from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from opfusion import fusion_learned_operator_controller as learned
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
DEFAULT_CONTROLLER_SEED = 739_000
DEFAULT_CONTROLLER_HOLDOUT_SEED = 739_500
_ACTIVE_CONTROLLER: "SelfTunedPromptController | None" = None


class SelfTunedPromptController(nn.Module):
    """Predict operator direction and its continuous prior scale from a stage prompt."""

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
        self.trunk = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
        )
        self.operator_head = nn.Linear(hidden_size, len(FUNCTIONAL_OPERATORS))
        self.scale_head = nn.Linear(hidden_size, 1)

    def _pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("controller inputs must have matching [batch, time] shape")
        embedded = self.embedding(input_ids)
        mask = attention_mask.to(embedded.dtype).unsqueeze(-1)
        return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def control(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(self._pooled(input_ids, attention_mask))
        operator_logits = self.operator_head(hidden)
        scale = torch.nn.functional.softplus(self.scale_head(hidden).squeeze(-1))
        return operator_logits, scale

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        operator_logits, _ = self.control(input_ids, attention_mask)
        return operator_logits

    def prompt_control(
        self,
        prompt: Sequence[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        logits, scale = self.control(ids, mask)
        probabilities = torch.softmax(logits.squeeze(0), dim=-1)
        return probabilities, scale.squeeze(0)

    def probabilities(self, prompt: Sequence[int], *, device: torch.device) -> torch.Tensor:
        probabilities, _ = self.prompt_control(prompt, device=device)
        return probabilities

    def predicted_scale(self, prompt: Sequence[int], *, device: torch.device) -> float:
        _, scale = self.prompt_control(prompt, device=device)
        return float(scale.detach().cpu())


@dataclass(frozen=True)
class TeacherForcedControlBatch:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    operator_labels: torch.Tensor
    base_scores: torch.Tensor
    target_source_probabilities: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.operator_labels.numel())


def _pad_prompts(
    prompts: Sequence[Sequence[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompts:
        raise ValueError("prompts must not be empty")
    width = max(len(prompt) for prompt in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        values = torch.tensor(list(prompt), dtype=torch.long, device=device)
        ids[row, : values.numel()] = values
        mask[row, : values.numel()] = True
    return ids, mask


def source_prior_from_control(
    operator_probabilities: torch.Tensor,
    scale: torch.Tensor,
    *,
    source_count: int,
) -> torch.Tensor:
    if operator_probabilities.ndim < 1 or operator_probabilities.shape[-1] != len(FUNCTIONAL_OPERATORS):
        raise ValueError("expected one probability per functional operator")
    if scale.shape != operator_probabilities.shape[:-1]:
        raise ValueError("scale must match operator-probability batch dimensions")
    if source_count != len(EXPERIMENT_OPERATORS) + 1:
        raise ValueError("expected Base plus all five specialist sources")
    prior = operator_probabilities.new_zeros((*operator_probabilities.shape[:-1], source_count))
    for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
        source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
        prior[..., source_index] = scale * operator_probabilities[..., operator_index]
    return prior


def _teacher_forced_rows_for_cohort(
    cohort: Cohort,
    *,
    root: Path,
    mixer: nn.Module,
    examples_per_operator: int,
    data_seed: int,
    max_positions: int,
    device: torch.device,
) -> tuple[list[list[int]], list[int], list[torch.Tensor], list[torch.Tensor], int]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    candidate = sequential.confirmatory.candidate_grid()[1]

    prompts: list[list[int]] = []
    labels: list[int] = []
    base_scores: list[torch.Tensor] = []
    target_source_probabilities: list[torch.Tensor] = []

    with torch.no_grad():
        for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
            for sample_index in range(examples_per_operator):
                prompt, expected, _, _ = factory.prompt_and_expected_ids(
                    operator,
                    seed=data_seed,
                    split="train",
                    step=operator_index,
                    sample_index=sample_index,
                )
                ids = torch.tensor([prompt], dtype=torch.long, device=device)
                fast_state = None
                slow_state = None
                for target_id in expected:
                    sources = implementation._source_logits(base=base, units=units, ids=ids)
                    _, instant_weights = mixer.compose(sources)
                    instant_state = instant_weights.clamp_min(1e-9).log()
                    if fast_state is None or slow_state is None:
                        fast_state = instant_state
                        slow_state = instant_state
                    else:
                        fast_state = (
                            float(candidate.fast_memory) * fast_state
                            + (1.0 - float(candidate.fast_memory)) * instant_state
                        )
                        slow_state = (
                            float(candidate.slow_memory) * slow_state
                            + (1.0 - float(candidate.slow_memory)) * instant_state
                        )
                    combined = (
                        (1.0 - float(candidate.slow_mix)) * fast_state
                        + float(candidate.slow_mix) * slow_state
                    ) / float(candidate.temperature)
                    source_probabilities = torch.softmax(sources.float(), dim=-1)
                    target_probabilities = source_probabilities[:, int(target_id)]

                    prompts.append(list(prompt))
                    labels.append(operator_index)
                    base_scores.append(combined.detach().cpu())
                    target_source_probabilities.append(target_probabilities.detach().cpu())

                    if candidate.feedback > 0.0:
                        token_support = torch.log_softmax(sources.float(), dim=-1)[:, int(target_id)]
                        token_support = token_support - token_support.mean()
                        fast_state = fast_state + float(candidate.feedback) * token_support
                        slow_state = slow_state + 0.25 * float(candidate.feedback) * token_support
                    ids = torch.cat(
                        [ids, torch.tensor([[int(target_id)]], dtype=torch.long, device=device)],
                        dim=1,
                    )
                    if len(labels) >= max_positions:
                        break
                if len(labels) >= max_positions:
                    break
            if len(labels) >= max_positions:
                break

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompts, labels, base_scores, target_source_probabilities, tokenizer.pad_id


def collect_teacher_forced_control_batch(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    mixer: nn.Module,
    examples_per_operator: int,
    data_seed: int,
    max_positions_per_cohort: int,
    device: torch.device,
) -> TeacherForcedControlBatch:
    all_prompts: list[list[int]] = []
    all_labels: list[int] = []
    all_base_scores: list[torch.Tensor] = []
    all_target_probabilities: list[torch.Tensor] = []
    pad_ids: list[int] = []
    for cohort in cohorts[:2]:
        prompts, labels, base_scores, target_probabilities, pad_id = _teacher_forced_rows_for_cohort(
            cohort,
            root=root,
            mixer=mixer,
            examples_per_operator=examples_per_operator,
            data_seed=data_seed,
            max_positions=max_positions_per_cohort,
            device=device,
        )
        all_prompts.extend(prompts)
        all_labels.extend(labels)
        all_base_scores.extend(base_scores)
        all_target_probabilities.extend(target_probabilities)
        pad_ids.append(pad_id)
    if len(set(pad_ids)) != 1:
        raise RuntimeError("self-tuned controller requires identical tokenizer padding ids")
    prompt_ids, prompt_mask = _pad_prompts(all_prompts, pad_id=pad_ids[0], device=device)
    return TeacherForcedControlBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        operator_labels=torch.tensor(all_labels, dtype=torch.long, device=device),
        base_scores=torch.stack(all_base_scores, dim=0).to(device),
        target_source_probabilities=torch.stack(all_target_probabilities, dim=0).to(device),
    )


def _batch_control_loss(
    controller: SelfTunedPromptController,
    batch: TeacherForcedControlBatch,
    indices: torch.Tensor,
    *,
    auxiliary_operator_weight: float,
    scale_l2_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ids = batch.prompt_ids.index_select(0, indices)
    mask = batch.prompt_mask.index_select(0, indices)
    labels = batch.operator_labels.index_select(0, indices)
    base_scores = batch.base_scores.index_select(0, indices)
    target_source_probabilities = batch.target_source_probabilities.index_select(0, indices)

    operator_logits, scale = controller.control(ids, mask)
    operator_probabilities = torch.softmax(operator_logits, dim=-1)
    prior = source_prior_from_control(
        operator_probabilities,
        scale,
        source_count=int(base_scores.shape[-1]),
    )
    weights = torch.softmax(base_scores + prior, dim=-1)
    target_probability = (weights * target_source_probabilities).sum(dim=-1).clamp_min(1e-12)
    nll = -target_probability.log().mean()
    operator_ce = torch.nn.functional.cross_entropy(operator_logits, labels)
    scale_l2 = scale.square().mean()
    loss = nll + float(auxiliary_operator_weight) * operator_ce + float(scale_l2_weight) * scale_l2
    return loss, {
        "nll": float(nll.detach().cpu()),
        "operator_ce": float(operator_ce.detach().cpu()),
        "scale_l2": float(scale_l2.detach().cpu()),
        "mean_scale": float(scale.mean().detach().cpu()),
    }


def fit_self_tuned_controller(
    *,
    batch: TeacherForcedControlBatch,
    holdout_examples: Sequence[tuple[Sequence[int], int]],
    vocabulary_size: int,
    pad_id: int,
    embedding_size: int,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    auxiliary_operator_weight: float,
    scale_l2_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[SelfTunedPromptController, dict[str, Any]]:
    if batch.positions <= 0:
        raise ValueError("teacher-forced batch is empty")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    controller = SelfTunedPromptController(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate, weight_decay=1e-4)
    first_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] | None = None
    first_loss: float | None = None
    last_loss: float | None = None
    controller.train()
    for _ in range(steps):
        if batch_positions >= batch.positions:
            indices = torch.arange(batch.positions, device=device)
        else:
            indices = torch.randint(0, batch.positions, (batch_positions,), device=device)
        loss, metrics = _batch_control_loss(
            controller,
            batch,
            indices,
            auxiliary_operator_weight=auxiliary_operator_weight,
            scale_l2_weight=scale_l2_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
            first_metrics = metrics
        last_loss = float(loss.detach().cpu())
        last_metrics = metrics
    controller.eval()
    holdout_metrics = learned.controller_metrics(
        controller,
        holdout_examples,
        pad_id=pad_id,
        device=device,
    )
    holdout_ids, holdout_mask, _ = learned._pad_examples(
        holdout_examples,
        pad_id=pad_id,
        device=device,
    )
    with torch.no_grad():
        _, holdout_scale = controller.control(holdout_ids, holdout_mask)
    return controller, {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_positions": batch_positions,
        "teacher_forced_positions": batch.positions,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "auxiliary_operator_weight": auxiliary_operator_weight,
        "scale_l2_weight": scale_l2_weight,
        "optimization_first": first_loss,
        "optimization_last": last_loss,
        "first_step_metrics": first_metrics,
        "last_step_metrics": last_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_scale_mean": float(holdout_scale.mean().detach().cpu()),
        "holdout_scale_min": float(holdout_scale.min().detach().cpu()),
        "holdout_scale_max": float(holdout_scale.max().detach().cpu()),
    }


def _standalone_controller_examples(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    examples_per_operator: int,
    seed: int,
    split: str,
) -> list[tuple[list[int], int]]:
    return learned._standalone_examples(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=examples_per_operator,
        seed=seed,
        split=split,
    )


def fit_controller_for_cohorts(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    mixer: nn.Module,
    train_examples_per_operator: int,
    holdout_examples_per_operator: int,
    max_positions_per_cohort: int,
    controller_seed: int,
    holdout_seed: int,
    embedding_size: int,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    auxiliary_operator_weight: float,
    scale_l2_weight: float,
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

    training_batch = collect_teacher_forced_control_batch(
        cohorts,
        root=root,
        mixer=mixer,
        examples_per_operator=train_examples_per_operator,
        data_seed=controller_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    holdout_examples = _standalone_controller_examples(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=holdout_examples_per_operator,
        seed=holdout_seed,
        split="validation",
    )
    controller, report = fit_self_tuned_controller(
        batch=training_batch,
        holdout_examples=holdout_examples,
        vocabulary_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        steps=steps,
        batch_positions=batch_positions,
        auxiliary_operator_weight=auxiliary_operator_weight,
        scale_l2_weight=scale_l2_weight,
        seed=controller_seed,
        device=device,
    )
    report.update(
        {
            "training_split": "train",
            "holdout_split": "validation",
            "train_examples_per_operator_per_cohort": train_examples_per_operator,
            "holdout_examples_per_operator": holdout_examples_per_operator,
            "training_model_cohorts": 2,
            "vocab_hash": tokenizer.vocab_hash,
        }
    )
    return controller, report


def _generate_self_tuned_operator(
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
    if _ACTIVE_CONTROLLER is None:
        raise RuntimeError("self-tuned controller is not fitted")
    predicted_scale = _ACTIVE_CONTROLLER.predicted_scale(prompt, device=device)
    previous_controller = learned._ACTIVE_CONTROLLER
    previous_strength = os.environ.get(learned.ENV_CONTROLLER_STRENGTH)
    learned._ACTIVE_CONTROLLER = _ACTIVE_CONTROLLER
    os.environ[learned.ENV_CONTROLLER_STRENGTH] = repr(predicted_scale)
    try:
        generated, diagnostics = learned._generate_learned_operator(
            base=base,
            units=units,
            mixer=mixer,
            candidate=candidate,
            operator=operator,
            prompt=prompt,
            eos_id=eos_id,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    finally:
        learned._ACTIVE_CONTROLLER = previous_controller
        if previous_strength is None:
            os.environ.pop(learned.ENV_CONTROLLER_STRENGTH, None)
        else:
            os.environ[learned.ENV_CONTROLLER_STRENGTH] = previous_strength
    diagnostics["controller_predicted_scale"] = predicted_scale
    return generated, diagnostics


def composition_prompt_control_metrics(
    controller: SelfTunedPromptController,
    cohort: Cohort,
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    correct = 0
    count = 0
    matching_probability = 0.0
    scales: list[float] = []
    per_operator_scales = {operator: [] for operator in FUNCTIONAL_OPERATORS}
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
                    probabilities, scale = controller.prompt_control(prompt, device=device)
                    expected = FUNCTIONAL_OPERATORS.index(operator)
                    correct += int(int(probabilities.argmax().item()) == expected)
                    matching_probability += float(probabilities[expected].detach().cpu())
                    scale_value = float(scale.detach().cpu())
                    scales.append(scale_value)
                    per_operator_scales[operator].append(scale_value)
                    count += 1
    return {
        "examples": count,
        "accuracy": correct / max(1, count),
        "mean_matching_probability": matching_probability / max(1, count),
        "mean_predicted_scale": sum(scales) / max(1, len(scales)),
        "min_predicted_scale": min(scales) if scales else None,
        "max_predicted_scale": max(scales) if scales else None,
        "per_operator_mean_scale": {
            operator: sum(values) / max(1, len(values))
            for operator, values in per_operator_scales.items()
        },
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
    controller_max_positions_per_cohort: int,
    controller_seed: int,
    controller_holdout_seed: int,
    controller_embedding_size: int,
    controller_hidden_size: int,
    controller_learning_rate: float,
    controller_steps: int,
    controller_batch_positions: int,
    auxiliary_operator_weight: float,
    scale_l2_weight: float,
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
        mixer=mixer,
        train_examples_per_operator=controller_train_examples_per_operator,
        holdout_examples_per_operator=controller_holdout_examples_per_operator,
        max_positions_per_cohort=controller_max_positions_per_cohort,
        controller_seed=controller_seed,
        holdout_seed=controller_holdout_seed,
        embedding_size=controller_embedding_size,
        hidden_size=controller_hidden_size,
        learning_rate=controller_learning_rate,
        steps=controller_steps,
        batch_positions=controller_batch_positions,
        auxiliary_operator_weight=auxiliary_operator_weight,
        scale_l2_weight=scale_l2_weight,
        device=device,
    )
    _ACTIVE_CONTROLLER = controller
    original_generator = oracle._generate_oracle_operator
    oracle._generate_oracle_operator = _generate_self_tuned_operator
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

    composition_control = composition_prompt_control_metrics(
        controller,
        cohorts[0],
        root=root,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        device=device,
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_self_tuned_operator_state_sequential_composition",
        "claim_boundary": (
            "controller learns operator direction and continuous prior scale from standalone train traces only; "
            "composition examples and composition labels are excluded from controller fitting; "
            "stage boundaries remain external and generation resets at each stage; explicit operator tokens remain visible; "
            "NEG, learned boundaries, single-pass nesting, final IID test, and OOD are untested"
        ),
        "operators": list(FUNCTIONAL_OPERATORS),
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "manual_controller_strength": None,
        "data_seed": data_seed,
        "calibration_seed": calibration_seed,
        "controller_seed": controller_seed,
        "controller_holdout_seed": controller_holdout_seed,
        "fixed_candidate": sequential.confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": mixer_fit,
        "controller_fit": controller_fit,
        "composition_prompt_controller": composition_control,
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
        description="Learn both operator direction and fusion-prior strength from standalone traces"
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
    parser.add_argument("--controller-train-examples-per-operator", type=int, default=24)
    parser.add_argument("--controller-holdout-examples-per-operator", type=int, default=32)
    parser.add_argument("--controller-max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--controller-seed", type=int, default=DEFAULT_CONTROLLER_SEED)
    parser.add_argument("--controller-holdout-seed", type=int, default=DEFAULT_CONTROLLER_HOLDOUT_SEED)
    parser.add_argument("--controller-embedding-size", type=int, default=16)
    parser.add_argument("--controller-hidden-size", type=int, default=16)
    parser.add_argument("--controller-learning-rate", type=float, default=0.02)
    parser.add_argument("--controller-steps", type=int, default=400)
    parser.add_argument("--controller-batch-positions", type=int, default=256)
    parser.add_argument("--auxiliary-operator-weight", type=float, default=0.25)
    parser.add_argument("--scale-l2-weight", type=float, default=0.0001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/self_tuned_operator_controller/summary.json")
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
            controller_max_positions_per_cohort=args.controller_max_positions_per_cohort,
            controller_seed=args.controller_seed,
            controller_holdout_seed=args.controller_holdout_seed,
            controller_embedding_size=args.controller_embedding_size,
            controller_hidden_size=args.controller_hidden_size,
            controller_learning_rate=args.controller_learning_rate,
            controller_steps=args.controller_steps,
            controller_batch_positions=args.controller_batch_positions,
            auxiliary_operator_weight=args.auxiliary_operator_weight,
            scale_l2_weight=args.scale_l2_weight,
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
