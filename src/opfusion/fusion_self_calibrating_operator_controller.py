from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
SOURCE_COUNT = len(EXPERIMENT_OPERATORS) + 1
DEFAULT_CONTROLLER_SEED = 739_000
DEFAULT_CONTROLLER_HOLDOUT_SEED = 739_500
ENV_CONTROLLER_MODE = "OPFUSION_SELF_CALIBRATING_MODE"
_ACTIVE_CONTROLLER: "SelfCalibratingController | None" = None


@dataclass(frozen=True)
class TeacherForcedTrace:
    prompt: tuple[int, ...]
    operator_index: int
    combined_states: torch.Tensor
    target_source_probabilities: torch.Tensor


class SelfCalibratingController(nn.Module):
    """Predict gauge-fixed log-prior offsets for all fusion sources.

    The controller is not given an operator label. It observes only the stage
    prompt and emits one additive log-weight offset for Base and every
    specialist source, including NEG. Offsets are mean-centered so the
    otherwise-unidentifiable common additive constant is removed.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_size: int = 16,
        hidden_size: int = 16,
        source_count: int = SOURCE_COUNT,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 1 or embedding_size <= 0 or hidden_size <= 0:
            raise ValueError("invalid controller dimensions")
        if source_count <= 1:
            raise ValueError("source_count must exceed one")
        self.source_count = int(source_count)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.backbone = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
        )
        self.source_head = nn.Linear(hidden_size, self.source_count)
        # Start exactly at the no-prior control. Symmetry is broken by the
        # learned prompt representation and source-dependent likelihood signal.
        nn.init.zeros_(self.source_head.weight)
        nn.init.zeros_(self.source_head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("controller inputs must have matching [batch, time] shape")
        embedded = self.embedding(input_ids)
        mask = attention_mask.to(embedded.dtype).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        raw = self.source_head(self.backbone(pooled))
        return raw - raw.mean(dim=-1, keepdim=True)

    def source_prior(self, prompt: Sequence[int], *, device: torch.device) -> torch.Tensor:
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return self(ids, mask).squeeze(0)


def _pad_prompts(
    traces: Sequence[TeacherForcedTrace],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not traces:
        raise ValueError("teacher-forced traces must not be empty")
    width = max(len(trace.prompt) for trace in traces)
    ids = torch.full((len(traces), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, trace in enumerate(traces):
        values = torch.tensor(trace.prompt, dtype=torch.long, device=device)
        ids[row, : values.numel()] = values
        mask[row, : values.numel()] = True
    return ids, mask


def _flatten_trace_positions(
    traces: Sequence[TeacherForcedTrace],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    owners: list[torch.Tensor] = []
    for index, trace in enumerate(traces):
        if trace.combined_states.ndim != 2 or trace.combined_states.shape[1] != SOURCE_COUNT:
            raise ValueError("combined_states must have shape [positions, sources]")
        if trace.target_source_probabilities.shape != trace.combined_states.shape:
            raise ValueError("target source probabilities must match combined_states")
        if trace.combined_states.shape[0] == 0:
            continue
        states.append(trace.combined_states.to(device=device, dtype=torch.float32))
        probabilities.append(
            trace.target_source_probabilities.to(device=device, dtype=torch.float32)
        )
        owners.append(
            torch.full(
                (trace.combined_states.shape[0],),
                index,
                dtype=torch.long,
                device=device,
            )
        )
    if not states:
        raise ValueError("teacher-forced traces contain no positions")
    return torch.cat(states, dim=0), torch.cat(probabilities, dim=0), torch.cat(owners, dim=0)


def teacher_forced_metrics(
    controller: SelfCalibratingController,
    traces: Sequence[TeacherForcedTrace],
    *,
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    ids, mask = _pad_prompts(traces, pad_id=pad_id, device=device)
    states, target_probabilities, owners = _flatten_trace_positions(traces, device=device)
    controller.eval()
    with torch.no_grad():
        priors = controller(ids, mask)
        position_priors = priors.index_select(0, owners)
        weights = torch.softmax(states + position_priors, dim=-1)
        mixture_probability = (weights * target_probabilities).sum(dim=-1).clamp_min(1e-12)
        nll = -mixture_probability.log().mean()
        matching = []
        argmax_correct = []
        margins = []
        per_operator: dict[str, list[float]] = {operator: [] for operator in FUNCTIONAL_OPERATORS}
        for row, trace in enumerate(traces):
            operator = FUNCTIONAL_OPERATORS[trace.operator_index]
            source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
            prior = priors[row]
            matching.append(float(prior[source_index].detach().cpu()))
            prediction = int(prior.argmax().item())
            correct = float(prediction == source_index)
            argmax_correct.append(correct)
            per_operator[operator].append(correct)
            other = torch.cat([prior[:source_index], prior[source_index + 1 :]])
            margins.append(float((prior[source_index] - other.max()).detach().cpu()))
    return {
        "traces": len(traces),
        "positions": int(states.shape[0]),
        "token_nll": float(nll.detach().cpu()),
        "mean_abs_prior": float(priors.abs().mean().detach().cpu()),
        "mean_prior_l2": float(priors.square().mean().detach().cpu()),
        "mean_matching_prior": sum(matching) / max(1, len(matching)),
        "mean_matching_margin": sum(margins) / max(1, len(margins)),
        "prior_argmax_matching_accuracy": sum(argmax_correct) / max(1, len(argmax_correct)),
        "per_operator_prior_argmax_accuracy": {
            operator: sum(values) / max(1, len(values))
            for operator, values in per_operator.items()
        },
    }


def fit_self_calibrating_controller(
    *,
    traces: Sequence[TeacherForcedTrace],
    holdout_traces: Sequence[TeacherForcedTrace],
    vocabulary_size: int,
    pad_id: int,
    embedding_size: int,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    prior_l2_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[SelfCalibratingController, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    controller = SelfCalibratingController(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        source_count=SOURCE_COUNT,
    ).to(device)
    ids, mask = _pad_prompts(traces, pad_id=pad_id, device=device)
    states, target_probabilities, owners = _flatten_trace_positions(traces, device=device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    losses: list[float] = []
    nlls: list[float] = []
    controller.train()
    for _ in range(steps):
        priors = controller(ids, mask)
        weights = torch.softmax(states + priors.index_select(0, owners), dim=-1)
        mixture_probability = (weights * target_probabilities).sum(dim=-1).clamp_min(1e-12)
        nll = -mixture_probability.log().mean()
        regularizer = priors.square().mean()
        loss = nll + float(prior_l2_weight) * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        nlls.append(float(nll.detach().cpu()))
    controller.eval()
    return controller, {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "prior_l2_weight": prior_l2_weight,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "optimization_first": losses[0] if losses else None,
        "optimization_last": losses[-1] if losses else None,
        "token_nll_first": nlls[0] if nlls else None,
        "token_nll_last": nlls[-1] if nlls else None,
        "train_metrics": teacher_forced_metrics(
            controller, traces, pad_id=pad_id, device=device
        ),
        "holdout_metrics": teacher_forced_metrics(
            controller, holdout_traces, pad_id=pad_id, device=device
        ),
    }


def combine_self_calibrating_states(
    fast_state: torch.Tensor,
    slow_state: torch.Tensor,
    *,
    candidate,
    source_prior: torch.Tensor,
) -> torch.Tensor:
    if fast_state.shape != slow_state.shape:
        raise ValueError("fast_state and slow_state must have the same shape")
    if source_prior.shape != fast_state.shape:
        raise ValueError("source_prior must match state shape")
    combined = (
        (1.0 - float(candidate.slow_mix)) * fast_state
        + float(candidate.slow_mix) * slow_state
    ) / float(candidate.temperature)
    return torch.softmax(combined + source_prior, dim=-1)


def _collect_trace_for_example(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    candidate,
    prompt: Sequence[int],
    expected: Sequence[int],
    operator_index: int,
    device: torch.device,
) -> TeacherForcedTrace:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    fast_state: torch.Tensor | None = None
    slow_state: torch.Tensor | None = None
    combined_rows: list[torch.Tensor] = []
    probability_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for next_id in expected:
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
            source_probabilities = torch.softmax(sources.float(), dim=-1)[:, int(next_id)]
            combined_rows.append(combined.detach().clone())
            probability_rows.append(source_probabilities.detach().clone())

            if candidate.feedback > 0.0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, int(next_id)]
                token_support = token_support - token_support.mean()
                fast_state = fast_state + float(candidate.feedback) * token_support
                slow_state = slow_state + 0.25 * float(candidate.feedback) * token_support
            ids = torch.cat(
                [ids, torch.tensor([[int(next_id)]], dtype=torch.long, device=device)], dim=1
            )
    return TeacherForcedTrace(
        prompt=tuple(int(value) for value in prompt),
        operator_index=int(operator_index),
        combined_states=torch.stack(combined_rows, dim=0),
        target_source_probabilities=torch.stack(probability_rows, dim=0),
    )


def collect_teacher_forced_traces(
    cohort: Cohort,
    *,
    root: Path,
    mixer: torch.nn.Module,
    examples_per_operator: int,
    seed: int,
    split: str,
    device: torch.device,
) -> tuple[list[TeacherForcedTrace], FixedVocabTokenizer]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    candidate = sequential.confirmatory.candidate_grid()[1]
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    traces: list[TeacherForcedTrace] = []
    for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
        for sample_index in range(examples_per_operator):
            prompt, expected, _, actual_operator = factory.prompt_and_expected_ids(
                operator,
                seed=seed,
                split=split,
                step=operator_index,
                sample_index=sample_index,
            )
            if actual_operator != operator:
                raise AssertionError("standalone trace operator mismatch")
            traces.append(
                _collect_trace_for_example(
                    base=base,
                    units=units,
                    mixer=mixer,
                    candidate=candidate,
                    prompt=prompt,
                    expected=expected,
                    operator_index=operator_index,
                    device=device,
                )
            )
    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return traces, tokenizer


def _generate_self_calibrating_operator(
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
    """Oracle-compatible generator; operator is diagnostic only."""
    if _ACTIVE_CONTROLLER is None:
        raise RuntimeError("self-calibrating controller is not fitted")
    mode = os.environ.get(ENV_CONTROLLER_MODE, "learned")
    if mode not in {"learned", "zero"}:
        raise ValueError(f"unsupported self-calibrating controller mode: {mode}")
    learned_prior = _ACTIVE_CONTROLLER.source_prior(prompt, device=device)
    source_prior = learned_prior if mode == "learned" else torch.zeros_like(learned_prior)
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

            weights = combine_self_calibrating_states(
                fast_state,
                slow_state,
                candidate=candidate,
                source_prior=source_prior,
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

    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
        "mean_timescale_gap": timescale_gap_sum / max(1, positions),
        "mean_oracle_source_weight": matching_weight_sum / max(1, positions),
        "controller_prior_l2": float(source_prior.square().mean().detach().cpu()),
    }


def composition_prompt_prior_metrics(
    controller: SelfCalibratingController,
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
    matching_prior = 0.0
    matching_margin = 0.0
    prior_l2 = 0.0
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
                    prior = controller.source_prior(prompt, device=device)
                    source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
                    correct += int(int(prior.argmax().item()) == source_index)
                    matching_prior += float(prior[source_index].detach().cpu())
                    other = torch.cat([prior[:source_index], prior[source_index + 1 :]])
                    matching_margin += float((prior[source_index] - other.max()).detach().cpu())
                    prior_l2 += float(prior.square().mean().detach().cpu())
                    count += 1
    return {
        "examples": count,
        "prior_argmax_matching_accuracy": correct / max(1, count),
        "mean_matching_prior": matching_prior / max(1, count),
        "mean_matching_margin": matching_margin / max(1, count),
        "mean_prior_l2": prior_l2 / max(1, count),
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
    controller_prior_l2_weight: float,
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

    train_trace_sets: list[list[TeacherForcedTrace]] = []
    train_tokenizers: list[FixedVocabTokenizer] = []
    for cohort in cohorts[:2]:
        traces, tokenizer = collect_teacher_forced_traces(
            cohort,
            root=root,
            mixer=mixer,
            examples_per_operator=controller_train_examples_per_operator,
            seed=controller_seed,
            split="train",
            device=device,
        )
        train_trace_sets.append(traces)
        train_tokenizers.append(tokenizer)
    holdout_traces, holdout_tokenizer = collect_teacher_forced_traces(
        cohorts[2],
        root=root,
        mixer=mixer,
        examples_per_operator=controller_holdout_examples_per_operator,
        seed=controller_holdout_seed,
        split="validation",
        device=device,
    )
    all_tokenizers = [*train_tokenizers, holdout_tokenizer]
    if len({tokenizer.vocab_hash for tokenizer in all_tokenizers}) != 1:
        raise RuntimeError("self-calibrating experiment requires identical tokenizers")
    tokenizer = train_tokenizers[0]
    train_traces = [trace for rows in train_trace_sets for trace in rows]
    controller, controller_fit = fit_self_calibrating_controller(
        traces=train_traces,
        holdout_traces=holdout_traces,
        vocabulary_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_size=controller_embedding_size,
        hidden_size=controller_hidden_size,
        learning_rate=controller_learning_rate,
        steps=controller_steps,
        prior_l2_weight=controller_prior_l2_weight,
        seed=controller_seed,
        device=device,
    )
    controller_fit.update(
        {
            "training_split": "train",
            "training_model_cohorts": 2,
            "train_examples_per_operator_per_cohort": controller_train_examples_per_operator,
            "holdout_split": "validation",
            "holdout_model_cohorts": 1,
            "holdout_examples_per_operator": controller_holdout_examples_per_operator,
            "vocab_hash": tokenizer.vocab_hash,
            "operator_labels_used_for_optimization": False,
        }
    )
    _ACTIVE_CONTROLLER = controller
    original_generator = oracle._generate_oracle_operator
    oracle._generate_oracle_operator = _generate_self_calibrating_operator
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

    mode = os.environ.get(ENV_CONTROLLER_MODE, "learned")
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_self_calibrating_operator_state_sequential_composition",
        "claim_boundary": (
            "controller optimization uses standalone teacher-forced token likelihood only; "
            "no composition examples or operator labels enter the controller objective; "
            "explicit operator tokens remain present in prompts; stage boundaries remain external "
            "and generation resets at each stage; final IID test, OOD, learned boundaries, "
            "single-pass nesting, arbitrary depth, loops, and branches are untested"
        ),
        "operators": list(FUNCTIONAL_OPERATORS),
        "source_order": ["base", *EXPERIMENT_OPERATORS],
        "controller_mode": mode,
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "data_seed": data_seed,
        "calibration_seed": calibration_seed,
        "controller_seed": controller_seed,
        "controller_holdout_seed": controller_holdout_seed,
        "fixed_candidate": sequential.confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": mixer_fit,
        "controller_fit": controller_fit,
        "composition_prompt_prior": composition_prompt_prior_metrics(
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
        description="Learn source log-prior calibration from standalone token likelihood and test zero-shot composition"
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
    parser.add_argument("--controller-holdout-examples-per-operator", type=int, default=16)
    parser.add_argument("--controller-seed", type=int, default=DEFAULT_CONTROLLER_SEED)
    parser.add_argument("--controller-holdout-seed", type=int, default=DEFAULT_CONTROLLER_HOLDOUT_SEED)
    parser.add_argument("--controller-embedding-size", type=int, default=16)
    parser.add_argument("--controller-hidden-size", type=int, default=16)
    parser.add_argument("--controller-learning-rate", type=float, default=0.03)
    parser.add_argument("--controller-steps", type=int, default=300)
    parser.add_argument("--controller-prior-l2-weight", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="evaluations/self_calibrating_operator_controller/summary.json"
    )
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
            controller_prior_l2_weight=args.controller_prior_l2_weight,
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
    print(json.dumps(report["composition_prompt_prior"], sort_keys=True))
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
