from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from opfusion.fusion_compose import _aggregate_reports, _ranking_key, fixed_compose
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_verify import (
    _dataset,
    _empty_gold_counter,
    _finalize_gold_counter,
    _next_logits,
    _update_gold_counter,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory, TrainingExample


DEFAULT_CALIBRATION_SEED = 711_000
DEFAULT_HOLDOUT_SEED = 711_500
DEFAULT_VERIFICATION_SEED = 712_000
FIXED_BASELINES = ("raw_sum", "bias_mean")


@dataclass(frozen=True)
class TrieNodeRecord:
    prefix: tuple[int, ...]
    valid_next: tuple[int, ...]


@dataclass(frozen=True)
class SparseValidBatch:
    base_logits: torch.Tensor
    unit_logits: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.base_logits.shape[0])


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.valid_next: set[int] = set()


def _reducer(operator: str):
    if operator == "aggregation.sum":
        return lambda left, right: left + right
    if operator == "scalar.min":
        return min
    if operator == "scalar.max":
        return max
    raise KeyError(operator)


def valid_state_paths(example: TrainingExample) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate every semantically valid state path from the prompt state.

    ADD and NEG are deterministic. SUM/MIN/MAX may reduce any adjacent pair, so
    every valid reduction order is retained. The returned paths include the
    prompt state as their first element.
    """

    start = tuple(example.prompt_state_values)
    operator = example.operator_id
    if example.task == "terminal_stop":
        return ((start,),)
    if operator == "scalar.add":
        if len(start) != 2:
            raise ValueError("ADD prompt state must contain two values")
        return ((start, (start[0] + start[1],)),)
    if operator == "scalar.neg":
        if len(start) != 1:
            raise ValueError("NEG prompt state must contain one value")
        return ((start, (-start[0],)),)
    if operator not in {"aggregation.sum", "scalar.min", "scalar.max"}:
        raise KeyError(operator)

    reduce_pair = _reducer(operator)

    @lru_cache(maxsize=None)
    def descend(state: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
        if len(state) == 1:
            return ((state,),)
        paths: list[tuple[tuple[int, ...], ...]] = []
        for index in range(len(state) - 1):
            reduced = reduce_pair(state[index], state[index + 1])
            next_state = (*state[:index], reduced, *state[index + 2 :])
            for suffix in descend(tuple(next_state)):
                paths.append((state, *suffix))
        return tuple(dict.fromkeys(paths))

    return descend(start)


def valid_response_sequences(
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    example: TrainingExample,
) -> tuple[tuple[int, ...], ...]:
    sequences: list[tuple[int, ...]] = []
    for path in valid_state_paths(example):
        response_tokens = factory._response_for_states(example.operator_id, path[1:])
        ids = tokenizer.encode_tokens(response_tokens, add_bos=False, add_eos=True)
        sequences.append(tuple(ids))
    return tuple(dict.fromkeys(sequences))


def build_valid_prefix_records(
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    example: TrainingExample,
) -> tuple[TrieNodeRecord, ...]:
    root = _TrieNode()
    for sequence in valid_response_sequences(factory, tokenizer, example):
        node = root
        for token in sequence:
            node.valid_next.add(int(token))
            node = node.children.setdefault(int(token), _TrieNode())

    rows: list[TrieNodeRecord] = []

    def visit(node: _TrieNode, prefix: tuple[int, ...]) -> None:
        if node.valid_next:
            rows.append(TrieNodeRecord(prefix=prefix, valid_next=tuple(sorted(node.valid_next))))
        for token, child in sorted(node.children.items()):
            visit(child, (*prefix, token))

    visit(root, ())
    return tuple(rows)


def sample_prefix_records(
    records: Sequence[TrieNodeRecord],
    *,
    maximum: int,
    seed: int,
) -> tuple[TrieNodeRecord, ...]:
    if maximum <= 0 or len(records) <= maximum:
        return tuple(records)
    root = [row for row in records if not row.prefix]
    branch = [row for row in records if len(row.valid_next) > 1]
    terminal = [row for row in records if len(row.valid_next) == 1 and row.valid_next[0] not in row.prefix[-1:]]
    selected: list[TrieNodeRecord] = []
    seen: set[tuple[int, ...]] = set()
    for row in (*root, *branch):
        if row.prefix not in seen and len(selected) < maximum:
            selected.append(row)
            seen.add(row.prefix)
    remaining = [row for row in records if row.prefix not in seen]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    # Bias the residual sample toward deeper prefixes so full trajectories are represented.
    remaining.sort(key=lambda row: len(row.prefix), reverse=True)
    for row in (*terminal, *remaining):
        if row.prefix not in seen and len(selected) < maximum:
            selected.append(row)
            seen.add(row.prefix)
    return tuple(selected)


class SparseEvidenceCompositor(nn.Module):
    """All-unit sparse logit compositor with no operator labels or routing signal.

    Every specialist is evaluated at every position. A shared evidence network
    maps only simultaneous logit-derived features to continuous confidences in
    [0, 1]. There is no positive floor. Raw centered fields are soft-thresholded;
    they are never RMS-equalized or promoted to a common magnitude.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 12,
        use_confidence: bool = True,
        allow_threshold: bool = True,
    ) -> None:
        super().__init__()
        self.use_confidence = bool(use_confidence)
        self.allow_threshold = bool(allow_threshold)
        self.raw_alpha = nn.Parameter(torch.tensor(-1.5077718))  # softplus ~= 0.2
        if self.allow_threshold:
            self.raw_threshold = nn.Parameter(torch.tensor(-2.2521685))  # softplus ~= 0.1
        else:
            self.register_buffer("raw_threshold", torch.tensor(float("-inf")), persistent=True)
        if self.use_confidence:
            self.score_network = nn.Sequential(
                nn.Linear(6, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.score_network = None

    @staticmethod
    def centered_biases(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        if unit_logits.shape[-2] != len(EXPERIMENT_OPERATORS):
            raise ValueError("all five specialist fields are required")
        if unit_logits.shape[:-2] != base_logits.shape[:-1] or unit_logits.shape[-1] != base_logits.shape[-1]:
            raise ValueError("base and unit logit shapes are incompatible")
        bias = unit_logits - base_logits.unsqueeze(-2)
        return bias - bias.mean(dim=-1, keepdim=True)

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top = logits.float().topk(k=2, dim=-1).values
        return top[..., 0] - top[..., 1]

    def evidence_features(
        self,
        base_logits: torch.Tensor,
        unit_logits: torch.Tensor,
        centered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rms = centered.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        normalized = centered / rms.unsqueeze(-1).to(centered.dtype)
        consensus = normalized.mean(dim=-2, keepdim=True)
        consensus_rms = consensus.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        cosine = (normalized * consensus).float().mean(dim=-1) / consensus_rms.squeeze(-2)

        base_prob = torch.softmax(base_logits.float(), dim=-1)
        unit_prob = torch.softmax(unit_logits.float(), dim=-1)
        base_entropy = -(base_prob * base_prob.clamp_min(1e-9).log()).sum(dim=-1)
        unit_entropy = -(unit_prob * unit_prob.clamp_min(1e-9).log()).sum(dim=-1)
        entropy_gain = base_entropy.unsqueeze(-1) - unit_entropy
        margin_gain = self._margin(unit_logits) - self._margin(base_logits).unsqueeze(-1)

        features = torch.stack(
            [
                rms.log(),
                normalized.amax(dim=-1),
                -normalized.amin(dim=-1),
                entropy_gain,
                margin_gain,
                cosine,
            ],
            dim=-1,
        )
        return features, rms

    def compose(
        self,
        base_logits: torch.Tensor,
        unit_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = self.centered_biases(base_logits, unit_logits)
        features, rms = self.evidence_features(base_logits, unit_logits, centered)
        if self.score_network is None:
            confidence = torch.ones_like(rms)
        else:
            confidence = torch.sigmoid(self.score_network(features).squeeze(-1))

        threshold_ratio = F.softplus(self.raw_threshold) if self.allow_threshold else centered.new_tensor(0.0)
        threshold = threshold_ratio * rms
        shrunk = centered.sign() * F.relu(centered.abs() - threshold.unsqueeze(-1).to(centered.dtype))
        residual = (confidence.unsqueeze(-1).to(shrunk.dtype) * shrunk).sum(dim=-2)
        alpha = F.softplus(self.raw_alpha) + 1e-5

        # Bound only the aggregate residual. Individual weak fields are not rescaled.
        cap = (4.0 * rms.median(dim=-1).values).clamp_min(1e-4).unsqueeze(-1).to(residual.dtype)
        bounded = cap * torch.tanh(alpha * residual / cap)
        fused = base_logits + bounded
        active_fraction = (shrunk.abs() > 0).float().mean(dim=-1)
        return fused, confidence, active_fraction, threshold_ratio

    def forward(self, base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        return self.compose(base_logits, unit_logits)[0]


def valid_set_loss(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have the same shape")
    if not bool(valid_mask.any(dim=-1).all()):
        raise ValueError("every position needs at least one valid next token")
    log_probs = F.log_softmax(logits, dim=-1)
    valid_log_mass = torch.logsumexp(log_probs.masked_fill(~valid_mask, float("-inf")), dim=-1)
    return -valid_log_mass.mean()


def batch_metrics(model: SparseEvidenceCompositor, batch: SparseValidBatch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        fused, confidence, active_fraction, threshold_ratio = model.compose(batch.base_logits, batch.unit_logits)
        loss = valid_set_loss(fused, batch.valid_mask)
        predictions = fused.argmax(dim=-1)
        correct = batch.valid_mask.gather(1, predictions.unsqueeze(-1)).squeeze(-1)
        valid_mass = (torch.softmax(fused.float(), dim=-1) * batch.valid_mask.float()).sum(dim=-1)
    return {
        "valid_set_nll": float(loss.detach().cpu()),
        "valid_top1_accuracy": float(correct.float().mean().detach().cpu()),
        "mean_valid_probability_mass": float(valid_mass.mean().detach().cpu()),
        "mean_confidence": float(confidence.mean().detach().cpu()),
        "mean_active_coordinate_fraction": float(active_fraction.mean().detach().cpu()),
        "threshold_ratio": float(threshold_ratio.detach().cpu()),
        "alpha": float((F.softplus(model.raw_alpha) + 1e-5).detach().cpu()),
    }


def export_parameters(model: SparseEvidenceCompositor) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def load_parameters(model: SparseEvidenceCompositor, values: Mapping[str, Any]) -> None:
    state = model.state_dict()
    for name, raw in values.items():
        if name not in state:
            raise KeyError(name)
        state[name] = torch.tensor(raw, dtype=state[name].dtype, device=state[name].device)
    model.load_state_dict(state)


def _collect_valid_batch(
    cohort: Cohort,
    *,
    root: Path,
    examples_per_operator: int,
    data_seed: int,
    max_prefixes_per_example: int,
    max_positions: int,
    device: torch.device,
) -> tuple[SparseValidBatch, dict[str, Any]]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }

    base_rows: list[torch.Tensor] = []
    unit_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    branch_positions = 0
    path_count = 0
    with torch.no_grad():
        for operator_index, operator in enumerate(EXPERIMENT_OPERATORS):
            for sample_index in range(examples_per_operator):
                example = factory.training_example(
                    operator,
                    seed=data_seed,
                    split="validation",
                    step=operator_index,
                    sample_index=sample_index,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                sequences = valid_response_sequences(factory, tokenizer, example)
                path_count += len(sequences)
                records = build_valid_prefix_records(factory, tokenizer, example)
                sampled = sample_prefix_records(
                    records,
                    maximum=max_prefixes_per_example,
                    seed=data_seed + 1009 * operator_index + sample_index,
                )
                branch_positions += sum(int(len(row.valid_next) > 1) for row in sampled)
                for row in sampled:
                    ids = torch.tensor([prompt + list(row.prefix)], dtype=torch.long, device=device)
                    base_logits = _next_logits(base, ids)
                    unit_logits = torch.stack(
                        [_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS],
                        dim=0,
                    )
                    mask = torch.zeros(len(tokenizer.tokens), dtype=torch.bool, device=device)
                    mask[list(row.valid_next)] = True
                    base_rows.append(base_logits.unsqueeze(0))
                    unit_rows.append(unit_logits.unsqueeze(0))
                    mask_rows.append(mask.unsqueeze(0))

    base_tensor = torch.cat(base_rows, dim=0)
    unit_tensor = torch.cat(unit_rows, dim=0)
    mask_tensor = torch.cat(mask_rows, dim=0)
    if max_positions > 0 and int(base_tensor.shape[0]) > max_positions:
        generator = torch.Generator(device="cpu").manual_seed(data_seed + int(cohort.metadata.get("seed", 0)))
        indices = torch.randperm(int(base_tensor.shape[0]), generator=generator)[:max_positions].to(device)
        base_tensor = base_tensor.index_select(0, indices)
        unit_tensor = unit_tensor.index_select(0, indices)
        mask_tensor = mask_tensor.index_select(0, indices)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return SparseValidBatch(base_tensor, unit_tensor, mask_tensor), {
        "cohort_id": cohort.cohort_id,
        "positions": int(base_tensor.shape[0]),
        "sampled_branch_positions": branch_positions,
        "enumerated_valid_paths": path_count,
    }


def _merge_batches(batches: Sequence[SparseValidBatch]) -> SparseValidBatch:
    return SparseValidBatch(
        base_logits=torch.cat([batch.base_logits for batch in batches], dim=0),
        unit_logits=torch.cat([batch.unit_logits for batch in batches], dim=0),
        valid_mask=torch.cat([batch.valid_mask for batch in batches], dim=0),
    )


def fit_sparse_compositor(
    *,
    batch: SparseValidBatch,
    hidden_size: int,
    use_confidence: bool,
    allow_threshold: bool,
    steps: int,
    batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    confidence_penalty: float,
    seed: int,
) -> tuple[SparseEvidenceCompositor, dict[str, Any]]:
    model = SparseEvidenceCompositor(
        hidden_size=hidden_size,
        use_confidence=use_confidence,
        allow_threshold=allow_threshold,
    ).to(batch.base_logits.device)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(batch.positions, generator=generator)
    cursor = 0
    losses: list[float] = []

    model.train()
    for _ in range(steps):
        if cursor + batch_positions > batch.positions:
            permutation = torch.randperm(batch.positions, generator=generator)
            cursor = 0
        cpu_indices = permutation[cursor : cursor + min(batch_positions, batch.positions)]
        cursor += int(cpu_indices.numel())
        indices = cpu_indices.to(batch.base_logits.device)
        fused, confidence, _, _ = model.compose(
            batch.base_logits.index_select(0, indices),
            batch.unit_logits.index_select(0, indices),
        )
        objective = valid_set_loss(fused, batch.valid_mask.index_select(0, indices))
        regularizer = sum(
            (parameter - initial[name]).float().pow(2).mean()
            for name, parameter in model.named_parameters()
        )
        loss = objective + float(l2_weight) * regularizer + float(confidence_penalty) * confidence.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    report = {
        "steps": steps,
        "batch_positions": batch_positions,
        "learning_rate": learning_rate,
        "l2_weight": l2_weight,
        "confidence_penalty": confidence_penalty,
        "use_confidence": use_confidence,
        "allow_threshold": allow_threshold,
        "optimization_first": losses[0] if losses else None,
        "optimization_last": losses[-1] if losses else None,
        "train_metrics": batch_metrics(model, batch),
        "parameters": export_parameters(model),
    }
    return model, report


def _generate_sparse(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    model: SparseEvidenceCompositor,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    confidence_sum = 0.0
    active_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            fused, confidence, active, _ = model.compose(base_logits, unit_logits)
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            confidence_sum += float(confidence.mean().detach().cpu())
            active_sum += float(active.mean().detach().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_confidence": confidence_sum / max(1, positions),
        "mean_active_coordinate_fraction": active_sum / max(1, positions),
    }


def _generate_fixed(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    mode: str,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            fused = fixed_compose(base_logits, unit_logits, mode=mode)
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    learned_specs: Mapping[str, Mapping[str, Any]],
    examples_per_operator: int,
    verification_seed: int,
    max_new_tokens: int,
    hidden_size: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    dataset = _dataset(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=examples_per_operator,
        evaluation_seed=verification_seed,
    )
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    learned: dict[str, SparseEvidenceCompositor] = {}
    for name, spec in learned_specs.items():
        model = SparseEvidenceCompositor(
            hidden_size=hidden_size,
            use_confidence=bool(spec["use_confidence"]),
            allow_threshold=bool(spec["allow_threshold"]),
        ).to(device)
        load_parameters(model, spec["parameters"])
        model.eval()
        learned[name] = model

    methods = [*FIXED_BASELINES, *learned]
    composition_metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    diagnostics: dict[str, dict[str, Any]] = {method: {} for method in learned}
    for operator in EXPERIMENT_OPERATORS:
        counters = {method: _empty_gold_counter() for method in methods}
        confidence_acc = {method: 0.0 for method in learned}
        active_acc = {method: 0.0 for method in learned}
        diag_count = {method: 0 for method in learned}
        for example, prompt, expected in dataset[operator]:
            for method in FIXED_BASELINES:
                generated = _generate_fixed(
                    base=base,
                    units=units,
                    mode=method,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    counters[method],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
            for method, model in learned.items():
                generated, row = _generate_sparse(
                    base=base,
                    units=units,
                    model=model,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    counters[method],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
                confidence_acc[method] += row["mean_confidence"]
                active_acc[method] += row["mean_active_coordinate_fraction"]
                diag_count[method] += 1
        for method in methods:
            composition_metrics[method][operator] = _finalize_gold_counter(counters[method])
        for method in learned:
            diagnostics[method][operator] = {
                "mean_confidence": confidence_acc[method] / max(1, diag_count[method]),
                "mean_active_coordinate_fraction": active_acc[method] / max(1, diag_count[method]),
            }

    del base, units, learned
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "composition_metrics": composition_metrics,
        "sparsity_diagnostics": diagnostics,
    }


def search_sparse_valid_composition(
    *,
    root: Path,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    verification_examples_per_operator: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    hidden_size: int,
    calibration_seed: int,
    holdout_seed: int,
    verification_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    if len({calibration_seed, holdout_seed, verification_seed}) != 3:
        raise ValueError("calibration, holdout, and verification seeds must differ")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete fusion-factory cohorts, found {len(cohorts)}")
    calibration_cohorts = cohorts[:2]
    held_out_cohort = cohorts[2]

    calibration_batches: list[SparseValidBatch] = []
    collection_reports: list[dict[str, Any]] = []
    for cohort in calibration_cohorts:
        batch, collection = _collect_valid_batch(
            cohort,
            root=root,
            examples_per_operator=calibration_examples_per_operator,
            data_seed=calibration_seed,
            max_prefixes_per_example=max_prefixes_per_example,
            max_positions=max_positions_per_cohort,
            device=device,
        )
        calibration_batches.append(batch)
        collection_reports.append(collection)
    merged = _merge_batches(calibration_batches)
    holdout_batch, holdout_collection = _collect_valid_batch(
        held_out_cohort,
        root=root,
        examples_per_operator=calibration_examples_per_operator,
        data_seed=holdout_seed,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions=max_positions_per_cohort,
        device=device,
    )

    candidate_settings = {
        "valid_uniform_sparse": {
            "use_confidence": False,
            "allow_threshold": True,
            "confidence_penalty": 0.0,
        },
        "valid_shared_sparse": {
            "use_confidence": True,
            "allow_threshold": True,
            "confidence_penalty": 0.005,
        },
        "valid_shared_sparse_strong": {
            "use_confidence": True,
            "allow_threshold": True,
            "confidence_penalty": 0.02,
        },
        "valid_shared_dense": {
            "use_confidence": True,
            "allow_threshold": False,
            "confidence_penalty": 0.005,
        },
    }
    fit_reports: dict[str, Any] = {}
    for index, (name, setting) in enumerate(candidate_settings.items()):
        model, report = fit_sparse_compositor(
            batch=merged,
            hidden_size=hidden_size,
            use_confidence=bool(setting["use_confidence"]),
            allow_threshold=bool(setting["allow_threshold"]),
            steps=fit_steps,
            batch_positions=fit_batch_positions,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            confidence_penalty=float(setting["confidence_penalty"]),
            seed=calibration_seed + index,
        )
        report["holdout_metrics"] = batch_metrics(model, holdout_batch)
        fit_reports[name] = report
        del model

    ranked_holdout = sorted(
        fit_reports,
        key=lambda name: (
            -float(fit_reports[name]["holdout_metrics"]["valid_top1_accuracy"]),
            float(fit_reports[name]["holdout_metrics"]["valid_set_nll"]),
            float(fit_reports[name]["holdout_metrics"]["mean_confidence"]),
        ),
    )
    selected_names = ranked_holdout[:2]
    learned_specs = {
        name: {
            "use_confidence": candidate_settings[name]["use_confidence"],
            "allow_threshold": candidate_settings[name]["allow_threshold"],
            "parameters": fit_reports[name]["parameters"],
        }
        for name in selected_names
    }
    cohort_reports = [
        _evaluate_cohort(
            cohort,
            root=root,
            learned_specs=learned_specs,
            examples_per_operator=verification_examples_per_operator,
            verification_seed=verification_seed,
            max_new_tokens=max_new_tokens,
            hidden_size=hidden_size,
            device=device,
        )
        for cohort in cohorts
    ]
    methods = [*FIXED_BASELINES, *selected_names]
    aggregate = _aggregate_reports(cohort_reports, methods)
    ranked = sorted(aggregate, key=_ranking_key)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_verifier_aware_sparse_composition_search",
        "claim_boundary": "all units evaluated at every token; no task labels or router; final IID/OOD splits unopened",
        "composition_constraint": (
            "raw centered Base-relative fields; shared evidence network; no RMS equalization; no positive weight floor"
        ),
        "objective": "negative log probability mass over the complete valid-next-token set",
        "device": str(device),
        "calibration_seed": calibration_seed,
        "holdout_seed": holdout_seed,
        "verification_seed": verification_seed,
        "calibration_model_seeds": [cohort.metadata.get("seed") for cohort in calibration_cohorts],
        "held_out_model_seed": held_out_cohort.metadata.get("seed"),
        "calibration_examples_per_operator": calibration_examples_per_operator,
        "verification_examples_per_operator": verification_examples_per_operator,
        "max_prefixes_per_example": max_prefixes_per_example,
        "collection_reports": collection_reports,
        "holdout_collection": holdout_collection,
        "candidate_settings": candidate_settings,
        "fit_reports": fit_reports,
        "holdout_ranking": ranked_holdout,
        "autoregressive_selected_candidates": selected_names,
        "cohort_reports": cohort_reports,
        "aggregate_methods": aggregate,
        "ranked_methods": ranked,
        "recommended_validation_composition": ranked[0] if ranked else None,
        "passing_methods": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search verifier-aware sparse all-unit logit composition laws"
    )
    parser.add_argument("--calibration-examples-per-operator", type=int, default=4)
    parser.add_argument("--max-prefixes-per-example", type=int, default=32)
    parser.add_argument("--max-positions-per-cohort", type=int, default=768)
    parser.add_argument("--verification-examples-per-operator", type=int, default=12)
    parser.add_argument("--fit-steps", type=int, default=220)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_sparse_valid/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = search_sparse_valid_composition(
        root=root,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        max_prefixes_per_example=args.max_prefixes_per_example,
        max_positions_per_cohort=args.max_positions_per_cohort,
        verification_examples_per_operator=args.verification_examples_per_operator,
        fit_steps=args.fit_steps,
        fit_batch_positions=args.fit_batch_positions,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
        hidden_size=args.hidden_size,
        calibration_seed=args.calibration_seed,
        holdout_seed=args.holdout_seed,
        verification_seed=args.verification_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report.get("recommended_validation_composition"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
