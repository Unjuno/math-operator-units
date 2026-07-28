from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from opfusion.fusion_compose import (
    CalibrationBatch,
    _aggregate_reports,
    _collect_calibration_batch,
    _merge_batches,
    _ranking_key,
    fixed_compose,
)
from opfusion.fusion_eval import _load_model, _teacher_forced_logits
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_verify import (
    _dataset,
    _empty_gold_counter,
    _finalize_gold_counter,
    _generate_model,
    _next_logits,
    _update_gold_counter,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


DEFAULT_CALIBRATION_SEED = 707_000
DEFAULT_VERIFICATION_SEED = 708_000
FIXED_BASELINES = ("bias_mean", "rms_mean")
LEARNED_METHODS = ("geometry_teacher_forced", "geometry_on_policy")


class BiasGeometryCompositor(nn.Module):
    """Continuous all-unit composition derived only from simultaneous bias geometry.

    Every specialist is evaluated at every position and receives a positive
    weight floor. The module has no operator label, task classifier, subset mask,
    or discrete routing output.
    """

    def __init__(self, *, hidden_size: int = 8, weight_floor: float = 0.03) -> None:
        super().__init__()
        operator_count = len(EXPERIMENT_OPERATORS)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= weight_floor < 1.0 / operator_count:
            raise ValueError("weight_floor must be in [0, 1/operator_count)")
        self.weight_floor = float(weight_floor)
        self.unit_prior = nn.Parameter(torch.zeros(operator_count))
        self.score_network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.raw_alpha = nn.Parameter(torch.tensor(0.54132485))  # softplus ~= 1
        self.raw_conflict = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _centered_biases(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        if unit_logits.shape[-2] != len(EXPERIMENT_OPERATORS):
            raise ValueError("all five specialist fields are required")
        if unit_logits.shape[:-2] != base_logits.shape[:-1] or unit_logits.shape[-1] != base_logits.shape[-1]:
            raise ValueError("base and specialist logit shapes are incompatible")
        biases = unit_logits - base_logits.unsqueeze(-2)
        return biases - biases.mean(dim=-1, keepdim=True)

    def compose(
        self,
        base_logits: torch.Tensor,
        unit_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = self._centered_biases(base_logits, unit_logits)
        rms = centered.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        normalized = centered / rms.to(centered.dtype)
        consensus = normalized.mean(dim=-2, keepdim=True)
        consensus_norm = consensus.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        cosine = (normalized * consensus).float().mean(dim=-1) / consensus_norm.squeeze(-1)
        features = torch.stack(
            [
                rms.squeeze(-1).log(),
                normalized.amax(dim=-1),
                -normalized.amin(dim=-1),
                cosine,
            ],
            dim=-1,
        )
        scores = self.score_network(features).squeeze(-1) + self.unit_prior
        soft_weights = torch.softmax(scores, dim=-1)
        floor = self.weight_floor
        weights = floor + (1.0 - floor * len(EXPERIMENT_OPERATORS)) * soft_weights
        signal = (weights.unsqueeze(-1) * normalized).sum(dim=-2)
        disagreement = (weights.unsqueeze(-1) * (normalized - signal.unsqueeze(-2)).pow(2)).sum(dim=-2)
        scale = rms.squeeze(-1).median(dim=-1).values.unsqueeze(-1).to(centered.dtype)
        alpha = F.softplus(self.raw_alpha) + 1e-4
        conflict = F.softplus(self.raw_conflict)
        corrected = alpha * signal / (1.0 + conflict * disagreement)
        residual = scale * (4.0 * torch.tanh(corrected / 4.0))
        return base_logits + residual, weights, disagreement

    def forward(self, base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self.compose(base_logits, unit_logits)
        return fused


def export_parameters(model: BiasGeometryCompositor) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def load_parameters(model: BiasGeometryCompositor, values: Mapping[str, Any]) -> None:
    state = model.state_dict()
    for name, raw in values.items():
        if name not in state:
            raise KeyError(f"unknown geometry compositor parameter: {name}")
        state[name] = torch.tensor(raw, dtype=state[name].dtype, device=state[name].device)
    model.load_state_dict(state)


def fit_geometry_compositor(
    model: BiasGeometryCompositor,
    *,
    batch: CalibrationBatch,
    steps: int,
    batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    entropy_penalty: float,
    entropy_target_ratio: float,
    seed: int,
) -> dict[str, Any]:
    if steps <= 0 or batch_positions <= 0:
        raise ValueError("fit steps and batch size must be positive")
    if not 0.0 <= entropy_target_ratio <= 1.0:
        raise ValueError("entropy_target_ratio must be in [0, 1]")
    device = batch.base_logits.device
    model.to(device)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    position_count = batch.positions
    permutation = torch.randperm(position_count, generator=generator)
    cursor = 0
    losses: list[float] = []
    entropy_values: list[float] = []
    target_entropy = entropy_target_ratio * math.log(len(EXPERIMENT_OPERATORS))

    model.train()
    for _ in range(steps):
        if cursor + batch_positions > position_count:
            permutation = torch.randperm(position_count, generator=generator)
            cursor = 0
        cpu_indices = permutation[cursor : cursor + min(batch_positions, position_count)]
        cursor += int(cpu_indices.numel())
        indices = cpu_indices.to(device)
        fused, weights, _ = model.compose(
            batch.base_logits.index_select(0, indices),
            batch.unit_logits.index_select(0, indices),
        )
        gold = batch.gold.index_select(0, indices)
        cross_entropy = F.cross_entropy(fused, gold)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        entropy_shortfall = F.relu(torch.tensor(target_entropy, device=device) - entropy)
        regularizer = sum((parameter - initial[name]).float().pow(2).mean() for name, parameter in model.named_parameters())
        loss = cross_entropy + l2_weight * regularizer + entropy_penalty * entropy_shortfall
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        entropy_values.append(float(entropy.detach().cpu()))

    model.eval()
    with torch.no_grad():
        fused, weights, disagreement = model.compose(batch.base_logits, batch.unit_logits)
        final_nll = float(F.cross_entropy(fused, batch.gold).detach().cpu())
        final_accuracy = float((fused.argmax(dim=-1) == batch.gold).float().mean().detach().cpu())
        final_entropy = float((-(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()).detach().cpu())
        final_disagreement = float(disagreement.mean().detach().cpu())
    return {
        "steps": steps,
        "batch_positions": batch_positions,
        "learning_rate": learning_rate,
        "l2_weight": l2_weight,
        "entropy_penalty": entropy_penalty,
        "entropy_target_ratio": entropy_target_ratio,
        "calibration_positions": position_count,
        "initial_loss": losses[0] if losses else None,
        "last_optimization_loss": losses[-1] if losses else None,
        "calibration_gold_nll": final_nll,
        "calibration_token_accuracy": final_accuracy,
        "mean_weight_entropy": final_entropy,
        "mean_coordinate_disagreement": final_disagreement,
        "parameters": export_parameters(model),
    }


def collect_scheduled_prefix_batch(
    cohort: Cohort,
    *,
    root: Path,
    compositor: BiasGeometryCompositor,
    examples_per_operator: int,
    data_seed: int,
    rollout_probability: float,
    max_positions: int,
    device: torch.device,
) -> CalibrationBatch:
    if not 0.0 <= rollout_probability <= 1.0:
        raise ValueError("rollout_probability must be in [0, 1]")
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    dataset = _dataset(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=examples_per_operator,
        evaluation_seed=data_seed,
    )
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    base_rows: list[torch.Tensor] = []
    unit_rows: list[torch.Tensor] = []
    gold_rows: list[int] = []
    generator = torch.Generator(device="cpu").manual_seed(
        data_seed + int(cohort.metadata.get("seed", 0)) * 1009 + int(rollout_probability * 1000)
    )
    compositor.eval()
    with torch.no_grad():
        for operator in EXPERIMENT_OPERATORS:
            for _, prompt, expected in dataset[operator]:
                ids = torch.tensor([prompt], dtype=torch.long, device=device)
                for gold_token in expected:
                    base_logits = _next_logits(base, ids)
                    unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
                    fused = compositor(base_logits, unit_logits)
                    predicted = int(torch.argmax(fused, dim=-1).item())
                    base_rows.append(base_logits.unsqueeze(0))
                    unit_rows.append(unit_logits.unsqueeze(0))
                    gold_rows.append(int(gold_token))
                    use_rollout = bool(torch.rand((), generator=generator).item() < rollout_probability)
                    next_token = predicted if use_rollout else int(gold_token)
                    ids = torch.cat(
                        [ids, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                        dim=1,
                    )
    base_tensor = torch.cat(base_rows, dim=0)
    unit_tensor = torch.cat(unit_rows, dim=0)
    gold_tensor = torch.tensor(gold_rows, dtype=torch.long, device=device)
    if max_positions > 0 and int(gold_tensor.numel()) > max_positions:
        indices = torch.randperm(int(gold_tensor.numel()), generator=generator)[:max_positions].to(device)
        base_tensor = base_tensor.index_select(0, indices)
        unit_tensor = unit_tensor.index_select(0, indices)
        gold_tensor = gold_tensor.index_select(0, indices)
    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return CalibrationBatch(base_tensor, unit_tensor, gold_tensor)


def _generate_geometry(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    compositor: BiasGeometryCompositor,
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
            fused = compositor(base_logits, unit_logits)
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _weight_diagnostics(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    compositor: BiasGeometryCompositor,
    dataset: Mapping[str, Sequence[tuple[Any, list[int], list[int]]]],
    device: torch.device,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    compositor.eval()
    with torch.no_grad():
        for operator in EXPERIMENT_OPERATORS:
            weight_rows: list[torch.Tensor] = []
            for _, prompt, expected in dataset[operator]:
                sequence = prompt + expected
                input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                response_start = len(prompt) - 1
                base_logits = _teacher_forced_logits(base, input_ids, response_start).squeeze(0).float()
                unit_logits = torch.stack(
                    [
                        _teacher_forced_logits(units[name], input_ids, response_start).squeeze(0).float()
                        for name in EXPERIMENT_OPERATORS
                    ],
                    dim=1,
                )
                _, weights, _ = compositor.compose(base_logits, unit_logits)
                weight_rows.append(weights)
            stacked = torch.cat(weight_rows, dim=0)
            mean_weights = stacked.mean(dim=0)
            entropy = -(stacked * stacked.clamp_min(1e-8).log()).sum(dim=-1).mean()
            output[operator] = {
                "mean_weights": {
                    name: float(mean_weights[index].detach().cpu())
                    for index, name in enumerate(EXPERIMENT_OPERATORS)
                },
                "mean_entropy": float(entropy.detach().cpu()),
                "minimum_observed_weight": float(stacked.min().detach().cpu()),
            }
    return output


def evaluate_geometry_cohort(
    cohort: Cohort,
    *,
    root: Path,
    parameter_sets: Mapping[str, Mapping[str, Any]],
    examples_per_operator: int,
    verification_seed: int,
    max_new_tokens: int,
    hidden_size: int,
    weight_floor: float,
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
    joint = (
        _load_model(cohort.joint_checkpoint, device=device, tokenizer=tokenizer)
        if cohort.joint_checkpoint is not None
        else None
    )
    learned: dict[str, BiasGeometryCompositor] = {}
    for method, values in parameter_sets.items():
        model = BiasGeometryCompositor(hidden_size=hidden_size, weight_floor=weight_floor).to(device)
        load_parameters(model, values)
        model.eval()
        learned[method] = model

    methods = list(FIXED_BASELINES) + list(LEARNED_METHODS)
    composition_metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    base_metrics: dict[str, Any] = {}
    joint_metrics: dict[str, Any] = {}
    for operator in EXPERIMENT_OPERATORS:
        base_counter = _empty_gold_counter()
        joint_counter = _empty_gold_counter()
        counters = {method: _empty_gold_counter() for method in methods}
        for example, prompt, expected in dataset[operator]:
            generated_base = _generate_model(
                base,
                prompt,
                eos_id=tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            _update_gold_counter(
                base_counter,
                factory=factory,
                example=example,
                generated=generated_base,
                expected=expected,
            )
            if joint is not None:
                generated_joint = _generate_model(
                    joint,
                    prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    joint_counter,
                    factory=factory,
                    example=example,
                    generated=generated_joint,
                    expected=expected,
                )
            for method in methods:
                if method in learned:
                    generated = _generate_geometry(
                        base=base,
                        units=units,
                        compositor=learned[method],
                        prompt=prompt,
                        eos_id=tokenizer.eos_id,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )
                else:
                    ids = torch.tensor([prompt], dtype=torch.long, device=device)
                    generated = []
                    with torch.no_grad():
                        for _ in range(max_new_tokens):
                            base_logits = _next_logits(base, ids)
                            unit_logits = torch.stack(
                                [_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0
                            )
                            fused = fixed_compose(base_logits, unit_logits, mode=method)
                            next_id = int(torch.argmax(fused, dim=-1).item())
                            generated.append(next_id)
                            ids = torch.cat(
                                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
                            )
                            if next_id == tokenizer.eos_id:
                                break
                _update_gold_counter(
                    counters[method],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
        base_metrics[operator] = _finalize_gold_counter(base_counter)
        if joint is not None:
            joint_metrics[operator] = _finalize_gold_counter(joint_counter)
        for method in methods:
            composition_metrics[method][operator] = _finalize_gold_counter(counters[method])

    weight_diagnostics = {
        method: _weight_diagnostics(
            base=base,
            units=units,
            compositor=model,
            dataset=dataset,
            device=device,
        )
        for method, model in learned.items()
    }
    del base, units, joint, learned
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "base_metrics": base_metrics,
        "joint_metrics": joint_metrics,
        "composition_metrics": composition_metrics,
        "weight_diagnostics": weight_diagnostics,
    }


def train_and_verify_onpolicy_compositor(
    *,
    root: Path,
    calibration_examples_per_operator: int,
    rollout_examples_per_operator: int,
    verification_examples_per_operator: int,
    max_positions_per_cohort: int,
    warmup_steps: int,
    round_steps: int,
    rollout_probabilities: Sequence[float],
    batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    entropy_penalty: float,
    entropy_target_ratio: float,
    hidden_size: int,
    weight_floor: float,
    calibration_seed: int,
    verification_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    if calibration_seed == verification_seed:
        raise ValueError("calibration and verification seeds must be distinct")
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
        raise RuntimeError(f"expected three complete fusion-factory cohorts, found {len(cohorts)}")
    calibration_cohorts = cohorts[:2]
    held_out_cohort = cohorts[2]

    teacher_batches = [
        _collect_calibration_batch(
            cohort,
            root=root,
            examples_per_operator=calibration_examples_per_operator,
            calibration_seed=calibration_seed,
            max_positions=max_positions_per_cohort,
            device=device,
        )[0]
        for cohort in calibration_cohorts
    ]
    teacher_batch = _merge_batches(teacher_batches)
    compositor = BiasGeometryCompositor(hidden_size=hidden_size, weight_floor=weight_floor).to(device)
    warmup_report = fit_geometry_compositor(
        compositor,
        batch=teacher_batch,
        steps=warmup_steps,
        batch_positions=batch_positions,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
        entropy_penalty=entropy_penalty,
        entropy_target_ratio=entropy_target_ratio,
        seed=calibration_seed,
    )
    teacher_parameters = export_parameters(compositor)

    round_reports: list[dict[str, Any]] = []
    for round_index, probability in enumerate(rollout_probabilities):
        rollout_batches = [
            collect_scheduled_prefix_batch(
                cohort,
                root=root,
                compositor=compositor,
                examples_per_operator=rollout_examples_per_operator,
                data_seed=calibration_seed + 100 + round_index,
                rollout_probability=float(probability),
                max_positions=max_positions_per_cohort,
                device=device,
            )
            for cohort in calibration_cohorts
        ]
        training_batch = _merge_batches([teacher_batch, *rollout_batches])
        report = fit_geometry_compositor(
            compositor,
            batch=training_batch,
            steps=round_steps,
            batch_positions=batch_positions,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            entropy_penalty=entropy_penalty,
            entropy_target_ratio=entropy_target_ratio,
            seed=calibration_seed + 1000 + round_index,
        )
        report["round"] = round_index + 1
        report["rollout_probability"] = float(probability)
        report["rollout_positions"] = sum(batch.positions for batch in rollout_batches)
        round_reports.append(report)
    on_policy_parameters = export_parameters(compositor)

    parameter_sets = {
        "geometry_teacher_forced": teacher_parameters,
        "geometry_on_policy": on_policy_parameters,
    }
    cohort_reports = [
        evaluate_geometry_cohort(
            cohort,
            root=root,
            parameter_sets=parameter_sets,
            examples_per_operator=verification_examples_per_operator,
            verification_seed=verification_seed,
            max_new_tokens=max_new_tokens,
            hidden_size=hidden_size,
            weight_floor=weight_floor,
            device=device,
        )
        for cohort in cohorts
    ]
    methods = list(FIXED_BASELINES) + list(LEARNED_METHODS)
    aggregate = _aggregate_reports(cohort_reports, methods)
    ranked = sorted(aggregate, key=_ranking_key)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_on_policy_composition_learning",
        "claim_boundary": "no external router or task labels; final IID/OOD splits remain unopened",
        "composition_constraint": (
            "all five units have strictly positive weights at every position; weights are continuous functions of bias geometry"
        ),
        "calibration_seed": calibration_seed,
        "verification_seed": verification_seed,
        "calibration_model_seeds": [cohort.metadata.get("seed") for cohort in calibration_cohorts],
        "held_out_model_seed": held_out_cohort.metadata.get("seed"),
        "calibration_examples_per_operator": calibration_examples_per_operator,
        "rollout_examples_per_operator": rollout_examples_per_operator,
        "verification_examples_per_operator": verification_examples_per_operator,
        "rollout_probabilities": [float(value) for value in rollout_probabilities],
        "hidden_size": hidden_size,
        "weight_floor": weight_floor,
        "warmup_report": warmup_report,
        "round_reports": round_reports,
        "parameter_sets": parameter_sets,
        "cohort_reports": cohort_reports,
        "aggregate_methods": aggregate,
        "ranked_methods": ranked,
        "recommended_validation_composition": ranked[0] if ranked else None,
        "passing_methods": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def parse_probabilities(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("rollout probabilities must be in [0, 1]")
    return values


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a conflict-aware all-five compositor on teacher and self-generated prefixes"
    )
    parser.add_argument("--calibration-examples-per-operator", type=int, default=8)
    parser.add_argument("--rollout-examples-per-operator", type=int, default=6)
    parser.add_argument("--verification-examples-per-operator", type=int, default=16)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=160)
    parser.add_argument("--round-steps", type=int, default=80)
    parser.add_argument("--rollout-probabilities", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--entropy-penalty", type=float, default=0.02)
    parser.add_argument("--entropy-target-ratio", type=float, default=0.70)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--weight-floor", type=float, default=0.03)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_onpolicy_composition/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = train_and_verify_onpolicy_compositor(
        root=root,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        rollout_examples_per_operator=args.rollout_examples_per_operator,
        verification_examples_per_operator=args.verification_examples_per_operator,
        max_positions_per_cohort=args.max_positions_per_cohort,
        warmup_steps=args.warmup_steps,
        round_steps=args.round_steps,
        rollout_probabilities=parse_probabilities(args.rollout_probabilities),
        batch_positions=args.batch_positions,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
        entropy_penalty=args.entropy_penalty,
        entropy_target_ratio=args.entropy_target_ratio,
        hidden_size=args.hidden_size,
        weight_floor=args.weight_floor,
        calibration_seed=args.calibration_seed,
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
