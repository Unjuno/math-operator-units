from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

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


DEFAULT_CALIBRATION_SEED = 705_000
DEFAULT_VERIFICATION_SEED = 706_000
PAIR_INDICES = tuple(itertools.combinations(range(len(EXPERIMENT_OPERATORS)), 2))
LEARNED_FAMILIES = ("global_linear", "token_class_linear", "pairwise_polynomial")
FIXED_BASELINES = ("raw_sum", "bias_mean", "rms_mean")
TOKEN_CLASSES = ("numeric", "special", "syntax")


@dataclass(frozen=True)
class CalibrationBatch:
    base_logits: torch.Tensor
    unit_logits: torch.Tensor
    gold: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.gold.numel())


class AlgebraicCompositor(nn.Module):
    """Static all-five logit-bias compositor.

    The compositor never receives an operator id, task label, subset mask, or
    routing decision. The same parameters are applied to all prompts and all
    autoregressive positions. Only the five simultaneous specialist logit fields
    and the common Base field are available.
    """

    def __init__(
        self,
        family: str,
        *,
        token_class_ids: torch.Tensor,
        operator_count: int = len(EXPERIMENT_OPERATORS),
    ) -> None:
        super().__init__()
        if family not in LEARNED_FAMILIES:
            raise ValueError(f"unsupported learned compositor family: {family}")
        if token_class_ids.ndim != 1:
            raise ValueError("token_class_ids must have shape [vocabulary]")
        self.family = family
        self.operator_count = operator_count
        self.register_buffer("token_class_ids", token_class_ids.to(dtype=torch.long), persistent=True)

        initial = torch.full((operator_count,), 1.0 / operator_count)
        if family == "token_class_linear":
            self.class_weights = nn.Parameter(initial.repeat(len(TOKEN_CLASSES), 1))
        else:
            self.weights = nn.Parameter(initial)
        if family == "pairwise_polynomial":
            self.interactions = nn.Parameter(torch.zeros(len(PAIR_INDICES)))

    @staticmethod
    def _centered_biases(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        if unit_logits.shape[-2] != len(EXPERIMENT_OPERATORS):
            raise ValueError("unit_logits must contain all five specialists")
        if unit_logits.shape[:-2] != base_logits.shape[:-1] or unit_logits.shape[-1] != base_logits.shape[-1]:
            raise ValueError("base and specialist logit shapes are incompatible")
        biases = unit_logits - base_logits.unsqueeze(-2)
        return biases - biases.mean(dim=-1, keepdim=True)

    @staticmethod
    def _normalized(centered: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        rms = centered.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        scale = rms.squeeze(-1).median(dim=-1).values.unsqueeze(-1)
        return centered / rms.to(centered.dtype), scale.to(centered.dtype)

    def residual(self, base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        centered = self._centered_biases(base_logits, unit_logits)
        if self.family == "global_linear":
            return torch.einsum("...kv,k->...v", centered, self.weights)

        if self.family == "token_class_linear":
            vocabulary = int(base_logits.shape[-1])
            if int(self.token_class_ids.numel()) != vocabulary:
                raise ValueError("token class table does not match vocabulary")
            per_token_weights = self.class_weights[self.token_class_ids]
            return torch.einsum("...kv,vk->...v", centered, per_token_weights)

        normalized, scale = self._normalized(centered)
        signal = torch.einsum("...kv,k->...v", normalized, self.weights)
        for pair_index, (left, right) in enumerate(PAIR_INDICES):
            signal = signal + self.interactions[pair_index] * normalized[..., left, :] * normalized[..., right, :]
        # The bounded polynomial keeps an interaction correction from producing
        # unbounded logits on an off-trajectory autoregressive context.
        return scale * (4.0 * torch.tanh(signal / 4.0))

    def forward(self, base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + self.residual(base_logits, unit_logits)


def token_class_ids(tokenizer: FixedVocabTokenizer, *, device: torch.device | None = None) -> torch.Tensor:
    classes: list[int] = []
    for token in tokenizer.tokens:
        if token.startswith("<N_"):
            classes.append(0)
        elif token.startswith("<") and token.endswith(">"):
            classes.append(1)
        else:
            classes.append(2)
    return torch.tensor(classes, dtype=torch.long, device=device)


def fixed_compose(base_logits: torch.Tensor, unit_logits: torch.Tensor, *, mode: str) -> torch.Tensor:
    centered = AlgebraicCompositor._centered_biases(base_logits, unit_logits)
    if mode == "raw_sum":
        residual = centered.sum(dim=-2)
    elif mode == "bias_mean":
        residual = centered.mean(dim=-2)
    elif mode == "rms_mean":
        normalized, scale = AlgebraicCompositor._normalized(centered)
        residual = scale * normalized.mean(dim=-2)
    else:
        raise ValueError(f"unsupported fixed composition mode: {mode}")
    return base_logits + residual


def export_parameters(model: AlgebraicCompositor) -> dict[str, Any]:
    return {
        name: value.detach().cpu().tolist()
        for name, value in model.state_dict().items()
        if name != "token_class_ids"
    }


def load_parameters(model: AlgebraicCompositor, values: Mapping[str, Any]) -> None:
    state = model.state_dict()
    for name, raw in values.items():
        if name not in state:
            raise KeyError(f"unknown compositor parameter: {name}")
        state[name] = torch.tensor(raw, dtype=state[name].dtype, device=state[name].device)
    model.load_state_dict(state)


def _collect_calibration_batch(
    cohort: Cohort,
    *,
    root: Path,
    examples_per_operator: int,
    calibration_seed: int,
    max_positions: int,
    device: torch.device,
) -> tuple[CalibrationBatch, FixedVocabTokenizer]:
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
    gold_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for operator_index, operator in enumerate(EXPERIMENT_OPERATORS):
            for sample_index in range(examples_per_operator):
                example = factory.training_example(
                    operator,
                    seed=calibration_seed,
                    split="validation",
                    step=operator_index,
                    sample_index=sample_index,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                sequence = prompt + expected
                input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                targets = torch.tensor(sequence[1:], dtype=torch.long, device=device)
                response_start = len(prompt) - 1
                base_logits = _teacher_forced_logits(base, input_ids, response_start).squeeze(0).float()
                specialist_logits = torch.stack(
                    [
                        _teacher_forced_logits(units[name], input_ids, response_start).squeeze(0).float()
                        for name in EXPERIMENT_OPERATORS
                    ],
                    dim=1,
                )
                base_rows.append(base_logits)
                unit_rows.append(specialist_logits)
                gold_rows.append(targets[response_start:])

    base_tensor = torch.cat(base_rows, dim=0)
    unit_tensor = torch.cat(unit_rows, dim=0)
    gold_tensor = torch.cat(gold_rows, dim=0)
    if max_positions > 0 and int(gold_tensor.numel()) > max_positions:
        generator = torch.Generator(device="cpu").manual_seed(calibration_seed + int(cohort.metadata.get("seed", 0)))
        indices = torch.randperm(int(gold_tensor.numel()), generator=generator)[:max_positions].to(device)
        base_tensor = base_tensor.index_select(0, indices)
        unit_tensor = unit_tensor.index_select(0, indices)
        gold_tensor = gold_tensor.index_select(0, indices)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return CalibrationBatch(base_tensor, unit_tensor, gold_tensor), tokenizer


def _merge_batches(batches: Sequence[CalibrationBatch]) -> CalibrationBatch:
    if not batches:
        raise ValueError("at least one calibration batch is required")
    return CalibrationBatch(
        base_logits=torch.cat([batch.base_logits for batch in batches], dim=0),
        unit_logits=torch.cat([batch.unit_logits for batch in batches], dim=0),
        gold=torch.cat([batch.gold for batch in batches], dim=0),
    )


def fit_compositor(
    family: str,
    *,
    batch: CalibrationBatch,
    token_classes: torch.Tensor,
    steps: int,
    batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    seed: int,
) -> tuple[AlgebraicCompositor, dict[str, Any]]:
    if steps <= 0 or batch_positions <= 0:
        raise ValueError("fit steps and batch size must be positive")
    device = batch.base_logits.device
    model = AlgebraicCompositor(family, token_class_ids=token_classes).to(device)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    position_count = batch.positions
    permutation = torch.randperm(position_count, generator=generator)
    cursor = 0
    losses: list[float] = []

    model.train()
    for _ in range(steps):
        if cursor + batch_positions > position_count:
            permutation = torch.randperm(position_count, generator=generator)
            cursor = 0
        cpu_indices = permutation[cursor : cursor + min(batch_positions, position_count)]
        cursor += int(cpu_indices.numel())
        indices = cpu_indices.to(device)
        fused = model(
            batch.base_logits.index_select(0, indices),
            batch.unit_logits.index_select(0, indices),
        )
        gold = batch.gold.index_select(0, indices)
        cross_entropy = F.cross_entropy(fused, gold)
        regularizer = sum((parameter - initial[name]).float().pow(2).mean() for name, parameter in model.named_parameters())
        loss = cross_entropy + float(l2_weight) * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        final_logits = model(batch.base_logits, batch.unit_logits)
        final_nll = float(F.cross_entropy(final_logits, batch.gold).detach().cpu())
        final_accuracy = float((final_logits.argmax(dim=-1) == batch.gold).float().mean().detach().cpu())
    return model, {
        "family": family,
        "steps": steps,
        "batch_positions": batch_positions,
        "learning_rate": learning_rate,
        "l2_weight": l2_weight,
        "calibration_positions": position_count,
        "initial_loss": losses[0] if losses else None,
        "last_optimization_loss": losses[-1] if losses else None,
        "calibration_gold_nll": final_nll,
        "calibration_token_accuracy": final_accuracy,
        "parameters": export_parameters(model),
    }


def _generate_composed(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    learned: AlgebraicCompositor | None,
    fixed_mode: str | None,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            if learned is not None:
                fused = learned(base_logits, unit_logits)
            elif fixed_mode is not None:
                fused = fixed_compose(base_logits, unit_logits, mode=fixed_mode)
            else:
                raise ValueError("one composition method is required")
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _evaluate_cohort_autoregressive(
    cohort: Cohort,
    *,
    root: Path,
    learned_parameters: Mapping[str, Mapping[str, Any]],
    examples_per_operator: int,
    verification_seed: int,
    max_new_tokens: int,
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
    classes = token_class_ids(tokenizer, device=device)
    learned_models: dict[str, AlgebraicCompositor] = {}
    for family, parameters in learned_parameters.items():
        model = AlgebraicCompositor(family, token_class_ids=classes).to(device)
        load_parameters(model, parameters)
        model.eval()
        learned_models[family] = model

    methods = list(FIXED_BASELINES) + list(learned_models)
    method_metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
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
                generated = _generate_composed(
                    base=base,
                    units=units,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    learned=learned_models.get(method),
                    fixed_mode=method if method in FIXED_BASELINES else None,
                    device=device,
                )
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
            method_metrics[method][operator] = _finalize_gold_counter(counters[method])

    del base, units, joint, learned_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "base_metrics": base_metrics,
        "joint_metrics": joint_metrics,
        "composition_metrics": method_metrics,
    }


def _aggregate_reports(reports: Sequence[dict[str, Any]], methods: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        per_operator: dict[str, Any] = {}
        for operator in EXPERIMENT_OPERATORS:
            metrics = [report["composition_metrics"][method][operator] for report in reports]
            finals = [float(item["final_value_accuracy"] or 0.0) for item in metrics]
            traces = [float(item["trace_validity_accuracy"]) for item in metrics]
            exacts = [float(item["response_exact_accuracy"]) for item in metrics]
            tokens = [float(item["response_token_accuracy"]) for item in metrics]
            per_operator[operator] = {
                "final_value_accuracy_mean": sum(finals) / len(finals),
                "trace_validity_accuracy_mean": sum(traces) / len(traces),
                "response_exact_accuracy_mean": sum(exacts) / len(exacts),
                "response_token_accuracy_mean": sum(tokens) / len(tokens),
            }
        final_values = [float(row["final_value_accuracy_mean"]) for row in per_operator.values()]
        trace_values = [float(row["trace_validity_accuracy_mean"]) for row in per_operator.values()]
        exact_values = [float(row["response_exact_accuracy_mean"]) for row in per_operator.values()]
        rows.append(
            {
                "method": method,
                "per_operator": per_operator,
                "final_value_accuracy_macro": sum(final_values) / len(final_values),
                "final_value_accuracy_min": min(final_values),
                "trace_validity_accuracy_macro": sum(trace_values) / len(trace_values),
                "trace_validity_accuracy_min": min(trace_values),
                "response_exact_accuracy_macro": sum(exact_values) / len(exact_values),
                "passes_validation_gate": min(final_values) >= 0.80 and min(trace_values) >= 0.80,
            }
        )
    return rows


def _ranking_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -float(bool(row["passes_validation_gate"])),
        -float(row["final_value_accuracy_min"]),
        -float(row["final_value_accuracy_macro"]),
        -float(row["trace_validity_accuracy_min"]),
        -float(row["response_exact_accuracy_macro"]),
    )


def learn_and_verify_composition(
    *,
    root: Path,
    calibration_examples_per_operator: int,
    verification_examples_per_operator: int,
    max_calibration_positions_per_cohort: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    l2_weight: float,
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
    cohorts = discover_cohorts(root, "fusion-factory")
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete fusion-factory cohorts, found {len(cohorts)}")
    cohorts = sorted(cohorts, key=lambda item: int(item.metadata.get("seed", 0)))
    calibration_cohorts = cohorts[:2]
    held_out_model_cohort = cohorts[2]

    calibration_batches: list[CalibrationBatch] = []
    tokenizer: FixedVocabTokenizer | None = None
    for cohort in calibration_cohorts:
        batch, current_tokenizer = _collect_calibration_batch(
            cohort,
            root=root,
            examples_per_operator=calibration_examples_per_operator,
            calibration_seed=calibration_seed,
            max_positions=max_calibration_positions_per_cohort,
            device=device,
        )
        calibration_batches.append(batch)
        tokenizer = current_tokenizer
    if tokenizer is None:
        raise RuntimeError("calibration tokenizer was not loaded")
    merged = _merge_batches(calibration_batches)
    classes = token_class_ids(tokenizer, device=device)

    fit_reports: dict[str, Any] = {}
    parameters: dict[str, Mapping[str, Any]] = {}
    for family_index, family in enumerate(LEARNED_FAMILIES):
        model, fit_report = fit_compositor(
            family,
            batch=merged,
            token_classes=classes,
            steps=fit_steps,
            batch_positions=fit_batch_positions,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            seed=calibration_seed + family_index,
        )
        fit_reports[family] = fit_report
        parameters[family] = fit_report["parameters"]
        del model

    cohort_reports = [
        _evaluate_cohort_autoregressive(
            cohort,
            root=root,
            learned_parameters=parameters,
            examples_per_operator=verification_examples_per_operator,
            verification_seed=verification_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts
    ]
    methods = list(FIXED_BASELINES) + list(LEARNED_FAMILIES)
    aggregate = _aggregate_reports(cohort_reports, methods)
    ranked = sorted(aggregate, key=_ranking_key)
    held_out_report = next(
        report for report in cohort_reports if report["cohort_id"] == held_out_model_cohort.cohort_id
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_global_composition_learning",
        "claim_boundary": "no task routing or subset selection; final IID/OOD splits remain unopened",
        "composition_constraint": (
            "all five specialists are evaluated at every generated position; the compositor receives no operator id, task label, or gate"
        ),
        "calibration_seed": calibration_seed,
        "verification_seed": verification_seed,
        "calibration_model_seeds": [cohort.metadata.get("seed") for cohort in calibration_cohorts],
        "held_out_model_seed": held_out_model_cohort.metadata.get("seed"),
        "calibration_examples_per_operator": calibration_examples_per_operator,
        "verification_examples_per_operator": verification_examples_per_operator,
        "max_calibration_positions_per_cohort": max_calibration_positions_per_cohort,
        "max_new_tokens": max_new_tokens,
        "device": str(device),
        "operator_order": list(EXPERIMENT_OPERATORS),
        "token_classes": list(TOKEN_CLASSES),
        "learned_families": list(LEARNED_FAMILIES),
        "fixed_baselines": list(FIXED_BASELINES),
        "fit_reports": fit_reports,
        "cohort_reports": cohort_reports,
        "held_out_model_report": held_out_report,
        "aggregate_methods": aggregate,
        "ranked_methods": ranked,
        "recommended_validation_composition": ranked[0] if ranked else None,
        "passing_methods": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learn and autoregressively verify static all-five algebraic bias composition laws"
    )
    parser.add_argument("--calibration-examples-per-operator", type=int, default=16)
    parser.add_argument("--verification-examples-per-operator", type=int, default=16)
    parser.add_argument("--max-calibration-positions-per-cohort", type=int, default=2048)
    parser.add_argument("--fit-steps", type=int, default=250)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_composition_learning/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = learn_and_verify_composition(
        root=root,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        verification_examples_per_operator=args.verification_examples_per_operator,
        max_calibration_positions_per_cohort=args.max_calibration_positions_per_cohort,
        fit_steps=args.fit_steps,
        fit_batch_positions=args.fit_batch_positions,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
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
