from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from opfusion.fusion_eval import _load_model, _teacher_forced_logits
from opfusion.fusion_search import (
    BIAS_FACTORY_SNAPSHOTS,
    Cohort,
    discover_cohorts,
)
from opfusion.fusion_verify import _generate_model
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory, TrainingExample


DEFAULT_EVALUATION_SEED = 709_000
AGGREGATE_OPERATORS = ("aggregation.sum", "scalar.min", "scalar.max")
SNAPSHOT_ORDER = {label: index for index, (label, _) in enumerate(BIAS_FACTORY_SNAPSHOTS)}


@dataclass(frozen=True)
class ModelTarget:
    target_id: str
    source: str
    role: str
    config_path: Path
    checkpoint: Path
    model_seed: int
    parameter_scale: str
    training_examples_label: str | None = None
    checkpoint_step: int | None = None
    target_operator: str | None = None


@dataclass
class MetricCounter:
    examples: int = 0
    exact: int = 0
    token_correct: int = 0
    token_count: int = 0
    final_correct: int = 0
    final_count: int = 0
    trace_valid: int = 0
    stop_correct: int = 0
    generated_tokens: int = 0
    teacher_correct: int = 0
    teacher_tokens: int = 0
    teacher_nll_sum: float = 0.0
    correct_prefix_tokens: int = 0
    expected_tokens: int = 0


def _token_match(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    width = max(len(left), len(right))
    correct = sum(
        int(index < len(left) and index < len(right) and left[index] == right[index])
        for index in range(width)
    )
    return correct, width


def _correct_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    width = min(len(left), len(right))
    for index in range(width):
        if left[index] != right[index]:
            return index
    return width if len(left) == len(right) else width


def update_counter(
    counter: MetricCounter,
    *,
    factory: SyntheticTraceFactory,
    example: TrainingExample,
    generated: Sequence[int],
    expected: Sequence[int],
    teacher_logits: torch.Tensor,
    gold: torch.Tensor,
) -> None:
    verification = factory.verify_generated_ids(example, list(generated))
    correct, count = _token_match(generated, expected)
    counter.examples += 1
    counter.exact += int(list(generated) == list(expected))
    counter.token_correct += correct
    counter.token_count += count
    counter.trace_valid += int(bool(verification.get("valid")))
    counter.stop_correct += int(bool(verification.get("stop_correct")))
    counter.generated_tokens += len(generated)
    counter.correct_prefix_tokens += _correct_prefix(generated, expected)
    counter.expected_tokens += len(expected)
    if example.final_value is not None:
        counter.final_count += 1
        counter.final_correct += int(bool(verification.get("final_correct")))

    predictions = teacher_logits.argmax(dim=-1)
    counter.teacher_correct += int((predictions == gold).sum().item())
    counter.teacher_tokens += int(gold.numel())
    counter.teacher_nll_sum += float(F.cross_entropy(teacher_logits, gold, reduction="sum").item())


def finalize_counter(counter: MetricCounter) -> dict[str, float | int | None]:
    examples = max(1, counter.examples)
    return {
        "examples": counter.examples,
        "response_exact_accuracy": counter.exact / examples,
        "response_token_accuracy": counter.token_correct / max(1, counter.token_count),
        "final_value_accuracy": (
            counter.final_correct / counter.final_count if counter.final_count else None
        ),
        "trace_validity_accuracy": counter.trace_valid / examples,
        "stop_accuracy": counter.stop_correct / examples,
        "mean_generated_tokens": counter.generated_tokens / examples,
        "teacher_forced_token_accuracy": counter.teacher_correct / max(1, counter.teacher_tokens),
        "teacher_forced_nll": counter.teacher_nll_sum / max(1, counter.teacher_tokens),
        "mean_correct_prefix_fraction": counter.correct_prefix_tokens / max(1, counter.expected_tokens),
    }


def passes_metrics(metrics: Mapping[str, Any], *, identity: bool = False) -> bool:
    if identity:
        return (
            float(metrics["response_exact_accuracy"]) >= 0.95
            and float(metrics["trace_validity_accuracy"]) >= 0.95
            and float(metrics["stop_accuracy"]) >= 0.95
            and float(metrics["teacher_forced_token_accuracy"]) >= 0.95
        )
    final = metrics.get("final_value_accuracy")
    return (
        final is not None
        and float(final) >= 0.80
        and float(metrics["trace_validity_accuracy"]) >= 0.80
        and float(metrics["stop_accuracy"]) >= 0.95
        and float(metrics["teacher_forced_token_accuracy"]) >= 0.80
    )


def _targets_from_cohort(cohort: Cohort) -> list[ModelTarget]:
    metadata = cohort.metadata
    model_seed = int(metadata.get("seed", 0))
    parameter_scale = str(metadata.get("parameter_scale", "unknown"))
    label = metadata.get("training_examples_label")
    step = metadata.get("checkpoint_step")
    suffix = f"{cohort.cohort_id}"
    targets = [
        ModelTarget(
            target_id=f"{suffix}:base",
            source=cohort.source,
            role="base",
            config_path=cohort.config_path,
            checkpoint=cohort.base_checkpoint,
            model_seed=model_seed,
            parameter_scale=parameter_scale,
            training_examples_label=str(label) if label is not None else None,
            checkpoint_step=int(step) if step is not None else None,
        )
    ]
    for operator, checkpoint in cohort.unit_checkpoints.items():
        targets.append(
            ModelTarget(
                target_id=f"{suffix}:specialist:{operator}",
                source=cohort.source,
                role="specialist",
                config_path=cohort.config_path,
                checkpoint=checkpoint,
                model_seed=model_seed,
                parameter_scale=parameter_scale,
                training_examples_label=str(label) if label is not None else None,
                checkpoint_step=int(step) if step is not None else None,
                target_operator=operator,
            )
        )
    if cohort.joint_checkpoint is not None:
        targets.append(
            ModelTarget(
                target_id=f"{suffix}:joint",
                source=cohort.source,
                role="joint",
                config_path=cohort.config_path,
                checkpoint=cohort.joint_checkpoint,
                model_seed=model_seed,
                parameter_scale=parameter_scale,
                training_examples_label=str(label) if label is not None else None,
                checkpoint_step=int(step) if step is not None else None,
            )
        )
    return targets


def discover_targets(root: Path, scope: str) -> list[ModelTarget]:
    source = "fusion-factory" if scope == "fusion-factory" else "bias-factory"
    cohorts = discover_cohorts(root, source)
    targets: dict[str, ModelTarget] = {}
    for cohort in cohorts:
        for target in _targets_from_cohort(cohort):
            targets[target.target_id] = target
    return [targets[key] for key in sorted(targets)]


def _dataset(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    job_id: str,
    forced_operator: str | None,
    examples: int,
    evaluation_seed: int,
    step: int,
) -> list[tuple[TrainingExample, list[int], list[int]]]:
    rows: list[tuple[TrainingExample, list[int], list[int]]] = []
    for sample_index in range(examples):
        example = factory.training_example(
            job_id,
            seed=evaluation_seed,
            split="validation",
            step=step,
            sample_index=sample_index,
            forced_operator=forced_operator,
        )
        prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
        expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
        rows.append((example, prompt, expected))
    return rows


def evaluate_rows(
    *,
    model: torch.nn.Module,
    factory: SyntheticTraceFactory,
    rows: Sequence[tuple[TrainingExample, list[int], list[int]]],
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    counter = MetricCounter()
    with torch.no_grad():
        for example, prompt, expected in rows:
            generated = _generate_model(
                model,
                prompt,
                eos_id=tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            sequence = [*prompt, *expected]
            input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
            response_start = len(prompt) - 1
            teacher_logits = _teacher_forced_logits(model, input_ids, response_start).squeeze(0).float()
            gold = torch.tensor(expected, dtype=torch.long, device=device)
            update_counter(
                counter,
                factory=factory,
                example=example,
                generated=generated,
                expected=expected,
                teacher_logits=teacher_logits,
                gold=gold,
            )
    return finalize_counter(counter)


def _quality_gate(
    target: ModelTarget,
    arithmetic: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if target.role == "specialist":
        metrics = arithmetic[target.target_operator or ""]
        return {
            "scope": target.target_operator,
            "passed": passes_metrics(metrics),
            "metrics": dict(metrics),
        }
    if target.role == "joint":
        passed_by_operator = {
            operator: passes_metrics(metrics) for operator, metrics in arithmetic.items()
        }
        return {
            "scope": "all_arithmetic_operators",
            "passed": all(passed_by_operator.values()),
            "passed_by_operator": passed_by_operator,
        }
    passed_by_operator = {
        operator: passes_metrics(metrics, identity=True) for operator, metrics in identity.items()
    }
    return {
        "scope": "identity_equivalence_all_operators",
        "passed": all(passed_by_operator.values()),
        "passed_by_operator": passed_by_operator,
    }


def evaluate_target(
    target: ModelTarget,
    *,
    root: Path,
    examples_per_operator: int,
    length_examples: int,
    evaluation_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(target.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    model = _load_model(target.checkpoint, device=device, tokenizer=tokenizer)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    arithmetic: dict[str, Any] = {}
    for operator_index, operator in enumerate(EXPERIMENT_OPERATORS):
        rows = _dataset(
            factory=factory,
            tokenizer=tokenizer,
            job_id=operator,
            forced_operator=None,
            examples=examples_per_operator,
            evaluation_seed=evaluation_seed,
            step=operator_index,
        )
        arithmetic[operator] = evaluate_rows(
            model=model,
            factory=factory,
            rows=rows,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    identity: dict[str, Any] = {}
    if target.role == "base":
        for operator_index, operator in enumerate(EXPERIMENT_OPERATORS):
            rows = _dataset(
                factory=factory,
                tokenizer=tokenizer,
                job_id="base.common",
                forced_operator=operator,
                examples=examples_per_operator,
                evaluation_seed=evaluation_seed + 1,
                step=operator_index,
            )
            identity[operator] = evaluate_rows(
                model=model,
                factory=factory,
                rows=rows,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                device=device,
            )

    length_profile: dict[str, Any] = {}
    profile_operators: tuple[str, ...]
    if target.role == "joint":
        profile_operators = AGGREGATE_OPERATORS
    elif target.role == "specialist" and target.target_operator in AGGREGATE_OPERATORS:
        profile_operators = (target.target_operator,)
    else:
        profile_operators = ()
    for operator_index, operator in enumerate(profile_operators):
        operator_profile: dict[str, Any] = {}
        for term_count in range(run.data.min_terms, run.data.max_terms + 1):
            length_data = replace(run.data, min_terms=term_count, max_terms=term_count)
            length_factory = SyntheticTraceFactory(tokenizer, length_data)
            rows = _dataset(
                factory=length_factory,
                tokenizer=tokenizer,
                job_id=operator,
                forced_operator=None,
                examples=length_examples,
                evaluation_seed=evaluation_seed + 10_000 + term_count * 100,
                step=operator_index,
            )
            operator_profile[str(term_count)] = evaluate_rows(
                model=model,
                factory=length_factory,
                rows=rows,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        length_profile[operator] = operator_profile

    gate = _quality_gate(target, arithmetic, identity)
    report = {
        "target_id": target.target_id,
        "source": target.source,
        "role": target.role,
        "target_operator": target.target_operator,
        "model_seed": target.model_seed,
        "parameter_scale": target.parameter_scale,
        "parameter_count": parameter_count,
        "training_examples_label": target.training_examples_label,
        "checkpoint_step": target.checkpoint_step,
        "config": str(target.config_path.relative_to(root)),
        "checkpoint": str(target.checkpoint.relative_to(root)),
        "arithmetic_metrics": arithmetic,
        "identity_metrics": identity,
        "length_profile": length_profile,
        "quality_gate": gate,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def aggregate_fusion_reports(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for report in reports:
        key = (str(report["role"]), report.get("target_operator"))
        grouped.setdefault(key, []).append(report)
    rows: list[dict[str, Any]] = []
    for (role, target_operator), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        per_operator: dict[str, Any] = {}
        source_key = "identity_metrics" if role == "base" else "arithmetic_metrics"
        operators = EXPERIMENT_OPERATORS if role in {"base", "joint"} else (target_operator,)
        for operator in operators:
            metrics = [report[source_key][operator] for report in group]
            finals = [float(metric["final_value_accuracy"] or 0.0) for metric in metrics]
            per_operator[str(operator)] = {
                "response_exact_accuracy_mean": _mean([float(metric["response_exact_accuracy"]) for metric in metrics]),
                "final_value_accuracy_mean": None if role == "base" else _mean(finals),
                "trace_validity_accuracy_mean": _mean([float(metric["trace_validity_accuracy"]) for metric in metrics]),
                "teacher_forced_token_accuracy_mean": _mean([float(metric["teacher_forced_token_accuracy"]) for metric in metrics]),
                "mean_correct_prefix_fraction": _mean([float(metric["mean_correct_prefix_fraction"]) for metric in metrics]),
            }
        rows.append(
            {
                "role": role,
                "target_operator": target_operator,
                "seed_count": len(group),
                "parameter_count": int(group[0]["parameter_count"]),
                "quality_gate_pass_rate": _mean([float(bool(report["quality_gate"]["passed"])) for report in group]),
                "per_operator": per_operator,
            }
        )
    return rows


def bias_scaling_rows(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        role = str(report["role"])
        target_operator = report.get("target_operator")
        if role == "specialist":
            metric = report["arithmetic_metrics"][target_operator]
            score_scope = target_operator
        elif role == "base":
            identity_metrics = report["identity_metrics"]
            metric = {
                "response_exact_accuracy": min(float(value["response_exact_accuracy"]) for value in identity_metrics.values()),
                "final_value_accuracy": None,
                "trace_validity_accuracy": min(float(value["trace_validity_accuracy"]) for value in identity_metrics.values()),
                "stop_accuracy": min(float(value["stop_accuracy"]) for value in identity_metrics.values()),
                "teacher_forced_token_accuracy": min(float(value["teacher_forced_token_accuracy"]) for value in identity_metrics.values()),
                "mean_correct_prefix_fraction": min(float(value["mean_correct_prefix_fraction"]) for value in identity_metrics.values()),
            }
            score_scope = "identity_worst_operator"
        else:
            values = report["arithmetic_metrics"]
            metric = {
                "response_exact_accuracy": _mean([float(value["response_exact_accuracy"]) for value in values.values()]),
                "final_value_accuracy": _mean([float(value["final_value_accuracy"] or 0.0) for value in values.values()]),
                "trace_validity_accuracy": _mean([float(value["trace_validity_accuracy"]) for value in values.values()]),
                "stop_accuracy": _mean([float(value["stop_accuracy"]) for value in values.values()]),
                "teacher_forced_token_accuracy": _mean([float(value["teacher_forced_token_accuracy"]) for value in values.values()]),
                "mean_correct_prefix_fraction": _mean([float(value["mean_correct_prefix_fraction"]) for value in values.values()]),
            }
            score_scope = "all_operator_macro"
        rows.append(
            {
                "target_id": report["target_id"],
                "parameter_scale": report["parameter_scale"],
                "parameter_count": report["parameter_count"],
                "training_examples_label": report["training_examples_label"],
                "checkpoint_step": report["checkpoint_step"],
                "role": role,
                "target_operator": target_operator,
                "score_scope": score_scope,
                **metric,
                "quality_gate_passed": bool(report["quality_gate"]["passed"]),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["parameter_scale"]),
            SNAPSHOT_ORDER.get(str(row["training_examples_label"]), 999),
            str(row["role"]),
            str(row["target_operator"]),
        )
    )
    return rows


def minimum_passing_snapshots(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["role"] != "specialist":
            continue
        grouped.setdefault((str(row["parameter_scale"]), str(row["target_operator"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (size, operator), group in sorted(grouped.items()):
        passing = [row for row in group if bool(row["quality_gate_passed"])]
        passing.sort(key=lambda row: SNAPSHOT_ORDER.get(str(row["training_examples_label"]), 999))
        best = passing[0] if passing else None
        output.append(
            {
                "parameter_scale": size,
                "target_operator": operator,
                "minimum_passing_training_examples_label": (
                    best["training_examples_label"] if best is not None else None
                ),
                "minimum_passing_checkpoint_step": best["checkpoint_step"] if best is not None else None,
            }
        )
    return output


def evaluate_models(
    *,
    root: Path,
    scope: str,
    examples_per_operator: int,
    length_examples: int,
    evaluation_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    if scope not in {"fusion-factory", "bias-factory"}:
        raise ValueError("scope must be fusion-factory or bias-factory")
    if min(examples_per_operator, length_examples, max_new_tokens) <= 0:
        raise ValueError("evaluation sizes must be positive")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    targets = discover_targets(root, scope)
    if not targets:
        raise RuntimeError(f"no complete targets found for {scope}")
    reports = [
        evaluate_target(
            target,
            root=root,
            examples_per_operator=examples_per_operator,
            length_examples=length_examples,
            evaluation_seed=evaluation_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for target in targets
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "standalone_model_quality_validation",
        "claim_boundary": "validation only; iid_test, operand_ood, and length_ood remain unopened",
        "scope": scope,
        "evaluation_seed": evaluation_seed,
        "examples_per_operator": examples_per_operator,
        "length_examples": length_examples,
        "max_new_tokens": max_new_tokens,
        "device": str(device),
        "target_count": len(reports),
        "reports": reports,
        "production_go": False,
    }
    if scope == "fusion-factory":
        payload["cross_seed_summary"] = aggregate_fusion_reports(reports)
        payload["fusion_eligible_specialists"] = [
            {
                "target_id": report["target_id"],
                "model_seed": report["model_seed"],
                "operator": report["target_operator"],
            }
            for report in reports
            if report["role"] == "specialist" and report["quality_gate"]["passed"]
        ]
    else:
        scaling = bias_scaling_rows(reports)
        payload["scaling_rows"] = scaling
        payload["minimum_passing_snapshots"] = minimum_passing_snapshots(scaling)
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autoregressively evaluate Base, specialist, and Joint checkpoints on validation"
    )
    parser.add_argument("--scope", choices=("fusion-factory", "bias-factory"), required=True)
    parser.add_argument("--examples-per-operator", type=int, default=16)
    parser.add_argument("--length-examples", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = evaluate_models(
        root=root,
        scope=args.scope,
        examples_per_operator=args.examples_per_operator,
        length_examples=args.length_examples,
        evaluation_seed=args.evaluation_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    output = root / (
        args.out
        or f"evaluations/model_quality/{args.scope.replace('-', '_')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({"scope": args.scope, "target_count": report["target_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
