from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts, subset_operators
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


DEFAULT_EVALUATION_SEED = 704_000


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    subset_mask: int
    mode: str
    alpha: float

    @property
    def operators(self) -> tuple[str, ...]:
        return subset_operators(self.subset_mask)


DEFAULT_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("add_raw_1.00", 1, "raw_sum", 1.0),
    CandidateSpec("sum_raw_1.25", 2, "raw_sum", 1.25),
    CandidateSpec("neg_raw_0.50", 4, "raw_sum", 0.5),
    CandidateSpec("min_raw_1.25", 8, "raw_sum", 1.25),
    CandidateSpec("max_raw_1.25", 16, "raw_sum", 1.25),
    CandidateSpec("add_min_causal_rms_0.75", 9, "causal_rms_equalized_sum", 0.75),
    CandidateSpec("add_max_causal_rms_0.75", 17, "causal_rms_equalized_sum", 0.75),
    CandidateSpec("min_max_raw_0.75", 24, "raw_sum", 0.75),
    CandidateSpec("add_min_max_raw_0.25", 25, "raw_sum", 0.25),
    CandidateSpec("add_min_max_causal_rms_0.50", 25, "causal_rms_equalized_sum", 0.5),
    CandidateSpec("all_five_raw_0.125", 31, "raw_sum", 0.125),
    CandidateSpec("all_five_causal_rms_0.125", 31, "causal_rms_equalized_sum", 0.125),
)


def causal_rms_equalize_biases(biases: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Causally equalize next-token specialist fields over the vocabulary axis.

    The input is ``[specialists, vocabulary]``. Vocabulary-wise additive
    constants are removed because they do not affect softmax. Each nonzero
    specialist is scaled to the median active RMS for the current generation
    position. No future response positions are used.
    """

    if biases.ndim != 2:
        raise ValueError("causal bias equalization requires [specialists, vocabulary]")
    centered = biases - biases.mean(dim=-1, keepdim=True)
    rms = centered.float().pow(2).mean(dim=-1).sqrt()
    nonzero = rms > eps
    if not bool(nonzero.any()):
        return torch.zeros_like(centered)
    target = rms[nonzero].median()
    scales = torch.where(nonzero, target / rms.clamp_min(eps), torch.zeros_like(rms))
    return centered * scales[:, None].to(centered.dtype)


def fuse_next_logits(
    base_logits: torch.Tensor,
    specialist_logits: Sequence[torch.Tensor],
    *,
    mode: str,
    alpha: float,
) -> torch.Tensor:
    if base_logits.ndim != 1:
        raise ValueError("base_logits must have shape [vocabulary]")
    if not specialist_logits:
        return base_logits
    stack = torch.stack([logits - base_logits for logits in specialist_logits], dim=0)
    if mode == "raw_sum":
        combined = stack.sum(dim=0)
    elif mode == "causal_rms_equalized_sum":
        combined = causal_rms_equalize_biases(stack).sum(dim=0)
    else:
        raise ValueError(f"unsupported fusion mode: {mode}")
    return base_logits + float(alpha) * combined


def _next_logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    condition = ids[:, -model.config.max_seq_len :]
    return model(condition)[:, -1, :].squeeze(0).float()


def _generate_model(
    model: torch.nn.Module,
    prompt: Sequence[int],
    *,
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_id = int(torch.argmax(_next_logits(model, ids), dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _generate_fused(
    *,
    base: torch.nn.Module,
    units: Sequence[torch.nn.Module],
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    mode: str,
    alpha: float,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = [_next_logits(model, ids) for model in units]
            fused = fuse_next_logits(base_logits, unit_logits, mode=mode, alpha=alpha)
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _token_match(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    width = max(len(left), len(right))
    correct = sum(
        int(index < len(left) and index < len(right) and left[index] == right[index])
        for index in range(width)
    )
    return correct, width


def _empty_gold_counter() -> dict[str, float]:
    return {
        "examples": 0,
        "exact": 0,
        "token_correct": 0,
        "token_count": 0,
        "final_correct": 0,
        "final_count": 0,
        "trace_valid": 0,
        "stop_correct": 0,
        "generated_tokens": 0,
    }


def _update_gold_counter(
    counter: dict[str, float],
    *,
    factory: SyntheticTraceFactory,
    example: Any,
    generated: Sequence[int],
    expected: Sequence[int],
) -> None:
    verification = factory.verify_generated_ids(example, list(generated))
    correct, count = _token_match(generated, expected)
    counter["examples"] += 1
    counter["exact"] += int(list(generated) == list(expected))
    counter["token_correct"] += correct
    counter["token_count"] += count
    counter["trace_valid"] += int(bool(verification.get("valid")))
    counter["stop_correct"] += int(bool(verification.get("stop_correct")))
    counter["generated_tokens"] += len(generated)
    if example.final_value is not None:
        counter["final_count"] += 1
        counter["final_correct"] += int(bool(verification.get("final_correct")))


def _finalize_gold_counter(counter: dict[str, float]) -> dict[str, float | None]:
    examples = max(1, int(counter["examples"]))
    return {
        "examples": int(counter["examples"]),
        "response_exact_accuracy": float(counter["exact"]) / examples,
        "response_token_accuracy": float(counter["token_correct"]) / max(1, int(counter["token_count"])),
        "final_value_accuracy": (
            float(counter["final_correct"]) / int(counter["final_count"])
            if counter["final_count"]
            else None
        ),
        "trace_validity_accuracy": float(counter["trace_valid"]) / examples,
        "stop_accuracy": float(counter["stop_correct"]) / examples,
        "mean_generated_tokens": float(counter["generated_tokens"]) / examples,
    }


def _empty_agreement_counter() -> dict[str, int]:
    return {"examples": 0, "exact": 0, "token_correct": 0, "token_count": 0}


def _update_agreement_counter(
    counter: dict[str, int], candidate: Sequence[int], base: Sequence[int]
) -> None:
    correct, count = _token_match(candidate, base)
    counter["examples"] += 1
    counter["exact"] += int(list(candidate) == list(base))
    counter["token_correct"] += correct
    counter["token_count"] += count


def _finalize_agreement_counter(counter: dict[str, int]) -> dict[str, float | int]:
    return {
        "examples": counter["examples"],
        "base_response_exact_agreement": counter["exact"] / max(1, counter["examples"]),
        "base_response_token_agreement": counter["token_correct"] / max(1, counter["token_count"]),
    }


def _dataset(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    examples_per_operator: int,
    evaluation_seed: int,
) -> dict[str, list[tuple[Any, list[int], list[int]]]]:
    result: dict[str, list[tuple[Any, list[int], list[int]]]] = {}
    for operator_index, operator in enumerate(EXPERIMENT_OPERATORS):
        rows = []
        for sample_index in range(examples_per_operator):
            example = factory.training_example(
                operator,
                seed=evaluation_seed,
                split="validation",
                step=operator_index,
                sample_index=sample_index,
            )
            prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
            expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
            rows.append((example, prompt, expected))
        result[operator] = rows
    return result


def _evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    candidates: Sequence[CandidateSpec],
    examples_per_operator: int,
    evaluation_seed: int,
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
        evaluation_seed=evaluation_seed,
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

    base_outputs: dict[str, list[list[int]]] = {}
    base_metrics: dict[str, Any] = {}
    joint_metrics: dict[str, Any] = {}
    for operator in EXPERIMENT_OPERATORS:
        base_counter = _empty_gold_counter()
        joint_counter = _empty_gold_counter()
        base_rows: list[list[int]] = []
        for example, prompt, expected in dataset[operator]:
            generated_base = _generate_model(
                base,
                prompt,
                eos_id=tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            base_rows.append(generated_base)
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
        base_outputs[operator] = base_rows
        base_metrics[operator] = _finalize_gold_counter(base_counter)
        if joint is not None:
            joint_metrics[operator] = _finalize_gold_counter(joint_counter)

    candidate_reports: dict[str, Any] = {}
    for candidate in candidates:
        active = set(candidate.operators)
        active_metrics: dict[str, Any] = {}
        inactive_metrics: dict[str, Any] = {}
        selected_units = [units[operator] for operator in candidate.operators]
        for operator in EXPERIMENT_OPERATORS:
            if operator in active:
                counter = _empty_gold_counter()
            else:
                counter = _empty_agreement_counter()
            for sample_index, (example, prompt, expected) in enumerate(dataset[operator]):
                generated = _generate_fused(
                    base=base,
                    units=selected_units,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    mode=candidate.mode,
                    alpha=candidate.alpha,
                    device=device,
                )
                if operator in active:
                    _update_gold_counter(
                        counter,
                        factory=factory,
                        example=example,
                        generated=generated,
                        expected=expected,
                    )
                else:
                    _update_agreement_counter(counter, generated, base_outputs[operator][sample_index])
            if operator in active:
                active_metrics[operator] = _finalize_gold_counter(counter)
            else:
                inactive_metrics[operator] = _finalize_agreement_counter(counter)

        candidate_reports[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "subset_mask": candidate.subset_mask,
            "operators": list(candidate.operators),
            "mode": candidate.mode,
            "alpha": candidate.alpha,
            "active_metrics": active_metrics,
            "inactive_metrics": inactive_metrics,
        }

    del base, units, joint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "base_metrics": base_metrics,
        "joint_metrics": joint_metrics,
        "candidates": candidate_reports,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _aggregate_candidates(
    cohort_reports: Sequence[dict[str, Any]], candidates: Sequence[CandidateSpec]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        per_operator: dict[str, Any] = {}
        for operator in candidate.operators:
            metrics = [
                report["candidates"][candidate.candidate_id]["active_metrics"][operator]
                for report in cohort_reports
            ]
            per_operator[operator] = {
                "response_exact_accuracy_mean": _mean([float(item["response_exact_accuracy"]) for item in metrics]),
                "response_token_accuracy_mean": _mean([float(item["response_token_accuracy"]) for item in metrics]),
                "final_value_accuracy_mean": _mean([float(item["final_value_accuracy"] or 0.0) for item in metrics]),
                "trace_validity_accuracy_mean": _mean([float(item["trace_validity_accuracy"]) for item in metrics]),
                "stop_accuracy_mean": _mean([float(item["stop_accuracy"]) for item in metrics]),
            }

        inactive_exact_values: list[float] = []
        inactive_token_values: list[float] = []
        for report in cohort_reports:
            inactive = report["candidates"][candidate.candidate_id]["inactive_metrics"]
            inactive_exact_values.extend(float(item["base_response_exact_agreement"]) for item in inactive.values())
            inactive_token_values.extend(float(item["base_response_token_agreement"]) for item in inactive.values())

        finals = [float(item["final_value_accuracy_mean"]) for item in per_operator.values()]
        traces = [float(item["trace_validity_accuracy_mean"]) for item in per_operator.values()]
        exacts = [float(item["response_exact_accuracy_mean"]) for item in per_operator.values()]
        inactive_exact = _mean(inactive_exact_values) if inactive_exact_values else 1.0
        inactive_token = _mean(inactive_token_values) if inactive_token_values else 1.0
        strict_final = min(finals, default=0.0)
        strict_trace = min(traces, default=0.0)
        row = {
            "candidate_id": candidate.candidate_id,
            "subset_mask": candidate.subset_mask,
            "operators": list(candidate.operators),
            "mode": candidate.mode,
            "alpha": candidate.alpha,
            "seed_count": len(cohort_reports),
            "per_operator": per_operator,
            "active_final_accuracy_macro": _mean(finals),
            "active_final_accuracy_min": strict_final,
            "active_trace_validity_macro": _mean(traces),
            "active_trace_validity_min": strict_trace,
            "active_response_exact_macro": _mean(exacts),
            "inactive_base_exact_agreement_macro": inactive_exact,
            "inactive_base_token_agreement_macro": inactive_token,
            "passes_validation_gate": (
                strict_final >= 0.90
                and strict_trace >= 0.90
                and inactive_exact >= 0.90
            ),
        }
        rows.append(row)
    return rows


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        -float(bool(row["passes_validation_gate"])),
        -float(row["active_final_accuracy_min"]),
        -float(row["active_final_accuracy_macro"]),
        -float(row["active_trace_validity_min"]),
        -float(row["inactive_base_exact_agreement_macro"]),
        -float(row["active_response_exact_macro"]),
        float(len(row["operators"])),
    )


def verify_candidates(
    *,
    root: Path,
    candidates: Sequence[CandidateSpec] = DEFAULT_CANDIDATES,
    examples_per_operator: int = 8,
    evaluation_seed: int = DEFAULT_EVALUATION_SEED,
    max_new_tokens: int = 64,
    device_name: str = "auto",
) -> dict[str, Any]:
    if examples_per_operator <= 0 or max_new_tokens <= 0:
        raise ValueError("evaluation sizes must be positive")
    if evaluation_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")
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
    cohort_reports = [
        _evaluate_cohort(
            cohort,
            root=root,
            candidates=candidates,
            examples_per_operator=examples_per_operator,
            evaluation_seed=evaluation_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts
    ]
    aggregate = _aggregate_candidates(cohort_reports, candidates)
    ranked = sorted(aggregate, key=_candidate_sort_key)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "independent_validation_autoregressive_verification",
        "claim_boundary": "independent validation seed only; final IID/OOD splits remain unopened",
        "evaluation_seed": evaluation_seed,
        "examples_per_operator": examples_per_operator,
        "max_new_tokens": max_new_tokens,
        "device": str(device),
        "causal_normalization": "per generated position; vocabulary-centered specialist RMS equalized across active units",
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "subset_mask": candidate.subset_mask,
                "operators": list(candidate.operators),
                "mode": candidate.mode,
                "alpha": candidate.alpha,
            }
            for candidate in candidates
        ],
        "cohort_reports": cohort_reports,
        "aggregate_candidates": aggregate,
        "ranked_candidates": ranked,
        "recommended_candidate": ranked[0] if ranked else None,
        "passing_candidates": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autoregressively verify selected fusion candidates on an independent validation seed"
    )
    parser.add_argument("--examples-per-operator", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_combination_search/autoregressive_verification.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = verify_candidates(
        root=root,
        examples_per_operator=args.examples_per_operator,
        evaluation_seed=args.evaluation_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report.get("recommended_candidate"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
