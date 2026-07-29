from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from opfusion.fusion_compose import _aggregate_reports, _ranking_key, fixed_compose
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_sparse_valid import SparseValidBatch, _collect_valid_batch, valid_set_loss
from opfusion.fusion_verify import (
    _dataset,
    _empty_gold_counter,
    _finalize_gold_counter,
    _next_logits,
    _update_gold_counter,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


DEFAULT_CALIBRATION_SEED = 713_000
DEFAULT_HOLDOUT_SEED = 713_500
DEFAULT_VERIFICATION_SEED = 714_000


@dataclass(frozen=True)
class PoolCandidate:
    candidate_id: str
    mode: str
    alpha: float
    threshold: float = 0.0
    temperature: float = 1.0
    veto: float = 0.0


def centered_biases(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
    biases = unit_logits - base_logits.unsqueeze(-2)
    return biases - biases.mean(dim=-1, keepdim=True)


def coordinate_residual(
    base_logits: torch.Tensor,
    unit_logits: torch.Tensor,
    candidate: PoolCandidate,
) -> torch.Tensor:
    biases = centered_biases(base_logits, unit_logits)
    rms = biases.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    scale = rms.median(dim=-1).values.clamp_min(1e-6)
    cutoff = float(candidate.threshold) * scale.unsqueeze(-1).unsqueeze(-1)

    positive = F.relu(biases - cutoff.to(biases.dtype))
    negative = F.relu(-biases - cutoff.to(biases.dtype))
    mode = candidate.mode
    if mode == "positive_max":
        residual = positive.amax(dim=-2)
    elif mode == "positive_top2":
        residual = positive.topk(k=2, dim=-2).values.mean(dim=-2)
    elif mode == "positive_logmeanexp":
        tau = (float(candidate.temperature) * scale).clamp_min(1e-5).unsqueeze(-1).unsqueeze(-1)
        residual = tau.squeeze(-2) * (
            torch.logsumexp(positive / tau.to(positive.dtype), dim=-2) - math.log(positive.shape[-2])
        )
    elif mode == "max_veto":
        residual = positive.amax(dim=-2) - float(candidate.veto) * negative.amax(dim=-2)
    elif mode == "signed_absmax":
        indices = biases.abs().argmax(dim=-2, keepdim=True)
        residual = biases.gather(dim=-2, index=indices).squeeze(-2)
        if candidate.threshold > 0:
            absolute_cutoff = float(candidate.threshold) * scale.unsqueeze(-1)
            residual = residual.sign() * F.relu(residual.abs() - absolute_cutoff.to(residual.dtype))
    elif mode == "median":
        residual = biases.median(dim=-2).values
    elif mode == "trimmed_mean":
        ordered = biases.sort(dim=-2).values
        residual = ordered[..., 1:-1, :].mean(dim=-2)
    elif mode == "sparse_sum":
        signed = biases.sign() * F.relu(biases.abs() - cutoff.to(biases.dtype))
        residual = signed.sum(dim=-2)
    else:
        raise ValueError(f"unsupported pooling mode: {mode}")

    residual = residual - residual.mean(dim=-1, keepdim=True)
    cap = (4.0 * scale).clamp_min(1e-4).unsqueeze(-1).to(residual.dtype)
    return cap * torch.tanh(float(candidate.alpha) * residual / cap)


def compose_logits(
    base_logits: torch.Tensor,
    unit_logits: torch.Tensor,
    candidate: PoolCandidate,
) -> torch.Tensor:
    return base_logits + coordinate_residual(base_logits, unit_logits, candidate)


def candidate_grid() -> tuple[PoolCandidate, ...]:
    rows: list[PoolCandidate] = []
    for threshold in (0.0, 0.25, 0.5, 1.0):
        for alpha in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            rows.append(PoolCandidate(f"positive_max_t{threshold}_a{alpha}", "positive_max", alpha, threshold))
    for threshold in (0.0, 0.25, 0.5):
        for alpha in (0.5, 1.0, 1.5, 2.0):
            rows.append(PoolCandidate(f"positive_top2_t{threshold}_a{alpha}", "positive_top2", alpha, threshold))
    for temperature in (0.25, 0.5, 1.0, 2.0):
        for threshold in (0.0, 0.25, 0.5):
            for alpha in (0.5, 1.0, 1.5):
                rows.append(
                    PoolCandidate(
                        f"positive_lme_k{temperature}_t{threshold}_a{alpha}",
                        "positive_logmeanexp",
                        alpha,
                        threshold,
                        temperature,
                    )
                )
    for veto in (0.25, 0.5, 1.0):
        for threshold in (0.0, 0.25, 0.5):
            for alpha in (0.5, 1.0, 1.5):
                rows.append(
                    PoolCandidate(
                        f"max_veto_v{veto}_t{threshold}_a{alpha}",
                        "max_veto",
                        alpha,
                        threshold,
                        veto=veto,
                    )
                )
    for threshold in (0.0, 0.25, 0.5):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            rows.append(PoolCandidate(f"absmax_t{threshold}_a{alpha}", "signed_absmax", alpha, threshold))
    for mode in ("median", "trimmed_mean"):
        for alpha in (0.5, 1.0, 1.5, 2.0):
            rows.append(PoolCandidate(f"{mode}_a{alpha}", mode, alpha))
    for threshold in (0.25, 0.5, 1.0):
        for alpha in (0.1, 0.25, 0.5, 0.75):
            rows.append(PoolCandidate(f"sparse_sum_t{threshold}_a{alpha}", "sparse_sum", alpha, threshold))
    return tuple(rows)


def batch_candidate_metrics(batch: SparseValidBatch, candidate: PoolCandidate) -> dict[str, float]:
    with torch.no_grad():
        logits = compose_logits(batch.base_logits, batch.unit_logits, candidate)
        nll = valid_set_loss(logits, batch.valid_mask)
        prediction = logits.argmax(dim=-1)
        top1 = batch.valid_mask.gather(1, prediction.unsqueeze(-1)).squeeze(-1).float().mean()
        mass = (torch.softmax(logits.float(), dim=-1) * batch.valid_mask.float()).sum(dim=-1).mean()
    return {
        "valid_set_nll": float(nll.detach().cpu()),
        "valid_top1_accuracy": float(top1.detach().cpu()),
        "valid_probability_mass": float(mass.detach().cpu()),
    }


def oracle_diagnostics(batch: SparseValidBatch) -> dict[str, Any]:
    with torch.no_grad():
        logits = torch.cat([batch.base_logits.unsqueeze(-2), batch.unit_logits], dim=-2)
        probabilities = torch.softmax(logits.float(), dim=-1)
        valid_mass = (probabilities * batch.valid_mask.unsqueeze(-2).float()).sum(dim=-1)
        predictions = logits.argmax(dim=-1)
        valid_top1 = batch.valid_mask.unsqueeze(-2).expand_as(logits).gather(
            dim=-1, index=predictions.unsqueeze(-1)
        ).squeeze(-1)
        oracle_indices = valid_mass.argmax(dim=-1)
        oracle_top1 = valid_top1.gather(1, oracle_indices.unsqueeze(-1)).squeeze(-1)
        any_top1 = valid_top1.any(dim=-1)
    labels = ["base", *EXPERIMENT_OPERATORS]
    return {
        "per_model_valid_top1": {
            label: float(valid_top1[:, index].float().mean().detach().cpu())
            for index, label in enumerate(labels)
        },
        "per_model_valid_mass": {
            label: float(valid_mass[:, index].mean().detach().cpu())
            for index, label in enumerate(labels)
        },
        "oracle_by_valid_mass_top1": float(oracle_top1.float().mean().detach().cpu()),
        "any_model_valid_top1": float(any_top1.float().mean().detach().cpu()),
    }


def select_diverse_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    maximum: int,
) -> list[PoolCandidate]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    winners: list[dict[str, Any]] = []
    for mode_rows in by_mode.values():
        winners.append(
            min(
                mode_rows,
                key=lambda row: (
                    -float(row["holdout"]["valid_top1_accuracy"]),
                    float(row["holdout"]["valid_set_nll"]),
                ),
            )
        )
    winners.sort(
        key=lambda row: (
            -float(row["holdout"]["valid_top1_accuracy"]),
            float(row["holdout"]["valid_set_nll"]),
        )
    )
    return [PoolCandidate(**row["candidate"]) for row in winners[:maximum]]


def _generate_candidate(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    candidate: PoolCandidate,
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
            fused = compose_logits(base_logits, unit_logits, candidate)
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _generate_bias_mean(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
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
            fused = fixed_compose(base_logits, unit_logits, mode="bias_mean")
            next_id = int(torch.argmax(fused, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    candidates: Sequence[PoolCandidate],
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
    methods = ["bias_mean", *[candidate.candidate_id for candidate in candidates]]
    metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    for operator in EXPERIMENT_OPERATORS:
        counters = {method: _empty_gold_counter() for method in methods}
        for example, prompt, expected in dataset[operator]:
            generated = _generate_bias_mean(
                base=base,
                units=units,
                prompt=prompt,
                eos_id=tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            _update_gold_counter(
                counters["bias_mean"],
                factory=factory,
                example=example,
                generated=generated,
                expected=expected,
            )
            for candidate in candidates:
                generated = _generate_candidate(
                    base=base,
                    units=units,
                    candidate=candidate,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    counters[candidate.candidate_id],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
        for method in methods:
            metrics[method][operator] = _finalize_gold_counter(counters[method])
    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "composition_metrics": metrics,
    }


def search_coordinate_pooling(
    *,
    root: Path,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    verification_examples_per_operator: int,
    maximum_autoregressive_candidates: int,
    calibration_seed: int,
    holdout_seed: int,
    verification_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
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
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")
    calibration_batches = [
        _collect_valid_batch(
            cohort,
            root=root,
            examples_per_operator=calibration_examples_per_operator,
            data_seed=calibration_seed,
            max_prefixes_per_example=max_prefixes_per_example,
            max_positions=max_positions_per_cohort,
            device=device,
        )[0]
        for cohort in cohorts[:2]
    ]
    calibration = SparseValidBatch(
        base_logits=torch.cat([batch.base_logits for batch in calibration_batches], dim=0),
        unit_logits=torch.cat([batch.unit_logits for batch in calibration_batches], dim=0),
        valid_mask=torch.cat([batch.valid_mask for batch in calibration_batches], dim=0),
    )
    holdout, holdout_collection = _collect_valid_batch(
        cohorts[2],
        root=root,
        examples_per_operator=calibration_examples_per_operator,
        data_seed=holdout_seed,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions=max_positions_per_cohort,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for candidate in candidate_grid():
        rows.append(
            {
                "candidate": {
                    "candidate_id": candidate.candidate_id,
                    "mode": candidate.mode,
                    "alpha": candidate.alpha,
                    "threshold": candidate.threshold,
                    "temperature": candidate.temperature,
                    "veto": candidate.veto,
                },
                "mode": candidate.mode,
                "calibration": batch_candidate_metrics(calibration, candidate),
                "holdout": batch_candidate_metrics(holdout, candidate),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["holdout"]["valid_top1_accuracy"]),
            float(row["holdout"]["valid_set_nll"]),
        )
    )
    selected = select_diverse_candidates(rows, maximum=maximum_autoregressive_candidates)
    cohort_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            candidates=selected,
            examples_per_operator=verification_examples_per_operator,
            verification_seed=verification_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts
    ]
    methods = ["bias_mean", *[candidate.candidate_id for candidate in selected]]
    aggregate = _aggregate_reports(cohort_reports, methods)
    ranked = sorted(aggregate, key=_ranking_key)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_coordinate_pooling_search",
        "claim_boundary": "all five units evaluated at every token; no task labels or unit switching; final splits unopened",
        "algebra": "coordinate-wise robust pooling of raw centered bias fields",
        "calibration_seed": calibration_seed,
        "holdout_seed": holdout_seed,
        "verification_seed": verification_seed,
        "calibration_examples_per_operator": calibration_examples_per_operator,
        "verification_examples_per_operator": verification_examples_per_operator,
        "candidate_count": len(rows),
        "holdout_collection": holdout_collection,
        "holdout_oracle": oracle_diagnostics(holdout),
        "candidate_rows": rows,
        "autoregressive_candidates": [candidate.__dict__ for candidate in selected],
        "cohort_reports": cohort_reports,
        "aggregate_methods": aggregate,
        "ranked_methods": ranked,
        "recommended_validation_composition": ranked[0] if ranked else None,
        "passing_methods": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search coordinate-wise robust all-unit bias pooling laws")
    parser.add_argument("--calibration-examples-per-operator", type=int, default=3)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=512)
    parser.add_argument("--verification-examples-per-operator", type=int, default=8)
    parser.add_argument("--maximum-autoregressive-candidates", type=int, default=6)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_coordinate_pool/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = search_coordinate_pooling(
        root=root,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        max_prefixes_per_example=args.max_prefixes_per_example,
        max_positions_per_cohort=args.max_positions_per_cohort,
        verification_examples_per_operator=args.verification_examples_per_operator,
        maximum_autoregressive_candidates=args.maximum_autoregressive_candidates,
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
    print(json.dumps(report.get("holdout_oracle"), ensure_ascii=False, sort_keys=True))
    print(json.dumps(report.get("recommended_validation_composition"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
