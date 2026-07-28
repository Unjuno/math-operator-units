from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from opfusion.fusion_eval import Jensen_shannon_divergence, _load_model, _teacher_forced_logits
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


DEFAULT_EVALUATION_SEED = 703_000
DEFAULT_ALPHA_GRID = (0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BIAS_FACTORY_SNAPSHOTS: tuple[tuple[str, int], ...] = (
    ("4K", 39),
    ("16K", 156),
    ("65K", 635),
    ("262K", 2559),
    ("1M", 9800),
)
BIAS_FACTORY_SIZES = ("nano", "small", "medium", "1m")
OPERATOR_SHORT = {
    "scalar.add": "add",
    "aggregation.sum": "sum",
    "scalar.neg": "neg",
    "scalar.min": "min",
    "scalar.max": "max",
}


@dataclass(frozen=True)
class Cohort:
    cohort_id: str
    source: str
    config_path: Path
    base_checkpoint: Path
    unit_checkpoints: dict[str, Path]
    joint_checkpoint: Path | None
    metadata: dict[str, Any]


@dataclass
class CandidateCounter:
    active_correct: int = 0
    active_tokens: int = 0
    active_nll_sum: float = 0.0
    inactive_argmax_agree: int = 0
    inactive_tokens: int = 0
    inactive_delta_sq_sum: float = 0.0
    inactive_delta_elements: int = 0
    all_five_joint_jsd_sum: float = 0.0
    all_five_joint_examples: int = 0


def enumerate_nonempty_subsets(operator_count: int) -> tuple[int, ...]:
    if operator_count <= 0:
        raise ValueError("operator_count must be positive")
    return tuple(range(1, 1 << operator_count))


def subset_operators(mask: int, operators: Sequence[str] = EXPERIMENT_OPERATORS) -> tuple[str, ...]:
    return tuple(operator for bit, operator in enumerate(operators) if mask & (1 << bit))


def parse_alpha_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("alpha grid must contain finite nonnegative values")
    return tuple(dict.fromkeys(values))


def rms_equalize_biases(biases: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Equalize specialist centered-bias RMS while preserving cohort scale.

    ``biases`` has shape ``[specialists, positions, vocabulary]``. Vocabulary-wise
    constants are removed because they do not affect softmax. Every nonzero unit is
    rescaled to the median RMS of the active cohort. Zero-bias units remain zero.
    """

    if biases.ndim < 3:
        raise ValueError("biases must have [specialists, ..., vocabulary] dimensions")
    centered = biases - biases.mean(dim=-1, keepdim=True)
    reduce_dims = tuple(range(1, centered.ndim))
    rms = centered.float().pow(2).mean(dim=reduce_dims).sqrt()
    nonzero = rms > eps
    if not bool(nonzero.any()):
        return torch.zeros_like(centered)
    target = rms[nonzero].median()
    scales = torch.where(nonzero, target / rms.clamp_min(eps), torch.zeros_like(rms))
    shape = (scales.shape[0],) + (1,) * (centered.ndim - 1)
    return centered * scales.reshape(shape).to(centered.dtype)


def _checkpoint_from_job_dir(job_dir: Path, *, step: int | None = None) -> Path | None:
    if step is not None:
        exact = job_dir / "checkpoints" / f"step_{step:09d}.pt"
        return exact if exact.is_file() else None
    for name in ("selected.pt", "last.pt", "checkpoints/final.pt"):
        candidate = job_dir / name
        if candidate.is_file():
            return candidate
    complete_path = job_dir / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        for key in ("selected_checkpoint", "final_checkpoint"):
            raw = complete.get(key)
            if not raw:
                continue
            candidate = Path(str(raw))
            if candidate.is_file():
                return candidate
            local = job_dir / candidate.name
            if local.is_file():
                return local
    return None


def _fusion_factory_cohorts(root: Path) -> list[Cohort]:
    config = root / "configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml"
    run_root = root / "runs/gpt_bias_fusion_factory_surface_v3"
    cohorts: list[Cohort] = []
    for seed in (0, 1, 2):
        seed_root = run_root / f"seed_{seed}"
        base = _checkpoint_from_job_dir(seed_root / "base_common")
        units = {
            operator: _checkpoint_from_job_dir(seed_root / operator.replace(".", "_"))
            for operator in EXPERIMENT_OPERATORS
        }
        joint = _checkpoint_from_job_dir(seed_root / "joint_all_five_exposure_matched")
        if base is None or joint is None or any(path is None for path in units.values()):
            continue
        cohorts.append(
            Cohort(
                cohort_id=f"fusion_factory_seed_{seed}",
                source="fusion_factory",
                config_path=config,
                base_checkpoint=base,
                unit_checkpoints={key: value for key, value in units.items() if value is not None},
                joint_checkpoint=joint,
                metadata={
                    "seed": seed,
                    "parameter_scale": "1M",
                    "training_regime": "shared-base specialists",
                    "training_steps": 50_000,
                },
            )
        )
    return cohorts


def _bias_factory_cohorts(root: Path) -> list[Cohort]:
    cohorts: list[Cohort] = []
    for size in BIAS_FACTORY_SIZES:
        config = root / f"configs/experiments/bias_factory/joint_{size}.yaml"
        base_dir = root / f"runs/bias_factory/{size}/seed_0/base_common"
        unit_dirs = {
            operator: root
            / f"runs/bias_factory/spec_{size}_{OPERATOR_SHORT[operator]}/seed_0/{operator.replace('.', '_')}"
            for operator in EXPERIMENT_OPERATORS
        }
        found_exact = False
        for label, step in BIAS_FACTORY_SNAPSHOTS:
            base = _checkpoint_from_job_dir(base_dir, step=step)
            units = {operator: _checkpoint_from_job_dir(path, step=step) for operator, path in unit_dirs.items()}
            if base is None or any(path is None for path in units.values()):
                continue
            found_exact = True
            cohorts.append(
                Cohort(
                    cohort_id=f"bias_factory_{size}_{label}",
                    source="bias_factory",
                    config_path=config,
                    base_checkpoint=base,
                    unit_checkpoints={key: value for key, value in units.items() if value is not None},
                    joint_checkpoint=None,
                    metadata={
                        "seed": 0,
                        "parameter_scale": size,
                        "training_examples_label": label,
                        "checkpoint_step": step,
                        "training_regime": "scratch specialists; output-logit delta diagnostic",
                    },
                )
            )
        if not found_exact:
            base = _checkpoint_from_job_dir(base_dir)
            units = {operator: _checkpoint_from_job_dir(path) for operator, path in unit_dirs.items()}
            if base is not None and all(path is not None for path in units.values()):
                cohorts.append(
                    Cohort(
                        cohort_id=f"bias_factory_{size}_selected",
                        source="bias_factory",
                        config_path=config,
                        base_checkpoint=base,
                        unit_checkpoints={key: value for key, value in units.items() if value is not None},
                        joint_checkpoint=None,
                        metadata={
                            "seed": 0,
                            "parameter_scale": size,
                            "training_examples_label": "selected",
                            "training_regime": "scratch specialists; output-logit delta diagnostic",
                            "snapshot_warning": "exact data-amount checkpoints were not available",
                        },
                    )
                )
    return cohorts


def discover_cohorts(root: Path, source: str) -> list[Cohort]:
    cohorts: list[Cohort] = []
    if source in {"all", "fusion-factory"}:
        cohorts.extend(_fusion_factory_cohorts(root))
    if source in {"all", "bias-factory"}:
        cohorts.extend(_bias_factory_cohorts(root))
    return cohorts


def _mask_matrix(device: torch.device) -> tuple[tuple[int, ...], torch.Tensor]:
    masks = enumerate_nonempty_subsets(len(EXPERIMENT_OPERATORS))
    matrix = torch.tensor(
        [[float(bool(mask & (1 << bit))) for bit in range(len(EXPERIMENT_OPERATORS))] for mask in masks],
        dtype=torch.float32,
        device=device,
    )
    return masks, matrix


def _candidate_key(mask: int, mode: str, alpha: float) -> tuple[int, str, float]:
    return mask, mode, float(alpha)


def _evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    alpha_grid: Sequence[float],
    examples_per_operator: int,
    evaluation_seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
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

    masks, mask_matrix = _mask_matrix(device)
    full_mask = (1 << len(EXPERIMENT_OPERATORS)) - 1
    modes = ("raw_sum", "rms_equalized_sum")
    counters = {
        _candidate_key(mask, mode, alpha): CandidateCounter()
        for mask in masks
        for mode in modes
        for alpha in alpha_grid
    }
    base_by_operator: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for operator_index, target_operator in enumerate(EXPERIMENT_OPERATORS):
            base_correct = 0
            base_tokens = 0
            base_nll_sum = 0.0
            for sample_index in range(examples_per_operator):
                example = factory.training_example(
                    target_operator,
                    seed=evaluation_seed,
                    split="validation",
                    step=operator_index,
                    sample_index=sample_index,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                sequence = prompt + expected
                input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                gold = torch.tensor(sequence[1 + len(prompt) - 1 :], dtype=torch.long, device=device)
                response_start = len(prompt) - 1

                base_logits = _teacher_forced_logits(base, input_ids, response_start).squeeze(0).float()
                unit_logits = torch.stack(
                    [
                        _teacher_forced_logits(units[operator], input_ids, response_start).squeeze(0).float()
                        for operator in EXPERIMENT_OPERATORS
                    ],
                    dim=0,
                )
                biases = unit_logits - base_logits.unsqueeze(0)
                raw_subset_bias = torch.einsum("sk,klv->slv", mask_matrix, biases)
                equalized = rms_equalize_biases(biases)
                normalized_subset_bias = torch.einsum("sk,klv->slv", mask_matrix, equalized)

                base_pred = base_logits.argmax(dim=-1)
                base_correct += int((base_pred == gold).sum().item())
                base_tokens += int(gold.numel())
                base_nll_sum += float(F.cross_entropy(base_logits, gold, reduction="sum").item())
                joint_logits = (
                    _teacher_forced_logits(joint, input_ids, response_start).squeeze(0).float()
                    if joint is not None
                    else None
                )

                for mode, subset_bias in (
                    ("raw_sum", raw_subset_bias),
                    ("rms_equalized_sum", normalized_subset_bias),
                ):
                    for alpha in alpha_grid:
                        fused = base_logits.unsqueeze(0) + float(alpha) * subset_bias
                        predictions = fused.argmax(dim=-1)
                        log_probs = F.log_softmax(fused, dim=-1)
                        gold_index = gold.view(1, -1, 1).expand(fused.shape[0], -1, 1)
                        nll_per_subset = -log_probs.gather(dim=-1, index=gold_index).squeeze(-1).sum(dim=-1)
                        centered_delta = fused - base_logits.unsqueeze(0)
                        centered_delta = centered_delta - centered_delta.mean(dim=-1, keepdim=True)
                        delta_sq = centered_delta.pow(2).sum(dim=(1, 2))
                        argmax_agreement = (predictions == base_pred.unsqueeze(0)).sum(dim=1)

                        for subset_index, mask in enumerate(masks):
                            counter = counters[_candidate_key(mask, mode, alpha)]
                            is_active = bool(mask & (1 << operator_index))
                            if is_active:
                                counter.active_correct += int((predictions[subset_index] == gold).sum().item())
                                counter.active_tokens += int(gold.numel())
                                counter.active_nll_sum += float(nll_per_subset[subset_index].item())
                            else:
                                counter.inactive_argmax_agree += int(argmax_agreement[subset_index].item())
                                counter.inactive_tokens += int(gold.numel())
                                counter.inactive_delta_sq_sum += float(delta_sq[subset_index].item())
                                counter.inactive_delta_elements += int(centered_delta[subset_index].numel())
                            if mask == full_mask and joint_logits is not None:
                                counter.all_five_joint_jsd_sum += float(
                                    Jensen_shannon_divergence(
                                        fused[subset_index].unsqueeze(0), joint_logits.unsqueeze(0)
                                    ).item()
                                )
                                counter.all_five_joint_examples += 1

            base_by_operator[target_operator] = {
                "token_accuracy": base_correct / max(1, base_tokens),
                "token_nll": base_nll_sum / max(1, base_tokens),
                "tokens": base_tokens,
            }

    rows: list[dict[str, Any]] = []
    for (mask, mode, alpha), counter in counters.items():
        active_accuracy = counter.active_correct / max(1, counter.active_tokens)
        active_nll = counter.active_nll_sum / max(1, counter.active_tokens)
        inactive_agreement = (
            counter.inactive_argmax_agree / max(1, counter.inactive_tokens)
            if counter.inactive_tokens
            else None
        )
        inactive_rms = (
            math.sqrt(counter.inactive_delta_sq_sum / max(1, counter.inactive_delta_elements))
            if counter.inactive_delta_elements
            else None
        )
        rows.append(
            {
                "cohort_id": cohort.cohort_id,
                "source": cohort.source,
                "subset_mask": mask,
                "subset_id": f"subset_{mask:02d}",
                "operators": list(subset_operators(mask)),
                "cardinality": len(subset_operators(mask)),
                "mode": mode,
                "alpha": float(alpha),
                "active_token_accuracy": active_accuracy,
                "active_token_nll": active_nll,
                "inactive_base_argmax_agreement": inactive_agreement,
                "inactive_centered_delta_rms": inactive_rms,
                "all_five_joint_jsd": (
                    counter.all_five_joint_jsd_sum / counter.all_five_joint_examples
                    if counter.all_five_joint_examples
                    else None
                ),
            }
        )

    del base, units, joint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, {"cohort_id": cohort.cohort_id, "base_by_operator": base_by_operator}


def _winner_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    inactive_rms = row["inactive_centered_delta_rms"]
    joint_jsd = row["all_five_joint_jsd"]
    return (
        -float(row["active_token_accuracy"]),
        float(row["active_token_nll"]),
        float(inactive_rms) if inactive_rms is not None else 0.0,
        float(joint_jsd) if joint_jsd is not None else 0.0,
    )


def select_winners(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["cohort_id"]), int(row["subset_mask"])), []).append(row)
    return [min(group, key=_winner_sort_key) for _, group in sorted(grouped.items())]


def aggregate_fusion_factory(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row["source"] != "fusion_factory":
            continue
        key = (int(row["subset_mask"]), str(row["mode"]), float(row["alpha"]))
        grouped.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (mask, mode, alpha), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        accuracies = torch.tensor([float(row["active_token_accuracy"]) for row in group])
        nlls = torch.tensor([float(row["active_token_nll"]) for row in group])
        inactive = [row["inactive_centered_delta_rms"] for row in group if row["inactive_centered_delta_rms"] is not None]
        joint = [row["all_five_joint_jsd"] for row in group if row["all_five_joint_jsd"] is not None]
        aggregate.append(
            {
                "subset_mask": mask,
                "subset_id": f"subset_{mask:02d}",
                "operators": list(subset_operators(mask)),
                "cardinality": len(subset_operators(mask)),
                "mode": mode,
                "alpha": alpha,
                "seed_count": len(group),
                "active_token_accuracy_mean": float(accuracies.mean().item()),
                "active_token_accuracy_std": float(accuracies.std(unbiased=False).item()),
                "active_token_nll_mean": float(nlls.mean().item()),
                "inactive_centered_delta_rms_mean": sum(float(value) for value in inactive) / len(inactive) if inactive else None,
                "all_five_joint_jsd_mean": sum(float(value) for value in joint) / len(joint) if joint else None,
            }
        )
    return aggregate


def select_aggregate_winners(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["subset_mask"]), []).append(row)

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        stability_adjusted = float(row["active_token_accuracy_mean"]) - 0.25 * float(
            row["active_token_accuracy_std"]
        )
        inactive = row["inactive_centered_delta_rms_mean"]
        joint = row["all_five_joint_jsd_mean"]
        return (
            -stability_adjusted,
            float(row["active_token_nll_mean"]),
            float(inactive) if inactive is not None else 0.0,
            float(joint) if joint is not None else 0.0,
        )

    return [min(group, key=key) for _, group in sorted(grouped.items())]


def search_combinations(
    *,
    root: Path,
    source: str,
    alpha_grid: Sequence[float],
    examples_per_operator: int,
    evaluation_seed: int,
    device_name: str,
) -> dict[str, Any]:
    if examples_per_operator <= 0:
        raise ValueError("examples_per_operator must be positive")
    if evaluation_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    )
    cohorts = discover_cohorts(root, source)
    if not cohorts:
        raise RuntimeError("no complete model cohorts were found")

    all_rows: list[dict[str, Any]] = []
    base_reports: list[dict[str, Any]] = []
    for cohort in cohorts:
        rows, base_report = _evaluate_cohort(
            cohort,
            root=root,
            alpha_grid=alpha_grid,
            examples_per_operator=examples_per_operator,
            evaluation_seed=evaluation_seed,
            device=device,
        )
        all_rows.extend(rows)
        base_reports.append(base_report)

    aggregate = aggregate_fusion_factory(all_rows)
    aggregate_winners = select_aggregate_winners(aggregate)
    full_mask = (1 << len(EXPERIMENT_OPERATORS)) - 1
    all_five = next((row for row in aggregate_winners if row["subset_mask"] == full_mask), None)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_combination_search",
        "claim_boundary": "validation-only calibration; final IID/OOD splits remain unopened",
        "source": source,
        "device": str(device),
        "evaluation_seed": evaluation_seed,
        "examples_per_operator": examples_per_operator,
        "alpha_grid": [float(value) for value in alpha_grid],
        "modes": ["raw_sum", "rms_equalized_sum"],
        "algebra_note": "bias_mean at alpha a is exactly raw_sum at alpha a/cardinality and is not searched separately",
        "cohorts": [
            {
                "cohort_id": cohort.cohort_id,
                "source": cohort.source,
                "config": str(cohort.config_path.relative_to(root)),
                "base_checkpoint": str(cohort.base_checkpoint.relative_to(root)),
                "unit_checkpoints": {
                    operator: str(path.relative_to(root)) for operator, path in cohort.unit_checkpoints.items()
                },
                "joint_checkpoint": str(cohort.joint_checkpoint.relative_to(root)) if cohort.joint_checkpoint else None,
                **cohort.metadata,
            }
            for cohort in cohorts
        ],
        "base_reports": base_reports,
        "rows": all_rows,
        "winners_by_cohort_and_subset": select_winners(all_rows),
        "fusion_factory_cross_seed_rows": aggregate,
        "fusion_factory_cross_seed_winners": aggregate_winners,
        "recommended_all_five_validation_candidate": all_five,
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustively search specialist subsets, scales, and RMS equalization on validation data"
    )
    parser.add_argument("--source", choices=("all", "fusion-factory", "bias-factory"), default="all")
    parser.add_argument("--alpha-grid", default=",".join(str(value) for value in DEFAULT_ALPHA_GRID))
    parser.add_argument("--examples-per-operator", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_combination_search/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = search_combinations(
        root=root,
        source=args.source,
        alpha_grid=parse_alpha_grid(args.alpha_grid),
        examples_per_operator=args.examples_per_operator,
        evaluation_seed=args.evaluation_seed,
        device_name=args.device,
    )
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    recommended = report.get("recommended_all_five_validation_candidate")
    if recommended is not None:
        print(json.dumps(recommended, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
