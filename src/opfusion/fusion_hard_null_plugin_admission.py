from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_mixed_novel_plugin_admission as mixed
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_unseen_source_behavioral_admission as behavioral
from opfusion import fusion_unseen_source_residual_gating as residual
from opfusion import fusion_unseen_source_rollout_consensus as rollout
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
NUISANCE_SOURCE = mixed.NUISANCE_SOURCE
DEFAULT_CALIBRATION_SEED = 746_000
_ACTIVE_THRESHOLD: float | None = None
_ORIGINAL_GENERATE_VALUE = mixed._generate_value


def choose_hard_threshold(rows: Sequence[Mapping[str, float | int]]) -> dict[str, Any]:
    """Choose a source-identity-free threshold by balanced accuracy on seen-source calibration rows."""
    if not rows:
        raise ValueError("calibration rows must be non-empty")
    scores = sorted({float(row["consensus"]) for row in rows})
    positives = sum(int(row["majority_correct"]) for row in rows)
    negatives = len(rows) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("calibration requires positive and negative rows")

    candidates = [0.0]
    for left, right in zip(scores, scores[1:]):
        candidates.append((left + right) / 2.0)
    candidates.extend([scores[-1], 1.000001])

    evaluated: list[dict[str, float | int]] = []
    for threshold in sorted(set(candidates)):
        tp = fp = tn = fn = 0
        for row in rows:
            predicted = float(row["consensus"]) >= threshold
            actual = bool(int(row["majority_correct"]))
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1
        tpr = tp / positives
        tnr = tn / negatives
        balanced_accuracy = 0.5 * (tpr + tnr)
        evaluated.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": tpr,
                "tnr": tnr,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    # Conservative tie-break: prefer the higher threshold when balanced accuracy ties.
    best = max(evaluated, key=lambda row: (float(row["balanced_accuracy"]), float(row["threshold"])))
    return {
        "threshold": float(best["threshold"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "tpr": float(best["tpr"]),
        "tnr": float(best["tnr"]),
        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
        "tn": int(best["tn"]),
        "fn": int(best["fn"]),
        "rows": len(rows),
        "positive_rows": positives,
        "negative_rows": negatives,
        "candidate_table": evaluated,
    }


def calibrate_threshold(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    heldout: str,
    examples_per_operator: int,
    calibration_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    seen = [operator for operator in FUNCTIONAL_OPERATORS if operator != heldout]
    rows: list[dict[str, Any]] = []
    for cohort in cohorts[:2]:
        run = load_run_config(cohort.config_path)
        tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
        factory = SyntheticTraceFactory(tokenizer, run.data)
        units = {
            source: _load_model(cohort.unit_checkpoints[source], device=device, tokenizer=tokenizer)
            for source in seen
        }
        for task_index, operator in enumerate(seen):
            for sample_index in range(examples_per_operator):
                values, _ = factory._initial_values(
                    operator,
                    seed=calibration_seed,
                    split="validation",
                    step=task_index,
                    sample_index=sample_index,
                )
                expected = sequential.apply_operator(operator, values)
                for source in seen:
                    outputs: list[int | None] = []
                    for view_values in rollout.equivalent_views(values):
                        prompt = sequential.prompt_ids_for_values(
                            factory=factory,
                            tokenizer=tokenizer,
                            operator=operator,
                            values=view_values,
                        )
                        outputs.append(
                            rollout._generate_model_value(
                                units[source],
                                prompt=prompt,
                                tokenizer=tokenizer,
                                max_new_tokens=max_new_tokens,
                                device=device,
                            )
                        )
                    parsed = [value for value in outputs if value is not None]
                    modal_count = max(Counter(parsed).values(), default=0)
                    correct_count = sum(value == expected for value in outputs)
                    rows.append(
                        {
                            "cohort_id": cohort.cohort_id,
                            "task_operator": operator,
                            "source": source,
                            "consensus": modal_count / len(outputs),
                            "majority_correct": int(correct_count * 2 > len(outputs)),
                        }
                    )
        del units
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = choose_hard_threshold(rows)
    report["calibration_seed"] = calibration_seed
    report["examples_per_operator_per_cohort"] = examples_per_operator
    report["seen_operators"] = seen
    report["heldout_operator_excluded"] = heldout
    return report


def hard_admission_factor(consensus: float, *, power: float, threshold: float) -> float:
    if consensus < threshold:
        return 0.0
    return float(consensus) ** float(power)


def _generate_value_hard(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    scorer: residual.IndependentResidualGate,
    heldout: str,
    appended: Sequence[str],
    behavioral_sources: frozenset[str],
    consensus_power: float,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
    consensus_cache: dict[tuple[str, str, tuple[int, ...]], float],
) -> tuple[int | None, dict[str, dict[str, float]]]:
    if _ACTIVE_THRESHOLD is None:
        raise RuntimeError("hard admission threshold is not active")
    prompt = sequential.prompt_ids_for_values(
        factory=factory,
        tokenizer=tokenizer,
        operator=operator,
        values=values,
    )
    ids = torch.tensor([prompt], dtype=torch.long, device=device)

    factors: dict[str, float] = {}
    consensuses: dict[str, float] = {}
    for source in appended:
        consensus = 1.0
        if source in behavioral_sources:
            key = (source, operator, tuple(int(value) for value in values))
            if key not in consensus_cache:
                consensus_cache[key] = behavioral.behavioral_consensus(
                    units[source],
                    operator=operator,
                    values=values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            consensus = float(consensus_cache[key])
        consensuses[source] = consensus
        factors[source] = (
            hard_admission_factor(
                consensus,
                power=consensus_power,
                threshold=float(_ACTIVE_THRESHOLD),
            )
            if source in behavioral_sources
            else 1.0
        )

    generated: list[int] = []
    raw_gate_sum = defaultdict(float)
    effective_gate_sum = defaultdict(float)
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            names, sources = mixed._selected_logits(
                base=base,
                units=units,
                ids=ids,
                heldout=heldout,
                appended=appended,
            )
            gates = scorer.gates_from_logits(sources)
            effective = gates.clone()
            specialist_names = names[1:]
            for source in appended:
                index = specialist_names.index(source)
                raw_gate_sum[source] += float(gates[index].detach().cpu())
                effective[index] = effective[index] * factors[source]
                effective_gate_sum[source] += float(effective[index].detach().cpu())
            fused = residual.fuse_residual_logits(sources, effective)
            next_id = int(fused.argmax().item())
            generated.append(next_id)
            positions += 1
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == tokenizer.eos_id:
                break

    diagnostics: dict[str, dict[str, float]] = {}
    for source in appended:
        diagnostics[source] = {
            "consensus": consensuses[source],
            "admission_factor": factors[source],
            "mean_raw_gate": raw_gate_sum[source] / max(1, positions),
            "mean_effective_gate": effective_gate_sum[source] / max(1, positions),
        }
    return sequential.parse_final_numeric_token(generated, tokenizer), diagnostics


def evaluate_hard_mode(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    scorer: residual.IndependentResidualGate,
    heldout: str,
    appended: Sequence[str],
    behavioral_sources: frozenset[str],
    consensus_power: float,
    threshold: float,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    global _ACTIVE_THRESHOLD
    previous_generator = mixed._generate_value
    previous_threshold = _ACTIVE_THRESHOLD
    mixed._generate_value = _generate_value_hard
    _ACTIVE_THRESHOLD = float(threshold)
    try:
        return mixed.evaluate_mode(
            cohorts,
            root=root,
            scorer=scorer,
            heldout=heldout,
            appended=appended,
            behavioral_sources=behavioral_sources,
            consensus_power=consensus_power,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    finally:
        mixed._generate_value = previous_generator
        _ACTIVE_THRESHOLD = previous_threshold


def _count(block: Mapping[str, Any], subset: str) -> int:
    return int(block["subsets"][subset]["end_to_end_correct"])


def _delta(mode: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, int]:
    result = {
        subset: _count(mode, subset) - _count(baseline, subset)
        for subset in ("heldout_involved", "heldout_neither")
    }
    result["net"] = result["heldout_involved"] + result["heldout_neither"]
    return result


def _marginal(mixed_mode: Mapping[str, Any], useful_mode: Mapping[str, Any]) -> dict[str, int]:
    result = {
        subset: _count(mixed_mode, subset) - _count(useful_mode, subset)
        for subset in ("heldout_involved", "heldout_neither")
    }
    result["net"] = result["heldout_involved"] + result["heldout_neither"]
    return result


def run_experiment(
    *,
    root: Path,
    heldout: str,
    scorer_seed: int,
    train_seed: int,
    calibration_seed: int,
    calibration_examples: int,
    data_seed: int,
    consensus_power: float,
    examples_per_pair: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    runtime = mixed.deterministic.configure_deterministic_runtime()
    device = torch.device("cpu")
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    calibration = calibrate_threshold(
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=calibration_examples,
        calibration_seed=calibration_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    threshold = float(calibration["threshold"])

    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    torch.manual_seed(scorer_seed)
    scorer = residual.IndependentResidualGate(
        vocabulary_size=tokenizer.vocab_size,
        sketch_size=32,
        hidden_size=64,
        sketch_seed=743000,
        max_gate=4.0,
    ).to(device)
    batch = residual.collect_training_batch(
        scorer,
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=24,
        data_seed=train_seed,
        max_positions_per_cohort=3072,
        device=device,
    )
    fit_report = residual.fit_residual_gate(
        scorer,
        batch,
        learning_rate=0.01,
        steps=600,
        batch_positions=256,
        gate_penalty=0.01,
        seed=scorer_seed,
        device=device,
    )

    without = mixed.evaluate_mode(
        cohorts,
        root=root,
        scorer=scorer,
        heldout=heldout,
        appended=(),
        behavioral_sources=frozenset(),
        consensus_power=consensus_power,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    continuous_useful = mixed.evaluate_mode(
        cohorts,
        root=root,
        scorer=scorer,
        heldout=heldout,
        appended=(heldout,),
        behavioral_sources=frozenset({heldout}),
        consensus_power=consensus_power,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    continuous_mixed = mixed.evaluate_mode(
        cohorts,
        root=root,
        scorer=scorer,
        heldout=heldout,
        appended=(heldout, NUISANCE_SOURCE),
        behavioral_sources=frozenset({heldout, NUISANCE_SOURCE}),
        consensus_power=consensus_power,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    hard_useful = evaluate_hard_mode(
        cohorts,
        root=root,
        scorer=scorer,
        heldout=heldout,
        appended=(heldout,),
        behavioral_sources=frozenset({heldout}),
        consensus_power=consensus_power,
        threshold=threshold,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    hard_mixed = evaluate_hard_mode(
        cohorts,
        root=root,
        scorer=scorer,
        heldout=heldout,
        appended=(heldout, NUISANCE_SOURCE),
        behavioral_sources=frozenset({heldout, NUISANCE_SOURCE}),
        consensus_power=consensus_power,
        threshold=threshold,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "seen_calibrated_hard_null_mixed_plugin_admission_pilot",
        "heldout_operator": heldout,
        "nuisance_source": NUISANCE_SOURCE,
        "scorer_seed": scorer_seed,
        "runtime": runtime,
        "calibration": calibration,
        "scorer_fit": fit_report,
        "modes": {
            "without_plugins": without,
            "continuous_useful": continuous_useful,
            "continuous_mixed": continuous_mixed,
            "hard_useful": hard_useful,
            "hard_mixed": hard_mixed,
        },
        "correct_count_deltas": {
            "continuous_useful": _delta(continuous_useful, without),
            "continuous_mixed": _delta(continuous_mixed, without),
            "hard_useful": _delta(hard_useful, without),
            "hard_mixed": _delta(hard_mixed, without),
            "nuisance_marginal_continuous": _marginal(continuous_mixed, continuous_useful),
            "nuisance_marginal_hard": _marginal(hard_mixed, hard_useful),
        },
        "claim_boundary": (
            "hard threshold is selected only from seen functional specialists on a separate calibration seed and first two cohorts; "
            "held-out operator examples/sources and NEG do not participate in threshold selection; threshold uses correctness labels "
            "during meta-calibration but evaluation admission remains source-identity-free; deterministic one-seed mixed-plugin pilot"
        ),
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test seen-calibrated hard-null admission for useful+nuisance novel plugins"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--scorer-seed", type=int, default=741000)
    parser.add_argument("--train-seed", type=int, default=741000)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--calibration-examples", type=int, default=6)
    parser.add_argument("--data-seed", type=int, default=741500)
    parser.add_argument("--consensus-power", type=float, default=2.0)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_experiment(
        root=args.root,
        heldout=args.heldout,
        scorer_seed=args.scorer_seed,
        train_seed=args.train_seed,
        calibration_seed=args.calibration_seed,
        calibration_examples=args.calibration_examples,
        data_seed=args.data_seed,
        consensus_power=args.consensus_power,
        examples_per_pair=args.examples_per_pair,
        max_new_tokens=args.max_new_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "heldout_operator": report["heldout_operator"],
                "calibration": {key: report["calibration"][key] for key in ("threshold", "balanced_accuracy", "tpr", "tnr", "rows")},
                "correct_count_deltas": report["correct_count_deltas"],
                "hard_mixed_diagnostics": report["modes"]["hard_mixed"]["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
