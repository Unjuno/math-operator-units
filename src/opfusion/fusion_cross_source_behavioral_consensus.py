from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_behavioral_admission_seed_replication as deterministic
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_unseen_source_rollout_consensus as rollout
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
NUISANCE_SOURCE = "scalar.neg"
DEFAULT_DATA_SEED = 747_000
SCORE_KEYS = (
    "consensus",
    "peer_support",
    "peer_support_weighted",
    "consensus_x_peer",
    "consensus_x_weighted_peer",
)


def modal_output(outputs: Sequence[int | None]) -> tuple[int | None, float]:
    if not outputs:
        raise ValueError("outputs must be non-empty")
    parsed = [int(value) for value in outputs if value is not None]
    counts = Counter(parsed)
    if not counts:
        return None, 0.0
    best_count = max(counts.values())
    best = sorted(value for value, count in counts.items() if count == best_count)
    modal = best[0] if len(best) == 1 else None
    return modal, best_count / len(outputs)


def attach_peer_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach source-order-equivariant agreement scores within one semantic example."""
    if not rows:
        return []
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        modal = row.get("modal_value")
        peers = [peer for peer_index, peer in enumerate(rows) if peer_index != index]
        if modal is None or not peers:
            support = 0.0
            weighted = 0.0
            agreeing = 0
        else:
            agreeing_rows = [peer for peer in peers if peer.get("modal_value") == modal]
            agreeing = len(agreeing_rows)
            support = agreeing / len(peers)
            denominator = sum(float(peer["consensus"]) for peer in peers)
            numerator = sum(
                float(peer["consensus"])
                for peer in peers
                if peer.get("modal_value") == modal
            )
            weighted = numerator / denominator if denominator > 0.0 else 0.0
        item = dict(row)
        item.update(
            {
                "peer_agreeing_sources": agreeing,
                "peer_sources": len(peers),
                "peer_support": support,
                "peer_support_weighted": weighted,
                "consensus_x_peer": float(row["consensus"]) * support,
                "consensus_x_weighted_peer": float(row["consensus"]) * weighted,
            }
        )
        enriched.append(item)
    return enriched


def _auc(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return rollout.pairwise_auc(rows, score_key=key, label_key="majority_correct")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[key]) for row in rows) / len(rows)


def score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if int(row["majority_correct"]) == 1]
    negative = [row for row in rows if int(row["majority_correct"]) == 0]
    stable = [row for row in rows if float(row["consensus"]) >= 1.0 - 1e-12]
    stable_positive = [row for row in stable if int(row["majority_correct"]) == 1]
    stable_negative = [row for row in stable if int(row["majority_correct"]) == 0]
    scores: dict[str, Any] = {}
    for key in SCORE_KEYS:
        scores[key] = {
            "auc_all": _auc(rows, key),
            "auc_consensus_one": _auc(stable, key),
            "mean_positive": _mean(positive, key),
            "mean_negative": _mean(negative, key),
            "mean_consensus_one_positive": _mean(stable_positive, key),
            "mean_consensus_one_negative": _mean(stable_negative, key),
        }
    return {
        "rows": len(rows),
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "consensus_one_rows": len(stable),
        "consensus_one_positive_rows": len(stable_positive),
        "consensus_one_negative_rows": len(stable_negative),
        "scores": scores,
    }


def selection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["cohort_id"], row["task_operator"], row["sample_index"])
        grouped.setdefault(key, []).append(row)
    result: dict[str, Any] = {}
    for score_key in SCORE_KEYS:
        examples = 0
        top_any_correct = 0
        top_all_correct = 0
        matching_in_top = 0
        matching_strict_top = 0
        for (_, operator, _), group in grouped.items():
            if not group:
                continue
            examples += 1
            best = max(float(row[score_key]) for row in group)
            top = [row for row in group if float(row[score_key]) == best]
            top_any_correct += int(any(int(row["majority_correct"]) for row in top))
            top_all_correct += int(all(int(row["majority_correct"]) for row in top))
            matching = [row for row in group if row["source"] == operator]
            if matching:
                matching_score = float(matching[0][score_key])
                matching_in_top += int(matching_score == best)
                other_best = max(
                    [float(row[score_key]) for row in group if row["source"] != operator],
                    default=float("-inf"),
                )
                matching_strict_top += int(matching_score > other_best)
        result[score_key] = {
            "examples": examples,
            "top_set_contains_correct_rate": top_any_correct / max(1, examples),
            "all_top_sources_correct_rate": top_all_correct / max(1, examples),
            "matching_specialist_in_top_rate": matching_in_top / max(1, examples),
            "matching_specialist_strict_top_rate": matching_strict_top / max(1, examples),
        }
    return result


def contextual_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in [*FUNCTIONAL_OPERATORS, NUISANCE_SOURCE, "Base"]:
        source_rows = [row for row in rows if row["source"] == source]
        matching = [row for row in source_rows if row["task_operator"] == source]
        nonmatching = [row for row in source_rows if row["task_operator"] != source]
        stable_wrong = [
            row
            for row in source_rows
            if float(row["consensus"]) >= 1.0 - 1e-12
            and int(row["majority_correct"]) == 0
        ]
        output[source] = {
            "rows": len(source_rows),
            "matching_rows": len(matching),
            "nonmatching_rows": len(nonmatching),
            "stable_wrong_rows": len(stable_wrong),
            "matching_mean_consensus": _mean(matching, "consensus"),
            "nonmatching_mean_consensus": _mean(nonmatching, "consensus"),
            "stable_wrong_mean_peer_support": _mean(stable_wrong, "peer_support"),
            "stable_wrong_mean_weighted_peer_support": _mean(
                stable_wrong, "peer_support_weighted"
            ),
        }
    return output


def evaluate_cohort(
    cohort,
    *,
    root: Path,
    examples_per_operator: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
        if operator in FUNCTIONAL_OPERATORS or operator == NUISANCE_SOURCE
    }
    sources: list[tuple[str, torch.nn.Module]] = [("Base", base)]
    sources.extend((operator, units[operator]) for operator in FUNCTIONAL_OPERATORS)
    if NUISANCE_SOURCE in units:
        sources.append((NUISANCE_SOURCE, units[NUISANCE_SOURCE]))

    rows: list[dict[str, Any]] = []
    for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
        for sample_index in range(examples_per_operator):
            values, _ = factory._initial_values(
                operator,
                seed=data_seed,
                split="validation",
                step=operator_index,
                sample_index=sample_index,
            )
            views = rollout.equivalent_views(values)
            expected = sequential.apply_operator(operator, values)
            group: list[dict[str, Any]] = []
            for source_name, model in sources:
                outputs: list[int | None] = []
                for view_values in views:
                    prompt = sequential.prompt_ids_for_values(
                        factory=factory,
                        tokenizer=tokenizer,
                        operator=operator,
                        values=view_values,
                    )
                    outputs.append(
                        rollout._generate_model_value(
                            model,
                            prompt=prompt,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            device=device,
                        )
                    )
                modal, consensus = modal_output(outputs)
                correct_count = sum(value == expected for value in outputs)
                group.append(
                    {
                        "cohort_id": cohort.cohort_id,
                        "model_seed": cohort.metadata.get("seed"),
                        "task_operator": operator,
                        "sample_index": sample_index,
                        "source": source_name,
                        "matching_specialist": int(source_name == operator),
                        "values": list(values),
                        "expected": expected,
                        "outputs": outputs,
                        "modal_value": modal,
                        "consensus": consensus,
                        "majority_correct": int(correct_count * 2 > len(outputs)),
                        "correct_fraction": correct_count / len(outputs),
                    }
                )
            rows.extend(attach_peer_scores(group))

    del base, units, sources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_experiment(
    *,
    root: Path,
    examples_per_operator: int,
    data_seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    runtime = deterministic.configure_deterministic_runtime()
    device = torch.device("cpu")
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")
    rows: list[dict[str, Any]] = []
    for cohort in cohorts[:3]:
        rows.extend(
            evaluate_cohort(
                cohort,
                root=root,
                examples_per_operator=examples_per_operator,
                data_seed=data_seed,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "cross_source_semantic_view_consensus_diagnostic",
        "runtime": runtime,
        "data_seed": data_seed,
        "examples_per_operator_per_cohort": examples_per_operator,
        "cohorts": 3,
        "source_count": 6,
        "score_summary": score_summary(rows),
        "selection_summary": selection_summary(rows),
        "contextual_summary": contextual_summary(rows),
        "claim_boundary": (
            "diagnostic only: no admission gate is trained; peer agreement uses modal parsed outputs from Base, four functional "
            "specialists and failed NEG on the same semantic example; correctness and matching labels are posthoc only; source "
            "identity is not an input to any score; all sources share the arithmetic tokenizer/architecture family and hand-known "
            "commutative semantic views"
        ),
        "rows": rows,
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether cross-source agreement resolves stable-wrong semantic-view consensus"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--examples-per-operator", type=int, default=6)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_experiment(
        root=args.root,
        examples_per_operator=args.examples_per_operator,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        "score_summary": report["score_summary"],
        "selection_summary": report["selection_summary"],
        "contextual_summary": report["contextual_summary"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
