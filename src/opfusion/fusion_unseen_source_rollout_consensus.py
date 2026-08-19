from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_verify import _next_logits
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
DEFAULT_DATA_SEED = 745_000


def equivalent_views(values: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Deterministic meaning-preserving permutations for commutative operators."""
    original = tuple(int(value) for value in values)
    candidates = [original, tuple(reversed(original))]
    if len(original) >= 3:
        candidates.append((*original[1:], original[0]))
    unique: list[tuple[int, ...]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _generate_model_value(
    model: torch.nn.Module,
    *,
    prompt: Sequence[int],
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> int | None:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _next_logits(model, ids)
            next_id = int(logits.argmax().item())
            generated.append(next_id)
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id == tokenizer.eos_id:
                break
    return sequential.parse_final_numeric_token(generated, tokenizer)


def summarize_view_outputs(
    outputs: Sequence[int | None], *, expected: int
) -> dict[str, float | int | None]:
    view_count = len(outputs)
    if view_count <= 0:
        raise ValueError("outputs must be non-empty")
    parsed = [value for value in outputs if value is not None]
    counts = Counter(parsed)
    modal_count = max(counts.values(), default=0)
    modal_values = sorted(value for value, count in counts.items() if count == modal_count)
    correct_count = sum(value == expected for value in outputs)
    return {
        "views": view_count,
        "parse_count": len(parsed),
        "parse_rate": len(parsed) / view_count,
        "consensus": modal_count / view_count,
        "unique_parsed_values": len(counts),
        "correct_count": correct_count,
        "correct_fraction": correct_count / view_count,
        "majority_correct": int(correct_count * 2 > view_count),
        "all_views_correct": int(correct_count == view_count),
        "modal_value": modal_values[0] if len(modal_values) == 1 else None,
        "modal_tied": int(len(modal_values) > 1),
    }


def pairwise_auc(rows: Sequence[Mapping[str, Any]], *, score_key: str, label_key: str) -> float | None:
    positive = [float(row[score_key]) for row in rows if int(row[label_key]) == 1]
    negative = [float(row[score_key]) for row in rows if int(row[label_key]) == 0]
    if not positive or not negative:
        return None
    wins = 0.0
    pairs = 0
    for positive_score in positive:
        for negative_score in negative:
            pairs += 1
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / pairs


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if int(row["majority_correct"]) == 1]
    negatives = [row for row in rows if int(row["majority_correct"]) == 0]
    by_operator: dict[str, Any] = {}
    for operator in FUNCTIONAL_OPERATORS:
        subset = [row for row in rows if row["task_operator"] == operator]
        positive_subset = [row for row in subset if int(row["majority_correct"]) == 1]
        negative_subset = [row for row in subset if int(row["majority_correct"]) == 0]
        by_operator[operator] = {
            "rows": len(subset),
            "positive_rows": len(positive_subset),
            "consensus_auc_for_majority_correct": pairwise_auc(
                subset, score_key="consensus", label_key="majority_correct"
            ),
            "mean_consensus_positive": _mean(positive_subset, "consensus"),
            "mean_consensus_negative": _mean(negative_subset, "consensus"),
            "mean_correct_fraction": _mean(subset, "correct_fraction"),
        }

    contextual: dict[str, Any] = {}
    for source in FUNCTIONAL_OPERATORS:
        matching = [
            row for row in rows if row["source"] == source and row["task_operator"] == source
        ]
        nonmatching = [
            row for row in rows if row["source"] == source and row["task_operator"] != source
        ]
        matching_mean = _mean(matching, "consensus")
        nonmatching_mean = _mean(nonmatching, "consensus")
        contextual[source] = {
            "matching_rows": len(matching),
            "nonmatching_rows": len(nonmatching),
            "matching_mean_consensus": matching_mean,
            "nonmatching_mean_consensus": nonmatching_mean,
            "matching_minus_nonmatching_consensus": (
                None
                if matching_mean is None or nonmatching_mean is None
                else matching_mean - nonmatching_mean
            ),
            "matching_mean_correct_fraction": _mean(matching, "correct_fraction"),
            "nonmatching_mean_correct_fraction": _mean(nonmatching, "correct_fraction"),
        }

    return {
        "rows": len(rows),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "consensus_auc_for_majority_correct": pairwise_auc(
            rows, score_key="consensus", label_key="majority_correct"
        ),
        "parse_rate_auc_for_majority_correct": pairwise_auc(
            rows, score_key="parse_rate", label_key="majority_correct"
        ),
        "mean_consensus_positive": _mean(positives, "consensus"),
        "mean_consensus_negative": _mean(negatives, "consensus"),
        "mean_parse_rate_positive": _mean(positives, "parse_rate"),
        "mean_parse_rate_negative": _mean(negatives, "parse_rate"),
        "mean_correct_fraction": _mean(rows, "correct_fraction"),
        "by_task_operator": by_operator,
        "within_source_context_contrast": contextual,
    }


def summarize_example_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["cohort_id"], row["task_operator"], row["sample_index"])
        grouped[key].append(row)

    top_any_correct = 0
    top_all_correct = 0
    top_mean_correct_fraction = 0.0
    matching_in_top = 0
    matching_strict_top = 0
    examples = 0
    for (_, operator, _), group in grouped.items():
        if not group:
            continue
        examples += 1
        best_consensus = max(float(row["consensus"]) for row in group)
        top = [row for row in group if float(row["consensus"]) == best_consensus]
        top_correct = [int(row["majority_correct"]) for row in top]
        top_any_correct += int(any(top_correct))
        top_all_correct += int(all(top_correct))
        top_mean_correct_fraction += sum(float(row["correct_fraction"]) for row in top) / len(top)
        matching_rows = [row for row in group if row["source"] == operator]
        if matching_rows:
            matching_consensus = float(matching_rows[0]["consensus"])
            matching_in_top += int(matching_consensus == best_consensus)
            other_best = max(
                [float(row["consensus"]) for row in group if row["source"] != operator],
                default=float("-inf"),
            )
            matching_strict_top += int(matching_consensus > other_best)

    return {
        "examples": examples,
        "top_consensus_set_contains_majority_correct_source_rate": top_any_correct / max(1, examples),
        "all_top_consensus_sources_majority_correct_rate": top_all_correct / max(1, examples),
        "mean_correct_fraction_within_top_consensus_set": top_mean_correct_fraction / max(1, examples),
        "matching_specialist_in_top_consensus_set_rate": matching_in_top / max(1, examples),
        "matching_specialist_strictly_highest_consensus_rate": matching_strict_top / max(1, examples),
    }


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    examples_per_operator: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
    include_neg: bool,
) -> list[dict[str, Any]]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
        if operator in FUNCTIONAL_OPERATORS or (include_neg and operator == "scalar.neg")
    }
    sources: list[tuple[str, torch.nn.Module]] = [("Base", base)]
    sources.extend((operator, units[operator]) for operator in FUNCTIONAL_OPERATORS if operator in units)
    if include_neg and "scalar.neg" in units:
        sources.append(("scalar.neg", units["scalar.neg"]))

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
            views = equivalent_views(values)
            expected = sequential.apply_operator(operator, values)
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
                        _generate_model_value(
                            model,
                            prompt=prompt,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            device=device,
                        )
                    )
                summary = summarize_view_outputs(outputs, expected=expected)
                rows.append(
                    {
                        "cohort_id": cohort.cohort_id,
                        "model_seed": cohort.metadata.get("seed"),
                        "task_operator": operator,
                        "sample_index": sample_index,
                        "source": source_name,
                        "matching_specialist": int(source_name == operator),
                        "values": list(values),
                        "view_values": [list(view) for view in views],
                        "expected": expected,
                        "outputs": outputs,
                        **summary,
                    }
                )

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
    include_neg: bool,
    device_name: str,
) -> dict[str, Any]:
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
                include_neg=include_neg,
            )
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "semantic_equivalence_rollout_consensus_diagnostic",
        "claim_boundary": (
            "no fusion gate is trained; each candidate source is greedily rolled out alone on deterministic operand/list "
            "permutations that preserve the final commutative computation; consensus uses only parsed rollout outputs, while "
            "correctness and matching-source labels are posthoc diagnostics; arithmetic-family transformations do not establish "
            "general paraphrase invariance for arbitrary models"
        ),
        "data_seed": data_seed,
        "examples_per_operator_per_cohort": examples_per_operator,
        "cohorts": 3,
        "include_neg": include_neg,
        "summary": summarize_rows(rows),
        "selection_diagnostic": summarize_example_selection(rows),
        "rows": rows,
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether semantic-view rollout consensus predicts unseen-source usefulness"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--examples-per-operator", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--include-neg", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_experiment(
        root=args.root,
        examples_per_operator=args.examples_per_operator,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
        include_neg=args.include_neg,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "selection_diagnostic": report["selection_diagnostic"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
