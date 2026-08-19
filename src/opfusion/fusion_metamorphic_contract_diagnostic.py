from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
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
DEFAULT_DATA_SEED = 748_000
SCORE_KEYS = (
    "consensus",
    "contract_rate",
    "contract_weighted",
    "consensus_x_contract",
    "consensus_x_weighted_contract",
)


@dataclass(frozen=True)
class ContractProbe:
    probe_id: str
    values: tuple[int, ...]
    output_delta: int


def _bounded_step(value: int, *, positive: bool) -> int:
    if positive:
        return 1 if value < 16 else -1
    return -1 if value > -16 else 1


def contract_probes(operator: str, values: Sequence[int]) -> tuple[ContractProbe, ...]:
    """Task-contract transformations whose output delta is known without knowing f(values)."""
    original = tuple(int(value) for value in values)
    if len(original) < 2:
        raise ValueError("contract probes require at least two values")
    probes: list[ContractProbe] = []

    if operator in ("scalar.add", "aggregation.sum"):
        for index, value in enumerate(original):
            delta = _bounded_step(value, positive=True)
            transformed = list(original)
            transformed[index] += delta
            probes.append(
                ContractProbe(
                    probe_id=f"single_operand_delta_{index}",
                    values=tuple(transformed),
                    output_delta=delta,
                )
            )
        return tuple(probes)

    if operator == "scalar.min":
        min_value = min(original)
        min_index = original.index(min_value)
        if min_value > -16:
            transformed = list(original)
            transformed[min_index] -= 1
            probes.append(ContractProbe("lower_min", tuple(transformed), -1))
        else:
            transformed = tuple(value + 1 for value in original)
            probes.append(ContractProbe("translate_min_up", transformed, 1))
        for index, value in enumerate(original):
            if value > min_value and value < 16:
                transformed = list(original)
                transformed[index] += 1
                probes.append(ContractProbe("raise_non_min", tuple(transformed), 0))
                break
        return tuple(probes)

    if operator == "scalar.max":
        max_value = max(original)
        max_index = original.index(max_value)
        if max_value < 16:
            transformed = list(original)
            transformed[max_index] += 1
            probes.append(ContractProbe("raise_max", tuple(transformed), 1))
        else:
            transformed = tuple(value - 1 for value in original)
            probes.append(ContractProbe("translate_max_down", transformed, -1))
        for index, value in enumerate(original):
            if value < max_value and value > -16:
                transformed = list(original)
                transformed[index] -= 1
                probes.append(ContractProbe("lower_non_max", tuple(transformed), 0))
                break
        return tuple(probes)

    raise KeyError(operator)


def modal_output(outputs: Sequence[int | None]) -> tuple[int | None, float]:
    if not outputs:
        raise ValueError("outputs must be non-empty")
    parsed = [int(value) for value in outputs if value is not None]
    counts = Counter(parsed)
    if not counts:
        return None, 0.0
    count = max(counts.values())
    winners = sorted(value for value, value_count in counts.items() if value_count == count)
    return (winners[0] if len(winners) == 1 else None), count / len(outputs)


def summarize_contract(
    original_modal: int | None,
    original_consensus: float,
    probe_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    if not probe_rows:
        return {
            "contract_probes": 0,
            "contract_passes": 0,
            "contract_rate": 0.0,
            "contract_weighted": 0.0,
            "consensus_x_contract": 0.0,
            "consensus_x_weighted_contract": 0.0,
        }
    passes = 0
    weighted = 0.0
    for probe in probe_rows:
        probe_modal = probe["modal_value"]
        passed = int(
            original_modal is not None
            and probe_modal is not None
            and int(probe_modal) == int(original_modal) + int(probe["output_delta"])
        )
        passes += passed
        weighted += passed * min(float(original_consensus), float(probe["consensus"]))
    rate = passes / len(probe_rows)
    weighted_rate = weighted / len(probe_rows)
    return {
        "contract_probes": len(probe_rows),
        "contract_passes": passes,
        "contract_rate": rate,
        "contract_weighted": weighted_rate,
        "consensus_x_contract": float(original_consensus) * rate,
        "consensus_x_weighted_contract": float(original_consensus) * weighted_rate,
    }


def _auc(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return rollout.pairwise_auc(rows, score_key=key, label_key="majority_correct")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if int(row["majority_correct"]) == 1]
    negative = [row for row in rows if int(row["majority_correct"]) == 0]
    stable = [row for row in rows if float(row["consensus"]) >= 1.0 - 1e-12]
    stable_positive = [row for row in stable if int(row["majority_correct"]) == 1]
    stable_negative = [row for row in stable if int(row["majority_correct"]) == 0]
    return {
        "rows": len(rows),
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "consensus_one_rows": len(stable),
        "consensus_one_positive_rows": len(stable_positive),
        "consensus_one_negative_rows": len(stable_negative),
        "scores": {
            key: {
                "auc_all": _auc(rows, key),
                "auc_consensus_one": _auc(stable, key),
                "mean_positive": _mean(positive, key),
                "mean_negative": _mean(negative, key),
                "mean_consensus_one_positive": _mean(stable_positive, key),
                "mean_consensus_one_negative": _mean(stable_negative, key),
            }
            for key in SCORE_KEYS
        },
    }


def selection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["cohort_id"], row["task_operator"], row["sample_index"])
        grouped.setdefault(key, []).append(row)
    output: dict[str, Any] = {}
    for score_key in SCORE_KEYS:
        examples = 0
        top_any_correct = top_all_correct = matching_in_top = matching_strict = 0
        for (_, operator, _), group in grouped.items():
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
                matching_strict += int(matching_score > other_best)
        output[score_key] = {
            "examples": examples,
            "top_set_contains_correct_rate": top_any_correct / max(1, examples),
            "all_top_sources_correct_rate": top_all_correct / max(1, examples),
            "matching_specialist_in_top_rate": matching_in_top / max(1, examples),
            "matching_specialist_strict_top_rate": matching_strict / max(1, examples),
        }
    return output


def by_task_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        operator: {
            key: {
                "auc_all": _auc([row for row in rows if row["task_operator"] == operator], key),
                "auc_consensus_one": _auc(
                    [
                        row
                        for row in rows
                        if row["task_operator"] == operator
                        and float(row["consensus"]) >= 1.0 - 1e-12
                    ],
                    key,
                ),
            }
            for key in SCORE_KEYS
        }
        for operator in FUNCTIONAL_OPERATORS
    }


def _rollout_summary(
    model: torch.nn.Module,
    *,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[int | None, float, list[int | None]]:
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
                model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        )
    modal, consensus = modal_output(outputs)
    return modal, consensus, outputs


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
            expected = sequential.apply_operator(operator, values)
            probes = contract_probes(operator, values)
            for source_name, model in sources:
                original_modal, consensus, outputs = _rollout_summary(
                    model,
                    operator=operator,
                    values=values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                correct_count = sum(value == expected for value in outputs)
                probe_rows: list[dict[str, Any]] = []
                for probe in probes:
                    probe_modal, probe_consensus, probe_outputs = _rollout_summary(
                        model,
                        operator=operator,
                        values=probe.values,
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )
                    probe_rows.append(
                        {
                            "probe_id": probe.probe_id,
                            "values": list(probe.values),
                            "output_delta": probe.output_delta,
                            "modal_value": probe_modal,
                            "consensus": probe_consensus,
                            "outputs": probe_outputs,
                        }
                    )
                contract = summarize_contract(original_modal, consensus, probe_rows)
                rows.append(
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
                        "modal_value": original_modal,
                        "consensus": consensus,
                        "majority_correct": int(correct_count * 2 > len(outputs)),
                        "correct_fraction": correct_count / len(outputs),
                        "probe_rows": probe_rows,
                        **contract,
                    }
                )

    del base, units, sources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_experiment(
    *, root: Path, examples_per_operator: int, data_seed: int, max_new_tokens: int
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
        "evaluation_role": "task_conditioned_metamorphic_contract_diagnostic",
        "runtime": runtime,
        "data_seed": data_seed,
        "examples_per_operator_per_cohort": examples_per_operator,
        "score_summary": summarize_scores(rows),
        "selection_summary": selection_summary(rows),
        "by_task": by_task_summary(rows),
        "claim_boundary": (
            "diagnostic only; admission scores do not use the exact expected answer or source semantic identity, but contract "
            "transformations are selected from the known requested arithmetic operator semantics; correctness and matching labels "
            "are posthoc; all sources share the arithmetic tokenizer/architecture family and stage execution remains external"
        ),
        "rows": rows,
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose task-conditioned metamorphic contracts for plugin admission")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--examples-per-operator", type=int, default=4)
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
    print(json.dumps({
        "score_summary": report["score_summary"],
        "selection_summary": report["selection_summary"],
        "by_task": report["by_task"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
