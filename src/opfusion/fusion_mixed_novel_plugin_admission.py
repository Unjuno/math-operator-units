from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_behavioral_admission_seed_replication as deterministic
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_unseen_source_behavioral_admission as behavioral
from opfusion import fusion_unseen_source_generalization as unseen
from opfusion import fusion_unseen_source_residual_gating as residual
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_verify import _next_logits
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
NUISANCE_SOURCE = "scalar.neg"


def _selected_logits(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    ids: torch.Tensor,
    heldout: str,
    appended: Sequence[str],
) -> tuple[list[str], torch.Tensor]:
    seen = [operator for operator in FUNCTIONAL_OPERATORS if operator != heldout]
    ordered = [*seen]
    for source in appended:
        if source not in ordered:
            ordered.append(source)
    names = ["Base", *ordered]
    logits = [_next_logits(base, ids)]
    logits.extend(_next_logits(units[source], ids) for source in ordered)
    return names, torch.stack(logits, dim=0)


def _generate_value(
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
        factors[source] = consensus ** float(consensus_power) if source in behavioral_sources else 1.0

    generated: list[int] = []
    raw_gate_sum = defaultdict(float)
    effective_gate_sum = defaultdict(float)
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            names, sources = _selected_logits(
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


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    scorer: residual.IndependentResidualGate,
    heldout: str,
    appended: Sequence[str],
    behavioral_sources: frozenset[str],
    consensus_power: float,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
        if operator in set(FUNCTIONAL_OPERATORS) | {NUISANCE_SOURCE}
    }
    for source in [heldout, *appended]:
        if source not in units:
            raise RuntimeError(f"missing checkpoint for source {source}")

    aggregate = sequential._empty_counter()
    pair_rows: dict[str, dict[str, float | int | None]] = {}
    consensus_cache: dict[tuple[str, str, tuple[int, ...]], float] = {}
    diagnostic_sums: dict[str, dict[str, float]] = {
        source: defaultdict(float) for source in appended
    }

    def record(operator: str, diag: Mapping[str, Mapping[str, float]]) -> None:
        for source, row in diag.items():
            sums = diagnostic_sums[source]
            sums["count"] += 1
            sums["consensus"] += float(row["consensus"])
            sums["admission_factor"] += float(row["admission_factor"])
            sums["raw_gate"] += float(row["mean_raw_gate"])
            sums["effective_gate"] += float(row["mean_effective_gate"])
            if source == heldout:
                key = "matching" if operator == heldout else "nonmatching"
                sums[f"{key}_count"] += 1
                sums[f"{key}_consensus"] += float(row["consensus"])

    for inner_operator in FUNCTIONAL_OPERATORS:
        for outer_operator in FUNCTIONAL_OPERATORS:
            pair_id = f"{inner_operator}->{outer_operator}"
            counter = sequential._empty_counter()
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = sequential.composition_operands(
                    inner_operator=inner_operator,
                    outer_operator=outer_operator,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = sequential.apply_operator(inner_operator, inner_values)
                true_outer_values = (true_inner, *outer_extras)
                true_final = sequential.apply_operator(outer_operator, true_outer_values)

                generated_inner, inner_diag = _generate_value(
                    base=base,
                    units=units,
                    scorer=scorer,
                    heldout=heldout,
                    appended=appended,
                    behavioral_sources=behavioral_sources,
                    consensus_power=consensus_power,
                    operator=inner_operator,
                    values=inner_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    consensus_cache=consensus_cache,
                )
                record(inner_operator, inner_diag)

                oracle_outer, oracle_diag = _generate_value(
                    base=base,
                    units=units,
                    scorer=scorer,
                    heldout=heldout,
                    appended=appended,
                    behavioral_sources=behavioral_sources,
                    consensus_power=consensus_power,
                    operator=outer_operator,
                    values=true_outer_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    consensus_cache=consensus_cache,
                )
                record(outer_operator, oracle_diag)

                chained_outer: int | None = None
                if generated_inner is not None:
                    chained_outer, chained_diag = _generate_value(
                        base=base,
                        units=units,
                        scorer=scorer,
                        heldout=heldout,
                        appended=appended,
                        behavioral_sources=behavioral_sources,
                        consensus_power=consensus_power,
                        operator=outer_operator,
                        values=(generated_inner, *outer_extras),
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        consensus_cache=consensus_cache,
                    )
                    record(outer_operator, chained_diag)

                inner_correct = generated_inner == true_inner
                counter["cases"] += 1
                counter["inner_parse"] += int(generated_inner is not None)
                counter["inner_correct"] += int(inner_correct)
                counter["oracle_intermediate_outer_parse"] += int(oracle_outer is not None)
                counter["oracle_intermediate_outer_correct"] += int(oracle_outer == true_final)
                counter["end_to_end_outer_parse"] += int(chained_outer is not None)
                counter["end_to_end_correct"] += int(chained_outer == true_final)
                if inner_correct:
                    counter["inner_correct_cases"] += 1
                    counter["end_to_end_correct_given_inner_correct"] += int(chained_outer == true_final)

            sequential._merge_counter(aggregate, counter)
            pair_rows[pair_id] = sequential._finalize_counter(counter)

    diagnostics: dict[str, Any] = {}
    for source, sums in diagnostic_sums.items():
        count = max(1.0, sums["count"])
        diagnostics[source] = {
            "mean_consensus": sums["consensus"] / count,
            "mean_admission_factor": sums["admission_factor"] / count,
            "mean_raw_gate": sums["raw_gate"] / count,
            "mean_effective_gate": sums["effective_gate"] / count,
        }
        if source == heldout:
            diagnostics[source]["matching_mean_consensus"] = sums["matching_consensus"] / max(
                1.0, sums["matching_count"]
            )
            diagnostics[source]["nonmatching_mean_consensus"] = sums["nonmatching_consensus"] / max(
                1.0, sums["nonmatching_count"]
            )

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "aggregate": sequential._finalize_counter(aggregate),
        "pairs": pair_rows,
        "diagnostics": diagnostics,
        "consensus_cache_entries": len(consensus_cache),
    }


def aggregate_mode(
    reports: Sequence[Mapping[str, Any]], *, heldout: str, appended: Sequence[str]
) -> dict[str, Any]:
    aggregate, pairs = unseen._aggregate_reports(reports)
    diagnostics: dict[str, Any] = {}
    for source in appended:
        keys = sorted(
            {
                key
                for report in reports
                for key in report["diagnostics"].get(source, {}).keys()
            }
        )
        diagnostics[source] = {
            key: sum(float(report["diagnostics"][source][key]) for report in reports) / len(reports)
            for key in keys
        }
    return {
        "aggregate": aggregate,
        "pairs": pairs,
        "subsets": unseen._subset_summary(pairs, heldout=heldout),
        "diagnostics": diagnostics,
        "cohort_reports": list(reports),
    }


def evaluate_mode(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    scorer: residual.IndependentResidualGate,
    heldout: str,
    appended: Sequence[str],
    behavioral_sources: frozenset[str],
    consensus_power: float,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    reports = [
        evaluate_cohort(
            cohort,
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
        for cohort in cohorts[:3]
    ]
    return aggregate_mode(reports, heldout=heldout, appended=appended)


def _correct_count(block: Mapping[str, Any], subset: str) -> int:
    return int(block["subsets"][subset]["end_to_end_correct"])


def run_experiment(
    *,
    root: Path,
    heldout: str,
    scorer_seed: int,
    train_seed: int,
    data_seed: int,
    consensus_power: float,
    examples_per_pair: int,
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

    mode_specs = {
        "without_plugins": ((), frozenset()),
        "raw_useful_only": ((heldout,), frozenset()),
        "behavioral_useful_only": ((heldout,), frozenset({heldout})),
        "raw_mixed": ((heldout, NUISANCE_SOURCE), frozenset()),
        "behavioral_mixed": (
            (heldout, NUISANCE_SOURCE),
            frozenset({heldout, NUISANCE_SOURCE}),
        ),
    }
    modes: dict[str, Any] = {}
    for name, (appended, behavioral_sources) in mode_specs.items():
        modes[name] = evaluate_mode(
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

    subsets = ("heldout_involved", "heldout_neither")
    deltas: dict[str, Any] = {}
    baseline = modes["without_plugins"]
    for mode_name in ("raw_useful_only", "behavioral_useful_only", "raw_mixed", "behavioral_mixed"):
        deltas[mode_name] = {
            subset: _correct_count(modes[mode_name], subset) - _correct_count(baseline, subset)
            for subset in subsets
        }
        deltas[mode_name]["net"] = sum(deltas[mode_name][subset] for subset in subsets)
    deltas["nuisance_marginal_raw"] = {
        subset: _correct_count(modes["raw_mixed"], subset)
        - _correct_count(modes["raw_useful_only"], subset)
        for subset in subsets
    }
    deltas["nuisance_marginal_raw"]["net"] = sum(
        deltas["nuisance_marginal_raw"][subset] for subset in subsets
    )
    deltas["nuisance_marginal_behavioral"] = {
        subset: _correct_count(modes["behavioral_mixed"], subset)
        - _correct_count(modes["behavioral_useful_only"], subset)
        for subset in subsets
    }
    deltas["nuisance_marginal_behavioral"]["net"] = sum(
        deltas["nuisance_marginal_behavioral"][subset] for subset in subsets
    )

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "mixed_novel_useful_plus_failed_plugin_admission_pilot",
        "heldout_operator": heldout,
        "nuisance_source": NUISANCE_SOURCE,
        "scorer_seed": scorer_seed,
        "runtime": runtime,
        "scorer_fit": fit_report,
        "modes": modes,
        "correct_count_deltas": deltas,
        "claim_boundary": (
            "both held-out functional specialist and NEG are absent from gate training source pool; semantic identities are not "
            "used by behavioral consensus, but the system knows which handles are newly appended; NEG is a failed sibling unit, "
            "not an arbitrary external architecture; deterministic single scorer seed pilot"
        ),
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test independent behavioral admission with a useful and nuisance novel plugin"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--scorer-seed", type=int, default=741000)
    parser.add_argument("--train-seed", type=int, default=741000)
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
                "correct_count_deltas": report["correct_count_deltas"],
                "behavioral_mixed_diagnostics": report["modes"]["behavioral_mixed"]["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
