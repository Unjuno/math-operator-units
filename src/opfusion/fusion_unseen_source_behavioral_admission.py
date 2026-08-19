from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_unseen_source_generalization as unseen
from opfusion import fusion_unseen_source_residual_gating as residual
from opfusion import fusion_unseen_source_rollout_consensus as rollout
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
DEFAULT_TRAIN_SEED = 741_000
DEFAULT_EVAL_SEED = 741_500
DEFAULT_SKETCH_SEED = 743_000


def behavioral_consensus(
    model: torch.nn.Module,
    *,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> float:
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
    parsed = [value for value in outputs if value is not None]
    if not parsed:
        return 0.0
    counts: dict[int, int] = {}
    for value in parsed:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.values()) / len(outputs)


def _generate_value(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    scorer: residual.IndependentResidualGate,
    heldout: str,
    include_heldout: bool,
    consensus_power: float,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
    consensus_cache: dict[tuple[str, tuple[int, ...]], float],
) -> tuple[int | None, dict[str, float]]:
    prompt = sequential.prompt_ids_for_values(
        factory=factory,
        tokenizer=tokenizer,
        operator=operator,
        values=values,
    )
    ids = torch.tensor([prompt], dtype=torch.long, device=device)

    consensus = 1.0
    admission_factor = 1.0
    if include_heldout and consensus_power > 0.0:
        cache_key = (operator, tuple(int(value) for value in values))
        if cache_key not in consensus_cache:
            consensus_cache[cache_key] = behavioral_consensus(
                units[heldout],
                operator=operator,
                values=values,
                factory=factory,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        consensus = float(consensus_cache[cache_key])
        admission_factor = consensus ** float(consensus_power)

    generated: list[int] = []
    heldout_gate_sum = 0.0
    heldout_effective_gate_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            names, sources = residual._selected_logits(
                base=base,
                units=units,
                ids=ids,
                heldout=heldout,
                include_heldout=include_heldout,
            )
            gates = scorer.gates_from_logits(sources)
            effective_gates = gates.clone()
            if include_heldout and heldout in names[1:]:
                heldout_index = names[1:].index(heldout)
                raw_gate = float(gates[heldout_index].detach().cpu())
                effective_gates[heldout_index] = effective_gates[heldout_index] * admission_factor
                effective_gate = float(effective_gates[heldout_index].detach().cpu())
                heldout_gate_sum += raw_gate
                heldout_effective_gate_sum += effective_gate
            fused = residual.fuse_residual_logits(sources, effective_gates)
            next_id = int(fused.argmax().item())
            generated.append(next_id)
            positions += 1
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id == tokenizer.eos_id:
                break

    return sequential.parse_final_numeric_token(generated, tokenizer), {
        "behavioral_consensus": consensus,
        "admission_factor": admission_factor,
        "mean_heldout_raw_gate": heldout_gate_sum / max(1, positions),
        "mean_heldout_effective_gate": heldout_effective_gate_sum / max(1, positions),
    }


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    scorer: residual.IndependentResidualGate,
    heldout: str,
    include_heldout: bool,
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
        if operator in FUNCTIONAL_OPERATORS
    }
    if heldout not in units:
        raise RuntimeError(f"held-out source checkpoint missing: {heldout}")

    aggregate = sequential._empty_counter()
    pair_rows: dict[str, dict[str, float | int | None]] = {}
    consensus_cache: dict[tuple[str, tuple[int, ...]], float] = {}
    diagnostic_sums = {
        "matching_consensus": 0.0,
        "matching_consensus_count": 0,
        "nonmatching_consensus": 0.0,
        "nonmatching_consensus_count": 0,
        "admission_factor": 0.0,
        "admission_factor_count": 0,
        "raw_gate": 0.0,
        "effective_gate": 0.0,
        "gate_count": 0,
    }

    def record_diag(operator: str, row: Mapping[str, float]) -> None:
        if not include_heldout:
            return
        key = "matching" if operator == heldout else "nonmatching"
        diagnostic_sums[f"{key}_consensus"] += float(row["behavioral_consensus"])
        diagnostic_sums[f"{key}_consensus_count"] += 1
        diagnostic_sums["admission_factor"] += float(row["admission_factor"])
        diagnostic_sums["admission_factor_count"] += 1
        diagnostic_sums["raw_gate"] += float(row["mean_heldout_raw_gate"])
        diagnostic_sums["effective_gate"] += float(row["mean_heldout_effective_gate"])
        diagnostic_sums["gate_count"] += 1

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
                    include_heldout=include_heldout,
                    consensus_power=consensus_power,
                    operator=inner_operator,
                    values=inner_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    consensus_cache=consensus_cache,
                )
                record_diag(inner_operator, inner_diag)

                oracle_outer, oracle_diag = _generate_value(
                    base=base,
                    units=units,
                    scorer=scorer,
                    heldout=heldout,
                    include_heldout=include_heldout,
                    consensus_power=consensus_power,
                    operator=outer_operator,
                    values=true_outer_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    consensus_cache=consensus_cache,
                )
                record_diag(outer_operator, oracle_diag)

                chained_outer: int | None = None
                if generated_inner is not None:
                    chained_outer, chained_diag = _generate_value(
                        base=base,
                        units=units,
                        scorer=scorer,
                        heldout=heldout,
                        include_heldout=include_heldout,
                        consensus_power=consensus_power,
                        operator=outer_operator,
                        values=(generated_inner, *outer_extras),
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        consensus_cache=consensus_cache,
                    )
                    record_diag(outer_operator, chained_diag)

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

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "aggregate": sequential._finalize_counter(aggregate),
        "pairs": pair_rows,
        "diagnostics": {
            "mean_matching_heldout_consensus": diagnostic_sums["matching_consensus"]
            / max(1, diagnostic_sums["matching_consensus_count"]),
            "mean_nonmatching_heldout_consensus": diagnostic_sums["nonmatching_consensus"]
            / max(1, diagnostic_sums["nonmatching_consensus_count"]),
            "mean_admission_factor": diagnostic_sums["admission_factor"]
            / max(1, diagnostic_sums["admission_factor_count"]),
            "mean_heldout_raw_gate": diagnostic_sums["raw_gate"]
            / max(1, diagnostic_sums["gate_count"]),
            "mean_heldout_effective_gate": diagnostic_sums["effective_gate"]
            / max(1, diagnostic_sums["gate_count"]),
            "consensus_cache_entries": len(consensus_cache),
        },
    }


def aggregate_mode(reports: Sequence[Mapping[str, Any]], *, heldout: str) -> dict[str, Any]:
    aggregate, pairs = unseen._aggregate_reports(reports)
    diagnostics: dict[str, float] = {}
    diagnostic_keys = (
        "mean_matching_heldout_consensus",
        "mean_nonmatching_heldout_consensus",
        "mean_admission_factor",
        "mean_heldout_raw_gate",
        "mean_heldout_effective_gate",
    )
    for key in diagnostic_keys:
        diagnostics[key] = sum(float(report["diagnostics"][key]) for report in reports) / len(reports)
    return {
        "aggregate": aggregate,
        "pairs": pairs,
        "subsets": unseen._subset_summary(pairs, heldout=heldout),
        "diagnostics": diagnostics,
        "cohort_reports": list(reports),
    }


def run_experiment(
    *,
    root: Path,
    heldout: str,
    train_examples_per_operator: int,
    max_positions_per_cohort: int,
    train_seed: int,
    sketch_size: int,
    sketch_seed: int,
    hidden_size: int,
    max_gate: float,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    gate_penalty: float,
    scorer_seed: int,
    consensus_power: float,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
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
    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    torch.manual_seed(scorer_seed)
    scorer = residual.IndependentResidualGate(
        vocabulary_size=tokenizer.vocab_size,
        sketch_size=sketch_size,
        hidden_size=hidden_size,
        sketch_seed=sketch_seed,
        max_gate=max_gate,
    ).to(device)
    batch = residual.collect_training_batch(
        scorer,
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=train_examples_per_operator,
        data_seed=train_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    fit_report = residual.fit_residual_gate(
        scorer,
        batch,
        learning_rate=learning_rate,
        steps=steps,
        batch_positions=batch_positions,
        gate_penalty=gate_penalty,
        seed=scorer_seed,
        device=device,
    )

    without_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            scorer=scorer,
            heldout=heldout,
            include_heldout=False,
            consensus_power=0.0,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]
    raw_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            scorer=scorer,
            heldout=heldout,
            include_heldout=True,
            consensus_power=0.0,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]
    behavioral_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            scorer=scorer,
            heldout=heldout,
            include_heldout=True,
            consensus_power=consensus_power,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]

    without = aggregate_mode(without_reports, heldout=heldout)
    raw = aggregate_mode(raw_reports, heldout=heldout)
    behavioral = aggregate_mode(behavioral_reports, heldout=heldout)

    def subset_e2e(block: Mapping[str, Any], subset: str) -> float:
        return float(block["subsets"][subset]["end_to_end_accuracy"])

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "unseen_source_semantic_view_behavioral_admission_pilot",
        "claim_boundary": (
            "held-out specialist remains absent from residual-gate training data/source pool; its semantic identity is not used "
            "by the behavioral score, but the system knows which source handle is newly appended; behavioral consensus is based "
            "on standalone greedy rollouts over exact commutative prompt permutations; Base remains privileged and stage boundaries external"
        ),
        "heldout_operator": heldout,
        "consensus_power": consensus_power,
        "scorer_fit": fit_report,
        "without_heldout_source": without,
        "raw_heldout_insertion": raw,
        "behavioral_heldout_insertion": behavioral,
        "insertion_deltas": {
            "raw_involved": subset_e2e(raw, "heldout_involved") - subset_e2e(without, "heldout_involved"),
            "raw_neither": subset_e2e(raw, "heldout_neither") - subset_e2e(without, "heldout_neither"),
            "behavioral_involved": subset_e2e(behavioral, "heldout_involved") - subset_e2e(without, "heldout_involved"),
            "behavioral_neither": subset_e2e(behavioral, "heldout_neither") - subset_e2e(without, "heldout_neither"),
        },
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Use semantic-view rollout consensus to calibrate a never-trained residual source"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--train-examples-per-operator", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--sketch-size", type=int, default=32)
    parser.add_argument("--sketch-seed", type=int, default=DEFAULT_SKETCH_SEED)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-gate", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-positions", type=int, default=256)
    parser.add_argument("--gate-penalty", type=float, default=0.01)
    parser.add_argument("--scorer-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--consensus-power", type=float, default=2.0)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_experiment(
        root=args.root,
        heldout=args.heldout,
        train_examples_per_operator=args.train_examples_per_operator,
        max_positions_per_cohort=args.max_positions_per_cohort,
        train_seed=args.train_seed,
        sketch_size=args.sketch_size,
        sketch_seed=args.sketch_seed,
        hidden_size=args.hidden_size,
        max_gate=args.max_gate,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_positions=args.batch_positions,
        gate_penalty=args.gate_penalty,
        scorer_seed=args.scorer_seed,
        consensus_power=args.consensus_power,
        examples_per_pair=args.examples_per_pair,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "heldout_operator": report["heldout_operator"],
        "insertion_deltas": report["insertion_deltas"],
        "behavioral_diagnostics": report["behavioral_heldout_insertion"]["diagnostics"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
