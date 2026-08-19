from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_hard_null_plugin_admission as hard
from opfusion import fusion_metamorphic_contract_diagnostic as meta
from opfusion import fusion_mixed_novel_plugin_admission as mixed
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_unseen_source_behavioral_admission as behavioral
from opfusion import fusion_unseen_source_residual_gating as residual
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
NUISANCE_SOURCE = mixed.NUISANCE_SOURCE
DEFAULT_CALIBRATION_SEED = 749_000


def metamorphic_score_and_outputs(
    model: torch.nn.Module,
    *,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[float, list[int | None]]:
    original_modal, consensus, outputs = meta._rollout_summary(
        model,
        operator=operator,
        values=values,
        factory=factory,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    probe_rows: list[dict[str, Any]] = []
    for probe in meta.contract_probes(operator, values):
        probe_modal, probe_consensus, _ = meta._rollout_summary(
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
                "modal_value": probe_modal,
                "consensus": probe_consensus,
                "output_delta": probe.output_delta,
            }
        )
    contract = meta.summarize_contract(original_modal, consensus, probe_rows)
    return float(contract["consensus_x_contract"]), outputs


def metamorphic_behavioral_score(
    model: torch.nn.Module,
    *,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    score, _ = metamorphic_score_and_outputs(
        model,
        operator=operator,
        values=values,
        factory=factory,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return score


@contextmanager
def patched_behavioral_score():
    original = behavioral.behavioral_consensus
    behavioral.behavioral_consensus = metamorphic_behavioral_score
    try:
        yield
    finally:
        behavioral.behavioral_consensus = original


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
                    score, outputs = metamorphic_score_and_outputs(
                        units[source],
                        operator=operator,
                        values=values,
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )
                    correct_count = sum(value == expected for value in outputs)
                    rows.append(
                        {
                            "cohort_id": cohort.cohort_id,
                            "task_operator": operator,
                            "source": source,
                            "consensus": score,
                            "majority_correct": int(correct_count * 2 > len(outputs)),
                        }
                    )
        del units
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = hard.choose_hard_threshold(rows)
    report.update(
        {
            "score_name": "consensus_x_contract",
            "calibration_seed": calibration_seed,
            "examples_per_operator_per_cohort": examples_per_operator,
            "seen_operators": seen,
            "heldout_operator_excluded": heldout,
            "nuisance_source_excluded": NUISANCE_SOURCE,
        }
    )
    return report


def _count(block: Mapping[str, Any], subset: str) -> int:
    return int(block["subsets"][subset]["end_to_end_correct"])


def _delta(mode: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, int]:
    row = {
        subset: _count(mode, subset) - _count(baseline, subset)
        for subset in ("heldout_involved", "heldout_neither")
    }
    row["net"] = row["heldout_involved"] + row["heldout_neither"]
    return row


def _marginal(mixed_mode: Mapping[str, Any], useful_mode: Mapping[str, Any]) -> dict[str, int]:
    row = {
        subset: _count(mixed_mode, subset) - _count(useful_mode, subset)
        for subset in ("heldout_involved", "heldout_neither")
    }
    row["net"] = row["heldout_involved"] + row["heldout_neither"]
    return row


def run_experiment(
    *,
    root: Path,
    heldout: str,
    scorer_seed: int,
    train_seed: int,
    calibration_seed: int,
    calibration_examples: int,
    data_seed: int,
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
        consensus_power=1.0,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    with patched_behavioral_score():
        continuous_useful = mixed.evaluate_mode(
            cohorts,
            root=root,
            scorer=scorer,
            heldout=heldout,
            appended=(heldout,),
            behavioral_sources=frozenset({heldout}),
            consensus_power=1.0,
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
            consensus_power=1.0,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        hard_useful = hard.evaluate_hard_mode(
            cohorts,
            root=root,
            scorer=scorer,
            heldout=heldout,
            appended=(heldout,),
            behavioral_sources=frozenset({heldout}),
            consensus_power=1.0,
            threshold=threshold,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        hard_mixed = hard.evaluate_hard_mode(
            cohorts,
            root=root,
            scorer=scorer,
            heldout=heldout,
            appended=(heldout, NUISANCE_SOURCE),
            behavioral_sources=frozenset({heldout, NUISANCE_SOURCE}),
            consensus_power=1.0,
            threshold=threshold,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    modes = {
        "without_plugins": without,
        "metamorphic_continuous_useful": continuous_useful,
        "metamorphic_continuous_mixed": continuous_mixed,
        "metamorphic_hard_useful": hard_useful,
        "metamorphic_hard_mixed": hard_mixed,
    }
    deltas = {
        name: _delta(block, without)
        for name, block in modes.items()
        if name != "without_plugins"
    }
    deltas["nuisance_marginal_continuous"] = _marginal(continuous_mixed, continuous_useful)
    deltas["nuisance_marginal_hard"] = _marginal(hard_mixed, hard_useful)

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "seen_calibrated_metamorphic_contract_mixed_plugin_admission",
        "heldout_operator": heldout,
        "nuisance_source": NUISANCE_SOURCE,
        "scorer_seed": scorer_seed,
        "runtime": runtime,
        "calibration": calibration,
        "residual_gate_fit": fit_report,
        "modes": modes,
        "correct_count_deltas": deltas,
        "claim_boundary": (
            "held-out useful source and NEG are absent from residual-gate training and threshold calibration; threshold calibration "
            "uses only the other three functional specialists on a separate validation seed; admission score uses task-defined "
            "metamorphic output relations and semantic-view rollouts but not the exact answer or source identity at evaluation; "
            "hand-derived arithmetic contracts, sibling checkpoints, Base anchor and external stage boundaries remain"
        ),
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test metamorphic-contract admission for mixed novel plugins")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--scorer-seed", type=int, default=741000)
    parser.add_argument("--train-seed", type=int, default=741000)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--calibration-examples", type=int, default=6)
    parser.add_argument("--data-seed", type=int, default=741500)
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
        examples_per_pair=args.examples_per_pair,
        max_new_tokens=args.max_new_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "heldout_operator": report["heldout_operator"],
        "calibration": report["calibration"],
        "correct_count_deltas": report["correct_count_deltas"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
