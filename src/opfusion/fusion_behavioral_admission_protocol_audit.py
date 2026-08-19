from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from opfusion import fusion_unseen_source_behavioral_admission as behavioral
from opfusion import fusion_unseen_source_residual_gating as residual
from opfusion import fusion_unseen_source_residual_gating_runner as residual_runner
from opfusion.fusion_search import discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config


def _pair_integer_signature(row: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(row[key])
        for key in behavioral.sequential._empty_counter()
    }


def _diff_pairs(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    differences: dict[str, Any] = {}
    for pair_id in sorted(set(left) | set(right)):
        lrow = _pair_integer_signature(left[pair_id])
        rrow = _pair_integer_signature(right[pair_id])
        if lrow != rrow:
            differences[pair_id] = {"custom": lrow, "reference": rrow}
    return differences


def run_audit(
    *,
    root: Path,
    heldout: str,
    examples_per_pair: int,
    train_seed: int,
    data_seed: int,
    scorer_seed: int,
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
    fit = residual.fit_residual_gate(
        scorer,
        batch,
        learning_rate=0.01,
        steps=600,
        batch_positions=256,
        gate_penalty=0.01,
        seed=scorer_seed,
        device=device,
    )

    custom_without_reports = [
        behavioral.evaluate_cohort(
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
    custom_raw_reports = [
        behavioral.evaluate_cohort(
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
    custom_without = behavioral.aggregate_mode(custom_without_reports, heldout=heldout)
    custom_raw = behavioral.aggregate_mode(custom_raw_reports, heldout=heldout)

    original = residual._generate_residual
    residual._generate_residual = residual_runner._generate_residual_compatible
    try:
        reference_without = residual.evaluate_mode(
            scorer,
            cohorts,
            heldout=heldout,
            include_heldout=False,
            root=root,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        reference_raw = residual.evaluate_mode(
            scorer,
            cohorts,
            heldout=heldout,
            include_heldout=True,
            root=root,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    finally:
        residual._generate_residual = original

    without_diff = _diff_pairs(custom_without["pairs"], reference_without["pairs"])
    raw_diff = _diff_pairs(custom_raw["pairs"], reference_raw["pairs"])
    return {
        "schema_version": 1,
        "status": "completed",
        "heldout_operator": heldout,
        "fit": fit,
        "custom_without": custom_without["aggregate"],
        "reference_without": reference_without["aggregate"],
        "custom_raw": custom_raw["aggregate"],
        "reference_raw": reference_raw["aggregate"],
        "without_pair_differences": without_diff,
        "raw_pair_differences": raw_diff,
        "without_exact_match": not without_diff,
        "raw_exact_match": not raw_diff,
        "claim_boundary": "protocol audit only; no behavioral-admission result is tested here",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit custom behavioral evaluator against residual reference")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=("scalar.min", "scalar.max"), required=True)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--train-seed", type=int, default=741000)
    parser.add_argument("--data-seed", type=int, default=741500)
    parser.add_argument("--scorer-seed", type=int, default=741000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_audit(
        root=args.root,
        heldout=args.heldout,
        examples_per_pair=args.examples_per_pair,
        train_seed=args.train_seed,
        data_seed=args.data_seed,
        scorer_seed=args.scorer_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "heldout_operator": report["heldout_operator"],
        "without_exact_match": report["without_exact_match"],
        "raw_exact_match": report["raw_exact_match"],
        "without_different_pairs": len(report["without_pair_differences"]),
        "raw_different_pairs": len(report["raw_pair_differences"]),
        "custom_without": report["custom_without"],
        "reference_without": report["reference_without"],
        "custom_raw": report["custom_raw"],
        "reference_raw": report["reference_raw"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
