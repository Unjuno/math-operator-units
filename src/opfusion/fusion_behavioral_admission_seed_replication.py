from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

# Configure CPU execution before importing torch/model modules.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from opfusion import fusion_unseen_source_behavioral_admission as behavioral


def configure_deterministic_runtime() -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe when a parent/import path initialized the interop pool already.
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    return {
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replicate behavioral unseen-source admission under deterministic CPU execution"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=behavioral.FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--scorer-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--consensus-power", type=float, default=2.0)
    parser.add_argument("--train-seed", type=int, default=741000)
    parser.add_argument("--data-seed", type=int, default=741500)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(list(argv) if argv is not None else None)

    runtime = configure_deterministic_runtime()
    report = behavioral.run_experiment(
        root=args.root,
        heldout=args.heldout,
        train_examples_per_operator=24,
        max_positions_per_cohort=3072,
        train_seed=args.train_seed,
        sketch_size=32,
        sketch_seed=743000,
        hidden_size=64,
        max_gate=4.0,
        learning_rate=0.01,
        steps=600,
        batch_positions=256,
        gate_penalty=0.01,
        scorer_seed=args.scorer_seed,
        consensus_power=args.consensus_power,
        examples_per_pair=args.examples_per_pair,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
        device_name="cpu",
    )
    report["replication"] = {
        "runtime": runtime,
        "scorer_seed": args.scorer_seed,
        "train_seed": args.train_seed,
        "data_seed": args.data_seed,
        "training_data_fixed_across_scorer_seeds": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "heldout_operator": report["heldout_operator"],
                "scorer_seed": args.scorer_seed,
                "runtime": runtime,
                "insertion_deltas": report["insertion_deltas"],
                "behavioral_diagnostics": report["behavioral_heldout_insertion"]["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
