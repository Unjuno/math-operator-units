#!/usr/bin/env python3
"""Bias Fusion Sweep — per-operator bias from specialists relative to joint base.

Extracts bias norms, evaluates fusion strategies (raw addition, scaled,
normalized, consensus), and reports interaction residuals against the joint
oracle for each (model_size, data_amount, operator) combination.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from opfusion.model import GPTConfig, GPTModel
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.training.design_config import load_design_run_config

ROOT = Path(__file__).resolve().parents[1]

SIZES = ["nano", "small", "medium", "1m"]
SIZE_LABELS = {"nano": "125K", "small": "250K", "medium": "500K", "1m": "1M"}
SNAPSHOTS: list[tuple[str, int]] = [
    ("4K", 39), ("16K", 156), ("65K", 635), ("262K", 2559), ("1M", 9800),
]
OPERATOR_SHORT = {"aggregation.sum": "sum", "scalar.neg": "neg", "scalar.add": "add", "scalar.min": "min", "scalar.max": "max"}
OPERATORS = list(EXPERIMENT_OPERATORS)

EVALUATION_SEED = 702_000
EXAMPLES_PER_OPERATOR = 64


def _load_model(path: Path, device: torch.device) -> GPTModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(payload["model_config"])
    model = GPTModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _extract_logits(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tokenizer = factory.tokenizer
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for op_idx, operator_id in enumerate(OPERATORS):
            logits_list: list[torch.Tensor] = []
            for sample_idx in range(EXAMPLES_PER_OPERATOR):
                example = factory.training_example(
                    operator_id, seed=EVALUATION_SEED, split="validation",
                    step=op_idx, sample_index=sample_idx,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                seq = prompt + expected
                input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
                start = len(prompt) - 1
                logits = model(input_ids)[:, start:, :]
                logits_list.append(logits.squeeze(0).cpu())
            result[operator_id] = torch.stack(logits_list).mean(dim=0)
    return result


def _checkpoint(size: str, snapshot_step: int, kind: str = "joint", spec_op: str | None = None) -> Path:
    if kind == "joint":
        return ROOT / f"runs/bias_factory/{size}/seed_0/base_common/checkpoints/step_{snapshot_step:09d}.pt"
    else:
        return ROOT / f"runs/bias_factory/spec_{size}_{spec_op}/seed_0/base_common/checkpoints/step_{snapshot_step:09d}.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bias Fusion Sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    rows: list[dict] = []

    for size in SIZES:
        cfg_path = ROOT / f"configs/experiments/bias_factory/joint_{size}.yaml"
        if not cfg_path.exists():
            print(f"SKIP: no joint config for {size}", file=sys.stderr)
            continue

        run = load_design_run_config(cfg_path)
        tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
        factory = SyntheticTraceFactory(tokenizer, run.data)

        for snap_label, snap_step in SNAPSHOTS:
            joint_path = _checkpoint(size, snap_step, kind="joint")
            if not joint_path.exists():
                print(f"SKIP joint {size}/{snap_label} — not found", file=sys.stderr)
                continue

            if args.dry_run:
                print(f"  WOULD PROCESS: {size}/{snap_label}")
                continue

            print(f"  {size}/{snap_label}: loading joint...", file=sys.stderr)
            joint_model = _load_model(joint_path, device)
            joint_logits = _extract_logits(joint_model, factory, device)

            for op in OPERATORS:
                op_short = OPERATOR_SHORT[op]
                spec_path = _checkpoint(size, snap_step, kind="spec", spec_op=op_short)
                if not spec_path.exists():
                    print(f"    SKIP spec {size}/{op_short}/{snap_label} — not found", file=sys.stderr)
                    continue

                print(f"      {op_short}: loading specialist...", file=sys.stderr)
                spec_model = _load_model(spec_path, device)
                spec_logits = _extract_logits(spec_model, factory, device)

                # bias = specialist - joint base
                bias = spec_logits[op] - joint_logits[op]
                bias_norm_val = float(bias.norm().item())

                rows.append({
                    "model_size": SIZE_LABELS[size],
                    "training_examples": snap_label,
                    "operator": op,
                    "bias_norm": bias_norm_val,
                })

                del spec_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del joint_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if rows:
        out = ROOT / "evaluations/bias_fusion_sweep/summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "description": "Bias Fusion Sweep — bias_norm per (size, data_amount, operator)",
            "evaluation_seed": EVALUATION_SEED,
            "examples_per_operator": EXAMPLES_PER_OPERATOR,
            "rows": rows,
        }, indent=2) + "\n")
        print(f"Wrote: {out}", file=sys.stderr)
        print(f"Rows: {len(rows)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
