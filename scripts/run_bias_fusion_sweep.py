#!/usr/bin/env python3
"""Bias Fusion Sweep — extract logit biases from Model Factory checkpoints.

Usage:
    python scripts/run_bias_fusion_sweep.py
    python scripts/run_bias_fusion_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from opfusion.model import GPTConfig, GPTModel
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.training.design_config import load_design_run_config

ROOT = Path(__file__).resolve().parents[1]

SIZES = ["nano", "small", "medium", "1m"]
SIZE_LABELS = {"nano": "125K", "small": "250K", "medium": "500K", "1m": "1M"}
SNAPSHOTS: list[tuple[str, int]] = [
    ("4K", 39),
    ("16K", 156),
    ("65K", 635),
    ("262K", 2559),
    ("1M", 9800),
]
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


def _extract_biases(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    device: torch.device,
) -> dict[str, dict]:
    tokenizer = factory.tokenizer
    results: dict[str, dict] = {}

    with torch.no_grad():
        for op_idx, operator_id in enumerate(OPERATORS):
            logits_list: list[torch.Tensor] = []
            for sample_idx in range(EXAMPLES_PER_OPERATOR):
                example = factory.training_example(
                    operator_id,
                    seed=EVALUATION_SEED,
                    split="validation",
                    step=op_idx,
                    sample_index=sample_idx,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                sequence = prompt + expected
                input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                response_start = len(prompt) - 1
                logits = model(input_ids)[:, response_start:, :]
                logits_list.append(logits.squeeze(0).cpu())

            stacked = torch.stack(logits_list)
            results[operator_id] = {
                "mean_logits": stacked.mean(dim=0),
                "std_logits": stacked.std(dim=0),
            }

    return results


def _bias_norm(bias: torch.Tensor) -> float:
    return float(bias.norm().item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Bias Fusion Sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    rows: list[dict] = []

    for size in SIZES:
        cfg_path = ROOT / f"configs/experiments/bias_factory/bias_factory_{size}.yaml"
        if not cfg_path.exists():
            print(f"SKIP: no config for {size}", file=sys.stderr)
            continue

        run = load_design_run_config(cfg_path)
        tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
        factory = SyntheticTraceFactory(tokenizer, run.data)

        for snap_label, snap_step in SNAPSHOTS:
            ckpt = ROOT / f"runs/bias_factory/{size}/seed_0/base_common/checkpoints/step_{snap_step:09d}.pt"
            if not ckpt.exists():
                print(f"SKIP: {size}/{snap_label} — {ckpt.name} not found", file=sys.stderr)
                continue

            if args.dry_run:
                print(f"  WOULD PROCESS: {size}/{snap_label}")
                continue

            print(f"  Processing: {size}/{snap_label} ...", file=sys.stderr)
            model = _load_model(ckpt, device)
            biases = _extract_biases(model, factory, device)

            base_logits = biases[OPERATORS[0]]["mean_logits"]
            base_logits.zero_()

            for op in OPERATORS:
                op_bias = biases[op]["mean_logits"] - base_logits
                rows.append({
                    "model_size": SIZE_LABELS[size],
                    "training_examples": snap_label,
                    "operator": op,
                    "bias_norm": _bias_norm(op_bias),
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if rows:
        out = ROOT / "evaluations/bias_fusion_sweep/summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "description": "Bias Fusion Sweep — per-operator bias norms by model size and data amount",
            "evaluation_seed": EVALUATION_SEED,
            "examples_per_operator": EXAMPLES_PER_OPERATOR,
            "rows": rows,
        }, indent=2) + "\n")
        print(f"Wrote: {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
