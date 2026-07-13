from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import torch


def _floating_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    output: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            continue
        if (
            name == "lm_head.weight"
            and "token_embedding.weight" in state
            and torch.equal(tensor, state["token_embedding.weight"])
        ):
            continue
        output[name] = tensor.double().reshape(-1)
    return output


def _distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    diff_sq = 0.0
    base_sq = 0.0
    for name, left in a.items():
        right = b[name]
        diff = right - left
        diff_sq += float(torch.dot(diff, diff))
        base_sq += float(torch.dot(left, left))
    absolute = math.sqrt(diff_sq)
    return absolute, absolute / max(math.sqrt(base_sq), 1e-30)


def analyze(index_path: Path, output_path: Path) -> None:
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dedup: dict[int, dict] = {}
    for row in rows:
        dedup[int(row["step"])] = row
    ordered = [dedup[step] for step in sorted(dedup)]
    if not ordered:
        raise ValueError("checkpoint index is empty")
    initial = _floating_state(Path(ordered[0]["checkpoint"]))
    previous = initial
    output_rows = []
    for row in ordered:
        current = _floating_state(Path(row["checkpoint"]))
        from_initial, rel_initial = _distance(initial, current)
        from_previous, rel_previous = _distance(previous, current)
        output_rows.append(
            {
                "step": int(row["step"]),
                "checkpoint": row["checkpoint"],
                "initial_delta_l2": from_initial,
                "initial_relative_delta_l2": rel_initial,
                "previous_delta_l2": from_previous,
                "previous_relative_delta_l2": rel_previous,
                "train_loss": row.get("train_loss"),
                "validation_loss": json.dumps(row.get("validation_loss"), sort_keys=True),
            }
        )
        previous = current
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure parameter movement along a saved checkpoint trajectory")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    analyze(Path(args.index), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
