from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .data import EXPERIMENT_OPERATORS


PAIR_MAP = {
    "identity": ("identity_unanchored", "identity_retention"),
    "weak": ("weak_unanchored", "weak_retention"),
}
PAIRS = tuple(PAIR_MAP.values())
SHARED_JOBS = ("base.common", "joint.all_five.exposure_matched")


def _job_dir(root: Path, condition: str, job_id: str) -> Path:
    return root / condition / "seed_0" / job_id.replace(".", "_")


def _resolve(repo_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _selected_checkpoint(repo_root: Path, root: Path, condition: str, job_id: str) -> tuple[Path, dict[str, Any]]:
    complete_path = _job_dir(root, condition, job_id) / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_checkpoint")
    if selected is None:
        raise RuntimeError(f"selected checkpoint missing: {complete_path}")
    path = _resolve(repo_root, selected)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, payload


def _update_tensor_bytes(digest: Any, tensor: torch.Tensor, *, chunk_bytes: int = 1 << 20) -> None:
    flat = tensor.detach().cpu().contiguous().view(torch.uint8).flatten()
    for start in range(0, int(flat.numel()), chunk_bytes):
        chunk = flat[start : start + chunk_bytes]
        digest.update(bytes(chunk.tolist()))


def _state_hash(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"model_state_dict missing: {checkpoint}")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"non-tensor model state entry {name!r}: {checkpoint}")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        _update_tensor_bytes(digest, tensor)
    return digest.hexdigest()


def _runtime_state(root: Path, condition: str, job_id: str) -> dict[str, Any]:
    path = _job_dir(root, condition, job_id) / "runtime_state.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "micro_batch_size": payload.get("micro_batch_size"),
        "lr_scale": payload.get("lr_scale"),
        "oom_reductions": payload.get("oom_reductions"),
        "non_finite_restarts": payload.get("non_finite_restarts"),
    }


def audit_pilot_pairs(
    repo_root: str | Path = ".",
    *,
    pairs: Sequence[tuple[str, str]] = PAIRS,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    pilot_root = repo_root / "runs/model_design_pilot"
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []

    for left, right in pairs:
        shared: dict[str, Any] = {}
        for job_id in SHARED_JOBS:
            try:
                left_path, left_complete = _selected_checkpoint(repo_root, pilot_root, left, job_id)
                right_path, right_complete = _selected_checkpoint(repo_root, pilot_root, right, job_id)
                left_hash = _state_hash(left_path)
                right_hash = _state_hash(right_path)
                identical = left_hash == right_hash
                shared[job_id] = {
                    "identical_model_state": identical,
                    "left_checkpoint": str(left_path),
                    "right_checkpoint": str(right_path),
                    "left_step": left_complete.get("selected_step"),
                    "right_step": right_complete.get("selected_step"),
                    "left_sha256": left_hash,
                    "right_sha256": right_hash,
                }
                if not identical:
                    errors.append(
                        {
                            "kind": "paired_shared_endpoint_mismatch",
                            "pair": [left, right],
                            "job_id": job_id,
                            "left_sha256": left_hash,
                            "right_sha256": right_hash,
                        }
                    )
            except Exception as exc:
                shared[job_id] = {"error": repr(exc)}
                errors.append(
                    {
                        "kind": "paired_shared_endpoint_unreadable",
                        "pair": [left, right],
                        "job_id": job_id,
                        "error": repr(exc),
                    }
                )

        runtime_checks: dict[str, Any] = {}
        for operator in EXPERIMENT_OPERATORS:
            left_state = _runtime_state(pilot_root, left, operator)
            right_state = _runtime_state(pilot_root, right, operator)
            equal = left_state == right_state
            runtime_checks[operator] = {
                "left": left_state,
                "right": right_state,
                "equal": equal,
            }
            if left_state and right_state and not equal:
                warnings.append(
                    {
                        "kind": "paired_specialist_runtime_mismatch",
                        "pair": [left, right],
                        "operator": operator,
                        "left": left_state,
                        "right": right_state,
                        "interpretation": (
                            "effective batch is matched, but micro-batch, accumulation order, OOM recovery, "
                            "or learning-rate recovery differed between the paired Specialist runs"
                        ),
                    }
                )

        pair_results.append(
            {
                "left": left,
                "right": right,
                "shared_endpoint_checks": shared,
                "specialist_runtime_checks": runtime_checks,
            }
        )

    return {
        "status": "passed" if not errors else "failed",
        "pilot_root": str(pilot_root),
        "pairs": pair_results,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that retention/unanchored pilot pairs share exact Base and Joint endpoints"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pair", choices=("all", *PAIR_MAP), default="all")
    parser.add_argument("--out", default="audits/model_design_pilot/pair_consistency.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    pairs = PAIRS if args.pair == "all" else (PAIR_MAP[args.pair],)
    report = audit_pilot_pairs(args.repo_root, pairs=pairs)
    report["pair_scope"] = args.pair
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)
    else:
        print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
