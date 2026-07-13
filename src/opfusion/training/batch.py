from __future__ import annotations

import argparse
import itertools
import time
import traceback
from pathlib import Path
from typing import Iterable

from opfusion.tokenizer import FixedVocabTokenizer
from .config import load_run_config
from .data import EXPERIMENT_OPERATORS
from .trainer import _find_repo_root, _json_dump, _resolve_repo_path, train_job


def _write_subset_manifests(repo_root: Path, config, seed: int, final_checkpoints: dict[str, Path]) -> Path:
    output_root = _resolve_repo_path(repo_root, config.output_dir)
    target = output_root / f"seed_{seed}" / "fusion_subsets"
    target.mkdir(parents=True, exist_ok=True)
    tokenizer = FixedVocabTokenizer.from_config(_resolve_repo_path(repo_root, config.tokenizer_config))
    records = []
    for mask in range(1 << len(EXPERIMENT_OPERATORS)):
        active = [operator for bit, operator in enumerate(EXPERIMENT_OPERATORS) if mask & (1 << bit)]
        record = {
            "subset_id": f"subset_{mask:02d}",
            "bitmask": mask,
            "operators": active,
            "calibration_mode": "raw",
            "dispatch": False,
            "tokenizer_profile": tokenizer.profile,
            "vocab_hash": tokenizer.vocab_hash,
            "shared_initial_checkpoint": str(output_root / f"seed_{seed}" / "shared_initial.pt"),
            "unit_checkpoints": {operator: str(final_checkpoints[operator]) for operator in active},
            "joint_reference_checkpoint": str(final_checkpoints[config.joint_model_id]),
        }
        _json_dump(target / f"subset_{mask:02d}.json", record)
        records.append(record)
    _json_dump(target / "index.json", {"count": len(records), "subsets": records})
    return target / "index.json"


def run_batch(config_path: Path, *, allow_cpu: bool = False) -> int:
    config_path = config_path.resolve()
    repo_root = _find_repo_root(config_path.parent)
    config = load_run_config(config_path)
    output_root = _resolve_repo_path(repo_root, config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_state_path = output_root / "batch_state.json"
    failures: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []

    for seed, job_id in itertools.product(config.seeds, config.jobs):
        started = time.time()
        try:
            final = train_job(
                repo_root=repo_root,
                config=config,
                job_id=job_id,
                seed=seed,
                allow_cpu=allow_cpu,
            )
            completed.append({"seed": seed, "job_id": job_id, "final_checkpoint": str(final)})
        except Exception as exc:
            failure = {
                "seed": seed,
                "job_id": job_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "failed_unix": time.time(),
            }
            failures.append(failure)
            _json_dump(output_root / f"seed_{seed}" / job_id.replace(".", "_") / "failure.json", failure)
            _json_dump(batch_state_path, {"completed": completed, "failures": failures})
            if not config.continue_on_error:
                raise
        finally:
            _json_dump(
                batch_state_path,
                {
                    "experiment_id": config.experiment_id,
                    "completed": completed,
                    "failures": failures,
                    "updated_unix": time.time(),
                    "last_job_elapsed_seconds": time.time() - started,
                },
            )

    for seed in config.seeds:
        seed_results = {
            item["job_id"]: Path(str(item["final_checkpoint"]))
            for item in completed
            if int(item["seed"]) == seed
        }
        if all(job in seed_results for job in config.jobs):
            index_path = _write_subset_manifests(repo_root, config, seed, seed_results)
            completed.append({"seed": seed, "job_id": "fusion_subset_manifests", "final_checkpoint": str(index_path)})

    _json_dump(
        batch_state_path,
        {
            "experiment_id": config.experiment_id,
            "completed": completed,
            "failures": failures,
            "updated_unix": time.time(),
            "status": "completed" if not failures else "completed_with_failures",
        },
    )
    return 1 if failures else 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train five GPT operator models and one all-data joint reference model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke-test only; production config requires CUDA")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_batch(Path(args.config), allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    raise SystemExit(main())
