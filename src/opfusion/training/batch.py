from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from opfusion.tokenizer import FixedVocabTokenizer
from .config import RunConfig, load_run_config
from .data import EXPERIMENT_OPERATORS
from .trainer import (
    NonFiniteTrainingError,
    _find_repo_root,
    _json_dump,
    _resolve_repo_path,
    train_job,
)


def _subset_record(
    *,
    mask: int,
    checkpoints: dict[str, Path],
    joint_checkpoint: Path,
    initial: Path,
    tokenizer: Any,
    base_checkpoint: Path | None = None,
) -> dict[str, Any]:
    active = [operator for bit, operator in enumerate(EXPERIMENT_OPERATORS) if mask & (1 << bit)]
    base = base_checkpoint or initial
    return {
        "subset_id": f"subset_{mask:02d}",
        "bitmask": mask,
        "operators": active,
        "calibration_mode": "raw",
        "dispatch": False,
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "shared_initial_checkpoint": str(initial),
        "base_checkpoint": str(base),
        "bias_definition": "specialist_logits - base_logits",
        "unit_checkpoints": {operator: str(checkpoints[operator]) for operator in active},
        "joint_reference_checkpoint": str(joint_checkpoint),
    }


def _write_subset_directory(
    *,
    target: Path,
    checkpoints: dict[str, Path],
    joint_checkpoint: Path,
    initial: Path,
    tokenizer: Any,
    base_checkpoint: Path | None = None,
) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for mask in range(1 << len(EXPERIMENT_OPERATORS)):
        record = _subset_record(
            mask=mask,
            checkpoints=checkpoints,
            joint_checkpoint=joint_checkpoint,
            initial=initial,
            tokenizer=tokenizer,
            base_checkpoint=base_checkpoint,
        )
        _json_dump(target / f"subset_{mask:02d}.json", record)
        records.append(record)
    _json_dump(target / "index.json", {"count": len(records), "subsets": records})
    return target / "index.json"


def _checkpoint_map(index_path: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mapping[int(row["step"])] = Path(row["checkpoint"])
    return mapping


def _write_seed_manifests(repo_root: Path, config: RunConfig, seed: int, final_checkpoints: dict[str, Path]) -> list[Path]:
    output_root = _resolve_repo_path(repo_root, config.output_dir)
    seed_root = output_root / f"seed_{seed}"
    tokenizer = FixedVocabTokenizer.from_config(_resolve_repo_path(repo_root, config.tokenizer_config))
    initial = seed_root / "shared_initial.pt"
    base = final_checkpoints[config.base_model_id] if config.base_model_id else initial
    primary_joint = final_checkpoints[config.primary_joint_model_id]
    outputs = [
        _write_subset_directory(
            target=seed_root / "fusion_subsets",
            checkpoints={operator: final_checkpoints[operator] for operator in EXPERIMENT_OPERATORS},
            joint_checkpoint=primary_joint,
            initial=initial,
            base_checkpoint=base,
            tokenizer=tokenizer,
        )
    ]

    fusion_jobs = (*EXPERIMENT_OPERATORS, *config.joint_model_ids)
    maps = {
        job: _checkpoint_map(seed_root / job.replace(".", "_") / "checkpoint_index.jsonl")
        for job in fusion_jobs
    }
    common_steps = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    grid_index = []
    for step in common_steps:
        step_target = seed_root / "fusion_checkpoint_grid" / f"step_{step:09d}"
        index_path = _write_subset_directory(
            target=step_target,
            checkpoints={operator: maps[operator][step] for operator in EXPERIMENT_OPERATORS},
            joint_checkpoint=maps[config.primary_joint_model_id][step],
            initial=initial,
            base_checkpoint=base,
            tokenizer=tokenizer,
        )
        grid_index.append({"step": step, "index": str(index_path)})
    grid_path = seed_root / "fusion_checkpoint_grid" / "index.json"
    _json_dump(grid_path, {"common_steps": common_steps, "grids": grid_index})
    outputs.append(grid_path)

    inventory = {
        "seed": seed,
        "experiment_id": config.experiment_id,
        "random_initial_checkpoint": str(initial),
        "base_model_id": config.base_model_id,
        "base_checkpoint": str(base),
        "specialists": {operator: str(final_checkpoints[operator]) for operator in EXPERIMENT_OPERATORS},
        "joint_references": {job: str(final_checkpoints[job]) for job in config.joint_model_ids},
        "trained_model_count": len(config.jobs),
        "runtime_subset_count": 1 << len(EXPERIMENT_OPERATORS),
        "checkpoint_steps": list(config.resolved_checkpoint_steps),
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
    }
    inventory_path = seed_root / "model_inventory.json"
    _json_dump(inventory_path, inventory)
    outputs.append(inventory_path)
    return outputs


def _plan(config: RunConfig) -> dict[str, Any]:
    exposure_multiplier = {
        job: (len(EXPERIMENT_OPERATORS) if config.is_exposure_matched_joint(job) else 1)
        for job in config.jobs
    }
    return {
        "experiment_id": config.experiment_id,
        "seeds": list(config.seeds),
        "jobs_in_dependency_order": list(config.jobs),
        "trained_models_per_seed": len(config.jobs),
        "total_trained_models": len(config.jobs) * len(config.seeds),
        "runtime_subsets_per_seed": 1 << len(EXPERIMENT_OPERATORS),
        "max_steps_per_model": config.max_steps,
        "checkpoint_steps": list(config.resolved_checkpoint_steps),
        "logical_checkpoints_per_model": len(config.resolved_checkpoint_steps),
        "effective_batch_size": config.effective_batch_size,
        "exposure_multiplier_by_job": exposure_multiplier,
        "equivalent_specialist_steps": config.max_steps
        * len(config.seeds)
        * sum(exposure_multiplier.values()),
        "note": "wall-clock duration is measured on the target GPU during smoke/probe; it is not guaranteed by this static plan",
    }


def _adjust_non_finite_recovery(job_dir: Path, config: RunConfig) -> bool:
    state_path = job_dir / "runtime_state.json"
    raw = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    restarts = int(raw.get("non_finite_restarts", 0))
    if restarts >= config.recovery.max_lr_reductions:
        return False
    raw["non_finite_restarts"] = restarts + 1
    raw["lr_scale"] = float(raw.get("lr_scale", 1.0)) * config.recovery.non_finite_lr_factor
    if int(raw.get("micro_batch_size", 0)) <= 0:
        raw["micro_batch_size"] = config.micro_batch_size
    _json_dump(state_path, raw)
    recovery_path = job_dir / "recovery.jsonl"
    with recovery_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "non_finite_restart",
                    "restart": restarts + 1,
                    "new_lr_scale": raw["lr_scale"],
                    "action": "resume_last_good_checkpoint",
                    "unix": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    return True


def _train_with_recovery(
    *,
    repo_root: Path,
    config: RunConfig,
    job_id: str,
    seed: int,
    allow_cpu: bool,
) -> Path:
    job_dir = _resolve_repo_path(repo_root, config.output_dir) / f"seed_{seed}" / job_id.replace(".", "_")
    attempts = 0
    while True:
        try:
            return train_job(
                repo_root=repo_root,
                config=config,
                job_id=job_id,
                seed=seed,
                allow_cpu=allow_cpu,
            )
        except NonFiniteTrainingError:
            attempts += 1
            if attempts > config.recovery.max_retries_per_job or not _adjust_non_finite_recovery(job_dir, config):
                raise
            time.sleep(config.recovery.restart_delay_seconds)


def run_batch(config_path: Path, *, allow_cpu: bool = False, plan_only: bool = False) -> int:
    config_path = config_path.resolve()
    repo_root = _find_repo_root(config_path.parent)
    config = load_run_config(config_path)
    output_root = _resolve_repo_path(repo_root, config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = _plan(config)
    _json_dump(output_root / "experiment_plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if plan_only:
        return 0

    batch_state_path = output_root / "batch_state.json"
    failures: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []

    # config.jobs is dependency ordered: base, specialists, joint references.
    for seed in config.seeds:
        for job_id in config.jobs:
            started = time.time()
            try:
                final = _train_with_recovery(
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
            str(item["job_id"]): Path(str(item["final_checkpoint"]))
            for item in completed
            if int(item["seed"]) == seed and str(item["job_id"]) in config.jobs
        }
        if all(job in seed_results for job in config.jobs):
            for manifest_path in _write_seed_manifests(repo_root, config, seed, seed_results):
                completed.append({"seed": seed, "job_id": "fusion_manifests", "final_checkpoint": str(manifest_path)})

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
    parser = argparse.ArgumentParser(description="Train the dependency-ordered GPT model set for bias-fusion experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke-test only; production config requires CUDA")
    parser.add_argument("--plan-only", action="store_true", help="write and print the resolved model/checkpoint plan without training")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_batch(Path(args.config), allow_cpu=args.allow_cpu, plan_only=args.plan_only)


if __name__ == "__main__":
    raise SystemExit(main())
