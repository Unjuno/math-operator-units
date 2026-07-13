from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from . import batch as core_batch
from .design_config import load_design_run_config, model_design
from .experiment_contract import ensure_experiment_contract
from .trainer import _find_repo_root, _json_dump, _resolve_repo_path
from .trainer_design_hardened import train_job


_ORIGINAL_SUBSET_RECORD = core_batch._subset_record
_ORIGINAL_PLAN = core_batch._plan
_ACTIVE_FINGERPRINT: str | None = None
_ACTIVE_DESIGN: dict[str, Any] | None = None


def _subset_record_with_contract(**kwargs: Any) -> dict[str, Any]:
    record = _ORIGINAL_SUBSET_RECORD(**kwargs)
    if _ACTIVE_FINGERPRINT:
        record["experiment_fingerprint"] = _ACTIVE_FINGERPRINT
    if _ACTIVE_DESIGN:
        record["model_design"] = _ACTIVE_DESIGN
    record["endpoint_policy"] = "validation_selected"
    return record


def _plan_with_contract(config: Any) -> dict[str, Any]:
    plan = _ORIGINAL_PLAN(config)
    if _ACTIVE_FINGERPRINT:
        plan["experiment_fingerprint"] = _ACTIVE_FINGERPRINT
    plan["model_design"] = model_design(config).to_dict()
    plan["endpoint_policy"] = "validation_selected"
    plan["warning"] = (
        "final.pt is retained for trajectories, but dependency branches and final subset manifests use selected.pt"
    )
    return plan


def _install_batch_runtime() -> None:
    core_batch.load_run_config = load_design_run_config
    core_batch.train_job = train_job
    core_batch._subset_record = _subset_record_with_contract
    core_batch._plan = _plan_with_contract


def _stamp_outputs(repo_root: Path, config: Any, fingerprint: str) -> None:
    output_root = _resolve_repo_path(repo_root, config.output_dir)
    state_path = output_root / "batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["experiment_fingerprint"] = fingerprint
        state["model_design"] = model_design(config).to_dict()
        state["endpoint_policy"] = "validation_selected"
        _json_dump(state_path, state)
    for seed in config.seeds:
        inventory_path = output_root / f"seed_{seed}" / "model_inventory.json"
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["experiment_fingerprint"] = fingerprint
            inventory["model_design"] = model_design(config).to_dict()
            inventory["endpoint_policy"] = "validation_selected"
            _json_dump(inventory_path, inventory)


def run_batch(
    config_path: Path,
    *,
    allow_cpu: bool = False,
    plan_only: bool = False,
) -> int:
    global _ACTIVE_FINGERPRINT, _ACTIVE_DESIGN
    config_path = config_path.resolve()
    repo_root = _find_repo_root(config_path.parent)
    config = load_design_run_config(config_path)
    contract = ensure_experiment_contract(repo_root, config)
    _ACTIVE_FINGERPRINT = str(contract["fingerprint"])
    _ACTIVE_DESIGN = model_design(config).to_dict()
    _install_batch_runtime()
    try:
        result = core_batch.run_batch(config_path, allow_cpu=allow_cpu, plan_only=plan_only)
        _stamp_outputs(repo_root, config, _ACTIVE_FINGERPRINT)
        return result
    finally:
        _ACTIVE_FINGERPRINT = None
        _ACTIVE_DESIGN = None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a dependency-ordered model set with model-design controls and strict contracts"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke/pilot only")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_batch(
        Path(args.config),
        allow_cpu=args.allow_cpu,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
