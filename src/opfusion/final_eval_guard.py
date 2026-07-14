from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from opfusion.training.config import RunConfig


FINAL_AUTHORIZATION_ABI_VERSION = 1
ACTIVE_PLAN_PATH = Path("configs/experiments/experiment_plan_v2.yaml")
SURFACE_V4_EXPERIMENT_ID = "gpt_bias_fusion_factory_surface_v4"
FINAL_SPLITS = {"iid_test", "operand_ood", "length_ood"}
_ALLOWED_CALIBRATION_STATUSES = {
    "raw_preserved_no_rescue",
    "rescue_selected",
    "no_eligible_nonrouter_rescue",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_active_plan(repo_root: Path) -> tuple[dict[str, Any], Path, str]:
    plan_path = repo_root / ACTIVE_PLAN_PATH
    if not plan_path.is_file():
        raise RuntimeError(f"active experiment plan is missing: {plan_path}")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))["plan"]
    return plan, plan_path, _sha256_file(plan_path)


def _load_authorization(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError(f"final evaluation authorization is missing: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("final evaluation authorization must be a JSON object")
    return payload, _sha256_bytes(raw)


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(
            f"final evaluation authorization mismatch for {name}: "
            f"actual={actual!r} expected={expected!r}"
        )


def validate_evaluation_policy(
    *,
    repo_root: str | Path,
    config: RunConfig,
    manifest: dict[str, Any],
    split: str,
    evaluation_seed: int,
    examples_per_operator: int,
    final_authorization_path: str | Path | None,
) -> dict[str, Any] | None:
    """Fail closed before reserved final splits are generated.

    Pilot profiles are validation-only. The canonical surface-v4 profile may open
    final IID/OOD splits only after a machine-readable authorization produced by
    the preregistered calibration stage has been frozen for the current plan,
    code revision, and experiment fingerprint.
    """

    repo_root = Path(repo_root).resolve()
    if config.experiment_id.startswith("model_design_pilot_"):
        if split != "validation":
            raise RuntimeError("model-design pilot profiles are validation-only")
        return None

    if config.experiment_id != SURFACE_V4_EXPERIMENT_ID or split not in FINAL_SPLITS:
        return None

    plan, _plan_path, plan_sha256 = _load_active_plan(repo_root)
    if final_authorization_path is None:
        raise RuntimeError(
            "surface-v4 final IID/OOD evaluation is locked until fusion calibration "
            "is frozen; pass --final-authorization with the verified authorization JSON"
        )
    authorization_path = Path(final_authorization_path)
    if not authorization_path.is_absolute():
        authorization_path = repo_root / authorization_path
    authorization, authorization_sha256 = _load_authorization(authorization_path)

    _require_equal(
        "authorization_abi_version",
        authorization.get("authorization_abi_version"),
        FINAL_AUTHORIZATION_ABI_VERSION,
    )
    _require_equal("plan_id", authorization.get("plan_id"), plan["id"])
    _require_equal("plan_sha256", authorization.get("plan_sha256"), plan_sha256)
    _require_equal("experiment_id", authorization.get("experiment_id"), config.experiment_id)
    _require_equal(
        "experiment_fingerprint",
        authorization.get("experiment_fingerprint"),
        manifest.get("experiment_fingerprint"),
    )

    current_commit = _git_commit(repo_root)
    if current_commit is None:
        raise RuntimeError("cannot authorize final evaluation outside a Git checkout")
    _require_equal("git_commit", authorization.get("git_commit"), current_commit)

    production_seeds = list(plan["production"]["seeds"])
    _require_equal("production_seeds", authorization.get("production_seeds"), production_seeds)
    _require_equal(
        "completed_production_seeds",
        authorization.get("completed_production_seeds"),
        production_seeds,
    )

    calibration = authorization.get("calibration")
    if not isinstance(calibration, dict):
        raise RuntimeError("authorization.calibration must be an object")
    contingency = plan["contingency"]
    _require_equal("calibration.split", calibration.get("split"), contingency["calibration_split"])
    _require_equal(
        "calibration.evaluation_seed",
        calibration.get("evaluation_seed"),
        contingency["calibration_evaluation_seed"],
    )
    _require_equal(
        "calibration.examples_per_operator",
        calibration.get("examples_per_operator"),
        contingency["calibration_examples_per_operator"],
    )
    _require_equal(
        "calibration.completed_seed_folds",
        calibration.get("completed_seed_folds"),
        production_seeds,
    )
    status = calibration.get("status")
    if status not in _ALLOWED_CALIBRATION_STATUSES:
        raise RuntimeError(f"unsupported calibration status in authorization: {status!r}")
    if status == "rescue_selected":
        selected_family = calibration.get("selected_family")
        if selected_family not in contingency["stage_order"]:
            raise RuntimeError("rescue_selected authorization has an invalid selected_family")
        mixer_hash = calibration.get("mixer_contract_sha256")
        if not isinstance(mixer_hash, str) or len(mixer_hash) != 64:
            raise RuntimeError("rescue_selected authorization requires a SHA-256 mixer contract hash")
    else:
        _require_equal("calibration.selected_family", calibration.get("selected_family"), None)
        _require_equal(
            "calibration.mixer_contract_sha256",
            calibration.get("mixer_contract_sha256"),
            None,
        )

    final = plan["final"]
    _require_equal("split", split, split if split in final["splits"] else None)
    _require_equal("evaluation_seed", evaluation_seed, final["evaluation_seed"])
    _require_equal(
        "examples_per_operator",
        examples_per_operator,
        final["examples_per_operator"],
    )
    authorization_final = authorization.get("final")
    if not isinstance(authorization_final, dict):
        raise RuntimeError("authorization.final must be an object")
    _require_equal("final.splits", authorization_final.get("splits"), list(final["splits"]))
    _require_equal(
        "final.evaluation_seed",
        authorization_final.get("evaluation_seed"),
        final["evaluation_seed"],
    )
    _require_equal(
        "final.examples_per_operator",
        authorization_final.get("examples_per_operator"),
        final["examples_per_operator"],
    )

    return {
        "authorization_path": str(authorization_path.resolve()),
        "authorization_sha256": authorization_sha256,
        "plan_id": plan["id"],
        "plan_sha256": plan_sha256,
        "calibration_status": status,
        "selected_family": calibration.get("selected_family"),
    }
