from __future__ import annotations

import hashlib
import string
import subprocess
from pathlib import Path
from typing import Any

import yaml

from opfusion.final_eval_guard import (
    ACTIVE_PLAN_PATH,
    FINAL_SPLITS,
    SURFACE_V4_EXPERIMENT_ID,
    validate_evaluation_policy as _validate_authorization,
)
from opfusion.training.config import RunConfig
from opfusion.training.experiment_contract import build_contract


LOCKED_FINAL_NAMES = FINAL_SPLITS | {"test"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _git_is_clean(repo_root: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return not output.strip()
    except (OSError, subprocess.SubprocessError):
        return False


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
    repo_root = Path(repo_root).resolve()

    if config.experiment_id.startswith("model_design_pilot_") and split != "validation":
        raise RuntimeError("model-design pilot profiles are validation-only")

    if config.experiment_id != SURFACE_V4_EXPERIMENT_ID or split not in LOCKED_FINAL_NAMES:
        return _validate_authorization(
            repo_root=repo_root,
            config=config,
            manifest=manifest,
            split=split,
            evaluation_seed=evaluation_seed,
            examples_per_operator=examples_per_operator,
            final_authorization_path=final_authorization_path,
        )

    plan_path = repo_root / ACTIVE_PLAN_PATH
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))["plan"]
    if split not in plan["final"]["splits"]:
        raise RuntimeError(
            f"split {split!r} is not preregistered; use one of {plan['final']['splits']}"
        )

    expected_authorization = repo_root / plan["contingency"]["final_authorization"]["path"]
    if final_authorization_path is not None:
        supplied = Path(final_authorization_path)
        if not supplied.is_absolute():
            supplied = repo_root / supplied
        if supplied.resolve() != expected_authorization.resolve():
            raise RuntimeError(
                "final authorization must use the preregistered path: "
                f"{expected_authorization}"
            )

    fingerprint = manifest.get("experiment_fingerprint")
    if not _is_sha256(fingerprint):
        raise RuntimeError("surface-v4 final manifest has no valid experiment fingerprint")
    current_fingerprint = build_contract(repo_root, config)["fingerprint"]
    if fingerprint != current_fingerprint:
        raise RuntimeError(
            "surface-v4 final manifest fingerprint does not match the current code/config contract"
        )
    if not _git_is_clean(repo_root):
        raise RuntimeError("final evaluation requires a clean tracked Git checkout")

    result = _validate_authorization(
        repo_root=repo_root,
        config=config,
        manifest=manifest,
        split=split,
        evaluation_seed=evaluation_seed,
        examples_per_operator=examples_per_operator,
        final_authorization_path=final_authorization_path,
    )
    if result is None:
        raise RuntimeError("surface-v4 final evaluation authorization was not applied")
    if result.get("plan_sha256") != _sha256_file(plan_path):
        raise RuntimeError("active plan hash changed during final authorization validation")
    return result
