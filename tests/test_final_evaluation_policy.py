from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from opfusion.final_eval_policy import validate_evaluation_policy
from opfusion.training.config import load_run_config
from opfusion.training.experiment_contract import build_contract


ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs/experiments/experiment_plan_v2.yaml"
CONFIG_PATH = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml"
PILOT_CONFIG_PATH = ROOT / "configs/experiments/model_design_pilot_weak_retention.yaml"
AUTH_PATH = ROOT / "evaluations/fusion_calibration/final_authorization.json"


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _plan() -> dict:
    return yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))["plan"]


def _authorization(fingerprint: str) -> dict:
    plan = _plan()
    contingency = plan["contingency"]
    final = plan["final"]
    seeds = list(plan["production"]["seeds"])
    return {
        "authorization_abi_version": 1,
        "plan_id": plan["id"],
        "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "experiment_id": "gpt_bias_fusion_factory_surface_v4",
        "experiment_fingerprint": fingerprint,
        "production_seeds": seeds,
        "completed_production_seeds": seeds,
        "calibration": {
            "split": contingency["calibration_split"],
            "evaluation_seed": contingency["calibration_evaluation_seed"],
            "examples_per_operator": contingency["calibration_examples_per_operator"],
            "completed_seed_folds": seeds,
            "status": "raw_preserved_no_rescue",
            "selected_family": None,
            "mixer_contract_sha256": None,
        },
        "final": {
            "splits": list(final["splits"]),
            "evaluation_seed": final["evaluation_seed"],
            "examples_per_operator": final["examples_per_operator"],
        },
    }


@pytest.fixture
def clean_authorization_path():
    previous = AUTH_PATH.read_bytes() if AUTH_PATH.exists() else None
    AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_PATH.unlink(missing_ok=True)
    try:
        yield AUTH_PATH
    finally:
        if previous is None:
            AUTH_PATH.unlink(missing_ok=True)
        else:
            AUTH_PATH.write_bytes(previous)


def test_surface_v4_final_split_requires_authorization(clean_authorization_path: Path) -> None:
    config = load_run_config(CONFIG_PATH)
    fingerprint = build_contract(ROOT, config)["fingerprint"]
    with pytest.raises(RuntimeError, match="locked"):
        validate_evaluation_policy(
            repo_root=ROOT,
            config=config,
            manifest={"experiment_fingerprint": fingerprint},
            split="iid_test",
            evaluation_seed=700000,
            examples_per_operator=64,
            final_authorization_path=None,
        )


def test_surface_v4_rejects_test_alias_before_opening_data(clean_authorization_path: Path) -> None:
    config = load_run_config(CONFIG_PATH)
    fingerprint = build_contract(ROOT, config)["fingerprint"]
    with pytest.raises(RuntimeError, match="not preregistered"):
        validate_evaluation_policy(
            repo_root=ROOT,
            config=config,
            manifest={"experiment_fingerprint": fingerprint},
            split="test",
            evaluation_seed=700000,
            examples_per_operator=64,
            final_authorization_path=AUTH_PATH,
        )


def test_valid_authorization_unlocks_only_frozen_final_settings(clean_authorization_path: Path) -> None:
    config = load_run_config(CONFIG_PATH)
    fingerprint = build_contract(ROOT, config)["fingerprint"]
    AUTH_PATH.write_text(json.dumps(_authorization(fingerprint), sort_keys=True) + "\n", encoding="utf-8")

    result = validate_evaluation_policy(
        repo_root=ROOT,
        config=config,
        manifest={"experiment_fingerprint": fingerprint},
        split="iid_test",
        evaluation_seed=700000,
        examples_per_operator=64,
        final_authorization_path=AUTH_PATH,
    )
    assert result is not None
    assert result["plan_id"] == "bias_fusion_surface_v4_v2"
    assert result["calibration_status"] == "raw_preserved_no_rescue"

    with pytest.raises(RuntimeError, match="evaluation_seed"):
        validate_evaluation_policy(
            repo_root=ROOT,
            config=config,
            manifest={"experiment_fingerprint": fingerprint},
            split="iid_test",
            evaluation_seed=700001,
            examples_per_operator=64,
            final_authorization_path=AUTH_PATH,
        )


def test_stale_manifest_fingerprint_is_rejected(clean_authorization_path: Path) -> None:
    config = load_run_config(CONFIG_PATH)
    stale = "0" * 64
    AUTH_PATH.write_text(json.dumps(_authorization(stale), sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current code/config contract"):
        validate_evaluation_policy(
            repo_root=ROOT,
            config=config,
            manifest={"experiment_fingerprint": stale},
            split="iid_test",
            evaluation_seed=700000,
            examples_per_operator=64,
            final_authorization_path=AUTH_PATH,
        )


def test_pilot_profiles_are_validation_only() -> None:
    config = load_run_config(PILOT_CONFIG_PATH)
    with pytest.raises(RuntimeError, match="validation-only"):
        validate_evaluation_policy(
            repo_root=ROOT,
            config=config,
            manifest={},
            split="iid_test",
            evaluation_seed=701000,
            examples_per_operator=64,
            final_authorization_path=None,
        )
