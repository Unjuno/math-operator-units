from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from opfusion.training.design_config import load_design_run_config

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/d5_operator_unit_factory"
EVAL_ROOT = ROOT / "evaluations/d5_operator_unit_factory"
STATE = RUN_ROOT / "state.json"
SUMMARY = EVAL_ROOT / "summary.json"
BASE_CONFIG = "configs/experiments/d5_operator_unit_factory/shared_base.yaml"
EVALUATION_SEED = 705100
EXAMPLES_PER_OPERATOR = 128


@dataclass(frozen=True)
class Condition:
    name: str
    operator: str
    config: str
    initialization: str

    @property
    def output(self) -> Path:
        return RUN_ROOT / self.name

    @property
    def job_dir(self) -> Path:
        return self.output / "seed_0" / self.operator.replace(".", "_")


CONDITIONS = (
    Condition("sum_scratch", "aggregation.sum", "configs/experiments/d5_operator_unit_factory/sum_scratch.yaml", "scratch"),
    Condition("sum_base", "aggregation.sum", "configs/experiments/d5_operator_unit_factory/sum_base.yaml", "weak_base"),
    Condition("neg_scratch", "scalar.neg", "configs/experiments/d5_operator_unit_factory/neg_scratch.yaml", "scratch"),
    Condition("neg_base", "scalar.neg", "configs/experiments/d5_operator_unit_factory/neg_base.yaml", "weak_base"),
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def state(status: str, phase: str, condition: str | None, detail: str) -> None:
    atomic_json(STATE, {
        "status": status,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def model_state_hash(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise RuntimeError(f"missing model_state_dict: {path}")
    digest = hashlib.sha256()
    for name in sorted(model_state):
        tensor = model_state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(bytes(tensor.view(torch.uint8).tolist()))
    return digest.hexdigest()


def preflight() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("tracked checkout is dirty")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("index is dirty")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    for config in [BASE_CONFIG, *(condition.config for condition in CONDITIONS)]:
        loaded = load_design_run_config(ROOT / config)
        loaded.validate()


def train_shared_base() -> Path:
    source = RUN_ROOT / "shared_base/seed_0/base_common"
    if not (source / "complete.json").is_file():
        state("running", "training", "shared_base", "training one weak multitask Base")
        run([str(ROOT / ".venv/bin/opfusion-train-one-design"), "--config", BASE_CONFIG, "--job", "base.common", "--seed", "0"])
    if not (source / "selected.pt").is_file():
        raise RuntimeError("shared Base selected.pt is missing")
    return source


def train_condition(condition: Condition, base_source: Path) -> None:
    complete = condition.job_dir / "complete.json"
    if complete.is_file():
        return
    if condition.initialization == "weak_base":
        state("running", "staging", condition.name, "staging verified shared Base")
        run([
            str(ROOT / ".venv/bin/python"), "scripts/stage_d4_shared_base.py",
            "--source", str(base_source.relative_to(ROOT)),
            "--destination-output", str(condition.output.relative_to(ROOT)),
            "--base-config", BASE_CONFIG,
            "--condition-config", condition.config,
            "--seed", "0",
            "--role", "d5_shared_parent_base",
        ])
    state("running", "training", condition.name, f"training {condition.operator} from {condition.initialization}")
    run([str(ROOT / ".venv/bin/opfusion-train-one-design"), "--config", condition.config, "--job", condition.operator, "--seed", "0"])
    if not complete.is_file():
        raise RuntimeError(f"completion marker missing: {complete}")


def verify_scratch_initials() -> dict[str, Any]:
    paths = {
        condition.name: condition.output / "seed_0/shared_initial.pt"
        for condition in CONDITIONS if condition.initialization == "scratch"
    }
    hashes = {name: model_state_hash(path) for name, path in paths.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"scratch initial states differ: {hashes}")
    return {
        "verified_equal": True,
        "model_state_sha256": next(iter(hashes.values())),
        "paths": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }


def diagnose(condition: Condition) -> dict[str, Any]:
    selected = condition.job_dir / "selected.pt"
    contract = json.loads((condition.output / "experiment_contract.json").read_text(encoding="utf-8"))
    manifest = condition.output / "d5_selected_manifest.json"
    atomic_json(manifest, {
        "experiment_fingerprint": contract["fingerprint"],
        "unit_checkpoints": {condition.operator: str(selected.relative_to(ROOT))},
        "subset_id": "d5_selected",
    })
    report_path = EVAL_ROOT / f"{condition.name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state("running", "diagnostics", condition.name, "evaluating selected checkpoint")
    run([
        str(ROOT / ".venv/bin/opfusion-diagnose-specialist-failures"),
        "--config", condition.config,
        "--manifest", str(manifest),
        "--operators", condition.operator,
        "--split", "validation",
        "--evaluation-seed", str(EVALUATION_SEED),
        "--examples-per-operator", str(EXAMPLES_PER_OPERATOR),
        "--retain-examples", "20",
        "--out", str(report_path),
    ])
    result = json.loads(report_path.read_text(encoding="utf-8"))["operators"][condition.operator]
    generation = result["generation"]
    teacher = result["teacher_forced"]
    metrics = {
        "trace_validity": float(generation["trace_validity"]),
        "final_value_accuracy": float(generation["final_value_accuracy"]),
        "eos_accuracy": float(generation["stop_accuracy"]),
        "teacher_forced_token_accuracy": float(teacher["token_accuracy"]),
        "checkpoint": str(selected.relative_to(ROOT)),
    }
    metrics["passed"] = (
        metrics["trace_validity"] >= 0.80
        and metrics["final_value_accuracy"] >= 0.80
        and metrics["eos_accuracy"] >= 0.95
        and metrics["teacher_forced_token_accuracy"] >= 0.80
    )
    return metrics


def interpretation(scratch: dict[str, Any], base: dict[str, Any]) -> str:
    if scratch["passed"] and not base["passed"]:
        return "base_initialization_interferes"
    if not scratch["passed"] and base["passed"]:
        return "base_initialization_helps"
    if scratch["passed"] and base["passed"]:
        return "initialization_not_blocking"
    return "task_data_objective_or_capacity_failure"


def comparison(scratch: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    keys = ("trace_validity", "final_value_accuracy", "eos_accuracy", "teacher_forced_token_accuracy")
    return {
        "interpretation": interpretation(scratch, base),
        **{f"base_minus_scratch_{key}": base[key] - scratch[key] for key in keys},
    }


def main() -> int:
    started = time.time()
    try:
        state("running", "preflight", None, "checking fixed checkout, configs, and CUDA")
        preflight()
        base_source = train_shared_base()
        for condition in CONDITIONS:
            train_condition(condition, base_source)
        scratch = verify_scratch_initials()
        results = {condition.name: diagnose(condition) for condition in CONDITIONS}
        summary = {
            "status": "completed",
            "experiment": "d5_operator_unit_factory",
            "scientific_role": "scratch_vs_weak_base_initialization_diagnostic",
            "evaluation_seed": EVALUATION_SEED,
            "examples_per_operator": EXAMPLES_PER_OPERATOR,
            "thresholds": {"trace": 0.80, "final": 0.80, "eos": 0.95, "teacher_token": 0.80},
            "scratch_initialization": scratch,
            "conditions": results,
            "comparisons": {
                "aggregation.sum": comparison(results["sum_scratch"], results["sum_base"]),
                "scalar.neg": comparison(results["neg_scratch"], results["neg_base"]),
            },
            "production_go": False,
            "final_splits_opened": False,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(SUMMARY, summary)
        state("completed", "completed", None, str(SUMMARY.relative_to(ROOT)))
        return 0
    except Exception as exc:
        state("failed", "failed", None, repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
