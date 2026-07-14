from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs/d4_specialist_ablation/full_diagnostic_state.json"
SUMMARY = ROOT / "evaluations/d4_specialist_ablation/summary.json"
CONFIGS = {
    "sum_a": ("aggregation.sum", "configs/experiments/d4_specialist_ablation/sum_a.yaml"),
    "sum_b": ("aggregation.sum", "configs/experiments/d4_specialist_ablation/sum_b.yaml"),
    "sum_c": ("aggregation.sum", "configs/experiments/d4_specialist_ablation/sum_c.yaml"),
    "neg_a": ("scalar.neg", "configs/experiments/d4_specialist_ablation/neg_a.yaml"),
    "neg_b": ("scalar.neg", "configs/experiments/d4_specialist_ablation/neg_b.yaml"),
    "neg_c": ("scalar.neg", "configs/experiments/d4_specialist_ablation/neg_c.yaml"),
}


def write_state(status: str, phase: str, detail: str) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "status": status,
        "phase": phase,
        "detail": detail,
        "updated_unix": time.time(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def diagnose_selected(name: str, operator: str, config: str) -> dict:
    output = ROOT / "runs/d4_specialist_ablation" / name
    selected = output / "seed_0" / operator.replace(".", "_") / "selected.pt"
    if not selected.is_file():
        raise RuntimeError(f"selected checkpoint missing: {selected}")
    manifest = output / "selected_diagnostic_manifest.json"
    contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({
        "experiment_fingerprint": contract["fingerprint"],
        "unit_checkpoints": {operator: str(selected.relative_to(ROOT))},
        "subset_id": "d4_selected",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / "evaluations/d4_specialist_ablation" / f"{name}_selected.json"
    run([
        str(ROOT / ".venv/bin/opfusion-diagnose-specialist-failures"),
        "--config", config,
        "--manifest", str(manifest),
        "--operators", operator,
        "--split", "validation",
        "--evaluation-seed", "704000",
        "--examples-per-operator", "64",
        "--retain-examples", "20",
        "--out", str(report_path),
    ])
    return json.loads(report_path.read_text(encoding="utf-8"))["operators"][operator]


def main() -> int:
    try:
        write_state("running", "training", "shared Base and six specialist ablations")
        run(["bash", "scripts/run_d4_ablation.sh"])
        results = {}
        for name, (operator, config) in CONFIGS.items():
            write_state("running", "diagnostics", name)
            results[name] = diagnose_selected(name, operator, config)

        compact = {}
        for name, result in results.items():
            generation = result["generation"]
            teacher = result["teacher_forced"]
            passed = (
                generation["trace_validity"] >= 0.80
                and generation["final_value_accuracy"] >= 0.80
                and generation["stop_accuracy"] >= 0.95
                and teacher["token_accuracy"] >= 0.80
            )
            compact[name] = {
                "passed": passed,
                "trace_validity": generation["trace_validity"],
                "final_value_accuracy": generation["final_value_accuracy"],
                "eos_accuracy": generation["stop_accuracy"],
                "teacher_forced_token_accuracy": teacher["token_accuracy"],
                "checkpoint": result["checkpoint"],
            }
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps({
            "status": "completed",
            "thresholds": {"trace": 0.80, "final": 0.80, "eos": 0.95, "teacher_token": 0.80},
            "conditions": compact,
            "sum_any_passed": any(compact[n]["passed"] for n in ("sum_a", "sum_b", "sum_c")),
            "neg_any_passed": any(compact[n]["passed"] for n in ("neg_a", "neg_b", "neg_c")),
            "production_go": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_state("completed", "completed", str(SUMMARY.relative_to(ROOT)))
        return 0
    except Exception as exc:
        write_state("failed", "failed", repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
