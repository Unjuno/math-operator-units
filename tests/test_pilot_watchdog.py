from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
WATCHER = ROOT / "scripts/watch_model_design_pilot.sh"


def _watchdog_env(tmp_path: Path, worker: Path) -> dict[str, str]:
    log = tmp_path / "pilot.log"
    log.write_text("test start\n", encoding="utf-8")
    return {
        **os.environ,
        "PYTHON": sys.executable,
        "WORKER": str(worker),
        "MAX_RESTARTS": "3",
        "RESTART_DELAY_SECONDS": "0",
        "STALL_TIMEOUT_SECONDS": "30",
        "STALL_CHECK_SECONDS": "1",
        "LOCK_FILE": str(tmp_path / "pilot.lock"),
        "STATE_FILE": str(tmp_path / "pilot_state.json"),
        "PID_FILE": str(tmp_path / "pilot.pid"),
        "PILOT_LOG": str(log),
    }


def test_watchdog_reaps_a_successful_worker_without_false_stall(tmp_path: Path) -> None:
    worker = tmp_path / "success.sh"
    worker.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(WATCHER)],
        cwd=ROOT,
        env=_watchdog_env(tmp_path, worker),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    state = json.loads((tmp_path / "pilot_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["watchdog_attempt"] == 1


def test_watchdog_retries_one_failed_worker_then_completes(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    worker = tmp_path / "retry.sh"
    worker.write_text(
        "#!/usr/bin/env bash\n"
        f"counter={counter!s}\n"
        "count=0\n"
        "[[ -f \"$counter\" ]] && count=$(cat \"$counter\")\n"
        "count=$((count + 1))\n"
        "echo \"$count\" > \"$counter\"\n"
        "[[ $count -ge 2 ]]\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(WATCHER)],
        cwd=ROOT,
        env=_watchdog_env(tmp_path, worker),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert counter.read_text(encoding="utf-8").strip() == "2"
    state = json.loads((tmp_path / "pilot_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["watchdog_attempt"] == 2
