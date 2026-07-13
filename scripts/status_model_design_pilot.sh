#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PID_FILE="${PID_FILE:-$ROOT/runs/model_design_pilot/pilot.pid}"
STATE_FILE="${STATE_FILE:-$ROOT/runs/model_design_pilot/pilot_state.json}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "process: running (PID $pid)"
  else
    echo "process: not running (stale PID file: ${pid:-empty})"
  fi
else
  echo "process: not running (no PID file)"
fi

if [[ -f "$STATE_FILE" ]]; then
  echo "state:"
  if [[ -x "$PYTHON" ]]; then
    "$PYTHON" - "$STATE_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
updated = payload.get("updated_unix")
if updated is not None:
    payload["updated_utc"] = datetime.fromtimestamp(float(updated), tz=timezone.utc).isoformat()
for key in (
    "status",
    "condition",
    "phase",
    "detail",
    "watchdog_attempt",
    "watchdog_detail",
    "updated_utc",
    "log",
):
    value = payload.get(key)
    if value is not None:
        print(f"  {key}: {value}")
PY
  else
    cat "$STATE_FILE"
  fi
else
  echo "state: not created yet"
fi

if [[ -x "$PYTHON" ]]; then
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

conditions = ["identity_unanchored", "identity_retention", "weak_unanchored", "weak_retention"]
splits = ["validation"]
print("progress:")
for condition in conditions:
    root = Path("runs/model_design_pilot") / condition
    marker = root / "pilot_condition_complete.json"
    jobs = list((root / "seed_0").glob("*/complete.json")) if (root / "seed_0").exists() else []
    reports = sum(
        (Path("evaluations/model_design_pilot") / f"{condition}_{split}.json").is_file()
        for split in splits
    )
    diagnostics = sum(
        (Path("evaluations/model_design_pilot") / f"{condition}_{split}_units.json").is_file()
        for split in splits
    )
    status = "complete" if marker.is_file() else "incomplete"
    print(
        f"  {condition}: {status}; completed_models={len(jobs)}/7; "
        f"fusion_reports={reports}/1; unit_reports={diagnostics}/1"
    )
index = Path("evaluations/model_design_pilot/index.json")
pair = Path("audits/model_design_pilot/pair_consistency.json")
print(f"summary_index: {'present' if index.is_file() else 'not present'}")
if pair.is_file():
    try:
        payload = json.loads(pair.read_text(encoding="utf-8"))
        print(f"pair_consistency: {payload.get('status')}; warnings={len(payload.get('warnings', []))}")
    except Exception:
        print("pair_consistency: unreadable")
else:
    print("pair_consistency: not present")
print("final_splits: iid_test, operand_ood, length_ood reserved/not run by pilot")
PY
fi

latest_log="$(ls -1t logs/model_design_pilot_*.log 2>/dev/null | head -1 || true)"
if [[ -n "$latest_log" ]]; then
  echo "latest_log: $latest_log"
  echo "tail_command: tail -f $latest_log"
fi
