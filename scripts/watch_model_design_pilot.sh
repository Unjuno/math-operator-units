#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
WORKER="${WORKER:-$ROOT/scripts/run_model_design_pilot.sh}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-60}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-21600}"
STALL_CHECK_SECONDS="${STALL_CHECK_SECONDS:-300}"
LOCK_FILE="${LOCK_FILE:-$ROOT/runs/model_design_pilot/pilot.lock}"
STATE_FILE="${STATE_FILE:-$ROOT/runs/model_design_pilot/pilot_state.json}"
PID_FILE="${PID_FILE:-$ROOT/runs/model_design_pilot/pilot.pid}"
PILOT_LOG="${PILOT_LOG:-}"

mkdir -p runs/model_design_pilot logs
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another model-design pilot holds $LOCK_FILE" >&2
  exit 73
fi
if ! [[ "$MAX_RESTARTS" =~ ^[1-9][0-9]*$ && "$RESTART_DELAY_SECONDS" =~ ^[0-9]+$ && "$STALL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ && "$STALL_CHECK_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "watchdog timing values must be nonnegative/positive integers" >&2
  exit 64
fi

write_watch_state() {
  local status="$1"
  local attempt="$2"
  local detail="$3"
  if [[ ! -x "$PYTHON" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${status}: ${detail}" >&2
    return
  fi
  "$PYTHON" - "$STATE_FILE" "$status" "$attempt" "$detail" "$$" "$PILOT_LOG" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
existing = {}
if path.exists():
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
existing.update(
    {
        "status": sys.argv[2],
        "watchdog_attempt": int(sys.argv[3]),
        "watchdog_detail": sys.argv[4],
        "watchdog_pid": int(sys.argv[5]),
        "log": sys.argv[6] or existing.get("log"),
        "updated_unix": time.time(),
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

cleanup_pid() {
  if [[ -f "$PID_FILE" ]]; then
    recorded="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$recorded" == "$$" || "$recorded" == "$PPID" ]]; then
      rm -f "$PID_FILE"
    fi
  fi
}

attempt=0
child_pid=""
child_is_process_group=0
stop_worker() {
  if [[ -z "$child_pid" ]] || ! kill -0 "$child_pid" 2>/dev/null; then
    return
  fi
  if [[ "$child_is_process_group" == "1" ]]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || true
  else
    pkill -TERM -P "$child_pid" 2>/dev/null || true
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  wait "$child_pid" 2>/dev/null || true
}

on_signal() {
  local signal="$1"
  echo "received ${signal}; stopping pilot worker" >&2
  stop_worker
  write_watch_state interrupted "$attempt" "watchdog received ${signal}"
  cleanup_pid
  exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap cleanup_pid EXIT

while true; do
  attempt=$((attempt + 1))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] model-design pilot attempt ${attempt}/${MAX_RESTARTS}"
  write_watch_state running "$attempt" "starting or resuming the four-condition pilot"

  if command -v setsid >/dev/null 2>&1; then
    setsid env OPFUSION_PILOT_CHILD=1 OPFUSION_PILOT_WATCHED=1 \
      bash "$WORKER" foreground &
    child_is_process_group=1
  else
    env OPFUSION_PILOT_CHILD=1 OPFUSION_PILOT_WATCHED=1 \
      bash "$WORKER" foreground &
    child_is_process_group=0
  fi
  child_pid=$!
  stalled=0

  if [[ -n "$PILOT_LOG" ]]; then
    while kill -0 "$child_pid" 2>/dev/null; do
      sleep "$STALL_CHECK_SECONDS"
      if ! kill -0 "$child_pid" 2>/dev/null; then
        break
      fi
      now="$(date +%s)"
      last_activity="$(stat -c %Y "$PILOT_LOG" 2>/dev/null || echo "$now")"
      if (( now - last_activity >= STALL_TIMEOUT_SECONDS )); then
        echo "pilot log has not advanced for ${STALL_TIMEOUT_SECONDS}s; restarting worker" >&2
        write_watch_state restarting "$attempt" "stall timeout after ${STALL_TIMEOUT_SECONDS}s without log progress"
        stop_worker
        status=124
        stalled=1
        break
      fi
    done
  fi

  if [[ "$stalled" == "0" ]]; then
    wait "$child_pid"
    status=$?
  fi
  child_pid=""
  child_is_process_group=0

  if [[ $status -eq 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] model-design pilot completed"
    write_watch_state completed "$attempt" "all conditions and evaluations completed"
    exit 0
  fi

  case "$status" in
    64|65|66|73)
      echo "pilot stopped with permanent preflight status=${status}; not retrying" >&2
      write_watch_state failed "$attempt" "permanent preflight failure status=${status}"
      exit "$status"
      ;;
  esac

  if [[ $attempt -ge $MAX_RESTARTS ]]; then
    echo "giving up after ${attempt} attempts; last status=${status}" >&2
    write_watch_state failed "$attempt" "retry limit reached; last status=${status}"
    exit "$status"
  fi

  echo "pilot worker failed with status=${status}; resuming in ${RESTART_DELAY_SECONDS}s" >&2
  write_watch_state restarting "$attempt" "worker status=${status}; sleeping ${RESTART_DELAY_SECONDS}s"
  sleep "$RESTART_DELAY_SECONDS"
done
