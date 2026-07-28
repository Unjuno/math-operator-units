#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="runs/fusion_combination_search"
STATE="$RUN_DIR/state.json"
PID_FILE="$RUN_DIR/search.pid"
LOCK_FILE="$RUN_DIR/search.lock"
OUT="evaluations/fusion_combination_search/summary.json"
mkdir -p "$RUN_DIR" logs "$(dirname "$OUT")"

MODE="${1:-run}"
if [[ $# -gt 0 ]]; then
    shift
fi

write_state() {
    local status="$1"
    local phase="$2"
    local detail="$3"
    local log_path="${4:-}"
    STATUS="$status" PHASE="$phase" DETAIL="$detail" LOG_PATH="$log_path" STATE_PATH="$STATE" \
        .venv/bin/python - <<'PY'
import json
import os
import time
from pathlib import Path

path = Path(os.environ["STATE_PATH"])
payload = {
    "status": os.environ["STATUS"],
    "phase": os.environ["PHASE"],
    "detail": os.environ["DETAIL"],
    "log": os.environ.get("LOG_PATH") or None,
    "updated_unix": time.time(),
}
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

pid_is_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

run_search() {
    local log_path="${FUSION_SEARCH_LOG:-}"
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "Another fusion combination search holds $LOCK_FILE" >&2
        exit 1
    fi

    if [[ ! -x ".venv/bin/opfusion-search-fusion-combinations" ]]; then
        echo "Missing CLI. Run: .venv/bin/pip install -e ." >&2
        exit 1
    fi

    if ! .venv/bin/python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is required by the unattended search launcher", file=sys.stderr)
    raise SystemExit(1)
print(torch.cuda.get_device_name(0))
PY
    then
        write_state "failed" "preflight" "CUDA unavailable" "$log_path"
        exit 1
    fi

    write_state "running" "validation_search" "31 subsets x alpha grid x raw/RMS-equalized modes" "$log_path"
    trap 'code=$?; if [[ $code -ne 0 ]]; then write_state "failed" "validation_search" "exit_code=$code" "$log_path"; fi; exit $code' EXIT

    .venv/bin/opfusion-search-fusion-combinations \
        --source all \
        --device cuda \
        --out "$OUT" \
        "$@"

    write_state "completed" "completed" "$OUT" "$log_path"
    trap - EXIT
}

case "$MODE" in
    run|foreground)
        run_search "$@"
        ;;
    detach)
        if pid_is_running; then
            echo "Search is already running with PID $(cat "$PID_FILE")" >&2
            exit 1
        fi
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        log_path="logs/fusion_combination_search_${stamp}.log"
        FUSION_SEARCH_LOG="$log_path" nohup "$0" run "$@" >"$log_path" 2>&1 &
        pid=$!
        printf '%s\n' "$pid" > "$PID_FILE"
        write_state "running" "launching" "pid=$pid" "$log_path"
        echo "PID: $pid"
        echo "Log: $log_path"
        echo "State: $STATE"
        echo "Output: $OUT"
        ;;
    status)
        if [[ -f "$STATE" ]]; then
            cat "$STATE"
        else
            echo '{"status":"not_started"}'
        fi
        if pid_is_running; then
            echo "process: running (PID $(cat "$PID_FILE"))"
        else
            echo "process: not running"
        fi
        ;;
    *)
        echo "Usage: $0 {run|detach|status} [search options]" >&2
        echo "Example: $0 detach --examples-per-operator 64" >&2
        exit 2
        ;;
esac
