#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-detach}"
PID_FILE="runs/d4_specialist_ablation/full_diagnostic.pid"
mkdir -p runs/d4_specialist_ablation logs

if [[ "$MODE" == "detach" ]]; then
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "D4 full diagnostic already running: PID $(cat "$PID_FILE")"
        exit 0
    fi
    log="logs/d4_full_diagnostic_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup setsid bash "$0" foreground >"$log" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    echo "D4 full diagnostic started: PID $!"
    echo "log: $log"
    exit 0
fi

trap 'rm -f "$PID_FILE"' EXIT
exec .venv/bin/python scripts/d4_full_diagnostic.py
