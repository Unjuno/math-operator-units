#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/experiments/gpt_bias_fusion_factory_v2.yaml}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-60}"
TRAIN_BATCH="${TRAIN_BATCH:-$ROOT/.venv/bin/opfusion-train-batch}"
LOCK_FILE="${LOCK_FILE:-$ROOT/runs/gpt_bias_fusion_factory_v2/factory.lock}"
mkdir -p "$(dirname "$LOCK_FILE")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another bias-fusion factory process holds $LOCK_FILE" >&2
  exit 73
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] bias-fusion factory attempt ${attempt}"
  "$TRAIN_BATCH" --config "$CONFIG"
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] bias-fusion factory completed"
    exit 0
  fi
  if [[ $attempt -ge $MAX_RESTARTS ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] giving up after ${attempt} attempts; last status=${status}" >&2
    exit "$status"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed with status=${status}; retrying in ${RESTART_DELAY_SECONDS}s" >&2
  sleep "$RESTART_DELAY_SECONDS"
done
