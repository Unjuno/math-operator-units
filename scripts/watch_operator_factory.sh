#!/usr/bin/env bash
set -u

CONFIG="${1:-configs/experiments/gpt_operator_factory_v1.yaml}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-60}"
attempt=0

while true; do
  attempt=$((attempt + 1))
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] operator factory attempt ${attempt}"
  opfusion-train-batch --config "$CONFIG"
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] operator factory completed"
    exit 0
  fi
  if [[ $attempt -ge $MAX_RESTARTS ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] giving up after ${attempt} attempts; last status=${status}" >&2
    exit "$status"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed with status=${status}; retrying in ${RESTART_DELAY_SECONDS}s" >&2
  sleep "$RESTART_DELAY_SECONDS"
done
