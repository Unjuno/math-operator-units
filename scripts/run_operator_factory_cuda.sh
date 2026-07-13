#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/gpt_operator_factory_v1.yaml}"
MODE="${2:-foreground}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/experiments/gpt_operator_factory_smoke.yaml}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"

python - <<'PY'
import shutil
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
free_gb = shutil.disk_usage(".").free / (1024 ** 3)
print(f"Free disk: {free_gb:.1f} GiB")
PY

free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if [[ "$free_kb" -lt "$required_kb" ]]; then
  echo "Need at least ${MIN_FREE_GB} GiB free disk before starting" >&2
  exit 1
fi

python -m pytest -q

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "Running short CUDA smoke batch..."
  opfusion-train-batch --config "$SMOKE_CONFIG"
fi

mkdir -p logs runs/gpt_operator_factory_v1

if [[ "$MODE" == "detach" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="logs/operator_factory_${stamp}.log"
  nohup bash scripts/watch_operator_factory.sh "$CONFIG" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > runs/gpt_operator_factory_v1/batch.pid
  echo "started watchdog PID $pid; log: $log"
else
  exec bash scripts/watch_operator_factory.sh "$CONFIG"
fi
