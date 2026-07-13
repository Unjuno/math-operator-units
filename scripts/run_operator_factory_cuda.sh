#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/gpt_operator_factory_v1.yaml}"
MODE="${2:-foreground}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
PY

python -m pytest -q
mkdir -p logs runs/gpt_operator_factory_v1

if [[ "$MODE" == "detach" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="logs/operator_factory_${stamp}.log"
  nohup opfusion-train-batch --config "$CONFIG" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > runs/gpt_operator_factory_v1/batch.pid
  echo "started PID $pid; log: $log"
else
  exec opfusion-train-batch --config "$CONFIG"
fi
