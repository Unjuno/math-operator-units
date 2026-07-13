#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml}"
MODE="${2:-foreground}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/experiments/gpt_bias_fusion_factory_surface_v3_smoke.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_BATCH="${TRAIN_BATCH:-$ROOT/.venv/bin/opfusion-train-batch-surface}"
AUDIT_DATA="${AUDIT_DATA:-$ROOT/.venv/bin/opfusion-audit-data}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-512}"

if [[ ! -x "$PYTHON" || ! -x "$TRAIN_BATCH" || ! -x "$AUDIT_DATA" ]]; then
  echo "virtual environment is missing or stale; run: bash scripts/bootstrap_arch_linux.sh" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. On Arch Linux, verify the NVIDIA driver and running kernel module." >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import shutil
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
props = torch.cuda.get_device_properties(0)
print(f"CUDA device: {props.name}")
print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
print(f"PyTorch: {torch.__version__}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
print(f"Free disk: {shutil.disk_usage('.').free / 1024**3:.1f} GiB")
PY

free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if [[ "$free_kb" -lt "$required_kb" ]]; then
  echo "Need at least ${MIN_FREE_GB} GiB free disk before starting" >&2
  exit 1
fi

mkdir -p audits logs runs/gpt_bias_fusion_factory_surface_v3
"$PYTHON" -m pytest -q
"$AUDIT_DATA" \
  --config "$CONFIG" \
  --samples-per-operator "$AUDIT_SAMPLES" \
  --out audits/surface_v3_data_audit.json
"$TRAIN_BATCH" --config "$CONFIG" --plan-only

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "Running surface-v3 CUDA smoke batch..."
  if [[ "${KEEP_SMOKE:-0}" != "1" ]]; then
    rm -rf runs/gpt_bias_fusion_factory_surface_v3_smoke
  fi
  "$TRAIN_BATCH" --config "$SMOKE_CONFIG"
fi

WATCH=(bash scripts/watch_bias_fusion_factory_surface_v3.sh "$CONFIG")
if command -v systemd-inhibit >/dev/null 2>&1; then
  WATCH=(systemd-inhibit --what=sleep:shutdown --why="surface bias fusion model factory" --mode=block "${WATCH[@]}")
fi

if [[ "$MODE" == "detach" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="logs/bias_fusion_surface_v3_${stamp}.log"
  nohup "${WATCH[@]}" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > runs/gpt_bias_fusion_factory_surface_v3/batch.pid
  echo "started watchdog PID $pid; log: $log"
else
  exec "${WATCH[@]}"
fi
