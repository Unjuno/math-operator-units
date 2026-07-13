#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml}"
MODE="${2:-foreground}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/experiments/gpt_bias_fusion_factory_surface_v4_smoke.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_BATCH="${TRAIN_BATCH:-$ROOT/.venv/bin/opfusion-train-batch-design}"
REPO_AUDIT="${REPO_AUDIT:-$ROOT/.venv/bin/opfusion-audit}"
AUDIT_DATA="${AUDIT_DATA:-$ROOT/.venv/bin/opfusion-audit-data-design}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-512}"

if [[ "${OPFUSION_ALLOW_V4_PRODUCTION:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
surface-v4 production is intentionally gated.
Run the four-condition model-design pilot first and inspect its validation/test reports:
  bash scripts/run_model_design_pilot.sh detach
After selecting the weak-base/retention design, re-run with:
  OPFUSION_ALLOW_V4_PRODUCTION=1 bash scripts/run_bias_fusion_factory_surface_v4.sh ...
EOF
  exit 64
fi

if [[ ! -x "$PYTHON" || ! -x "$TRAIN_BATCH" || ! -x "$REPO_AUDIT" || ! -x "$AUDIT_DATA" ]]; then
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

mkdir -p audits logs runs/gpt_bias_fusion_factory_surface_v4
"$PYTHON" -m pytest -q
"$REPO_AUDIT" . --data-samples-per-operator 64
"$AUDIT_DATA" \
  --config "$CONFIG" \
  --samples-per-operator "$AUDIT_SAMPLES" \
  --out audits/surface_v4_data_audit.json
"$TRAIN_BATCH" --config "$CONFIG" --plan-only

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "Running surface-v4 CUDA smoke batch..."
  if [[ "${KEEP_SMOKE:-0}" != "1" ]]; then
    rm -rf runs/gpt_bias_fusion_factory_surface_v4_smoke
  fi
  "$TRAIN_BATCH" --config "$SMOKE_CONFIG"
fi

WATCH=(bash scripts/watch_bias_fusion_factory_surface_v4.sh "$CONFIG")
if command -v systemd-inhibit >/dev/null 2>&1; then
  WATCH=(systemd-inhibit --what=sleep:shutdown --why="surface-v4 bias fusion model factory" --mode=block "${WATCH[@]}")
fi

if [[ "$MODE" == "detach" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="logs/bias_fusion_surface_v4_${stamp}.log"
  nohup "${WATCH[@]}" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > runs/gpt_bias_fusion_factory_surface_v4/batch.pid
  echo "started watchdog PID $pid; log: $log"
else
  exec "${WATCH[@]}"
fi
