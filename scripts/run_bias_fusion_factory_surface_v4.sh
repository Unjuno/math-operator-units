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
PILOT_INDEX="${PILOT_INDEX:-evaluations/model_design_pilot/index.json}"
PILOT_PAIR_AUDIT="${PILOT_PAIR_AUDIT:-audits/model_design_pilot/pair_consistency.json}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ "${OPFUSION_ALLOW_V4_PRODUCTION:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
surface-v4 production is intentionally gated.
Run the four-condition model-design pilot first and inspect its validation reports:
  bash scripts/run_model_design_pilot.sh detach
The pilot reserves iid_test, operand_ood, and length_ood for final evaluation.
After selecting the weak-Base/retention design, re-run with:
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

# The environment variable records human approval. These checks additionally
# prove that the corrected pilot completed under this exact Git revision.
"$PYTHON" - "$PILOT_INDEX" "$PILOT_PAIR_AUDIT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
pair_path = Path(sys.argv[2])
conditions = ["identity_unanchored", "identity_retention", "weak_unanchored", "weak_retention"]
for path in (index_path, pair_path):
    if not path.is_file():
        raise SystemExit(f"corrected pilot artifact is missing: {path}")
index = json.loads(index_path.read_text(encoding="utf-8"))
pair = json.loads(pair_path.read_text(encoding="utf-8"))
if pair.get("status") != "passed":
    raise SystemExit(f"pilot pair consistency did not pass: {pair_path}")
if index.get("reserved_final_splits") != ["iid_test", "operand_ood", "length_ood"]:
    raise SystemExit("pilot index does not reserve all final IID/OOD splits")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
for condition in conditions:
    marker = Path("runs/model_design_pilot") / condition / "pilot_condition_complete.json"
    if not marker.is_file():
        raise SystemExit(f"pilot condition is incomplete: {condition}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("git_commit") != commit:
        raise SystemExit(f"pilot condition was not completed under current commit: {condition}")
    if payload.get("evaluation_splits") != ["validation"]:
        raise SystemExit(f"pilot condition used an unexpected evaluation split: {condition}")
print("corrected model-design pilot gate: passed")
PY

"$PYTHON" - <<'PY'
import os
import shutil
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
props = torch.cuda.get_device_properties(0)
print(f"CUDA device: {props.name}")
print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
print(f"PyTorch: {torch.__version__}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
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
