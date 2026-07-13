#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f /etc/arch-release ]]; then
  echo "warning: /etc/arch-release was not found; continuing with the portable Python setup" >&2
fi

if [[ "${INSTALL_SYSTEM_DEPS:-0}" == "1" ]]; then
  sudo pacman -S --needed python python-pip git base-devel util-linux procps-ng
fi

for command in python git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "missing command: $command" >&2
    echo "On Arch Linux: sudo pacman -S --needed python python-pip git base-devel util-linux procps-ng" >&2
    exit 1
  fi
done

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required; found {sys.version}")
print(f"Python: {sys.version.split()[0]}")
PY

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel

# An explicitly selected CUDA wheel channel must replace an existing CPU-only
# installation as well as install into a fresh environment.
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  .venv/bin/python -m pip install --upgrade --force-reinstall torch --index-url "$TORCH_INDEX_URL"
elif ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  .venv/bin/python -m pip install torch
fi
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/python - <<'PY'
import os
import platform
import torch

print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA device: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
elif os.environ.get("REQUIRE_CUDA", "0") == "1":
    raise SystemExit("REQUIRE_CUDA=1 but PyTorch cannot access CUDA")
else:
    print("Install/enable a compatible NVIDIA driver and CUDA-enabled PyTorch wheel before GPU runs.")
PY

cat <<'EOF'
Setup complete.

Required real-hardware preflight before a long run:
  bash scripts/run_surface_v4_cuda_smoke.sh

Required first long run: all four model-design conditions, unattended
  bash scripts/run_model_design_pilot.sh detach

Pilot status
  bash scripts/status_model_design_pilot.sh

Guarded production candidate after reviewing the pilot:
  OPFUSION_ALLOW_V4_PRODUCTION=1 \
    bash scripts/run_bias_fusion_factory_surface_v4.sh \
      configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
      detach

The CUDA smoke trains the seven-model dependency graph for two optimizer steps
with production batch, deterministic CUDA, retention, checkpoint-selection,
fusion-evaluation, and per-unit diagnostic paths. It is an operational test,
not scientific evidence.

The pilot watchdog runs conditions sequentially, resumes incomplete checkpoints,
and retries unexpected worker failures. The v4 path uses a weak multitask common
base, inactive-operator retention, validation-selected endpoints, and strict
experiment fingerprints. Legacy surface-v3 and typed-v2 launchers require opt-in.

On Arch Linux, verify `nvidia-smi` after every kernel/driver update. The script
does not assume Ubuntu packages or apt.
EOF
