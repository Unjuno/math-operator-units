#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f /etc/arch-release ]]; then
  echo "warning: /etc/arch-release was not found; continuing with the portable Python setup" >&2
fi

if [[ "${INSTALL_SYSTEM_DEPS:-0}" == "1" ]]; then
  sudo pacman -S --needed python python-pip git base-devel
fi

for command in python git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "missing command: $command" >&2
    echo "On Arch Linux: sudo pacman -S --needed python python-pip git base-devel" >&2
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

if ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    .venv/bin/python -m pip install torch --index-url "$TORCH_INDEX_URL"
  else
    .venv/bin/python -m pip install torch
  fi
fi
.venv/bin/python -m pip install -e '.[dev]'

.venv/bin/python - <<'PY'
import platform
import torch
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA device: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("Install/enable a compatible NVIDIA driver before GPU runs.")
PY

cat <<'EOF'
Setup complete.

Required first run: four-condition model-design pilot
  bash scripts/run_model_design_pilot.sh detach

Guarded production candidate after reviewing the pilot:
  OPFUSION_ALLOW_V4_PRODUCTION=1 \
    bash scripts/run_bias_fusion_factory_surface_v4.sh \
      configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
      detach

The v4 path uses a weak multitask common base, inactive-operator retention,
validation-selected endpoints, and strict experiment fingerprints. Legacy
surface-v3 and typed-v2 launchers require explicit opt-in.

On Arch Linux, verify `nvidia-smi` after every kernel/driver update. The script
does not assume Ubuntu packages or apt.
EOF
