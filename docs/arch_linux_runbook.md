# Arch Linux Runbook

## System prerequisites

Required commands:

```text
python 3.10+
git
nvidia-smi
bash
flock
```

`flock` is provided by `util-linux`, which is part of a normal Arch installation. `systemd-inhibit` is optional but normally available on systemd-based Arch systems.

Minimal user-space packages:

```bash
sudo pacman -S --needed python python-pip git base-devel
```

Install the NVIDIA driver appropriate to the GPU and kernel. The project intentionally does not automate selection among `nvidia`, `nvidia-open`, DKMS variants, or custom kernels. Reboot when required, then verify:

```bash
nvidia-smi
```

## Python environment

```bash
bash scripts/bootstrap_arch_linux.sh
```

For an explicitly selected PyTorch CUDA wheel channel:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  bash scripts/bootstrap_arch_linux.sh
```

The driver must support the CUDA runtime bundled by that wheel. The local CUDA toolkit package is not required merely to run a PyTorch wheel, but may be needed for custom extension builds.

## Preflight

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
.venv/bin/python -m pytest -q
.venv/bin/opfusion-train-batch \
  --config configs/experiments/gpt_bias_fusion_factory_v2.yaml \
  --plan-only
```

## Start

```bash
bash scripts/run_bias_fusion_factory_v2.sh \
  configs/experiments/gpt_bias_fusion_factory_v2.yaml \
  detach
```

Monitor:

```bash
cat runs/gpt_bias_fusion_factory_v2/batch.pid
ls -1t logs/bias_fusion_factory_*.log | head -1 | xargs tail -f
nvidia-smi
```

## Resume

Run the same launch command. The queue recognizes completed jobs and loads `last.pt` for incomplete jobs.

Do not delete `runtime_state.json`, `last.pt`, or `complete.json` while a run is active.

## Kernel and driver updates

Arch is rolling release. Before restarting a long run after a system upgrade:

```bash
uname -r
pacman -Q | grep -E '^(linux|nvidia|cuda|python|python-pytorch)'
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

If the Python minor version changed, recreate `.venv` with the bootstrap script rather than reusing a broken environment.

## Storage

The launcher requires at least 15 GiB free by default. Override only after estimating checkpoint size:

```bash
MIN_FREE_GB=10 bash scripts/run_bias_fusion_factory_v2.sh ...
```

## Failure diagnosis

- `torch.cuda.is_available() == false`: driver/module/PyTorch wheel mismatch.
- OOM recovery repeats at micro-batch 4: reduce context/model size or set a smaller declared minimum and re-run tests.
- Non-finite restart limit reached: inspect `recovery.jsonl`, `metrics.jsonl`, and the last good checkpoint; do not increase retries blindly.
- duplicate factory error: inspect the PID and lock holder before removing the lock file.
