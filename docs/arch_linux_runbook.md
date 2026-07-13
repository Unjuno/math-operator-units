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

`flock` is provided by `util-linux`. `systemd-inhibit` is optional but normally available on systemd-based Arch systems.

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

The driver must support the CUDA runtime bundled by that wheel. The local CUDA toolkit package is not required merely to run a PyTorch wheel.

## Model-design staging

The repository no longer treats the identity-base, unanchored specialist construction as a safe default. Before the three-seed production run, execute the four-condition pilot:

```text
identity base + unanchored specialists
identity base + retention-anchored specialists
weak multitask base + unanchored specialists
weak multitask base + retention-anchored specialists
```

Start it with:

```bash
bash scripts/run_model_design_pilot.sh detach
```

Monitor:

```bash
cat runs/model_design_pilot/pilot.pid
ls -1t logs/model_design_pilot_*.log | head -1 | xargs tail -f
nvidia-smi
```

The pilot trains one seed for 3,000 optimizer steps per model and evaluates validation and test outputs under `evaluations/model_design_pilot/`. It is a design-selection experiment, not a final result. Compare relevant-specialist accuracy, all-five raw/mean fusion, trace validity, EOS accuracy, matched-joint divergence, and selected-versus-final checkpoint steps.

## Guarded surface-v4 production candidate

The guarded candidate uses:

```text
config:   configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml
launcher: scripts/run_bias_fusion_factory_surface_v4.sh
output:   runs/gpt_bias_fusion_factory_surface_v4/
runner:   opfusion-train-batch-design
```

Its model construction is:

- weak multitask `base.common` on operands within ±8 and at most four terms;
- five full-domain specialists branching from the validation-selected base checkpoint;
- inactive-operator retention KL against the frozen selected base;
- a small parameter anchor to the selected base;
- validation-selected specialist and joint endpoints;
- strict experiment fingerprints that reject mixed configuration/code revisions.

Production is intentionally gated. After the pilot supports this design:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

## Manual preflight

The launcher repeats these checks automatically:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
.venv/bin/python -m pytest -q
.venv/bin/opfusion-audit .
.venv/bin/opfusion-audit-data-design \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --samples-per-operator 512 \
  --out audits/surface_v4_data_audit.json
.venv/bin/opfusion-train-batch-design \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --plan-only
```

Do not bypass a nonzero result.

## Checkpoint selection

Every job still keeps `final.pt`, but dependency branches and final fusion manifests use `selected.pt`, chosen by validation token NLL from positive-step permanent checkpoints.

```text
base.common selected.pt
        ├── specialist selected.pt files
        └── joint selected.pt
```

This prevents a fixed 50,000-step endpoint from silently becoming the scientific result when an earlier checkpoint validates better. Checkpoint-grid manifests remain available for step-matched trajectory analysis.

## Experiment fingerprints

Each output root contains:

```text
experiment_contract.json
```

The fingerprint includes normalized run configuration, model-design controls, model/tokenizer files, vocabulary hash, relevant training/evaluation source hashes, and the Git revision when available. Re-running with different code or configuration in the same output directory fails before checkpoint reuse.

Do not delete the contract to force adoption of old artifacts. Move the old run aside or choose a new `output_dir`.

## Resume

Run the same command with the same checkout and configuration. The queue loads `last.pt` for incomplete jobs and returns the existing validation-selected endpoint for completed jobs.

Do not delete `experiment_contract.json`, `runtime_state.json`, `last.pt`, `complete.json`, `checkpoint_index.jsonl`, or `batch_state.json` while a run is active.

## Legacy conditions

Surface v3 is the identity-base/unanchored legacy condition:

```bash
OPFUSION_ALLOW_LEGACY_SURFACE_V3=1 \
  bash scripts/run_bias_fusion_factory_surface_v3.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
    detach
```

Typed v2 remains a diagnostic output-token ablation:

```bash
OPFUSION_ALLOW_TYPED_V2=1 \
  bash scripts/run_bias_fusion_factory_v2.sh \
    configs/experiments/gpt_bias_fusion_factory_v2.yaml \
    detach
```

Never mix their checkpoints or manifests with surface-v4 output trees.

## Kernel and driver updates

Arch is rolling release. Before restarting after a system upgrade:

```bash
uname -r
pacman -Q | grep -E '^(linux|nvidia|cuda|python|python-pytorch)'
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

If Python minor version or the repository revision changed, recreate `.venv`. A changed repository revision will also change the experiment fingerprint, so resume only from the exact run checkout.

## Storage

Surface v4 requires at least 20 GiB free by default because it retains trajectory checkpoints, optimizer state, selected endpoints, and four pilot conditions. Override only after measuring actual checkpoint size:

```bash
MIN_FREE_GB=30 OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh ...
```

## Failure diagnosis

- `torch.cuda.is_available() == false`: driver/module/PyTorch wheel mismatch.
- OOM recovery reaches micro-batch 4: inspect whether retention inference is active and reduce the declared minimum only after a smoke test.
- fingerprint mismatch: do not overwrite; use a new output directory or restore the original checkout/config.
- no selectable checkpoint: inspect `checkpoint_index.jsonl`; at least one positive-step checkpoint with finite validation NLL is required.
- non-finite restart limit reached: inspect `recovery.jsonl`, `metrics.jsonl`, and `regularization.jsonl`.
- duplicate factory error: inspect the PID and lock holder before removing a lock file.
- data audit failure: preserve the JSON report and inspect the first failing invariant.
