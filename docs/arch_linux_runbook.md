# Arch Linux Runbook

## System prerequisites

Required commands:

```text
python 3.10+
git
nvidia-smi
bash
flock
ps
setsid
```

`flock` and `setsid` are provided by `util-linux`; `ps` is provided by `procps-ng`. `systemd-inhibit` is optional but normally available on systemd-based Arch systems.

```bash
sudo pacman -S --needed python python-pip git base-devel util-linux procps-ng
```

Install the NVIDIA driver appropriate to the GPU and kernel. The project does not automate selection among `nvidia`, `nvidia-open`, DKMS variants, or custom kernels. Reboot when required, then verify:

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

The driver must support the CUDA runtime bundled by that wheel. A local CUDA toolkit package is not required merely to run a PyTorch wheel.

## Model-design staging

Before the three-seed production run, execute the four-condition pilot:

```text
identity Base + unanchored Specialists
identity Base + retention-anchored Specialists
weak multitask Base + unanchored Specialists
weak multitask Base + retention-anchored Specialists
```

Start the entire queue with one command:

```bash
bash scripts/run_model_design_pilot.sh detach
```

The detached path starts a watchdog under `nohup`, acquires a global `flock`, blocks sleep/shutdown with `systemd-inhibit` when available, retries unexpected failures, and resumes incomplete jobs from `last.pt`.

Monitor:

```bash
bash scripts/status_model_design_pilot.sh
nvidia-smi
```

Follow the current log:

```bash
latest_log="$(ls -1t logs/model_design_pilot_*.log | head -1)"
tail -f "$latest_log"
```

The pilot trains one seed for 3,000 optimizer steps per model. Each condition contains one Base, five Specialists, and one all-five Joint. Conditions execute sequentially on one GPU.

### Deterministic pair controls

The pilot configs use:

```text
deterministic_algorithms: true
allow_tf32: false
CUBLAS_WORKSPACE_CONFIG=:4096:8
flash/memory-efficient SDPA disabled
math SDPA enabled
```

The retention and unanchored member of each Base type independently recompute their Base and Joint. At completion, `audits/model_design_pilot/pair_consistency.json` requires exact selected model-state equality for those shared endpoints. Exit status 67 indicates a scientific pair mismatch and is not retried by the watchdog.

The same audit records Specialist micro-batch, learning-rate scale, OOM reductions, and non-finite restarts. A mismatch is an interpretation warning even when the declared effective batch is unchanged.

### Pilot evaluation policy

The pilot evaluates only:

```text
validation
```

The following are reserved for final evaluation after the construction has been fixed:

```text
iid_test
operand_ood
length_ood
```

Each condition receives one validation fusion report and one per-unit inactive-drift report. The evaluator records its synthetic-data seed; model-design runs use seed `701000`, while final/default evaluation uses `700000`.

### Full-domain retention

For weak-multitask Base conditions, inactive retention prompts are sampled from the full Specialist domain rather than the Base's restricted ±8/four-term domain. The inactive arithmetic response is used only as a teacher-forcing path and response mask for KL; no inactive task cross-entropy is added.

## Corrected-pilot artifact rule

Pilot artifacts created before the current experiment-contract ABI must not be resumed into the corrected design. Before starting, move existing pilot trees aside:

```bash
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
mv runs/model_design_pilot "runs/model_design_pilot.pre_reaudit_$stamp" 2>/dev/null || true
mv audits/model_design_pilot "audits/model_design_pilot.pre_reaudit_$stamp" 2>/dev/null || true
mv evaluations/model_design_pilot "evaluations/model_design_pilot.pre_reaudit_$stamp" 2>/dev/null || true
```

Do this only when no pilot process is running.

## Guarded surface-v4 production candidate

The guarded candidate uses:

```text
config:   configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml
launcher: scripts/run_bias_fusion_factory_surface_v4.sh
output:   runs/gpt_bias_fusion_factory_surface_v4/
runner:   opfusion-train-batch-design
```

Its construction is:

- weak multitask `base.common` on operands within ±8 and at most four terms;
- five full-domain Specialists branching from the validation-selected Base;
- full-domain inactive-operator retention KL against the frozen selected Base;
- a small parameter anchor to the selected Base;
- validation-selected Specialist and Joint endpoints;
- strict experiment fingerprints.

Production is intentionally gated. After the pilot supports this design:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

## Manual preflight

The launchers repeat these checks automatically:

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

Every job keeps `final.pt`, but dependency branches and final fusion manifests use `selected.pt`, chosen by validation token NLL from positive-step permanent checkpoints.

```text
base.common selected.pt
        ├── Specialist selected.pt files
        └── Joint selected.pt
```

Checkpoint-grid manifests remain available for step-matched trajectory analysis.

## Experiment fingerprints

Each output root contains `experiment_contract.json`. The fingerprint includes normalized run configuration, model-design controls, model/tokenizer files, vocabulary hash, hardened training, seeded evaluation, diagnostics, and the Git revision when available.

Do not delete the contract to force adoption of old artifacts. Move the old run aside or choose a new `output_dir`.

## Resume

Run the same command with the same checkout and configuration. The queue loads `last.pt` for incomplete jobs and returns the existing validation-selected endpoint for completed jobs.

A logout or terminal close does not stop a detached run. A reboot, kernel panic, driver reset requiring reboot, power loss, or forced shutdown does. After a reboot:

```bash
nvidia-smi
bash scripts/run_model_design_pilot.sh detach
```

Do not delete `experiment_contract.json`, `runtime_state.json`, `last.pt`, `complete.json`, `checkpoint_index.jsonl`, `batch_state.json`, or condition completion markers while a run is active. Do not run `git pull` or edit configs during a run.

## Legacy conditions

Surface v3 identity-Base/unanchored control:

```bash
OPFUSION_ALLOW_LEGACY_SURFACE_V3=1 \
  bash scripts/run_bias_fusion_factory_surface_v3.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
    detach
```

Typed v2 output-token ablation:

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

A changed Python minor version or repository revision requires recreating `.venv`; a changed revision also changes the experiment fingerprint.

## Storage

The pilot requires at least 15 GiB free by default. Surface v4 production requires at least 20 GiB by default.

```bash
MIN_FREE_GB=30 bash scripts/run_model_design_pilot.sh detach
```

## Failure diagnosis

- `torch.cuda.is_available() == false`: driver/module/PyTorch wheel mismatch.
- deterministic CUDA error: preserve the log; do not disable deterministic settings merely to make the paired pilot pass.
- exit 67: paired Base/Joint states differ; inspect `audits/model_design_pilot/pair_consistency.json`.
- OOM recovery reaches micro-batch 4: inspect retention memory use before changing the minimum.
- fingerprint mismatch: use a new output directory or restore the original checkout/config.
- no selectable checkpoint: inspect `checkpoint_index.jsonl`.
- non-finite restart limit reached: inspect `recovery.jsonl`, `metrics.jsonl`, and `regularization.jsonl`.
- duplicate-run error: inspect the PID and lock holder before removing a lock file.
- data audit failure: preserve the JSON report and inspect the first failing invariant.
- pair runtime warning: include the differing micro-batch/recovery state in interpretation.
