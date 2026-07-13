# Math Operator Units

This repository builds the controlled GPT checkpoint set required to test **logit-space bias fusion**. Mathematical operators are the experimental instrument, not the final application: their prompts, equality transitions, final values, out-of-distribution cases, and failures can be generated and verified exactly.

For a shared prefix `x`, the experiment defines each specialist field relative to a trained common base:

```text
B_k(x) = z_k(x) - z_base(x)
z_fused(x) = z_base(x) + F(B_1(x), ..., B_n(x))
```

The v2 factory is designed to create the minimum useful model set first. It does not assume that raw addition works, and it does not hide a router or learned corrector inside model generation.

## V2 model set

For every seed, one dependency-ordered batch run creates seven trained GPT models:

```text
shared random initialization
        ↓
base.common
        ├── scalar.add
        ├── aggregation.sum
        ├── scalar.neg
        ├── scalar.min
        ├── scalar.max
        └── joint.all_five.exposure_matched
```

`base.common` learns the shared number/expression ABI and a deterministic copy/response protocol, but not arithmetic answers. All five specialists and the joint reference start from exactly the same completed base checkpoint.

The production configuration uses three seeds:

- 7 trained models per seed;
- 21 trained models total;
- 3 preserved random initial checkpoints;
- 32 runtime specialist subsets per seed;
- 16 logical observation checkpoints per trained model, or 336 model/checkpoint observations in total.

The 32 subsets are manifests over the five trained specialists. They are not 32 separately trained models.

## Equality-transition training

Specialists produce verified contractive equality traces:

```text
<OP_AGG_SUM> 1 + 2 + 3 + 4 <RESPONSE>
<EQ_STEP> 3 + 3 + 4
<EQ_STEP> 6 + 4
<EQ_STEP> 10
<TRACE_STOP>
```

The prompt is visible to the model but excluded from cross-entropy. Only response tokens are supervised. This prevents randomly sampled operands from creating an irreducible prompt-prediction loss floor.

The data generator is deterministic by seed, split, step, sample index, and operator. It provides:

- train and IID validation ranges;
- value OOD operands;
- length OOD expressions;
- exact canonical values;
- deterministic equality traces;
- explicit trace stopping.

The default operators are ADD, variable-length SUM, NEG, MIN reduction, and MAX reduction.

## Arch Linux setup

The scripts use Bash, Python virtual environments, `nvidia-smi`, `flock`, and optional `systemd-inhibit`. They do not assume Ubuntu or `apt`.

```bash
git clone https://github.com/Unjuno/math-operator-units.git
cd math-operator-units
bash scripts/bootstrap_arch_linux.sh
```

The bootstrap script creates `.venv`, installs the project, reports PyTorch/CUDA status, and prints detected VRAM. It does not silently change the NVIDIA kernel driver. On Arch Linux, verify `nvidia-smi` after kernel or driver upgrades.

A specific PyTorch wheel index can be supplied when required:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  bash scripts/bootstrap_arch_linux.sh
```

Use an index compatible with the locally installed NVIDIA driver. Omitting `TORCH_INDEX_URL` uses the normal PyPI resolver.

## One-command production run

Detached multi-day run:

```bash
bash scripts/run_bias_fusion_factory_v2.sh \
  configs/experiments/gpt_bias_fusion_factory_v2.yaml \
  detach
```

Foreground run:

```bash
bash scripts/run_bias_fusion_factory_v2.sh
```

Before production, the launcher performs:

1. CUDA, GPU, VRAM, BF16, and disk checks;
2. the complete pytest suite;
3. a resolved model/checkpoint plan;
4. a short v2 CUDA smoke batch;
5. launch through a restartable watchdog.

Logs are written under `logs/`. The detached PID is written to:

```text
runs/gpt_bias_fusion_factory_v2/batch.pid
```

Re-running the same command is the resume procedure. Completed jobs are skipped and incomplete jobs load `last.pt`.

## 16 GB and larger GPUs

The factory does not require a fixed 24 GB assumption. When `micro_batch_size: 0`, each job probes the configured candidates on the actual GPU:

```text
128, 64, 32, 16, 8, 4
```

It selects the largest fitting micro-batch and uses gradient accumulation to preserve the configured effective batch size of 128.

For the exposure-matched joint reference, every optimizer step accumulates one full effective batch for each of the five operators:

```text
ADD 128 + SUM 128 + NEG 128 + MIN 128 + MAX 128
```

This matches per-operator exposure without requiring all 640 examples to reside in VRAM simultaneously.

## Automatic scheduling and recovery

The batch queue is dependency ordered:

```text
base → five specialists → exposure-matched joint → next seed
```

Operational recovery is distinct from a learned bias corrector.

- CUDA OOM: restore the same step RNG, halve the micro-batch, preserve effective batch through accumulation, and retry the same step.
- Non-finite loss/gradient: resume the last good checkpoint with a predeclared learning-rate reduction, up to the configured limit.
- Process interruption: the watchdog restarts the batch and each job resumes from `last.pt`.
- Duplicate launch: `flock` prevents two factories from writing the same output tree.
- Sleep/shutdown inhibition: `systemd-inhibit` is used when available.

Every automatic change is recorded in `recovery.jsonl` and `runtime_state.json`. Recovery never changes architecture, tokenizer, data ranges, model identity, or effective per-operator exposure.

## Production configuration

The committed production profile uses:

```text
architecture: causal GPT decoder
parameters: 863,184
parameter limit: 1,000,000
context length: 256
vocabulary size: 2,066
precision: BF16 when supported, otherwise FP32
TF32: enabled for supported CUDA FP32 kernels
max optimizer steps: 50,000 per model
seeds: 0, 1, 2
```

The configured step count is a reproducible upper plan, not a guarantee of a four- or five-day duration. Actual wall time depends on the GPU, driver, PyTorch build, selected micro-batch, CPU data generation, storage, and evaluation overhead. The smoke run must pass on the target machine before leaving the process unattended.

## Checkpoints and outputs

The 16 observation locations are defined as fractions of total training:

```text
0%, 0.1%, 0.3%, 1%, 3%, 5%, 10%, 20%, 30%, 40%,
50%, 60%, 70%, 80%, 90%, 100%
```

Rolling resume state is updated every 500 steps. Each permanent checkpoint records:

- model and optimizer state;
- CPU and CUDA RNG state;
- parent base and random-initial checkpoint identities;
- task loss and generation metrics;
- cumulative examples and per-operator exposure;
- micro-batch and learning-rate recovery state;
- tokenizer profile and vocabulary hash;
- parameter distance from the parent base and random initialization.

Output layout:

```text
runs/gpt_bias_fusion_factory_v2/
├── experiment_plan.json
├── batch_state.json
└── seed_<n>/
    ├── shared_initial.pt
    ├── base_common/
    ├── scalar_add/
    ├── aggregation_sum/
    ├── scalar_neg/
    ├── scalar_min/
    ├── scalar_max/
    ├── joint_all_five_exposure_matched/
    ├── model_inventory.json
    ├── fusion_subsets/
    └── fusion_checkpoint_grid/
```

## Plan without training

```bash
.venv/bin/opfusion-train-batch \
  --config configs/experiments/gpt_bias_fusion_factory_v2.yaml \
  --plan-only
```

## Research boundary

This factory produces the checkpoint set and aligned manifests needed for the bias-fusion experiment. Raw, mean, weighted, routed, or learned-correction fusion rules must be evaluated as separate runtime conditions. A joint model is a reference distribution, not proof that any fusion rule exists.

Detailed operational contracts:

- [`docs/gpt_operator_model_factory.md`](docs/gpt_operator_model_factory.md)
- [`docs/arch_linux_runbook.md`](docs/arch_linux_runbook.md)
- [`docs/equivalence_trace_training_plan.md`](docs/equivalence_trace_training_plan.md)
- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md)
