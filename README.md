# Math Operator Units

This repository builds the controlled GPT checkpoint set required to test **logit-space bias fusion**. Mathematical operators are the experimental instrument, not the final application: their inputs, intermediate transformations, final answers, and failure cases can be generated and verified exactly.

For a shared prefix `x`, each specialist field is defined relative to a trained common base:

```text
B_k(x) = z_k(x) - z_base(x)
z_fused(x) = z_base(x) + F(B_1(x), ..., B_n(x))
```

The repository does not assume that raw bias addition works, and it does not hide a router or learned corrector inside model generation.

## Main condition: surface-form v3

The primary experiment uses ordinary model-facing equality punctuation and EOS:

```text
<OP_AGG_SUM> 1 + 2 + 3 + 4 <RESPONSE>
= 3 + 3 + 4
= 6 + 4
= 10
<EOS>
```

The model vocabulary contains the literal tokens `+`, `,`, `[`, `]`, and `=`. It does **not** contain `<EQ_STEP>` or `<TRACE_STOP>` as output classes. Generator-facing aliases resolve to `=` and `<EOS>` without increasing the vocabulary. This keeps the implementation typed while ensuring that fusion is evaluated on an ordinary next-token equality/EOS policy rather than on experiment-only control tokens.

The older typed-token v2 configuration remains in the repository as a diagnostic ablation and checkpoint-compatibility condition. It is not the main production run.

## Model set

For every seed, one dependency-ordered batch creates seven trained GPT models:

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

`base.common` learns the shared expression ABI, equality syntax, identity transformation, and EOS behavior, but not arithmetic answers. All five specialists and the joint reference start from exactly the same completed base checkpoint.

The production configuration uses three seeds:

- 7 trained models per seed;
- 21 trained models total;
- 3 preserved random initial checkpoints;
- 32 runtime specialist subsets per seed;
- 16 logical observation checkpoints per trained model, or 336 model/checkpoint observations in total.

The 32 subsets are manifests over the five specialists. They are not 32 separately trained models.

## Generated data contract

The prompt is visible to the model but excluded from cross-entropy. Only response tokens are supervised.

IID expressions are assigned to train, validation, or test by a stable hash of the normalized operator/input pair:

```text
bucket 0–69   → train
bucket 70–84  → validation
bucket 85–99  → IID test
```

The same normalized problem cannot appear in more than one IID split, even across different generator seeds.

Training mixes three views:

```text
full equality trace       60%
continuation from a state 25%
terminal state → EOS      15%
```

For SUM, MIN, and MAX, training uses deterministic randomized valid adjacent reductions. Validation and test use a canonical left-fold trace so checkpoints remain directly comparable. Every generated transition is checked by the exact verifier.

Evaluation includes:

- IID validation and IID test;
- operand-position OOD inputs;
- length OOD inputs;
- response exact accuracy;
- final-value accuracy;
- EOS stopping accuracy;
- exact trace-validity accuracy;
- fixed evaluation sample counts independent of the selected CUDA micro-batch.

The operand OOD split means that an input operand occurs outside the training operand range. It is not claimed to use entirely unseen numeric vocabulary tokens.

## Data preflight

Before a long run, audit the generator directly:

```bash
.venv/bin/opfusion-audit-data \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  --samples-per-operator 512 \
  --out audits/surface_v3_data_audit.json
```

The audit fails on split overlap, invalid arithmetic transitions, nondeterministic replay, prompt-label leakage, unknown tokens, context overflow, broken surface-token policy, or missing non-left reduction paths.

## Arch Linux setup

The scripts use Bash, Python virtual environments, `nvidia-smi`, `flock`, and optional `systemd-inhibit`. They do not assume Ubuntu or `apt`.

```bash
git clone https://github.com/Unjuno/math-operator-units.git
cd math-operator-units
bash scripts/bootstrap_arch_linux.sh
```

The bootstrap script creates `.venv`, installs the project, reports PyTorch/CUDA status, and prints detected VRAM. It does not modify the NVIDIA kernel driver. On Arch Linux, verify `nvidia-smi` after kernel or driver upgrades.

A specific PyTorch wheel index can be supplied when required:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  bash scripts/bootstrap_arch_linux.sh
```

Use an index compatible with the installed NVIDIA driver. Omitting `TORCH_INDEX_URL` uses the normal PyPI resolver.

## One-command production run

Detached multi-day run:

```bash
bash scripts/run_bias_fusion_factory_surface_v3.sh \
  configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  detach
```

Foreground run:

```bash
bash scripts/run_bias_fusion_factory_surface_v3.sh
```

Before production, the launcher performs:

1. CUDA, GPU, VRAM, BF16, and disk checks;
2. the complete pytest suite;
3. the generated-data audit;
4. a resolved model/checkpoint plan;
5. a short surface-v3 CUDA smoke batch;
6. launch through a restartable watchdog.

Logs are written under `logs/`. The detached PID is written to:

```text
runs/gpt_bias_fusion_factory_surface_v3/batch.pid
```

Re-running the same command is the resume procedure. Completed jobs are skipped and incomplete jobs load `last.pt`.

## 16 GB and larger GPUs

When `micro_batch_size: 0`, each job probes the configured candidates on the actual GPU:

```text
128, 64, 32, 16, 8, 4
```

The largest fitting micro-batch is selected and gradient accumulation preserves the configured effective batch size of 128.

For the exposure-matched joint reference, every optimizer step accumulates one full effective batch for each operator:

```text
ADD 128 + SUM 128 + NEG 128 + MIN 128 + MAX 128
```

This matches per-operator exposure without requiring all 640 examples to reside in VRAM simultaneously.

## Automatic scheduling and recovery

The queue is dependency ordered:

```text
base → five specialists → exposure-matched joint → next seed
```

Operational recovery is distinct from a learned bias corrector.

- CUDA OOM: restore the same-step RNG, halve the micro-batch, preserve effective batch through accumulation, and retry.
- Non-finite loss/gradient: resume the last good checkpoint with a predeclared learning-rate reduction.
- Process interruption: restart through the watchdog and resume from `last.pt`.
- Duplicate launch: `flock` prevents two factories from writing the same output tree.
- Sleep/shutdown inhibition: `systemd-inhibit` is used when available.

Every automatic change is recorded in `recovery.jsonl` and `runtime_state.json`. Recovery never changes architecture, tokenizer ABI, data ranges, model identity, or effective per-operator exposure.

## Production profile

```text
architecture: causal GPT decoder
parameters: 863,072
parameter limit: 1,000,000
context length: 256
vocabulary size: 2,065
precision: BF16 when supported, otherwise FP32
TF32: enabled for supported CUDA FP32 kernels
max optimizer steps: 50,000 per model
seeds: 0, 1, 2
```

The configured step count is a reproducible upper plan, not a guarantee of a four- or five-day wall-clock duration. Actual duration must be measured on the target machine during the smoke run.

## Checkpoints and outputs

The 16 observation locations are:

```text
0%, 0.1%, 0.3%, 1%, 3%, 5%, 10%, 20%, 30%, 40%,
50%, 60%, 70%, 80%, 90%, 100%
```

Rolling resume state is updated every 500 steps. Permanent checkpoints record model/optimizer state, RNG state, parent base identity, task and generation metrics, cumulative examples, per-operator exposure, recovery state, tokenizer hash, and parameter distances.

```text
runs/gpt_bias_fusion_factory_surface_v3/
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

Plan without training:

```bash
.venv/bin/opfusion-train-batch-surface \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  --plan-only
```

## Research boundary

The surface condition is still a controlled synthetic testbed, not a claim about arbitrary natural-language LLM fusion. Its purpose is to avoid making the result depend on bespoke trace-control output tokens while retaining exact verifiability.

Raw, mean, weighted, routed, and learned-correction fusion rules must be evaluated as separate runtime conditions. A joint model is a reference distribution, not proof that any fusion rule exists.

Detailed contracts:

- [`docs/gpt_operator_model_factory.md`](docs/gpt_operator_model_factory.md)
- [`docs/arch_linux_runbook.md`](docs/arch_linux_runbook.md)
- [`docs/equivalence_trace_training_plan.md`](docs/equivalence_trace_training_plan.md)
- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md)
