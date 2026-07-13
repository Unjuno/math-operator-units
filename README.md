# Math Operator Units

This repository builds controlled GPT checkpoints for testing **logit-space bias fusion**. Mathematical operators are the experimental instrument: inputs, intermediate transformations, final values, stopping behavior, and invalid traces can all be generated and verified exactly.

For one shared prefix `x`:

```text
B_k(x) = z_k(x) - z_base(x)
z_fused(x) = z_base(x) + F(B_1(x), ..., B_n(x))
```

The repository does not assume that raw addition works, and it does not hide an input router or learned fusion corrector inside the primary model factory.

## Surface policy

The model predicts ordinary equality punctuation and EOS:

```text
<OP_AGG_SUM> 1 + 2 + 3 + 4 <RESPONSE>
= 3 + 3 + 4
= 6 + 4
= 10
<EOS>
```

The surface vocabulary contains literal `+`, `,`, `[`, `]`, and `=` tokens. `<EQ_STEP>` and `<TRACE_STOP>` are implementation aliases only; they are not separate output classes.

## Do not start with the three-seed production run

The original surface-v3 construction used an identity common base and unrestricted specialist fine-tuning. That is a valid control, but a failed fusion result would be ambiguous because:

- every specialist may repeat the same correction away from the identity policy;
- inactive operators may drift without constraint;
- the fixed final training step may be worse than an earlier checkpoint.

The required first experiment is therefore a one-seed 2×2 model-design pilot:

| Base | Specialist construction |
|---|---|
| identity | unanchored |
| identity | retention-anchored |
| weak multitask | unanchored |
| weak multitask | retention-anchored |

Run it on Arch Linux with:

```bash
git clone https://github.com/Unjuno/math-operator-units.git
cd math-operator-units
bash scripts/bootstrap_arch_linux.sh
bash scripts/run_model_design_pilot.sh detach
```

Pilot outputs are written under:

```text
runs/model_design_pilot/
audits/model_design_pilot/
evaluations/model_design_pilot/
```

Detailed decision criteria are in [`docs/model_design_pilot.md`](docs/model_design_pilot.md).

## Guarded surface-v4 candidate

After the pilot supports the choice, the guarded production candidate uses:

- a weak multitask common base trained on all five operators only within operands ±8 and at most four terms;
- five full-domain specialists initialized from the validation-selected base checkpoint;
- teacher-KL retention on inactive operators;
- a small parameter anchor to the frozen base;
- an exposure-matched all-five joint reference;
- validation-selected endpoints;
- strict experiment fingerprints.

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

The environment variable is an operational acknowledgement, not evidence that the pilot succeeded.

## Model set

Each seed creates seven trained models:

```text
shared random initialization
        ↓
base.common selected.pt
        ├── scalar.add selected.pt
        ├── aggregation.sum selected.pt
        ├── scalar.neg selected.pt
        ├── scalar.min selected.pt
        ├── scalar.max selected.pt
        └── joint.all_five.exposure_matched selected.pt
```

The production candidate uses three seeds: 21 trained models total. The five specialists also define 32 runtime subset manifests; those are not 32 separately trained models.

Only the all-five subset has a matched all-five joint reference. Intermediate subsets are leakage/interference diagnostics unless a corresponding subset-joint model is trained.

## Why `selected.pt` exists

Every job keeps `final.pt`, but dependencies and final subset manifests use the positive-step permanent checkpoint with minimum validation token NLL:

- specialist selection uses its own operator validation NLL;
- joint selection uses mean validation NLL;
- base selection uses base validation NLL;
- test data are never used for selection.

`final.pt` and the checkpoint grid remain available for step-matched trajectory analysis.

## Specialist retention

For a specialist `k`, the anchored condition optimizes:

```text
L = L_task
  + lambda_KL * KL(p_base || p_k) on inactive operators
  + lambda_param * mean((theta_k - theta_base)^2)
```

The reference base is frozen. This is not a router and not a fusion-time corrector; it constrains how the specialist field is learned.

## Generated data

The prompt is visible but excluded from cross-entropy. Only response tokens are supervised.

IID problems are assigned by a stable hash of `(operator, initial values)`:

```text
0–69   train
70–84  validation
85–99  IID test
```

Training mixes:

```text
full trace          60%
continuation        25%
terminal → EOS      15%
```

SUM, MIN, and MAX use deterministic randomized valid adjacent reductions in training. Validation and test use canonical left-fold traces. Evaluation also includes operand-position OOD and length OOD conditions.

Design-aware audit:

```bash
.venv/bin/opfusion-audit-data-design \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --samples-per-operator 512 \
  --out audits/surface_v4_data_audit.json
```

The audit checks deterministic replay, split separation, arithmetic transitions, verifier acceptance, response-only labels, vocabulary/context bounds, surface `=`/EOS policy, weak-base limits, and prompt-schema compatibility.

## Experiment fingerprints

Every design-safe output root contains `experiment_contract.json`. The fingerprint covers:

- normalized run configuration;
- model-design controls;
- model/tokenizer files and vocabulary hash;
- relevant training/evaluation source hashes;
- Git revision when available.

A changed configuration or code revision cannot reuse the same output directory. Move the previous run aside or choose a new `output_dir`; do not delete the contract to force reuse.

## GPU and recovery behavior

The model is a causal GPT decoder with 863,072 parameters, context length 256, and vocabulary size 2,065. On the target GPU, micro-batch candidates are probed from:

```text
128, 64, 32, 16, 8, 4
```

Gradient accumulation preserves effective task batch 128. The all-five joint accumulates one effective batch per operator before one optimizer update.

Operational recovery includes:

- same-step CUDA OOM retry with a smaller micro-batch;
- bounded non-finite-loss restart with learning-rate reduction;
- `last.pt` resume;
- watchdog restart;
- `flock` duplicate-run protection;
- optional `systemd-inhibit` on Arch Linux.

Recovery events are recorded in `recovery.jsonl`; retention terms are recorded in `regularization.jsonl`.

## Fusion evaluation

After a seed completes:

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
  --split test \
  --examples-per-operator 64 \
  --out evaluations/seed_0_subset_31_test.json
```

The evaluator compares base, relevant specialist, raw bias sum, bias mean, and the matched all-five joint. It reports generation accuracy, final value, EOS, exact trace validity, NLL, joint divergence, and argmax agreement.

## Legacy controls

Identity-base/unanchored surface v3:

```bash
OPFUSION_ALLOW_LEGACY_SURFACE_V3=1 \
  bash scripts/run_bias_fusion_factory_surface_v3.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
    detach
```

Typed control-token v2:

```bash
OPFUSION_ALLOW_TYPED_V2=1 \
  bash scripts/run_bias_fusion_factory_v2.sh \
    configs/experiments/gpt_bias_fusion_factory_v2.yaml \
    detach
```

Do not mix checkpoints or manifests across these output trees.

## Research boundary

This is a controlled synthetic testbed, not a proof of arbitrary natural-language model fusion. It can establish whether independently trained fields compose on a shared autoregressive surface policy and can characterize inactive leakage, interference, stopping failures, distribution mismatch, and error amplification.

Further contracts:

- [`docs/model_design_pilot.md`](docs/model_design_pilot.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/fusion_evaluation_runbook.md`](docs/fusion_evaluation_runbook.md)
- [`docs/arch_linux_runbook.md`](docs/arch_linux_runbook.md)
- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md)
