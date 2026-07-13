# Math Operator Units

This repository studies whether small GPT operator models can be composed in logit space. Mathematical operators are used as a controlled system because targets, intermediate transformations, and failure cases can be generated and verified exactly.

## Current executable milestone

The immediate goal is to manufacture the model set required for the next fusion experiment. The model factory trains, from the same initialization for each seed:

1. `scalar.add`
2. `aggregation.sum`
3. `scalar.neg`
4. `scalar.min`
5. `scalar.max`
6. `joint.all_five`, trained on the balanced union of all five operator datasets

The five independent units define 32 runtime subsets, including the empty/base subset. The batch runner writes all 32 subset manifests after the six checkpoints for a seed are complete. It also writes checkpoint-aligned 32-subset grids at every training step shared by all six jobs.

The factory is deliberately GPT-only. It does not train perceptron, linear, MLP, router, or calibrator models. Reliability calibration remains in the repository as a later ablation, but it is not a prerequisite for generating the operator checkpoints.

## Hard experimental constraints

- Model architecture: causal GPT decoder only
- Parameter ceiling: `1,000,000` trainable parameters
- Current model: `848,624` parameters
- Training backend: CUDA by default; production runs fail fast without CUDA
- Tokenizer: fixed experiment-specific vocabulary with only the five operator families and required trace tokens
- Reserved/blank operator slots: disabled for this experiment tokenizer
- Initialization: shared exactly across the five independent models and the joint reference model for each seed
- Checkpoints: step 0, early steps, periodic steps, final step, optimizer state, RNG state, loss metrics, and parameter-distance summaries
- Resume: each job maintains an atomic `last.pt`

## Equality-trace training

Training records contain contractive equivalence traces. Resolved intermediate subexpressions are replaced and discarded from the next state.

```text
<OP_AGG_SUM>
1 + 2 + 3 + 4
= 3 + 3 + 4
= 6 + 4
= 10
<TRACE_STOP>
```

The model-facing representation uses `<EQ_STEP>` and `<TRACE_STOP>` rather than relying on an untyped equals token. This preserves the intended equality-chain behavior while making termination explicit.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Start the CUDA batch

Foreground:

```bash
bash scripts/run_operator_factory_cuda.sh
```

Detached:

```bash
bash scripts/run_operator_factory_cuda.sh configs/experiments/gpt_operator_factory_v1.yaml detach
```

The launcher performs three checks before starting the long run:

1. CUDA availability and device reporting;
2. the complete unit-test suite;
3. a three-step CUDA smoke batch covering all five units and the joint model.

Set `SKIP_SMOKE=1` only after the smoke batch has already passed on the same environment.

Direct production CLI, without the launcher preflight:

```bash
opfusion-train-batch --config configs/experiments/gpt_operator_factory_v1.yaml
```

Train or resume one job:

```bash
opfusion-train-one \
  --config configs/experiments/gpt_operator_factory_v1.yaml \
  --job scalar.add \
  --seed 0
```

The production configuration uses three seeds and six jobs per seed. Existing complete jobs are skipped; incomplete jobs resume from `last.pt`.

## Outputs

```text
runs/gpt_operator_factory_v1/
└── seed_<n>/
    ├── shared_initial.pt
    ├── scalar_add/
    ├── aggregation_sum/
    ├── scalar_neg/
    ├── scalar_min/
    ├── scalar_max/
    ├── joint_all_five/
    ├── fusion_subsets/
    │   ├── subset_00.json
    │   ├── ...
    │   ├── subset_31.json
    │   └── index.json
    └── fusion_checkpoint_grid/
        ├── step_000000000/
        ├── ...
        └── index.json
```

Each job directory contains:

- `run_manifest.json`
- `metrics.jsonl`
- `checkpoint_index.jsonl`
- `checkpoints/step_*.pt`
- `checkpoints/final.pt`
- `last.pt`
- `complete.json`
- `vocab.json`

Analyze parameter movement across checkpoints:

```bash
opfusion-checkpoint-trajectory \
  --index runs/gpt_operator_factory_v1/seed_0/scalar_add/checkpoint_index.jsonl \
  --out runs/gpt_operator_factory_v1/seed_0/scalar_add/trajectory.csv
```

## Export and optional Hugging Face upload

Create a self-contained bundle:

```bash
opfusion-export-bundle \
  --checkpoint runs/gpt_operator_factory_v1/seed_0/scalar_add/checkpoints/final.pt \
  --model-config configs/model/gpt_operator_1m_v1.yaml \
  --tokenizer-config configs/tokenizer/operator_experiment_v1.yaml \
  --out exports/scalar-add-seed0
```

Upload is optional and requires the publish extra plus normal Hugging Face authentication:

```bash
python -m pip install -e ".[publish]"
opfusion-export-bundle ... --repo-id USER/REPOSITORY
```

## Research boundary

The model factory does not claim that raw bias fusion works. It creates the controlled checkpoints needed to test that claim.

The later comparison is between:

- the joint reference model trained on all five data families;
- runtime compositions of independently trained operator models;
- raw, oracle-routed, and learned-calibration variants.

Earlier perceptron/NN/MLP proxy observations remain exploratory evidence. They do not establish the behavior of the GPT models produced here.

## Existing research documents

- [`docs/gpt_operator_model_factory.md`](docs/gpt_operator_model_factory.md): executable model-generation contract
- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md): logit-space framing
- [`docs/parallel_sequence_bias_control.md`](docs/parallel_sequence_bias_control.md): same-prefix composition
- [`docs/equivalence_trace_training_plan.md`](docs/equivalence_trace_training_plan.md): equality trace policy
- [`docs/raw_fusion_failure_observations.md`](docs/raw_fusion_failure_observations.md): prior proxy observations
- [`docs/generation_path_reliability_calibrator.md`](docs/generation_path_reliability_calibrator.md): optional later calibration design
- [`docs/reliability_calibrator_training_plan.md`](docs/reliability_calibrator_training_plan.md): optional later calibration training

## Core design rules

1. The fixed tokenizer and vocabulary hash are part of the model ABI.
2. All models compared within a seed use the same architecture, tokenizer, and initial parameters.
3. The one-million-parameter ceiling is enforced before training.
4. Intermediate checkpoints are evidence and must not be overwritten.
5. The joint model is a reference distribution, not proof that a fusion rule exists.
6. Correctors and routers are separate experimental axes, not hidden parts of the operator model.
7. Fusion conclusions require GPT results; proxy-model results are not substituted.
