# Math Operator Units

This repository studies **bias fusion for language models**.

The central operation is simple: obtain logit-space bias fields from several models or model variants, combine those fields, and apply the result to one next-token distribution.

```text
base logits z_base(x)
+ fused bias F(B_1(x), ..., B_n(x))
= fused logits z_fused(x)
```

The mathematical operator models in this repository are not the final application. They are a controlled experimental system chosen because inputs, intermediate transformations, final answers, and failure cases can be generated and verified exactly.

## Research question

For a shared prefix `x`, let a base model produce `z_base(x)` and a specialist model `k` produce `z_k(x)`. Define its bias field as:

```text
B_k(x) = z_k(x) - z_base(x)
```

A fusion rule produces:

```text
z_fused(x) = z_base(x) + F(B_1(x), ..., B_n(x))
p_fused(x) = softmax(z_fused(x))
```

The main question is whether a fusion rule can preserve the useful contribution of each specialist while avoiding amplification of irrelevant or incorrect contributions.

A jointly trained model is used as a reference distribution:

```text
L_match = D(p_joint, p_fused)
```

`L_match = 0` means that the fused and joint next-token distributions are identical under the selected divergence. It does not by itself prove that either model is correct, so task accuracy and trace validity must also be measured.

## Why error amplification matters

A model that is outside its trained domain does not necessarily emit a neutral bias. It may produce a structured, confident, and wrong logit shift. Adding several such fields can amplify an incorrect token rather than cancel noise.

This repository therefore separates:

1. raw bias fusion;
2. measurement of leakage, conflict, and error amplification;
3. optional routing, weighting, or calibration methods tested only after the raw baseline is understood.

A corrector or router is not part of the definition of bias fusion and is not assumed to be necessary in advance.

## Why mathematical operators are used

Mathematical operators provide a convenient testbed because:

- data can be generated without annotation noise;
- intermediate states can be checked exactly;
- final answers have exact verifiers;
- operator-specific and mixed distributions can be constructed deliberately;
- inactive-model leakage can be measured directly;
- a joint reference model can be trained on the exact union of specialist data.

The project is not primarily a calculator, symbolic solver, mixture-of-experts router, or claim that arbitrary LLMs can already be fused safely.

## Current model factory

The current implementation creates five specialist GPT checkpoints and one joint checkpoint for each seed:

1. `scalar.add`
2. `aggregation.sum`
3. `scalar.neg`
4. `scalar.min`
5. `scalar.max`
6. `joint.all_five`

The five specialists define `2^5 = 32` runtime subsets. These subsets are manifests referencing checkpoints; they are not 32 separately trained models.

The factory is GPT-only. Earlier perceptron, NN, and MLP experiments remain proxy observations and are not treated as GPT evidence.

## Model and tokenizer contract

- architecture: causal GPT decoder
- trainable-parameter ceiling: `1,000,000`
- current profile: `848,624` parameters
- vocabulary size: `2,064`
- context length: `128`
- training backend: CUDA for production runs
- tokenizer: fixed experiment-specific vocabulary
- reserved or blank operator slots: disabled
- checkpoints: initial, early, periodic, and final states with optimizer and RNG state

The current model is small relative to a 24 GB RTX 3090. Hardware capacity is therefore not the primary constraint; throughput, data generation, experiment balance, and correctness of the loss definition must be benchmarked before a long unattended run.

## Equality-trace data

Training examples use contractive equality traces. A resolved subexpression is replaced by its value and is not copied into the next state.

```text
<OP_AGG_SUM>
1 + 2 + 3 + 4
<EQ_STEP> 3 + 3 + 4
<EQ_STEP> 6 + 4
<EQ_STEP> 10
<TRACE_STOP>
```

`<EQ_STEP>` and `<TRACE_STOP>` make the transition and termination semantics explicit.

## Current status and known gaps

The repository currently contains a working checkpoint-generation scaffold, but the present production configuration should **not** be treated as the final four-week experiment yet.

Known issues that must be corrected before the long run:

1. `shared_initial.pt` is a random shared initialization, not a trained common base model.
2. The current language-model loss includes prompt tokens; randomly generated operands create an irreducible loss floor. Prompt tokens must be masked when the target is the generated trace.
3. At the same optimizer step, `joint.all_five` receives about one fifth as many examples per operator as each specialist. A fair comparison needs explicit exposure matching or separate step-matched and exposure-matched references.
4. The factory writes 32 subset manifests but does not yet execute bias fusion or calculate `L_match`.
5. Exact answer accuracy, complete-trace validity, stopping behavior, inactive leakage, and OOD tests are not yet part of the training evaluation.
6. The bias origin should be a trained common base when the experiment is intended to isolate operator-specific changes.

These are experimental-design issues, not merely documentation details.

## Intended experiment pipeline

```text
shared random initialization
        ↓
train common BaseGPT
        ↓
branch from the same BaseGPT checkpoint
        ├── five specialist GPT models
        └── one or more joint reference GPT models
        ↓
measure every saved checkpoint
        ↓
extract specialist bias fields relative to BaseGPT
        ↓
run all 32 subsets with candidate fusion rules
        ↓
compare fused distribution, joint distribution, and exact task behavior
        ↓
only then test routing, weighting, or calibration ablations
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

## CUDA smoke test

The existing launcher checks CUDA, disk space, tests, and a short six-job smoke batch:

```bash
bash scripts/run_operator_factory_cuda.sh \
  configs/experiments/gpt_operator_factory_smoke.yaml
```

The current `gpt_operator_factory_v1.yaml` remains useful for implementation testing and checkpoint-format validation, but it should be revised before starting the unattended production run described above.

## Existing documents

- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md): mathematical definition of bias fusion
- [`docs/parallel_sequence_bias_control.md`](docs/parallel_sequence_bias_control.md): same-prefix runtime formulation
- [`docs/gpt_operator_model_factory.md`](docs/gpt_operator_model_factory.md): current factory and required corrections
- [`docs/equivalence_trace_training_plan.md`](docs/equivalence_trace_training_plan.md): equality-trace policy
- [`docs/raw_fusion_failure_observations.md`](docs/raw_fusion_failure_observations.md): historical proxy observations and testable failure hypotheses
- [`docs/generation_path_reliability_calibrator.md`](docs/generation_path_reliability_calibrator.md): optional later calibration design

## Core rules

1. Bias fusion is the research target; mathematical operators are the controlled testbed.
2. All compared models must share architecture, tokenizer ABI, and a clearly defined base checkpoint.
3. Raw fusion must be measured before introducing routing or correction.
4. Error amplification is an empirical failure mode to quantify, not a reason to assume one correction method in advance.
5. Joint and specialist models must be compared under explicit step, token, and per-operator exposure accounting.
6. Distribution matching and task correctness are separate measurements.
7. Long CUDA runs require a passed smoke test and an approved experiment configuration.