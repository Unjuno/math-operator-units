# Bias-Fusion Experiment Protocol

## 1. Research question

For one model-facing prefix `x`, define each specialist field relative to the same trained common base:

```text
B_k(x) = z_k(x) - z_base(x)
z_raw(x) = z_base(x) + sum_k B_k(x)
```

The primary question is whether independently trained fields can be composed during autoregressive generation without destroying task correctness, trace validity, or stopping behavior. This is an existence and failure-analysis experiment. Raw addition is not assumed to work, and the primary condition contains no hidden input router or learned fusion corrector.

## 2. Surface policy

All main and pilot conditions use the ordinary surface tokenizer/model ABI:

```text
tokenizer: configs/tokenizer/operator_experiment_surface_v3.yaml
model:     configs/model/gpt_operator_1m_surface_v3.yaml
```

The model predicts ordinary `=`, arithmetic punctuation, numeric tokens, and EOS. `<EQ_STEP>` and `<TRACE_STOP>` are implementation aliases, not separate output classes.

Typed v2 is a diagnostic output-token ablation. Surface v3 is retained as the identity-base/unanchored control. Neither is the default production design.

## 3. Required model-design pilot

Before the three-seed run, train one seed under four conditions:

```text
identity base × unanchored specialists
identity base × retention-anchored specialists
weak multitask base × unanchored specialists
weak multitask base × retention-anchored specialists
```

The four configurations differ only in the base target mode and specialist regularization. Architecture, tokenizer, operator set, seed, task batch, optimizer family, evaluation protocol, and endpoint selection remain fixed.

The pilot determines whether fusion behavior is dominated by:

- repeated cancellation of an identity-base policy;
- inactive-operator specialist drift;
- or a more fundamental incompatibility among the learned fields.

The guarded production candidate is `surface_v4`, but it must not advance merely because it is implemented.

## 4. Minimum trained model set

For each condition and seed:

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

The production candidate uses three seeds, producing 21 trained models. All specialists and the joint reference start from the exact validation-selected `base.common` checkpoint for that seed. Tokenizer profile, vocabulary hash, architecture, experiment fingerprint, and parent parameter state must match.

## 5. Common-base conditions

Base and specialist inputs use the same prompt schema:

```text
<OP_*> expression <RESPONSE>
```

### Identity control

The identity base learns:

```text
expression = expression <EOS>
```

It teaches shared expression/equality/EOS syntax while withholding arithmetic transitions. It is retained because it maximizes the task-specific change required from each specialist.

### Weak multitask candidate

The weak base receives verified targets from all five operators only in a restricted domain:

```text
absolute input operand <= 8
input term count <= 4
```

Specialists receive the full training domain. The goal is to place common reduction and stopping behavior in the base while leaving domain extension and specialization in each field.

The weak base is not assumed superior. The pilot must establish whether it reduces shared cancellation and improves fusion without erasing specialist differentiation.

## 6. Specialist conditions

### Unanchored control

```text
L = L_task
```

All parameters are fine-tuned on the assigned operator. Behavior on inactive operators is unconstrained.

### Retention-anchored candidate

```text
L = L_task
  + lambda_KL * KL(p_base || p_specialist) on inactive operators
  + lambda_param * mean((theta_specialist - theta_base)^2)
```

The selected base is frozen. KL is applied on response-supervised positions for the four inactive operator families. This is specialist-training regularization, not routing or fusion-time correction.

Retention coefficients are global experimental hyperparameters. Any tuning must use validation data and a separate output directory; test metrics must not select them.

## 7. Training-data generation

Data are generated deterministically from seed, split, optimizer step, sample index, and operator.

### IID partitioning

A stable hash of normalized `(operator, initial values)` assigns examples to disjoint domains:

```text
0–69   train
70–84  validation
85–99  IID test
```

Changing generator seed does not move the same normalized problem into another IID split.

### Training views

```text
full trace          60%
continuation        25%
terminal → EOS      15%
```

SUM, MIN, and MAX use deterministic randomized valid adjacent reductions in training. Validation and test use canonical left-fold traces.

### OOD conditions

- `operand_ood`: input operands lie outside the full specialist training operand range.
- `length_ood`: reduction inputs are longer than the full specialist training range.

`operand_ood` is not an unseen-vocabulary claim because the same numeric tokens may occur in other positions during training.

### Required invariants

The design-aware preflight checks:

- deterministic replay;
- disjoint IID splits;
- exact arithmetic validity of each transition;
- no prompt-label leakage;
- no unknown tokens;
- no context overflow;
- non-left valid training paths when randomization is enabled;
- ordinary equality/EOS surface behavior;
- weak-base operand and length limits;
- verifier-valid base and specialist examples;
- compatible base/specialist prompt schemas.

## 8. Optimization and exposure matching

A specialist processes one effective task batch of 128 examples per optimizer step. Retention examples are auxiliary and recorded separately.

The exposure-matched joint processes one effective task batch for each operator before one optimizer update:

```text
ADD 128 + SUM 128 + NEG 128 + MIN 128 + MAX 128
```

This matches per-operator example exposure, not gradient magnitude or every notion of optimization effort. A step-matched mixed-batch joint or gradient-scale control must be reported as a separate ablation if added.

Micro-batch size is selected on the actual GPU. Gradient accumulation preserves the declared effective task batch and joint per-operator exposure.

## 9. Validation-selected endpoints

Every job retains `final.pt`, but scientific endpoint manifests and downstream branches use `selected.pt`.

```text
selected.pt = positive-step permanent checkpoint with minimum validation token NLL
```

Selection rules:

- specialist: its assigned operator validation NLL;
- joint: mean validation NLL across the five operators;
- base: base validation NLL.

Test data are never used for selection. `final.pt` and checkpoint-grid manifests remain available for step-matched trajectory analysis.

## 10. Experiment fingerprints

Every design-safe output root contains `experiment_contract.json`. The fingerprint includes:

- normalized run configuration;
- model-design controls;
- model/tokenizer configuration hashes;
- vocabulary hash;
- relevant training/evaluation source hashes;
- Git revision when available.

A changed configuration or implementation cannot reuse the same output directory. Legacy nonempty output directories without a contract are rejected. Do not delete the contract to force checkpoint adoption.

## 11. Runtime fusion conditions

At minimum, evaluate on identical prefixes and generation settings:

1. `base`
2. `relevant_specialist`
3. `raw_sum`: `z_base + sum B_k`
4. `bias_mean`: `z_base + mean B_k`
5. `joint_reference`, only where a matched joint exists

Global scalar weights may be selected on validation data and frozen for test as a secondary condition. Input-dependent routing and learned correction are later experiments.

Logit diagnostics should also use vocabulary-centered fields:

```text
B_centered = B - mean_vocab(B)
```

Centering leaves softmax unchanged and removes the irrelevant vocabulary-wise additive constant from norms and cosine measurements.

## 12. Evaluation

Generation is greedy until EOS or the declared maximum token limit. It is not forced to the reference length.

Report per operator, split, seed, checkpoint, model-design condition, and fusion condition:

- response exact accuracy;
- token accuracy;
- final-value accuracy;
- EOS stopping accuracy;
- exact trace-validity accuracy;
- mean generated length;
- gold-token negative log-likelihood;
- next-token agreement with the matched joint;
- Jensen-Shannon or KL divergence to the matched joint;
- parameter displacement from the selected base;
- retention KL and parameter-anchor diagnostics where active.

Raw autoregressive generation is primary. Verifier-assisted decoding must be reported separately.

## 13. Subset-manifest claims

Five specialists define 32 runtime subsets; they are not 32 trained joint models.

The available `joint.all_five.exposure_matched` checkpoint is matched only to the all-five subset. Therefore:

- empty subset may be checked against the base;
- singleton subsets may check specialist reconstruction;
- all-five fusion may be compared to the all-five joint;
- intermediate subsets may diagnose leakage, interference, and stability;
- intermediate-subset equivalence to joint training requires a corresponding `joint.S` model.

## 14. Go/no-go sequence

First execute the model-design pilot:

```bash
bash scripts/bootstrap_arch_linux.sh
bash scripts/run_model_design_pilot.sh detach
```

Review validation and test reports for all four conditions. The weak-base/retention candidate advances only if it preserves specialist performance while reducing inactive interference or improving fusion stability.

Then, and only then, acknowledge and start the guarded production candidate:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

The production launcher repeats tests, repository audit, design-aware data audit, static planning, and CUDA smoke training. Do not bypass a failed gate.

## 15. Scope of conclusions

This experiment can establish whether independently trained operator fields compose on a shared autoregressive surface policy and characterize inactive leakage, interference, stopping failures, distribution mismatch, and error amplification.

It cannot by itself establish arbitrary natural-language model fusion. Operator tags, atomic integer tokens, synthetic grammar, and limited model scale remain controlled simplifications. Those factors should be relaxed only after the surface and model-construction conditions are understood.
