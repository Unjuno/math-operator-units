# Bias-Fusion Experiment Protocol

## 1. Research question

For one model-facing prefix `x`, define each Specialist field relative to the same trained common Base:

```text
B_k(x) = z_k(x) - z_base(x)
z_raw(x) = z_base(x) + sum_k B_k(x)
```

The primary question is whether independently trained fields can be composed during autoregressive generation without destroying task correctness, trace validity, or stopping behavior. Raw addition is not assumed to work, and the primary condition contains no hidden input router or learned fusion corrector.

## 2. Surface policy

Main and pilot conditions use the ordinary surface tokenizer/model ABI:

```text
tokenizer: configs/tokenizer/operator_experiment_surface_v3.yaml
model:     configs/model/gpt_operator_1m_surface_v3.yaml
```

The model predicts ordinary `=`, arithmetic punctuation, numeric tokens, and EOS. `<EQ_STEP>` and `<TRACE_STOP>` are implementation aliases, not separate output classes.

Typed v2 is a diagnostic output-token ablation. Surface v3 is retained as the identity-Base/unanchored control. Neither is the default production design.

## 3. Required model-design pilot

Before the three-seed run, train one seed under four conditions:

```text
identity Base × unanchored Specialists
identity Base × retention-anchored Specialists
weak multitask Base × unanchored Specialists
weak multitask Base × retention-anchored Specialists
```

Architecture, tokenizer, operator set, seed, effective task batch, optimizer family, validation protocol, and endpoint selection remain fixed.

The pilot determines whether fusion behavior is dominated by:

- repeated cancellation of an identity-Base policy;
- inactive-operator Specialist drift;
- or a more fundamental incompatibility among learned fields.

The guarded `surface_v4` candidate must not advance merely because it is implemented.

## 4. Paired numerical controls

Retention and unanchored conditions for one Base type use deterministic numerical semantics:

```text
deterministic_algorithms: true
allow_tf32: false
CUBLAS_WORKSPACE_CONFIG=:4096:8
flash and memory-efficient SDPA: disabled
math SDPA: enabled
```

Each pair independently recomputes its Base and Joint. After training, the selected `base.common` and `joint.all_five.exposure_matched` model states must hash identically within the pair. The pair audit also records Specialist micro-batch, LR-scale, OOM-reduction, and non-finite-restart differences.

Production uses the same deterministic numerical policy so the pilot does not select a construction under one kernel regime and evaluate it under another.

## 5. Minimum trained model set

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

The production candidate uses three seeds, producing 21 trained models. All Specialists and the Joint start from the exact validation-selected Base checkpoint for that seed. Tokenizer profile, vocabulary hash, architecture, experiment fingerprint, and parent parameter state must match.

## 6. Common-Base conditions

Base and Specialist inputs use the same prompt schema:

```text
<OP_*> expression <RESPONSE>
```

### Identity control

```text
expression = expression <EOS>
```

It teaches shared expression, equality, and EOS syntax while withholding arithmetic transitions.

### Weak multitask candidate

The weak Base receives verified targets from all five operators only in a restricted domain:

```text
absolute input operand <= 8
input term count <= 4
```

Specialists receive the full training domain. The pilot must establish whether the weak Base reduces shared cancellation without erasing Specialist differentiation.

## 7. Specialist conditions

### Unanchored control

```text
L = L_task
```

Behavior on inactive operators is unconstrained.

### Retention-anchored candidate

```text
L = L_task
  + lambda_KL * KL(p_base || p_specialist) on inactive operators
  + lambda_param * mean((theta_specialist - theta_base)^2)
```

The selected Base is frozen. KL is applied on response-supervised positions for the four inactive families. Inactive prompts are sampled from the full Specialist domain, including outside the weak Base's training range. Their responses define only the teacher-forcing path and KL mask; no inactive task cross-entropy is added.

Retention coefficients are global hyperparameters. Any tuning must use validation and a separate output directory.

## 8. Training-data generation

Data are generated deterministically from seed, split, optimizer step, sample index, and operator.

### IID partitioning

A stable hash of normalized `(operator, initial values)` assigns disjoint domains:

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

SUM, MIN, and MAX use deterministic randomized valid adjacent reductions in training. Validation and final evaluation use canonical left-fold traces.

### Final OOD conditions

- `operand_ood`: input operands lie outside the full Specialist training range.
- `length_ood`: reduction inputs are longer than the full Specialist training range.

`operand_ood` is not an unseen-vocabulary claim because the same numeric tokens may occur in other positions during training.

### Required invariants

The design-aware preflight checks deterministic replay, IID split separation, exact transition validity, prompt-label masking, token/context bounds, non-left valid training paths, ordinary equality/EOS behavior, weak-Base limits, verifier validity, and Base/Specialist prompt-schema compatibility.

## 9. Optimization and exposure matching

A Specialist processes one effective task batch of 128 examples per optimizer step. Retention examples are auxiliary and recorded separately.

The exposure-matched Joint processes one effective task batch for each operator before one optimizer update:

```text
ADD 128 + SUM 128 + NEG 128 + MIN 128 + MAX 128
```

This matches per-operator example exposure, not gradient magnitude or every notion of optimization effort. Micro-batch size is selected on the actual GPU; gradient accumulation preserves the declared effective batch.

## 10. Validation-selected endpoints

Every job retains `final.pt`, but scientific manifests and downstream branches use `selected.pt`:

```text
selected.pt = positive-step permanent checkpoint with minimum validation token NLL
```

Selection rules:

- Specialist: assigned-operator validation NLL;
- Joint: mean validation NLL across five operators;
- Base: Base validation NLL.

Training-time generation metrics are validation-only. `iid_test`, `operand_ood`, and `length_ood` are not evaluated against the model during training or model-design selection.

## 11. Evaluation namespace policy

The model-design pilot evaluates validation only. The canonical evaluator records its data-generation seed:

```text
pilot validation seed: 701000
final evaluation seed: 700000
```

The pilot/final seed distinction provides provenance. The stronger protection is split reservation: IID test and both OOD conditions are not inspected until construction, endpoints, and global alpha are frozen.

`iid_test` is the canonical final IID name. `test` remains a backward-compatible alias but should not be mixed into primary tables.

## 12. Experiment fingerprints

Every design-safe output root contains `experiment_contract.json`. The fingerprint includes normalized configuration, model-design controls, model/tokenizer hashes, vocabulary hash, hardened training, seeded evaluation, diagnostics, and Git revision.

A changed configuration or implementation cannot reuse the same output directory. Legacy nonempty output directories without a compatible contract are rejected.

## 13. Runtime fusion conditions

At minimum, evaluate on identical prefixes and generation settings:

1. `base`
2. `relevant_specialist`
3. `raw_sum`: `z_base + sum B_k`
4. `bias_mean`: `z_base + mean B_k`
5. `joint_reference`, only where a matched Joint exists

A global scalar may be selected on validation and then frozen across all final seeds and splits. Input-dependent routing and learned correction are later experiments.

Logit diagnostics use vocabulary-centered fields where appropriate:

```text
B_centered = B - mean_vocab(B)
```

Centering leaves softmax unchanged and removes the irrelevant vocabulary-wise additive constant from norms.

## 14. Evaluation metrics

Generation is greedy until EOS or the declared maximum token limit. It is not forced to the reference length.

Report per operator, split, seed, checkpoint, model-design condition, and fusion condition:

- response exact and token accuracy;
- final-value accuracy;
- EOS stopping and exact trace validity;
- mean generated length;
- gold-token NLL;
- argmax agreement and divergence to the matched Joint;
- parameter displacement from the selected Base;
- retention KL and parameter-anchor diagnostics;
- per-unit inactive Base-to-unit JSD, KL, argmax agreement, and centered-bias magnitude;
- pair-audit runtime warnings.

Raw autoregressive generation is primary. Verifier-assisted decoding must be reported separately.

## 15. Subset-manifest claims

Five Specialists define 32 runtime subsets; they are not 32 trained Joint models.

The available all-five Joint is matched only to the all-five subset. Empty and singleton subsets support Base/Specialist checks; intermediate subsets support leakage, interference, and stability diagnostics. Intermediate-subset equivalence to Joint training requires a corresponding `joint.S` model.

## 16. Go/no-go sequence

Run the corrected pilot:

```bash
bash scripts/bootstrap_arch_linux.sh
bash scripts/run_model_design_pilot.sh detach
```

Review validation reports, per-unit diagnostics, and pair consistency for all four conditions. The weak-Base/retention candidate advances only if it preserves Relevant Specialist performance while reducing inactive interference or improving fusion stability.

The production launcher also verifies that all corrected-pilot markers and pair consistency belong to the current Git revision. Then, and only then:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

After production endpoint selection and alpha freezing, evaluate `iid_test`, `operand_ood`, and `length_ood` with the recorded final evaluation seed.
