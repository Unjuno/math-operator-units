# Bias-Fusion Experiment Protocol

## 1. Research question

For one model-facing prefix `x`, define each specialist field relative to the same trained common base:

```text
B_k(x) = z_k(x) - z_base(x)
z_raw(x) = z_base(x) + sum_k B_k(x)
```

The primary question is whether those independently trained fields can be composed during autoregressive generation without destroying task correctness, trace validity, or stopping behavior.

This is an existence and failure-analysis experiment. It does not assume that raw addition is correct, and it does not include a hidden router or learned corrector in the primary condition.

## 2. Canonical condition

The only production condition is `surface_v3`:

```text
config:    configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml
launcher:  scripts/run_bias_fusion_factory_surface_v3.sh
tokenizer: configs/tokenizer/operator_experiment_surface_v3.yaml
model:     configs/model/gpt_operator_1m_surface_v3.yaml
```

The model predicts ordinary `=`, arithmetic punctuation, numeric tokens, and EOS. `<EQ_STEP>` and `<TRACE_STOP>` are implementation aliases only and are not separate output classes in the surface vocabulary.

Typed v2 is a diagnostic ablation. It must not be substituted for the canonical condition.

## 3. Minimum trained model set

For each seed:

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

The production configuration uses three seeds, producing 21 trained models.

All specialists and the joint reference must start from the exact same completed `base.common` checkpoint for that seed. Tokenizer profile, vocabulary hash, architecture, and initial parameter state must match.

## 4. Common-base contract

The common base and every specialist use the same model-facing prefix:

```text
<OP_*> expression <RESPONSE>
```

The common base receives a neutral identity target:

```text
expression = expression <EOS>
```

A specialist receives the verified operator transition target. The base therefore learns the shared expression/equality/EOS protocol without receiving the arithmetic answer, while `B_k(x)` is still defined on the same prefix schema used to train the specialist.

The repository audit fails if base and specialist prompts diverge.

## 5. Training-data generation

Data are generated deterministically from seed, split, optimizer step, sample index, and operator.

### IID partitioning

A stable hash of the normalized `(operator, initial values)` key assigns examples to disjoint domains:

```text
0–69   train
70–84  validation
85–99  IID test
```

Changing generator seed does not move the same normalized problem into a different IID split.

### Training views

```text
full trace          60%
continuation        25%
terminal → EOS      15%
```

SUM, MIN, and MAX use deterministic randomized valid adjacent reductions in training. Validation and test use canonical left-fold traces for checkpoint comparability.

### OOD conditions

- `operand_ood`: input operands are outside the training operand range.
- `length_ood`: reduction inputs are longer than the training range.

`operand_ood` is not an unseen-vocabulary claim because numeric tokens may occur elsewhere in training.

### Required invariants

The preflight audit checks:

- deterministic replay;
- disjoint IID splits;
- exact arithmetic validity of every transition;
- no prompt-label leakage;
- no unknown tokens;
- no context overflow;
- non-left valid training paths when randomization is enabled;
- correct surface equality/EOS policy;
- valid common-base identity examples.

## 6. Optimization and exposure matching

Specialists process one effective batch of 128 examples per optimizer step.

The exposure-matched joint processes one effective batch for each of the five operators before one optimizer update:

```text
ADD 128 + SUM 128 + NEG 128 + MIN 128 + MAX 128
```

This makes per-operator example exposure comparable to a specialist trained for the same number of optimizer steps. A step-matched but non-exposure-matched joint may be added later only as a separate ablation.

Micro-batch size is selected on the actual GPU. Gradient accumulation preserves the declared effective batch and per-operator exposure.

## 7. Runtime fusion conditions

At minimum, evaluate these conditions on identical prefixes and generation settings:

1. `base`
2. `relevant_specialist`
3. `raw_sum`: `z_base + sum B_k`
4. `bias_mean`: `z_base + mean B_k`
5. `joint_reference`, only when a matched joint exists

Global scalar weights may be selected on validation data and reported as a secondary condition. Input-dependent routing and learned correction are later experiments, not part of the raw baseline.

Logit-space diagnostics should also use vocabulary-centered fields:

```text
B_centered = B - mean_vocab(B)
```

Centering does not change the softmax distribution and removes the irrelevant vocabulary-wise additive constant from norms and cosine measurements.

## 8. Evaluation

Generation is greedy until EOS or the declared maximum new-token limit. It is not forced to the reference response length.

Report per operator, split, seed, checkpoint, and fusion condition:

- response exact accuracy;
- token accuracy;
- final-value accuracy;
- EOS stopping accuracy;
- exact trace-validity accuracy;
- mean generated length;
- gold-token negative log-likelihood;
- next-token agreement with the matched joint;
- KL or Jensen-Shannon divergence to the matched joint, where available.

Raw autoregressive generation is primary. Verifier-assisted decoding must be reported separately.

## 9. What the 32 subset manifests mean

Five specialists produce 32 possible runtime subsets. They are not 32 trained joint models.

The available `joint.all_five.exposure_matched` checkpoint is a matched reference only for the all-five subset. It is not a valid joint reference for an arbitrary two-, three-, or four-specialist subset.

Therefore:

- empty subset may be checked against the base;
- singleton subsets may be checked for specialist reconstruction;
- all-five fusion may be compared to the all-five joint;
- intermediate subsets may be used for leakage, interference, and stability diagnostics;
- intermediate-subset equivalence to joint training cannot be claimed without a corresponding `joint.S` model.

A stronger follow-up should train selected diagnostic joint references such as ADD+SUM, ADD+NEG, and MIN+MAX rather than treating the all-five joint as universal.

## 10. Go/no-go sequence

Before a multi-day run:

```bash
bash scripts/bootstrap_arch_linux.sh
.venv/bin/opfusion-audit .
.venv/bin/opfusion-audit-data \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  --samples-per-operator 512 \
  --out audits/surface_v3_data_audit.json
bash scripts/run_bias_fusion_factory_surface_v3.sh \
  configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  detach
```

The launcher itself repeats tests, repository audit, data audit, static planning, and CUDA smoke training. Do not bypass a failed gate.

## 11. Scope of conclusions

This experiment can establish whether independently trained operator fields compose on a shared autoregressive surface policy and can characterize interference, leakage, stopping failures, and error amplification.

It cannot by itself establish arbitrary natural-language model fusion. Operator tags, atomic integer tokens, synthetic grammar, and the limited model scale remain controlled simplifications. Those factors should be relaxed in staged follow-up experiments only after the surface condition is understood.
