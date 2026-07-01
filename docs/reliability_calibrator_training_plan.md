# Reliability Calibrator Training Plan

This document defines how to train the paired reliability calibrator `R_k` for a fixed generator `M_k`.

## 1. Core principle

The generator should not self-certify its own output.

Training order:

```text
1. Train or instantiate generator M_k.
2. Freeze M_k.
3. Run M_k on owned, non-owned, and OOD inputs.
4. Record its generated bias fields and sequence paths.
5. Train R_k to judge whether M_k's generated path is reliable.
6. Use R_k to attenuate or remove unreliable parts before composition.
```

`R_k` is not a global applicability oracle. It is a model-specific auditor for `M_k`.

## 2. Inputs to R_k

The calibrator should not receive only the raw expression or only the final token.

Recommended input bundle:

```text
x:
  current prefix / current expression / trace state

B_k:
  bias field emitted by M_k

summary(B_k):
  top tokens
  pmax
  entropy
  logit margin
  norm
  centered norm
  rank profile

optional signals:
  verifier score
  progress score
  equality-validity score
  consensus or conflict with other fields
  length/depth/OOD indicators
```

The key is that `R_k` audits the path that `M_k` is actually trying to generate.

## 3. Outputs from R_k

There are three useful levels.

### 3.1 Scalar reliability

```text
R_k(x, B_k) -> r_k in [0, 1]
B_tilde_k = r_k B_k
```

This is the simplest form.

### 3.2 Token-wise reliability

```text
R_k(x, B_k) -> r_k(v | x) in [0, 1]
B_tilde_k(v | x) = r_k(v | x) B_k(v | x)
```

This can suppress only specific token directions.

### 3.3 Removal-field prediction

```text
R_k(x, B_k) -> E_k(v | x)
B_tilde_k(v | x) = B_k(v | x) - E_k(v | x)
```

This is the most expressive form. It allows the calibrator to remove an estimated error component rather than simply shrinking the whole field.

## 4. Data construction

For each generator `M_k`, create a calibrator dataset by running the frozen generator over several input families.

### 4.1 Owned positives

```text
inputs from M_k's own operator family
valid generated paths
fields matching the reference softmax effect
fields accepted by verifier/progress checks
```

Targets:

```text
reliability high
attenuation low
removal field near zero
```

### 4.2 Non-owned negatives

```text
inputs from other operator families
fields produced by M_k on non-owned contexts
wrong operator assimilation cases
```

Targets:

```text
reliability low
attenuation high
remove harmful components
```

### 4.3 OOD negatives

```text
length outside training range
depth outside training range
malformed expressions
ambiguous expressions
rare token combinations
mixed operators not seen by M_k
```

Targets depend on behavior:

```text
if M_k abstains or stays low-confidence:
  mild penalty

if M_k emits peaked wrong bias:
  strong attenuation / removal target
```

### 4.4 Hard negatives

Hard negatives are essential.

```text
cases where M_k is confident but wrong
cases where wrong output has high pmax
cases where wrong field has low entropy
cases where wrong field resembles a valid owned path
cases where another unit would be correct but M_k still emits a strong field
```

These are the examples that make `R_k` useful.

## 5. Label generation

Labels can be produced using exact evaluators and reference fields.

For a generated field `B_k`, compute:

```text
p_k = softmax(z_0 + B_k)
p_ref = softmax(z_0 + B_ref)
```

Possible labels:

```text
softmax_effect_distance = JSD(p_ref, p_k)
verifier_delta = V(p_k) - V(p_0)
reference_alignment = cos(Center(B_k), Center(B_ref))
wrong_top_token = top(p_k) not in accepted set
owned_path = 1 or 0
reliability = calibrated target in [0, 1]
```

A simple reliability target:

```text
r_star = exp(-alpha * JSD(p_ref, p_k)) * verifier_accept
```

For removal-field training:

```text
E_star = B_k - B_ref_projected
```

where `B_ref_projected` is a reference or allowed component in the same bias-field space.

## 6. Losses

A practical first loss:

```text
L = L_reliability + beta L_effect + gamma L_calibration
```

Where:

```text
L_reliability:
  BCE or MSE between r_k and r_star

L_effect:
  KL/JSD between softmax(z_0 + B_tilde_k) and softmax(z_0 + B_ref)

L_calibration:
  calibration loss for predicted reliability vs observed correctness
```

For token-wise or removal-field versions:

```text
L_remove = ||(B_k - E_k) - B_ref_projected||^2
```

or effect-space form:

```text
L_remove_effect = JSD(softmax(z_0 + B_k - E_k), softmax(z_0 + B_ref))
```

## 7. Phased implementation

### Phase 0: scalar reliability classifier

Train:

```text
R_k(x, summary(B_k)) -> r_k
```

Use:

```text
B_tilde_k = r_k B_k
```

This tests whether the calibrator can detect high-level failure paths.

### Phase 1: token-wise reliability mask

Train:

```text
R_k(x, B_k) -> r_k(v | x)
```

Use:

```text
B_tilde_k(v | x) = r_k(v | x) B_k(v | x)
```

This tests whether the calibrator can suppress only harmful token directions.

### Phase 2: removal-field predictor

Train:

```text
R_k(x, B_k) -> E_k(v | x)
```

Use:

```text
B_tilde_k = B_k - E_k
```

This tests whether the calibrator can learn a structured correction field.

### Phase 3: shared trunk + per-generator head

Use:

```text
R_shared(x, B_k) -> h
R_k_head(h) -> reliability / attenuation / removal
```

This shares generic reliability features while preserving generator-specific failure profiles.

## 8. Evaluation metrics

```text
path_reliability_auc
owned_vs_non_owned_auc
hard_negative_recall
false_attenuation_rate
attenuation_precision
attenuation_recall
post_calibration_softmax_jsd
raw_vs_calibrated_delta
wrong_top_token_suppression_rate
correct_top_token_preservation_rate
inactive_pmax_reduction
confidence_calibration_error
```

The key comparison is:

```text
raw composition:
  F_raw = O(B_1, ..., B_n)

calibrated composition:
  F_cal = O(B_tilde_1, ..., B_tilde_n)
```

Success means:

```text
calibrated composition reduces unreliable dominance
while preserving useful high-confidence paths
```

## 9. Minimal experiment

Use two generators:

```text
M_ADD
M_SUB
```

Train:

```text
R_ADD
R_SUB
```

Construct four regimes:

```text
ADD-owned inputs
SUB-owned inputs
mixed ADD/SUB inputs
OOD or malformed inputs
```

Compare:

```text
base only
raw M_ADD + M_SUB composition
scalar-calibrated composition
token-wise-calibrated composition
removal-field-calibrated composition
```

Primary question:

```text
Can R_ADD and R_SUB reduce confident wrong paths without erasing valid competing paths?
```

## 10. Short framing

```text
A reliability calibrator is trained after its generator is fixed. It learns the generator's failure profile by observing the fields and sequence paths that the generator actually emits on owned, non-owned, and OOD inputs. Its role is to attenuate or remove unreliable components before all calibrated fields are composed.
```

Japanese:

```text
信頼性校正器は、対応する生成モデルを固定した後に作る。生成モデルが owned / non-owned / OOD 入力で実際に出す bias field と系列経路を観測し、その失敗傾向を学習する。役割は、全ての校正済み field を合成する前に、信頼できない成分を減衰または除去することである。
```
