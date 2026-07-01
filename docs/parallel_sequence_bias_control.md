# Parallel Sequence Bias Control

This document corrects the runtime-control framing.

The goal is not keyword-based mode switching. The goal is to run multiple model or unit outputs over the same sequence context, fully compose their bias fields, and let the composed field change the next-token distribution.

## 1. Core idea

Given the same sequence prefix `x`, multiple models, units, or control heads can be run in parallel:

```text
z_0(v | x)      base logits
z_1(v | x)      model or unit 1 logits
z_2(v | x)      model or unit 2 logits
...
z_n(v | x)      model or unit n logits
```

Convert these outputs into bias fields:

```text
B_i(v | x) = z_i(v | x) - z_0(v | x)
```

or use already centered fields:

```text
B_i(v | x) in R^{|V|}
```

The operator calculator acts on these fields:

```text
F(v | x) = O(B_1, B_2, ..., B_n)(v | x)
```

The resulting control field is injected into the same sequence distribution:

```text
z_final(v | x) = z_0(v | x) + lambda F(v | x)
p_final(v | x) = softmax(z_final(v | x))
```

This is the runtime target: bias control over the same sequence context.

## 2. What this is not

This is not:

```text
keyword appears -> switch mode
parser reads an operator -> choose a symbolic unit
route the input to one expert
hard-select one bias and discard the others by default
replace symbolic computation with a neural calculator
```

Keyword or parser-driven switching can be useful for debugging, but it is not the research target.

The target is:

```text
parallel model or unit outputs
+ full bias-field composition
+ confidence / alignment / contribution calibration
+ softmax and verifier effect measurement
```

## 3. Relation to the mathematical operator calculator

The mathematical operator calculator is a proxy for learning and testing operators over bias fields.

Examples:

```text
ADD:              F = A + B
MEAN:             F = (A + B) / 2
SUB:              F = A - B
AGREE:            F = A_+ B_+
WEIGHTED_SUM:     F = w_A A + w_B B
ANGLE_SELECT:     F = weighted field by confidence/alignment
CONFLICT:         detect disagreement between A and B
REMOVE:           F = A - Proj_C(A)
COMPLETE:         F = T - A
RESIDUAL:         R = T - explained(T)
```

These operations are interesting because the same forms can be applied to logit-space control fields from parallel model outputs.

## 4. Same-prefix parallelism

The central runtime pattern is:

```text
same prefix x
  -> base model logits z_0
  -> control model or unit logits z_1, z_2, ...
  -> convert to bias fields B_i
  -> compose the fields with bias operator O
  -> inject F into z_0
  -> decode next token
```

This differs from routing:

```text
routing:
  choose one or a few experts

parallel bias control:
  keep the same sequence context and compose multiple control fields
```

The desired behavior is not that a symbolic controller chooses `ADD` or `SUB` first. The desired behavior is that both fields can exist in the same vocabulary space, are composed, and the final softmax makes tokens supported by the stronger or more coherent composed field more likely.

## 5. Confidence and angle-based composition

A central hypothesis is:

```text
When multiple bias fields are composed, the token direction with higher confidence, margin, or alignment becomes more likely after softmax.
```

For example, if `B_add` pushes `+` and `B_sub` pushes `-`, then the composed field determines which token becomes more likely:

```text
F = O(B_add, B_sub)
p_final = softmax(z_0 + lambda F)
```

The selected token should emerge from the composed distribution, not from a prior parser decision.

Possible confidence or alignment signals:

```text
pmax of the field-induced distribution
logit margin between top candidates
entropy reduction
cosine alignment with a reference or consensus field
agreement with verifier or progress field
stability across perturbations
```

A normalized field form is useful:

```text
B_hat_i = Center(B_i) / (||Center(B_i)|| + eps)
```

A confidence-weighted composition can be written as:

```text
F = sum_i w_i B_hat_i
w_i = softmax(tau q_i)
```

where `q_i` is a confidence, calibration, alignment, or verifier-supported quality score.

## 6. Corrector role

A corrector should not be understood as a parser-based symbolic operator selector.

The corrector is better understood as a contribution calibrator over a bias field.

General form:

```text
F(v | x) = sum_i c_i(v | x) B_i(v | x)
```

Lower-rank scalar form:

```text
F(v | x) = sum_i g_i(x) B_i(v | x)
```

The token-wise form is more general:

```text
c_i(v | x) in [0, 1]
```

Interpretation:

```text
corrector = contribution calibrator over a bias field
```

not:

```text
corrector = parser-based symbolic operator selector
```

Important distinction:

```text
composition first:
  compare and combine the parallel fields in the same space

calibration second:
  prevent raw confidence, OOD peakedness, or irrelevant fields from dominating incorrectly
```

## 7. Why direct summation is not enough

A naive assumption is:

```text
irrelevant control field -> neutral contribution
```

Early proxy experiments suggest this assumption can fail. Irrelevant units can emit peaked, structured wrong biases.

Therefore direct fusion:

```text
z_raw = z_0 + sum_i B_i
```

must be compared against calibrated composition:

```text
z_final = z_0 + O_calibrated(B_1, B_2, ..., B_n)
```

The goal is not to remove competition between fields. The goal is to make competition meaningful by calibrating field scale, confidence, angle, and reliability.

## 8. Slot-wise control as an implementation detail

Slot-wise control can be useful in mathematical generation:

```text
operator slot
operand slot
result slot
equality-boundary slot
stop slot
```

However, slot-wise control is an implementation of bias-field control, not the main definition.

The central object remains the same-prefix bias field:

```text
B_i(v | x)
```

and the central question remains:

```text
Can bias operators transform and compose parallel control fields so that the composed softmax prefers the intended high-confidence or high-alignment direction?
```

## 9. Minimal example

Suppose two parallel units produce fields over the same prefix:

```text
B_add(v | x)
B_sub(v | x)
```

At an operator-like position:

```text
B_add may push `+`
B_sub may push `-`
```

The intended experiment is not simply:

```text
turn ADD off and keep SUB
```

The intended experiment is:

```text
compose B_add and B_sub
calibrate their scale or confidence if needed
apply softmax
observe whether `+` or `-` becomes more likely
```

Example calibrated composition:

```text
F = w_add B_add + w_sub B_sub
p_final = softmax(z_0 + lambda F)
```

If the subtract field has higher calibrated confidence or stronger alignment with the desired control objective, then:

```text
w_sub B_sub(- | x) > w_add B_add(+ | x)
```

and `-` becomes more likely after softmax.

At a result-like position, the same mechanism applies:

```text
B_add may push `7`
B_sub may push `-1`
```

The composed and calibrated field determines whether `7` or `-1` becomes more likely.

## 10. Evaluation

Key measurements:

```text
softmax_effect_kl
softmax_effect_jsd
control_success_rate
raw_vs_calibrated_delta
inactive_bias_norm
inactive_pmax
wrong_control_projection_rate
parallel_field_agreement
parallel_field_conflict
confidence_calibration_error
angle_alignment_score
verifier_score_shift
```

For a reference field `F_ref` and learned field `F_model`:

```text
p_ref   = softmax(z_0 + F_ref)
p_model = softmax(z_0 + F_model)
```

Measure:

```text
KL(p_ref || p_model)
JSD(p_ref, p_model)
Delta verifier effect
```

## 11. Correct short framing

```text
This project uses a mathematical operator calculator as a controlled proxy for learning operators over parallel logit or bias fields. The aim is to run multiple model or unit outputs over the same sequence context, compose those outputs with bias algebra, calibrate their confidence and alignment when needed, and inject the resulting control field back into the next-token distribution.
```

Japanese:

```text
このプロジェクトでは、数学的演算器を、並列に得られたlogit/bias fieldを操作するための制御proxyとして使う。同じ系列prefixに対して複数のmodel/unit出力を並列に出し、それらをbias algebraで完全に合成し、必要なら確度・角度・信頼性を校正し、得られたcontrol fieldを次token分布に注入することが目的である。
```
