# Parallel Sequence Bias Control

This document corrects the runtime-control framing.

The goal is not keyword-based mode switching. The goal is to run multiple model or unit outputs over the same sequence context and apply bias-field operators to adjust the resulting next-token distribution.

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
replace symbolic computation with a neural calculator
```

Keyword or parser-driven switching can be useful for debugging, but it is not the research target.

The target is:

```text
parallel model or unit outputs
+ bias-field operators
+ contribution control
+ softmax and verifier effect measurement
```

## 3. Relation to the mathematical operator calculator

The mathematical operator calculator is a proxy for learning and testing operators over bias fields.

Examples:

```text
ADD:       F = A + B
SUB:       F = A - B
AGREE:     F = A_+ B_+
REMOVE:    F = A - Proj_C(A)
COMPLETE:  F = T - A
RESIDUAL:  R = T - explained(T)
```

These operations are interesting because the same forms can be applied to logit-space control fields from parallel model outputs.

## 4. Same-prefix parallelism

The central runtime pattern is:

```text
same prefix x
  -> base model logits z_0
  -> control model or unit logits z_1, z_2, ...
  -> convert to bias fields B_i
  -> apply bias operator O
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

## 5. Corrector role

A corrector should be understood as a contribution controller over a bias field.

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
corrector = contribution controller over a bias field
```

not:

```text
corrector = parser-based symbolic operator selector
```

## 6. Why direct summation is not enough

A naive assumption is:

```text
irrelevant control field -> neutral contribution
```

Early proxy experiments suggest this assumption can fail. Irrelevant units can emit peaked, structured wrong biases.

Therefore direct fusion:

```text
z_raw = z_0 + sum_i B_i
```

should be compared against contribution-controlled fusion:

```text
z_final = z_0 + sum_i c_i * B_i
```

where `*` denotes elementwise multiplication over the vocabulary.

## 7. Slot-wise control as an implementation detail

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
Can bias operators transform and compose parallel control fields in a measurable way?
```

## 8. Minimal example

Suppose two parallel units produce fields over the same prefix:

```text
B_add(v | x)
B_sub(v | x)
```

At an operator-like position, `B_add` may push `+` and `B_sub` may push `-`.

A control objective may reduce the add field and preserve the subtract field:

```text
F = c_add * B_add + c_sub * B_sub
c_add(+, x) approximately 0
c_sub(-, x) approximately 1
```

At a result-like position, the same mechanism controls result-token bias:

```text
c_add(7, x) approximately 0
c_sub(-1, x) approximately 1
```

The adjustment happens in the next-token distribution, not after a symbolic parser has already selected the operation.

## 9. Evaluation

Key measurements:

```text
softmax_effect_kl
softmax_effect_jsd
control_success_rate
raw_vs_corrected_delta
inactive_bias_norm
inactive_pmax
wrong_control_projection_rate
parallel_field_agreement
parallel_field_conflict
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

## 10. Correct short framing

```text
This project uses a mathematical operator calculator as a controlled proxy for learning operators over parallel logit or bias fields. The aim is to run multiple model or unit outputs over the same sequence context, transform those outputs with bias algebra, and inject the resulting control field back into the next-token distribution.
```

Japanese:

```text
このプロジェクトでは、数学的演算器を、並列に得られたlogit/bias fieldを操作するための制御proxyとして使う。同じ系列prefixに対して複数のmodel/unit出力を並列に出し、それらをbias algebraで変換・合成し、得られたcontrol fieldを次token分布に注入することが目的である。
```
