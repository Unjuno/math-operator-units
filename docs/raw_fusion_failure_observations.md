# Raw Fusion Failure Observations

This note records qualitative observations from early 0.1K-scale proxy experiments.

The purpose is not to claim a final result. The purpose is to document the failure mode that motivates corrector-gated fusion.

## 1. Initial expectation

A naive expectation was:

```text
inactive operator model on irrelevant input -> neutral output
```

If this were true, raw fusion might work by simple averaging or addition:

```text
z_raw = z_0 + b_ADD(x) + b_SUB(x) + b_SUM(x) + ...
```

The expected behavior was that only the relevant unit would produce a strong contribution, while irrelevant units would be flat, neutral, or close to zero.

## 2. Observed behavior

The observed behavior was different.

```text
inactive operator model on irrelevant input -> peaked wrong output
```

Irrelevant units did not reliably produce neutral distributions. They often emitted sharp, misleading biases on inputs outside their training domain.

This means raw fusion can accumulate structured wrong biases rather than canceling independent zero-mean noise.

## 3. Operator assimilation error

One observed error pattern was operator assimilation.

Example:

```text
model:
  trained only on addition

test input:
  expression containing a subtraction-like operator

expected possible behavior:
  neutral / uncertain / flat distribution

observed qualitative behavior:
  the unknown subtraction-like operator was treated as if it were addition
```

This is not just a wrong answer. It is a specific failure mode:

```text
unknown operator -> projected into known operator semantics
```

Name:

```text
operator assimilation error
```

In Japanese:

```text
未知演算子の既知演算子への同化エラー
```

## 4. Why this matters for fusion

Raw fusion assumes that irrelevant units are harmless.

The observation contradicts that assumption.

If an ADD-only unit sees a subtraction-like expression and interprets it through ADD semantics, then its bias contribution is not neutral:

```text
b_ADD(x_unknown_subtraction) != 0
```

Therefore:

```text
z_raw = z_0 + Σ_k b_k(x)
```

is unsafe when inactive units emit peaked wrong biases.

## 5. Corrector-gated fusion

The corrected fusion form is:

```text
z_final = z_0 + Σ_{k in S_runtime} g_k(x) b_k(x)
```

The corrector or gate should learn:

```text
if operator k is applicable:
  g_k(x) ≈ 1

if operator k is inactive, unknown, or out-of-domain:
  g_k(x) ≈ 0
```

The corrector does not merely improve accuracy. It is required to suppress structured bias leakage from irrelevant units.

## 6. Length out-of-distribution breakpoint

Another observed pattern was length-related failure.

For an addition-only model trained on expressions up to a fixed length, the model worked reliably near the training length but began to fail after crossing that range.

Example qualitative pattern:

```text
trained length:
  L_train = 3 terms

observed:
  3-term addition works
  4 or more terms begin to fail
```

With larger model or data scale `N`, the gap between the trained length and the failure point appeared to grow.

Useful metric:

```text
L_break:
  first expression length where accuracy drops below threshold

length_margin:
  L_break - L_train
```

The important distinction:

```text
larger N may increase length_margin
but this is not the same as proving unbounded algorithmic generalization
```

## 7. Metrics to track

The following metrics should be added to early proxy experiments:

```text
ood_entropy
ood_pmax
inactive_bias_norm
inactive_pmax
unknown_operator_neutrality
unknown_operator_assimilation_rate
wrong_operator_projection_rate
raw_fusion_accuracy
corrected_fusion_accuracy
inactive_leakage_mean
inactive_leakage_p95
length_breakpoint
length_margin
false_activation_rate
false_suppression_rate
```

Especially important:

```text
unknown_operator_assimilation_rate
inactive_pmax
inactive_leakage_mean
length_breakpoint
```

## 8. Research interpretation

These observations support the following interpretation:

```text
learned operator modules are not safely composable by raw addition
because inactive modules do not reliably produce neutral outputs
```

Stable composition requires a unit to be a pair:

```text
U_k = (M_k, C_k)
```

where:

```text
M_k:
  emits an operator-specific bias or proposal

C_k:
  suppresses the unit when the operator is not applicable
```

## 9. External-facing phrasing

A concise external statement:

```text
In preliminary 0.1K proxy experiments, inactive operator models did not produce neutral outputs on out-of-distribution operator tokens. For example, an ADD-only model exposed to subtraction-like inputs tended to assimilate the unknown operator into addition, producing peaked but wrong predictions rather than uncertainty. This explains why raw bias fusion fails and motivates pairing every operator model with an applicability corrector.
```

Japanese version:

```text
予備的な0.1K proxy実験では、未適用operator modelはOOD入力に対して中立出力を返さなかった。例えばADDのみを学習したモデルは、未学習の減算tokenを不確実性として扱うのではなく、加算として同化し、尖った誤分布を出した。このためraw bias fusionは失敗し、各operator modelに適用性補正器を付ける必要がある。
```

## 10. Final rule

```text
Do not assume irrelevant learned modules are neutral.
Measure their OOD peakedness and operator assimilation.
Fuse only corrected contributions g_k(x) b_k(x), not raw module outputs b_k(x).
```
