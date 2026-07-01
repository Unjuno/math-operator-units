# Math Operator Units

This repository studies **logit-space semantics for same-prefix parallel bias control** through operator-specific model units.

The goal is not primarily to build a neural calculator, a faster router, keyword-based mode switching, or a replacement for symbolic computation. The goal is to run multiple model/unit outputs over the same sequence context, treat their differences as bias fields, fully compose those fields with human-interpretable bias operations, and test whether the composed field changes the next-token distribution in the intended direction.

Each operator unit has two components:

```text
U_k = (M_k, R_k)
```

- `M_k`: generator that produces a sequence-level bias field, logit contribution, or proposal.
- `R_k`: generation-path reliability calibrator that audits whether `M_k`'s generated path is reliable for `M_k` and attenuates or removes unreliable components before composition.

The composition-first runtime rule is:

```text
F(v | x) = O_calibrated(B_1, B_2, ..., B_n)(v | x)
z_final(v | x) = z_0(v | x) + λ F(v | x)
```

A simple token-wise contribution form is:

```text
z_final(v | x) = z_0(v | x) + Σ_{k in S_runtime} c_k(v | x) b_k(v | x)
```

A scalar gate version is allowed as an approximation:

```text
z_final(v | x) = z_0(v | x) + Σ_{k in S_runtime} g_k(x) b_k(v | x)
```

where `S_runtime` is the selected runtime set for the current experiment. The main target is not to route to one expert or hard-disable competing fields. The target is to compose multiple bias fields over the same sequence prefix, calibrate confidence or alignment when needed, and let the final softmax prefer the high-confidence or high-alignment direction.

## Project framing

A control direction is represented as a bias field over the vocabulary:

```text
B(v | x) ∈ R^{|V|}
```

For a parallel model/unit output:

```text
B_i(v | x) = z_i(v | x) - z_0(v | x)
```

A bias operator transforms these fields:

```text
F(v | x) = O(B_1, B_2, ..., B_n)(v | x)
```

Its meaning is defined by its induced distributional and verifier effects:

```text
Δp_F = softmax(z_0 + F) - softmax(z_0)
ΔV(F) = E_{y ~ p_F}[V(y)] - E_{y ~ p_0}[V(y)]
```

The mathematical operator experiments are controlled proxies for this goal. They test whether learned bias operators such as composition, mean, weighted sum, confidence selection, difference, projection removal, agreement, completion, and residual decomposition can be learned, calibrated, and composed before moving to less transparent LLM settings.

## Core design rules

1. The operator registry is the source of truth.
2. A unit checkpoint is an implementation of a registry entry.
3. Primitive operators may have learned units.
4. Derived operators should usually be represented as programs over primitives.
5. Distilled derived operators are allowed only when explicitly marked.
6. The tokenizer is part of the model ABI and must be fixed per tokenizer version.
7. Fusion is allowed only between checkpoints with the same tokenizer profile and vocabulary hash.
8. Mathematical and non-mathematical operators are separated by `kind`.
9. `dispatch` must remain false in runtime fusion manifests.
10. Numbers, equality, and structural expression tokens are shared ABI tokens across all units.
11. Operator units learn transformation distributions over the shared numeric/equality ABI; they must not redefine numbers or equality.
12. Direct summation must not assume inactive fields are neutral; calibrated composition must measure and control inactive bias leakage.
13. A learned bias module has semantic force only through its measured softmax/verifier effect and its calibrated contribution to the same-prefix next-token distribution.
14. Runtime control should be framed as same-prefix parallel bias-field composition, not keyword-based mode switching.
15. A generator should not be required to self-certify its own path; reliability calibration is performed by the paired `R_k` model.

## Initial documents

- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md): primary research framing for logit-space bias semantics.
- [`docs/parallel_sequence_bias_control.md`](docs/parallel_sequence_bias_control.md): same-prefix parallel model/unit outputs and bias-field control.
- [`docs/generation_path_reliability_calibrator.md`](docs/generation_path_reliability_calibrator.md): generator + path reliability calibrator design.
- [`docs/tokenizer_design.md`](docs/tokenizer_design.md): tokenizer and vocabulary policy.
- [`docs/shared_numeric_equality_abi.md`](docs/shared_numeric_equality_abi.md): shared number/equality ABI policy for all units.
- [`docs/equivalence_trace_training_plan.md`](docs/equivalence_trace_training_plan.md): equality trace data and anti-shortcut / anti-loop training policy.
- [`docs/raw_fusion_failure_observations.md`](docs/raw_fusion_failure_observations.md): preliminary 0.1K proxy observations motivating calibrated/corrected fusion.
- [`configs/tokenizer/tokenizer_core_v1.yaml`](configs/tokenizer/tokenizer_core_v1.yaml): initial tokenizer profile.
- [`configs/operators/registry.yaml`](configs/operators/registry.yaml): initial operator registry scaffold.

## Initial operator focus

The first learned units should target:

```text
<OP_SCALAR_ZERO>
<OP_SCALAR_ID>
<OP_SCALAR_NEG>
<OP_SCALAR_ADD>
<OP_SCALAR_ABS>
<OP_SCALAR_POS>
<OP_SCALAR_MIN>
<OP_SCALAR_MAX>
<OP_BIAS_ADD>
<OP_BIAS_SUB>
<OP_BIAS_CENTER>
<OP_BIAS_POS>
<OP_CTRL_GATE>
<OP_CTRL_SUPPRESS>
<OP_CTRL_ABSTAIN>
```

The first evaluation target is not broad problem solving. It is reproducible verification that parallel same-prefix bias control composes competing fields, controls unreliable contributions, and preserves useful active contributions.

## Preliminary raw fusion observation

Early 0.1K proxy experiments suggest that inactive operator models do not reliably produce neutral outputs on out-of-distribution inputs. For example, an ADD-only model exposed to subtraction-like inputs can assimilate the unknown operator into addition, producing peaked but wrong predictions rather than uncertainty.

This motivates treating a stable fusion unit as a pair:

```text
operator unit = generator + generation-path reliability calibrator
```

The generator proposes an operator-specific bias field. The calibrator audits whether the generated path is reliable for that generator and attenuates or removes unreliable components so that raw peakedness alone does not dominate the final same-prefix next-token distribution.

## Shared numeric and equality ABI

Numbers and equality are common infrastructure, not operator-specific semantics.

```text
shared ABI:
  numeric tokens
  equality token
  structural expression tokens

operator-specific layer:
  ADD transformation distribution
  MUL transformation distribution
  NEG transformation distribution
  VERIFY / PROGRESS / STOP judgments
```

For example, `ADD` and `MUL` may propose different transformations, but they must use the same numeric token meanings and the same equality relation marker.

```text
<OP_ADD> <NEXT_EQ> 3 + 4 + 5 -> 7 + 5
<OP_MUL> <NEXT_EQ> 3 * 4 * 5 -> 12 * 5
<VERIFY_EQ> 6 = 6 -> VALID
<PROGRESS?> 6 = 6 -> NO_PROGRESS
```

Valid equivalence is not the same as useful progress. Equality traces must therefore separate next-step proposal, verification, progress scoring, and stop decisions.

## Key metrics

- `single_accuracy`
- `fusion_accuracy`
- `active_gate_mean`
- `inactive_gate_mean`
- `inactive_leakage_mean`
- `inactive_leakage_p95`
- `false_neutral_rate`
- `invalid_entropy_norm`
- `invalid_pmax`
- `short_mixed_leakage`
- `length_ood_accuracy`
- `depth_ood_accuracy`
- `first_equals_final_answer_rate`
- `equals_to_eos_rate`
- `equals_continuation_rate`
- `repeated_state_rate`
- `no_progress_window_rate`
- `ood_entropy`
- `ood_pmax`
- `unknown_operator_assimilation_rate`
- `wrong_operator_projection_rate`
- `length_breakpoint`
- `length_margin`
- `softmax_effect_kl`
- `softmax_effect_jsd`
- `verifier_score_shift`
- `residual_stability`
- `parallel_field_agreement`
- `parallel_field_conflict`
- `control_success_rate`
- `confidence_calibration_error`
- `angle_alignment_score`
- `path_reliability_auc`
- `attenuation_precision`
- `attenuation_recall`
