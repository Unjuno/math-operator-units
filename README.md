# Math Operator Units

This repository studies **operator-specific model units** for mathematical and non-mathematical operators.

Each operator unit has two components:

```text
U_k = (M_k, C_k)
```

- `M_k`: main operator model that produces an operator-specific bias, logit contribution, or proposal.
- `C_k`: corrector / gate that suppresses the unit when the operator is not applicable.

The runtime fusion rule is:

```text
z_final = z_0 + Σ_{k in S_runtime} g_k(x) b_k(x)
```

where `S_runtime` is the selected runtime fusion set for the current task or experiment. The full registry may contain many units, but only the selected runtime set is loaded for a run. Within that set, irrelevant units are suppressed by their own correctors.

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

## Initial documents

- [`docs/tokenizer_design.md`](docs/tokenizer_design.md): tokenizer and vocabulary policy.
- [`docs/shared_numeric_equality_abi.md`](docs/shared_numeric_equality_abi.md): shared number/equality ABI policy for all units.
- [`docs/equivalence_trace_training_plan.md`](docs/equivalence_trace_training_plan.md): equality trace data and anti-shortcut / anti-loop training policy.
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

The first evaluation target is not broad problem solving. It is reproducible verification that runtime-selected fusion suppresses inactive units and preserves active units.

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
