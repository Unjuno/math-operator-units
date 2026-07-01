# Math Operator Units

This repository studies **operator-specific model units** for mathematical and non-mathematical operators.

Each operator unit has two components:

```text
U_k = (M_k, C_k)
```

- `M_k`: main operator model that produces an operator-specific bias, logit contribution, or proposal.
- `C_k`: corrector / gate that suppresses the unit when the operator is not applicable.

The always-on fusion rule is:

```text
z_final = z_0 + Σ_k g_k(x) b_k(x)
```

where every unit is executed, but irrelevant units are suppressed by their own correctors.

## Core design rules

1. The operator registry is the source of truth.
2. A unit checkpoint is an implementation of a registry entry.
3. Primitive operators may have learned units.
4. Derived operators should usually be represented as programs over primitives.
5. Distilled derived operators are allowed only when explicitly marked.
6. The tokenizer is part of the model ABI and must be fixed per tokenizer version.
7. Fusion is allowed only between checkpoints with the same tokenizer profile and vocabulary hash.
8. Mathematical and non-mathematical operators are separated by `kind`.

## Initial documents

- [`docs/tokenizer_design.md`](docs/tokenizer_design.md): tokenizer and vocabulary policy.
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

The first evaluation target is not broad problem solving. It is reproducible verification that always-on fusion suppresses inactive units and preserves active units.

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
