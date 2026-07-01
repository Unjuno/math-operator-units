# Math Operator Count Estimate

There is no finite number of operators in mathematics.

Mathematically, once a domain and codomain are fixed, every function can be treated as an operator. Since there are infinitely many functions, there are infinitely many possible operators.

For this repository, the relevant question is engineering-oriented:

```text
How many operator slots should the tokenizer reserve so that future practical operator catalogs can be represented without changing tokenizer shape?
```

## 1. Practical catalog scale

A practical math/science/control registry is not infinite. It is a curated catalog.

Approximate engineering scale:

| catalog level | estimated operators |
|---|---:|
| core arithmetic + control | 50-150 |
| broad math primitives | 300-800 |
| math + tensor + graph + symbolic + PDE | 1000-3000 |
| math + science + verifier + tool + semantic | 3000-8000 |
| Wolfram-like broad symbolic environment | 6000+ symbols/functions |

The last row should not be interpreted as a target for direct learned units. It is a reference for how large a broad symbolic computational language can become.

## 2. Why doubling makes sense only for a curated catalog

For true mathematics:

```text
operator count = infinite
2 * infinite = still infinite
```

So doubling does not apply.

For an engineering catalog, doubling is useful:

```text
expected useful catalog size ≈ 3000-6000
reserve about 2x
recommended full reserve ≈ 8192-16384
```

## 3. Recommendation

Use two tokenizer profiles:

```text
tokenizer_core_v1:
  8192 reserved operator slots
  default for serious early checkpoints

tokenizer_full_v1:
  16384 reserved operator slots
  for long-term broad math/science/control/tool expansion
```

Do not use 16384 for tiny toy experiments unless full-vocab heads are avoided.

## 4. Direct tokens vs reserved slots

The tokenizer should not rely entirely on anonymous reserved slots.

Central operators should have named direct tokens.

Examples:

```text
<OP_SCALAR_ADD>
<OP_SCALAR_MUL>
<OP_LINALG_MATMUL>
<OP_CALC_LAPLACIAN>
<OP_PROB_SOFTMAX>
<OP_BIAS_REMOVE>
<OP_VERIFY_EXACT>
```

Reserved slots should handle future and speculative operators.

## 5. Final capacity choice

Recommended final setting before freeze:

```yaml
reserved_operator_slots:
  core_v1: 8192
  full_v1: 16384

runtime_fusion_units:
  default_max: 64
  stress_max: 256
  research_max: 1024
```

This gives a broad future-proof tokenizer while keeping runtime fusion sets small.
