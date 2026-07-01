# Capacity Planning

This document distinguishes two capacities that should not be confused:

```text
reserved token capacity:
  how many future operator tokens can be assigned without changing tokenizer shape

runtime fusion capacity:
  how many units are loaded and fused for a specific task
```

They should be sized differently.

## 1. Recommended defaults

Recommended v1 policy:

```text
reserved operator slots: 4096 minimum, 8192 preferred if full-vocab heads are not used everywhere
runtime fusion set size: 8-64 default, 128-256 stress test, 1024 only for scaling research
```

Interpretation:

```text
large registry capacity
small task-specific runtime set
```

## 2. Reserved token capacity

Reserved tokens preserve ABI compatibility. They are not all active at runtime.

Recommended choices:

| reserved slots | use case |
|---:|---|
| 1024 | too small for long-term expansion |
| 4096 | good minimum for tokenizer_core_v1 |
| 8192 | safer for broad math/science/control/tool expansion |
| 16384 | only if full long-term universal tokenizer is the goal |

For this repository:

```text
4096 is acceptable now.
8192 is the safer freeze target before serious checkpoint investment.
```

## 3. Parameter cost

Let:

```text
R = number of reserved tokens
D = model embedding dimension
```

If embedding and output head are untied, reserved-token parameter cost is approximately:

```text
P_reserved ≈ 2 R D
```

If tied, it is approximately:

```text
P_reserved ≈ R D
```

Examples with untied embedding/output head:

| R | D=64 | D=256 | D=1024 |
|---:|---:|---:|---:|
| 4096 | 0.52M | 2.10M | 8.39M |
| 8192 | 1.05M | 4.19M | 16.78M |
| 16384 | 2.10M | 8.39M | 33.55M |

This is why nano/micro proxy units should not always use full-vocab heads.

## 4. Runtime fusion capacity

Runtime fusion set size should be much smaller than total registry size.

Recommended runtime profiles:

```text
fusion_tiny:    4-8 units
fusion_small:   16 units
fusion_medium:  32-64 units
fusion_large:   128 units
fusion_stress:  256 units
fusion_research: 1024 units
```

Default experiments should target:

```text
16 -> 64 -> 256
```

## 5. Leakage scaling

If each inactive unit has false activation rate or average inactive gate value `p`, then expected inactive leakage is roughly:

```text
E[leakage] ≈ N_inactive * p
```

For a leakage budget `L`, the required average inactive gate is:

```text
p <= L / N_inactive
```

Example with `L = 0.05`:

| inactive units | required p |
|---:|---:|
| 16 | 0.003125 |
| 64 | 0.000781 |
| 256 | 0.000195 |
| 1024 | 0.000049 |

Therefore, full always-on fusion becomes harder as unit count increases.

Runtime set swapping reduces `N_inactive`, which directly reduces leakage pressure.

## 6. Correct policy

```text
Do not make runtime fusion set size equal to reserved token count.
```

The registry may support thousands of operators, but a given task should load only the relevant subset.

## 7. Suggested tokenizer freeze target

For `tokenizer_core_v1`:

```text
current acceptable value: 4096 reserved operator tokens
recommended freeze value: 8192 reserved operator tokens
```

Rationale:

```text
4096 is enough for early math/bias/control expansion.
8192 gives more room for graph, geometry, PDE, symbolic, verifier, candidate, semantic, and tool operators.
```

Do not go to 16384 unless the project commits to a full universal tokenizer, because it increases full-vocab model heads substantially.

## 8. Suggested runtime set limits

For manifests:

```yaml
runtime_limits:
  default_max_units: 64
  warning_above_units: 128
  stress_test_max_units: 256
  research_only_max_units: 1024
```

A runtime fusion set above 128 units should report explicit leakage-scaling metrics.

## 9. Final recommendation

```text
Use 8192 reserved operator slots before final tokenizer freeze.
Use 64 as the default runtime fusion set ceiling.
Use 256 for stress tests.
Use 1024 only to study scaling behavior.
```

This balances future expansion with runtime stability.
