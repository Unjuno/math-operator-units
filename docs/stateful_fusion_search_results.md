# Stateful non-routing fusion search results

## Scope

All experiments in this note use validation data only. The final `iid_test`,
`operand_ood`, and `length_ood` splits remain unopened. Every method evaluates the
Base and all five specialists at every generated token. No operator id, task label,
subset mask, external classifier, or discrete source switch is supplied.

The verified standalone inventory remains unchanged: ADD, SUM, MIN, and MAX are
usable units; NEG is a failed unit and is retained as a negative control.

## Memoryless follow-up experiments

### Scaled continuous source-validity mixture

A shared source scorer was trained on model seeds 0 and 1 and evaluated on held-out
model seed 2. Increasing the calibration set improved held-out valid-next-token
accuracy to `0.8000`, but the best autoregressive final-value macro remained only
`0.1333`; bias mean obtained `0.0944` on the same verification run.

### Scaled vocabulary-coordinate gate

The shared coordinate gate used raw Base-relative fields, a learned continuous
threshold, token metadata, and no specialist identity. Its best held-out
valid-next-token accuracy was `0.5833`. Autoregressive final-value macro was
`0.0083`, below bias mean at `0.0667`.

### Candidate-token evidence mixture

A permutation-invariant candidate-token scorer combined local source rank,
probability, Base-relative gain, cross-source support, and token metadata. Its best
held-out valid-next-token accuracy was `0.7756`. Autoregressive final-value macro was
`0.1000`, while bias mean obtained `0.1083`.

These results reject the tested memoryless coordinate laws. Better teacher-forced
valid-set accuracy alone does not preserve a coherent source of evidence over a
multi-token trajectory.

## Continuous stateful mixture

The successful change was to retain a continuous reliability state for every source:

```text
instant_state_t = log instantaneous_source_weights_t
state_t = memory * state_(t-1) + (1 - memory) * instant_state_t
weights_t = softmax(state_t / temperature)
mixture_t = sum_i weights_t[i] * source_probability_t[i]
state_t += feedback * centered_log_support_i(generated_token_t)
```

All six weights remain strictly positive. This is not a discrete router: sources are
never selected or disabled, and the update depends only on simultaneous model
outputs and the generated trajectory.

### Initial search

The initial 36-condition search selected:

```text
memory = 0.95
feedback = 0.15
temperature = 0.75
```

Independent autoregressive verification over three model seeds produced:

| Operator | Final-value / valid-trace accuracy |
|---|---:|
| ADD | 1.0000 |
| SUM | 0.5000 |
| MIN | 0.5556 |
| MAX | 0.1111 |
| NEG | 0.0000 |
| **Macro** | **0.4333** |

Bias mean obtained `0.0889` on the same run.

### Fine search and larger verification

A 48-condition fine search tested:

- memory: `0.90, 0.95, 0.98, 0.99`;
- feedback: `0.00, 0.10, 0.20, 0.35`;
- temperature: `0.50, 0.75, 1.00`.

The source-validity scorer reached held-out valid-next-token accuracy `0.8214`.
The best independently verified state law was:

```text
memory = 0.95
feedback = 0.35
temperature = 1.00
```

Verification used 12 examples per operator per model seed, or 36 examples per
operator across the three seeds:

| Operator | Final-value / valid-trace accuracy |
|---|---:|
| ADD | 0.8889 |
| SUM | 0.3889 |
| MIN | 0.3611 |
| MAX | 0.4722 |
| NEG | 0.0000 |
| **Macro** | **0.4222** |

Bias mean obtained `0.1000` on the same verification set. The stateful result is
therefore about `4.2x` the memoryless baseline in macro accuracy. Excluding the known
failed NEG control, the four-usable-unit macro is `0.5278`, versus `0.1250` for bias
mean.

## Interpretation

The current strongest empirical result is that the fusion law needs trajectory
state. A source whose field was useful over the preceding tokens should retain
influence, while unsupported fields should decay continuously. Independent
per-token fusion repeatedly loses this information and makes a single early error
that invalidates the remaining trajectory.

The result is not yet a general composition law:

- none of the methods passes the production threshold;
- NEG contributes no usable capability;
- SUM, MIN, and MAX remain materially below their standalone specialists;
- evaluation still covers single-operator traces rather than held-out compound
  programs.

## Next experiment

The next justified compositor is a token-role stateful mixture. It will keep separate
continuous reliability states for numeric, structural, and stopping-token
coordinates, while still evaluating every source and using no operator label. This
should preserve arithmetic-source evidence across numeric transitions without
forcing the same source mixture onto equality, separators, and EOS. After that
single-operator retention test, the same law should be evaluated on held-out
compound computation graphs.

## Artifacts

The completed runs are archived by GitHub Actions as:

- `fusion-validity-mixture-scaled`;
- `fusion-coordinate-gate-scaled`;
- `fusion-token-evidence`;
- `fusion-stateful-mixture`;
- `fusion-stateful-mixture-fine`.
