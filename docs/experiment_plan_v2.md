# Preregistered Experiment Plan v2

This plan prospectively extends v1 before the target-GPU pilot and before any final IID/OOD result is inspected. The raw all-five sum remains the confirmatory test; fallback mixers are secondary rescue analyses and cannot convert a failed raw-sum result into a successful primary result.

## Unchanged primary experiment

For `B_k(x)=z_k(x)-z_base(x)`, the primary condition is:

```text
z_raw(x) = z_base(x) + sum_k B_k(x)
alpha = 1.0
```

The comparator is the Relevant Specialist and the endpoint is exact trace validity. Seeds 0, 1, and 2 are the replication units. The final splits remain `iid_test`, `operand_ood`, and `length_ood`, with evaluation seed `700000`, 64 examples per operator, and `subset_31`.

## Added stage: fusion calibration

After model construction and every production `selected.pt` are frozen, but before final splits are opened, evaluate the mixing ladder on production validation. Escalate beyond fixed `bias_mean` when any validation condition holds:

```text
mean raw-minus-relevant trace gap < -0.02
worst-operator trace gap < -0.05
mean raw-minus-relevant EOS gap < -0.02
```

A failure first observed on final data cannot trigger tuning.

## Mixing ladder

Use the first family that satisfies the rescue rule. Parameters must be shared across seeds and operators.

### F0 — fixed baselines

```text
raw_sum:   z = z_base + sum_k B_k
bias_mean: z = z_base + (1/K) sum_k B_k
```

### F1 — global shrinkage

```text
z = z_base + alpha * sum_k B_k
alpha in {0.10, 0.20, 0.25, 0.50, 0.75, 1.00}
```

One alpha is used for every seed, operator, position, and final split.

### F2 — norm-controlled fields

First center each field over vocabulary:

```text
C_k = B_k - mean_vocab(B_k)
```

Evaluate:

- **RMS-equalized mean:** inverse-scale each field by its production-validation RMS, restore the median RMS, then average;
- **RMS-clipped sum:** cap response-position RMS at validation quantile `q in {0.90, 0.95, 0.99}` before summing.

No operator-specific scale is permitted.

### F3 — static nonnegative weighted mean

```text
z = z_base + alpha * sum_k pi_k B_k
pi_k = softmax(a_k)
```

The five weights and alpha are constant across prompts, operators, seeds, and token positions. Fit with regularization toward uniform weights. This tests systematic scale imbalance without introducing routing.

### F4 — deterministic consensus-tempered decoding

At each token position let `p_k=softmax(z_base+B_k)`, let `p_bar` be their uniform probability mean, and set:

```text
d_k = JSD(p_k, p_bar)
w_k proportional to exp(-beta*d_k)
beta in {0.5, 1.0, 2.0}
z = z_base + alpha * sum_k w_k B_k
```

This method is input-dependent but has no trained router. It must be labeled consensus-tempered decoding, not raw composition. Because the relevant unit can legitimately disagree with inactive units, improvement is not assumed.

### F5 — learned router or corrector

Prompt-dependent gating, a learned router, or a residual logit corrector changes the scientific question and is excluded from the present confirmatory final evaluation. If F0–F4 fail, begin a new versioned study with a dedicated fusion-calibration partition, new output roots, a linear mixer before nonlinear correction, and the known operator-tag router only as an oracle upper bound.

## Validation protocol

Assess F1–F4 with three-fold leave-one-training-seed-out validation. Tune on two production seeds and score on the held-out seed. Never tune per seed or per operator.

A family is eligible only if held-out results satisfy:

```text
mean trace-gap improvement over raw_sum >= 0.02
mean EOS gap to Relevant Specialist >= -0.02
mean final-value gap to Relevant Specialist >= -0.02
worst-operator trace gap >= -0.05
improvement on at least 2 of 3 held-out seeds
```

Select the earliest eligible family in order F1, F2, F3, F4. Within a family, maximize held-out trace validity, then final-value accuracy, then EOS accuracy, then prefer the simpler or more strongly shrunk setting. Refit only the selected family's permitted statistics on all production validation data, write a machine-readable mixer contract, and freeze it before final evaluation.

If no family qualifies, stop searching. The conclusion is that the fields are not recoverably composable under the preregistered non-router families.

## Final reporting

The primary table always contains Relevant Specialist, raw sum, bias mean, and matched Joint. A prospectively frozen rescue mixer is reported as a labeled secondary condition on every final split.

Report every seed and operator, equal-weight seed mean and standard deviation, worst-operator gap, rescue improvement over raw sum, absolute gap to Relevant Specialist, EOS/final-value guardrails, validation folds, hyperparameters, and mixer-contract hash.

A rescue may be practically useful if it meets the preserved-composition thresholds on final data, but the raw-sum result remains the confirmatory answer. Any method chosen after final data are viewed is post hoc.
