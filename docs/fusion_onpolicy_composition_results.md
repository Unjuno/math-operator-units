# On-policy conflict-aware composition result

## Constraint

This experiment did not use routing.

- all five specialists were evaluated at every generated position;
- every unit had a strictly positive weight floor of `0.03`;
- no operator id, task label, subset mask, or external classifier was supplied;
- weights were continuous functions of the five simultaneous Base-relative bias fields;
- final IID/OOD splits remained unopened.

## Composition law

For centered and RMS-normalized bias fields `n_i`, the compositor computed continuous unit weights from:

- bias RMS;
- normalized positive peak;
- normalized negative peak;
- cosine agreement with the five-field mean.

It formed a weighted signal and shrank vocabulary coordinates with high inter-specialist disagreement:

```text
signal = sum_i weight_i * normalized_bias_i
conflict = sum_i weight_i * (normalized_bias_i - signal)^2
residual = scale * bounded(alpha * signal / (1 + lambda * conflict))
composed_logits = base_logits + residual
```

All weights were positive and summed to one.

## Training

The compositor was calibrated on model seeds 0 and 1. Model seed 2 was held out from fitting.

1. teacher-forced warmup;
2. scheduled self-prefix rounds with rollout probabilities `0.25`, `0.50`, `0.75`, and `1.00`;
3. independent autoregressive verification on validation seed `708000`;
4. 16 examples per operator per model seed.

## Observed optimization behavior

| Stage | Calibration token accuracy | NLL | Mean weight entropy |
|---|---:|---:|---:|
| teacher-forced warmup | 0.5190 | 3.6570 | 1.1176 |
| rollout 0.25 | 0.5315 | 3.5954 | 1.0969 |
| rollout 0.50 | 0.5033 | 4.5566 | 1.0865 |
| rollout 0.75 | 0.4844 | 5.2669 | 1.0578 |
| rollout 1.00 | 0.4222 | 7.5155 | 1.0874 |

## Autoregressive result

| Method | Final-value macro | Worst operator | Trace-validity macro |
|---|---:|---:|---:|
| `bias_mean` | 0.0875 | 0.0000 | 0.0875 |
| `rms_mean` | 0.0417 | 0.0000 | 0.0417 |
| geometry compositor, teacher-forced | 0.0000 | 0.0000 | 0.0000 |
| geometry compositor, scheduled-prefix training | 0.0000 | 0.0000 | 0.0000 |

No evaluated method passed.

## Validity limitation discovered by standalone model evaluation

The reduction specialists often generate semantically valid traces that differ from the single canonical validation trace. Across the three Fusion Factory seeds:

| Specialist | Valid/final accuracy | Canonical response exact |
|---|---:|---:|
| SUM | 0.9167 | 0.0833 |
| MIN | 1.0000 | 0.1528 |
| MAX | 1.0000 | 0.1806 |

The scheduled-prefix collector iterated through the original canonical `expected` sequence. At each position it could append a compositor prediction to the current prefix, but it still stored the next token from that fixed canonical sequence as the target.

After the compositor follows another valid reduction branch, the next canonical token may no longer be valid for the current state. The rollout dataset can therefore contain contradictory prefix/target pairs. Increasing rollout probability necessarily increases this target corruption.

Consequently, the declining scheduled-prefix objective does **not** establish that specialist fields become uninformative off trajectory. It establishes only that this canonical-label rollout procedure failed. The scheduled-prefix experiment is not a valid rejection of on-policy bias composition.

## Additional design mismatch

The evaluated compositor also contradicted the intended weak-bias suppression mechanism in two ways:

1. RMS normalization equalized weak and strong fields, potentially amplifying weak or noisy biases;
2. the positive weight floor prevented an unsupported contribution from shrinking to zero.

These choices are not equivalent to external routing, but they block the continuous attenuation mechanism the experiment is intended to study.

## Correct next experiment

The next compositor remains non-routing: all five units are evaluated at every position and no task label is supplied. It should change both the objective and the algebra.

### Verifier-aware target

Construct the set of valid next tokens for the current generated prefix over every valid adjacent-reduction trajectory. Optimize

```text
-loss = log sum_{token in valid_next(prefix)} p(token | prefix)
```

rather than cross-entropy to one canonical path. On-policy prefixes must be retained only while they remain in the valid-prefix automaton, or be trained with a verifier-derived sequence objective.

### Sparse evidence composition

Use raw centered biases without RMS equalization and allow continuous coefficients to reach zero:

```text
shrunk_i = sign(bias_i) * relu(abs(bias_i) - threshold_i)
residual = sum_i confidence_i * shrunk_i
composed_logits = base_logits + bounded(residual)
```

`confidence_i` is computed from simultaneous logit evidence and optional persistent generation-state evidence. It is not an operator selector. All specialists are still evaluated; weak or contradictory fields are continuously attenuated rather than externally switched.

## Revised conclusion

The run validly records poor performance for raw mean, RMS mean, and the tested geometry compositor. It does not support the previous claim that on-policy recovery is impossible. The on-policy result must be rerun with valid-set supervision and without RMS promotion or a positive weight floor. A routing policy remains outside the supported conclusion.
