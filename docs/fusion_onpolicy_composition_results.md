# On-policy conflict-aware composition result

## Constraint

This experiment does not use routing.

- all five specialists are evaluated at every generated position;
- every unit has a strictly positive weight floor of `0.03`;
- no operator id, task label, subset mask, or external classifier is supplied;
- weights are continuous functions of the five simultaneous Base-relative bias fields;
- final IID/OOD splits remain unopened.

## Composition law

For centered and RMS-normalized bias fields `n_i`, the compositor computes continuous unit weights from:

- bias RMS;
- normalized positive peak;
- normalized negative peak;
- cosine agreement with the five-field mean.

It then forms a weighted signal and shrinks vocabulary coordinates with high inter-specialist disagreement:

```text
signal = sum_i weight_i * normalized_bias_i
conflict = sum_i weight_i * (normalized_bias_i - signal)^2
residual = scale * bounded(alpha * signal / (1 + lambda * conflict))
composed_logits = base_logits + residual
```

All weights remain positive and sum to one.

## Training

The compositor was calibrated on model seeds 0 and 1. Model seed 2 was held out from fitting.

1. teacher-forced warmup;
2. scheduled self-prefix rounds with rollout probabilities:
   - `0.25`
   - `0.50`
   - `0.75`
   - `1.00`
3. independent autoregressive verification on validation seed `708000`;
4. 16 examples per operator per model seed.

## Training behavior

| Stage | Calibration token accuracy | NLL | Mean weight entropy |
|---|---:|---:|---:|
| teacher-forced warmup | 0.5190 | 3.6570 | 1.1176 |
| rollout 0.25 | 0.5315 | 3.5954 | 1.0969 |
| rollout 0.50 | 0.5033 | 4.5566 | 1.0865 |
| rollout 0.75 | 0.4844 | 5.2669 | 1.0578 |
| rollout 1.00 | 0.4222 | 7.5155 | 1.0874 |

As self-prefix exposure increased, the objective became harder and the compositor did not learn recovery. This is evidence that the specialist fields become mutually uninformative or misleading on off-trajectory prefixes, rather than a simple teacher-forcing mismatch that a small conflict shrinker can repair.

## Autoregressive result

| Method | Final-value macro | Worst operator | Trace-validity macro |
|---|---:|---:|---:|
| `bias_mean` | 0.0875 | 0.0000 | 0.0875 |
| `rms_mean` | 0.0417 | 0.0000 | 0.0417 |
| geometry compositor, teacher-forced | 0.0000 | 0.0000 | 0.0000 |
| geometry compositor, on-policy | 0.0000 | 0.0000 | 0.0000 |

No method passed.

## Weight behavior

The learned continuous weights did not recover the relevant specialist reliably. Typical mean weights after on-policy training were:

- on `scalar.add` prompts:
  - sum approximately `0.39`
  - max approximately `0.32`
  - min approximately `0.19`
  - neg approximately `0.05`
  - add approximately `0.04`
- on `aggregation.sum` prompts:
  - sum approximately `0.48`
  - max approximately `0.22`
  - min approximately `0.18`
  - neg approximately `0.07`
  - add approximately `0.05`

The geometry features can recognize SUM to a degree but cannot identify ADD relevance. The weight floor remained active, so this is not a hidden hard router.

## Conclusion

The following composition classes are now empirically rejected for the current 1M shared-Base checkpoints:

- raw logit-bias sum;
- bias mean;
- RMS-normalized mean;
- global learned coefficients;
- token-class learned coefficients;
- static pairwise polynomial corrections;
- continuous conflict-aware geometry weighting;
- the same geometry weighting trained on self-generated prefixes.

The next justified composition experiment is a more expressive but still non-routing map that can use token identity and specialist confidence while remaining shared across prompts. Candidate forms are:

1. vocabulary-diagonal or low-rank vocabulary-by-specialist weights;
2. specialist-confidence features such as entropy reduction and top-logit margin;
3. a small recurrent residual compositor with persistent generation state;
4. hidden-state composition before the LM head.

A routing policy is not a supported conclusion.