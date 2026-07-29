# Standalone model quality evaluation

## Scope

The models were evaluated independently before drawing further conclusions about fusion.
All evaluations use the validation partition. The final `iid_test`, `operand_ood`, and
`length_ood` splits remain unopened.

Two inventories were evaluated autoregressively and with teacher forcing:

1. Fusion Factory: 21 checkpoints = 3 model seeds × (Base + 5 specialists + Joint).
   - 24 examples per operator per model seed;
   - 72 examples per operator in the cross-seed aggregate;
   - 12 examples per aggregate length per seed.
2. Bias Factory: 24 selected checkpoints = 4 parameter sizes × (Base + 5 specialists).
   - 8 examples per operator per checkpoint;
   - 4 examples per aggregate length per checkpoint.

The Bias Factory configuration declared 4K/16K/65K/262K/1M snapshot steps, but the
repository checkout exposes only selected checkpoints for these runs. Consequently,
the present Bias Factory result measures the parameter-size axis but not the intended
training-data snapshot axis.

## Fusion Factory model quality

All models contain 1,013,824 parameters.

| Model role | Operator | Final value | Trace validity | Teacher-forced token | Quality gate |
|---|---|---:|---:|---:|---:|
| Specialist | `scalar.add` | 0.9444 | 0.9444 | 0.9815 | pass in 3/3 seeds |
| Specialist | `aggregation.sum` | 0.9167 | 0.9167 | 0.8995 | pass in 3/3 seeds |
| Specialist | `scalar.neg` | 0.0000 | 0.0000 | 0.3611 | fail in 3/3 seeds |
| Specialist | `scalar.min` | 1.0000 | 1.0000 | 0.9335 | pass in 3/3 seeds |
| Specialist | `scalar.max` | 1.0000 | 1.0000 | 0.9335 | pass in 3/3 seeds |

The identity Base passed its own identity-equivalence task at 1.0 on all five prompt
surfaces in every model seed.

The matched all-five Joint model obtained:

| Operator | Final value | Trace validity |
|---|---:|---:|
| `scalar.add` | 0.9861 | 0.9861 |
| `aggregation.sum` | 0.8056 | 0.8056 |
| `scalar.neg` | 0.1667 | 0.1667 |
| `scalar.min` | 0.9861 | 0.9861 |
| `scalar.max` | 1.0000 | 1.0000 |

The Joint model therefore also fails because negation is not learned and SUM is
materially weaker than its specialist.

## Length behavior

Cross-seed specialist final-value accuracy by number of terms:

| Terms | SUM specialist | MIN specialist | MAX specialist | Joint SUM |
|---:|---:|---:|---:|---:|
| 3 | 1.0000 | 1.0000 | 1.0000 | 0.9444 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| 5 | 0.9722 | 1.0000 | 1.0000 | 0.9444 |
| 6 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| 7 | 0.9722 | 1.0000 | 1.0000 | 0.8333 |
| 8 | 0.8611 | 1.0000 | 1.0000 | 0.6389 |

The long-sequence weakness is concentrated in SUM. MIN and MAX remain semantically
correct through eight terms, while Joint SUM degrades sharply at lengths seven and
eight.

## Valid trace versus canonical trace

The reduction specialists are much stronger semantically than canonical exact-match
metrics suggest:

| Specialist | Valid/final accuracy | Canonical response exact |
|---|---:|---:|
| SUM | 0.9167 | 0.0833 |
| MIN | 1.0000 | 0.1528 |
| MAX | 1.0000 | 0.1806 |

The data generator permits randomized adjacent-reduction orders during training, and
the verifier accepts any valid reduction trajectory. These models often produce a
valid noncanonical trace. Therefore a compositor trained against one canonical token
sequence receives incorrect supervision after it follows a different valid branch.

This invalidates the strong interpretation previously attached to the on-policy
compositor failure. The previous scheduled-prefix collector continued pairing a
self-generated prefix with tokens from the original canonical trace. After a valid
noncanonical branch, those tokens need not be valid for the current state. The run
shows that that training procedure fails; it does not establish that all specialist
fields are uninformative off trajectory.

## Bias Factory parameter-size result

Selected-checkpoint standalone accuracy:

| Parameters | ADD | SUM | NEG | MIN | MAX |
|---:|---:|---:|---:|---:|---:|
| 131,440 | 0.625 | 0.000 | 0.000 | 1.000 | 0.875 |
| 250,952 | 0.750 | 0.125 | 0.000 | 1.000 | 1.000 |
| 493,600 | 0.875 | 0.750 | 0.000 | 1.000 | 1.000 |
| 1,013,824 | 1.000 | 0.875 | 0.000 | 1.000 | 1.000 |

This is a low-sample scaling diagnostic, but the trend is clear:

- ADD and SUM benefit strongly from parameter count;
- MIN is solved by the smallest model;
- MAX is nearly solved by the smallest model;
- NEG fails identically at every size, with teacher-forced token accuracy fixed at
  approximately 2/3, consistent with learning equality/EOS while missing the numeric
  output token.

## Consequence for fusion

The usable Fusion Factory units are ADD, SUM, MIN, and MAX. NEG must not be treated as
an available capability during composition analysis.

The next compositor must not use routing, but it must also avoid two choices made in
the earlier experiment:

1. RMS-equalizing weak fields, which amplifies precisely the weak biases that should
   be attenuated;
2. forcing every unit to retain a positive weight floor, which prevents weak or
   contradictory contributions from shrinking to zero.

Training must use a set-valued valid-next-token objective over all valid reduction
orders, or an equivalent verifier-aware sequence objective. Canonical single-trace
cross-entropy is not a valid objective for the multi-trajectory reduction tasks.
