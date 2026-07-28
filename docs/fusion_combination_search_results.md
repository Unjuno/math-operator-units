# Fusion combination search results

## Scope

The search used only the validation partition. The final `iid_test`, `operand_ood`, and `length_ood` splits were not opened.

Two stages were run over the three completed Fusion Factory seeds.

1. Broad teacher-forced search:
   - all 31 non-empty specialist subsets;
   - raw base-relative addition and RMS-equalized addition;
   - alpha grid `0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0`;
   - 32 validation examples per operator;
   - 1,488 cohort/candidate rows.
2. Independent autoregressive verification:
   - evaluation seed `704000`, distinct from the broad-search seed;
   - 32 validation examples per operator per model seed, or 96 examples per operator in aggregate;
   - maximum 64 generated tokens;
   - causal RMS normalization computed independently at each generated position.

## Autoregressive results

No multi-specialist candidate passed the validation gate.

| Candidate | Active operators | Final-value macro | Worst active operator | Trace-validity macro |
|---|---|---:|---:|---:|
| `add_raw_1.00` | add | 0.9896 | 0.9896 | 0.9896 |
| `sum_raw_1.25` | sum | 0.8854 | 0.8854 | 0.8854 |
| `min_raw_1.25` | min | 0.8438 | 0.8438 | 0.8438 |
| `max_raw_1.25` | max | 0.8125 | 0.8125 | 0.8125 |
| `add_max_causal_rms_0.75` | add, max | 0.3333 | 0.1979 | 0.3333 |
| `add_min_causal_rms_0.75` | add, min | 0.3073 | 0.2188 | 0.3073 |
| `min_max_raw_0.75` | min, max | 0.1875 | 0.1667 | 0.1875 |
| `add_min_max_causal_rms_0.50` | add, min, max | 0.1250 | 0.0208 | 0.1250 |
| `all_five_causal_rms_0.125` | all five | 0.0083 | 0.0000 | 0.0083 |
| `all_five_raw_0.125` | all five | 0.0021 | 0.0000 | 0.0021 |
| `neg_raw_0.50` | neg | 0.0000 | 0.0000 | 0.0000 |

The strongest multi-specialist candidate was `add_max_causal_rms_0.75`, but it retained only 0.1979 final-value accuracy on its weaker active operator. It is not viable.

## Reference models

The matched all-five Joint model obtained the following final-value accuracies on the same independent validation run:

| Operator | Joint final-value accuracy |
|---|---:|
| add | 1.0000 |
| sum | 0.8542 |
| neg | 0.1667 |
| min | 0.8438 |
| max | 0.8646 |

The relevant singleton specialist was competitive for add, sum, min, and max. The neg specialist remained unusable.

## Interpretation

The broad teacher-forced search produced apparently strong pair and triple candidates, but those gains did not survive autoregressive generation. The discrepancy is exposure instability: once a fused field selects an incorrect token, later specialist fields are evaluated on a context outside the teacher-forced path and rapidly interfere.

Static always-on addition also failed to preserve inactive Base behavior. Every non-full candidate had zero exact agreement with Base on inactive operator prompts in the autoregressive check. This is direct evidence that a fixed global subset is the wrong runtime policy for the current checkpoints.

## Current decision

There is no production-eligible static bias combination.

The supported validation policy is prompt-conditioned dispatch:

| Routed operator | Unit | Alpha |
|---|---|---:|
| add | `scalar.add` | 1.00 |
| sum | `aggregation.sum` | 1.25 |
| min | `scalar.min` | 1.25 |
| max | `scalar.max` | 1.25 |
| neg | none | n/a |

This is a routing result, not evidence that raw multi-bias superposition works. Negation remains blocked. Final splits remain reserved.
