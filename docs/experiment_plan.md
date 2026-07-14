# Preregistered Experiment Plan v1

> **Superseded prospectively:** the active plan is [`experiment_plan_v2.md`](experiment_plan_v2.md). This v1 text is retained as the historical record; no target-GPU or final IID/OOD result was used to define v2.

This plan was frozen before the target-GPU pilot and before final IID/OOD results were inspected.

## Question and scope

For a shared prefix `x`:

```text
B_k(x) = z_k(x) - z_base(x)
z_raw(x) = z_base(x) + sum_k B_k(x)
```

The primary question is whether the all-five raw sum preserves the Relevant Specialist's exact trace validity without causing stopping failures. The claim is limited to the five arithmetic operators, the surface tokenizer ABI, greedy generation, and the all-five subset.

The confirmatory replication unit is the independently initialized training seed. Generated examples within one seed are repeated measurements, not independent model replications.

## Fixed sequence

1. **CUDA smoke:** run `bash scripts/run_surface_v4_cuda_smoke.sh`. It must pass for the current commit, config, GPU, driver, PyTorch/CUDA stack, and deterministic settings. Smoke accuracy is operational only and cannot select the scientific condition.
2. **Model-design pilot:** run all four one-seed conditions on `validation` only: identity/unanchored, identity/retention, weak/unanchored, weak/retention. Pair consistency must pass. Do not inspect `iid_test`, `operand_ood`, or `length_ood`.
3. **Production:** after construction is frozen, train seeds 0, 1, and 2. Each seed has Base, five Specialists, and one all-five Joint: 21 models.
4. **Final evaluation:** after endpoints and any optional global alpha are frozen, evaluate all seeds on `iid_test`, `operand_ood`, and `length_ood` with evaluation seed `700000`, 64 examples per operator, and primary manifest `subset_31`.

## Pilot decision rule

Primary pilot quantity:

```text
G = trace_validity(raw_sum) - trace_validity(relevant_specialist)
```

A condition is eligible only if pair consistency passes, mean Relevant Specialist trace validity is at least 0.80, and Relevant Specialist trace validity, final-value accuracy, and EOS accuracy are each within 0.02 of the best pilot condition.

Among eligible conditions, prefer the largest `G`. A winner is clear if its `G` exceeds the runner-up by at least 0.02. A near tie within 0.01 may be resolved by at least 20% lower inactive mean JSD or centered-bias RMS, provided Relevant Specialist trace validity is not more than 0.01 lower.

Weak/retention advances only if inactive drift falls by at least 10% versus weak/unanchored and Relevant Specialist trace validity regresses by no more than 0.02.

If selection remains ambiguous, run all four conditions on exactly one additional pilot seed and aggregate seeds equally. Do not run only favored conditions. If no condition is eligible, stop and version a new plan before opening final splits.

## Production endpoints

Primary results use validation-selected `selected.pt`; `final.pt` remains trajectory evidence. Test and OOD data never select endpoints. All three production seeds must complete. Missing seeds are not imputed.

## Primary final analysis

Primary condition: `raw_sum` with `alpha=1.0`.

Primary comparator: `relevant_specialist`.

Primary endpoint: exact trace validity.

Report every seed and operator. The primary summary is the equal-weight mean of seed-level operator averages.

Interpretation:

- **preserved composition:** mean raw-minus-relevant trace gap is at least `-0.02`, and no operator-level mean gap is below `-0.05`;
- **partial interference:** mean gap is from `-0.10` to below `-0.02`;
- **material failure:** mean gap is below `-0.10`, or mean raw-minus-relevant EOS gap is below `-0.05`.

These are practical effect-size labels, not evidence of general natural-language composition.

## Secondary analyses

Key secondary metrics are final-value accuracy, EOS accuracy, exact response accuracy, gold-token NLL, divergence to Joint, and Joint argmax agreement. `bias_mean` is a fixed secondary condition.

A tuned global alpha is optional and secondary. Select one alpha for every seed, operator, and final split from `0.25, 0.50, 0.75, 1.00` using validation mean trace validity, then final-value accuracy, then EOS accuracy, then smaller alpha. Never select alpha per seed or operator.

Intermediate subsets and checkpoint trajectories are diagnostic. Joint-equivalence claims are restricted to the all-five subset with the matched Joint.

## Reporting and deviations

With three seeds, use descriptive effect sizes rather than decisive formal p-values. Report all seed values, equal-weight mean and standard deviation, operator values, and worst-operator gap. Example-level intervals are conditional within a trained seed and must not replace seed replication.

Before final splits are opened, freeze model construction, selected endpoints, optional alpha, evaluation sample count/seed, and this plan's Git commit. Changes before final evaluation require a versioned plan and new output roots. Changes after final splits are inspected are post hoc; preregistered results remain primary.

Preserve the plan commit, experiment contracts, CUDA marker, pilot reports, pair audit, model inventories, checkpoints, final reports, recovery logs, and regularization logs together.
