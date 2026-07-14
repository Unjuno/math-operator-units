# Model Construction Plan v3: Specialist Failure Diagnosis

This plan starts after the completed four-condition pilot found no eligible production construction. Final IID/OOD splits remain locked. The observed retention improvement is retained as evidence about inactive drift, but production is stopped because the Relevant Specialists for `aggregation.sum` and `scalar.neg` did not meet the preregistered trace-validity threshold.

## Decision status

- Production: no-go.
- Additional blind seed sweep: not the first action.
- Final `iid_test`, `operand_ood`, and `length_ood`: remain unopened.
- Existing pilot artifacts: immutable diagnostic inputs.

## D1: Failure-record extraction

For each of the four pilot conditions, run the specialist diagnostic on validation with evaluation seed `704000`, 64 examples per operator, and the selected all-five manifest. Retain at most 20 failed examples per operator.

The diagnostic records:

- prompt, gold response, and generated response tokens;
- verifier failure reason;
- first divergent token index;
- generated and gold lengths;
- EOS and final-value correctness;
- teacher-forced token, first-token, and full-sequence argmax accuracy;
- free-generation trace validity.

Primary classification:

1. teacher-forced failure: token accuracy below 0.95;
2. exposure failure: teacher-forced token accuracy at least 0.95 but generation trace validity below 0.80;
3. stopping failure: stop accuracy below 0.95;
4. arithmetic transition failure: verifier reason `invalid_transition` dominates;
5. format failure: parse/equality/segment reasons dominate.

## D2: Checkpoint trajectory

For `aggregation.sum` and `scalar.neg`, evaluate every available permanent checkpoint on the same validation problem set. Do not select a new checkpoint from final data.

Compare:

- validation NLL;
- teacher-forced token accuracy;
- first-token accuracy;
- free-generation trace validity;
- final-value and EOS accuracy.

If the NLL-selected checkpoint is not within 0.02 trace validity of the best validation checkpoint, revise endpoint selection to include a generation guardrail in a new versioned plan.

## D3: Minimal specialist-only ablations

Only after D1 and D2 identify the failure class, train seed 0 diagnostics for the two failed operators. Use new output roots.

For `aggregation.sum`:

- current randomized-reduction mixture;
- canonical left-fold training;
- full-trace-only training;
- canonical plus full-trace-only training;
- short curriculum restricted to three or four terms.

For `scalar.neg`:

- current mixture;
- full-trace-only training;
- terminal-stop examples removed;
- generation-aware checkpoint selection diagnostic.

Each ablation must preserve tokenizer, architecture, optimizer, effective batch, deterministic CUDA settings, and validation problem set unless the ablated field explicitly changes them.

## Advancement rule

A repaired operator configuration advances only when both failed operators satisfy on validation:

- Relevant Specialist trace validity at least 0.80;
- final-value accuracy at least 0.80;
- EOS accuracy at least 0.95;
- no more than 0.02 regression on `scalar.add`, `scalar.min`, or `scalar.max` when reintegrated;
- retention inactive-drift benefit remains at least 10 percent relative to the matched unanchored condition.

Then rerun the full four-condition pilot under a new experiment ID and output root. Production remains blocked until that corrected pilot passes the original eligibility rules.

## Canonical command

For one completed pilot condition:

```bash
.venv/bin/opfusion-diagnose-specialist-failures \
  --config configs/experiments/model_design_pilot_weak_retention.yaml \
  --manifest runs/model_design_pilot/weak_retention/seed_0/fusion_subsets/subset_31.json \
  --operators aggregation.sum scalar.neg \
  --split validation \
  --evaluation-seed 704000 \
  --examples-per-operator 64 \
  --retain-examples 20 \
  --out evaluations/model_design_pilot/weak_retention_specialist_failures.json
```

Adjust the manifest path to the actual path recorded by the completed pilot index. Do not guess or move pilot artifacts.
