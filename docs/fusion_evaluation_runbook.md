# Fusion Evaluation Runbook

## 1. Pilot reports first

Run the four-condition model-design pilot:

```bash
bash scripts/run_model_design_pilot.sh detach
```

Use only:

```text
evaluations/model_design_pilot/<condition>_validation.json
evaluations/model_design_pilot/<condition>_validation_units.json
evaluations/model_design_pilot/index.json
audits/model_design_pilot/pair_consistency.json
```

The pilot must not evaluate `iid_test`, `operand_ood`, or `length_ood`. Select model construction using the active v2 plan.

## 2. Complete production before mixer calibration

Train seeds 0, 1, and 2. Freeze every validation-selected `selected.pt`, the three experiment fingerprints, and the all-five `subset_31` manifests. Do not open a final split after only one seed.

## 3. Fusion calibration namespace

Before final data, evaluate validation with:

```text
evaluation seed: 703000
examples per operator: 128
training seeds: 0, 1, 2
```

The fixed baseline table contains Base, Relevant Specialist, raw sum, bias mean, and matched Joint. If raw sum crosses a v2 activation threshold, evaluate the preregistered families F1-F4 with three leave-one-training-seed-out folds.

In each fold, fit on two seeds and score on the third. The held-out seed is excluded from every fitted alpha, weight, RMS, median, clipping threshold, and optimizer result. Do not tune per seed or operator.

The ladder and exact formulas are in [`experiment_plan_v2.md`](experiment_plan_v2.md). The implementation must save fold reports and, when applicable, a machine-readable mixer contract.

## 4. Freeze final authorization

Before final IID/OOD data are generated, the calibration implementation must create:

```text
evaluations/fusion_calibration/final_authorization.json
```

The authorization records the active-plan hash, current Git commit, production experiment fingerprint, completed seeds and folds, calibration status, final settings, and the selected mixer-contract hash when a rescue is selected.

Allowed statuses are:

```text
raw_preserved_no_rescue
rescue_selected
no_eligible_nonrouter_rescue
```

The evaluator fails closed if authorization is absent or inconsistent. The current repository contains the guard, but the F2-F4 calibration runner and authorization generator still must be implemented before final evaluation.

## 5. Canonical final evaluation

After authorization is frozen, evaluate every production seed on every reserved split. For one seed/split:

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
  --split iid_test \
  --evaluation-seed 700000 \
  --examples-per-operator 64 \
  --final-authorization evaluations/fusion_calibration/final_authorization.json \
  --out evaluations/surface_v4_seed_0_subset_31_iid_test.json
```

Repeat for seeds 1 and 2 and for `operand_ood` and `length_ood`. The backward-compatible `test` alias is not part of the preregistered final namespace.

Final reports must preserve:

- Relevant Specialist;
- raw sum at `alpha=1.0` as confirmatory;
- fixed bias mean;
- matched all-five Joint;
- the frozen rescue mixer, when selected, as secondary.

## 6. Intermediate subsets and trajectories

Intermediate subset manifests have no matched Joint and support leakage/interference diagnostics only. Checkpoint-grid manifests support step-matched trajectory analysis. Neither changes the validation-selected endpoint policy or the all-five primary claim.

Do not evaluate intermediate subsets or alternative checkpoints on final data until the primary `subset_31` reports have been generated and archived. Any mixer selected after final data are viewed is post hoc.

## 7. Required provenance

Archive together:

```text
active plan and SHA-256
Git commit
CUDA smoke marker
pilot reports and pair audit
production experiment contracts
selected/final checkpoints
calibration fold reports
mixer contract, if any
final_authorization.json
all final reports
recovery and regularization logs
```

Legacy surface-v3 and typed-v2 controls use their own configs, output trees, and evaluation records. Never combine their checkpoints with surface-v4.
