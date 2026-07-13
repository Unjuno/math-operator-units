# Fusion Evaluation Runbook

## Pilot reports first

The model-design pilot evaluates all four one-seed conditions automatically:

```bash
bash scripts/run_model_design_pilot.sh detach
```

Pilot outputs are development-only:

```text
evaluations/model_design_pilot/<condition>_validation.json
evaluations/model_design_pilot/<condition>_validation_units.json
evaluations/model_design_pilot/index.json
audits/model_design_pilot/pair_consistency.json
```

Select the construction from validation only. The pilot does not evaluate IID test, operand OOD, or length OOD model performance.

## Surface-v4 all-five matched-Joint comparison

After the construction has been fixed and a production seed finishes, use the canonical final evaluation namespace:

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
  --split iid_test \
  --evaluation-seed 700000 \
  --examples-per-operator 64 \
  --out evaluations/surface_v4_seed_0_subset_31_iid_test.json
```

`test` remains a backward-compatible IID split alias, but final primary reports should use `iid_test` consistently. The report records `evaluation_seed`; do not omit it from result tables.

For `subset_31`, the report includes:

- Base;
- Relevant Specialist;
- raw bias sum;
- bias mean;
- exposure-matched all-five Joint;
- gold-token NLL;
- generation correctness and exact trace validity;
- EOS accuracy and generated length;
- Jensen–Shannon divergence and next-token argmax agreement to the Joint.

Preserve the associated `experiment_contract.json` and record its fingerprint with the evaluation output.

## Final OOD evaluation

Run OOD conditions only after model construction, endpoint selection, and any global alpha are frozen:

```bash
for split in operand_ood length_ood; do
  .venv/bin/opfusion-evaluate-fusion \
    --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
    --split "$split" \
    --evaluation-seed 700000 \
    --examples-per-operator 64 \
    --out "evaluations/surface_v4_seed_0_subset_31_${split}.json"
done
```

These are final stress tests, not design-tuning signals.

## Intermediate subset diagnostic

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_03.json \
  --split iid_test \
  --evaluation-seed 700000 \
  --examples-per-operator 64 \
  --out evaluations/surface_v4_seed_0_subset_03_iid_test.json
```

Intermediate manifests deliberately have `joint_reference_checkpoint: null`. Their results support leakage, interference, stability, and task-accuracy diagnostics, not claims of equivalence to Joint training.

## Global alpha

The primary raw condition uses `--alpha 1.0`. A global alpha may be selected on validation and then frozen for every final seed and split. Do not select a different alpha per seed.

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
  --split validation \
  --evaluation-seed 700000 \
  --alpha 0.5 \
  --examples-per-operator 64 \
  --out evaluations/surface_v4_seed_0_alpha_0_5_validation.json
```

Input-dependent alpha, routing, and learned correction are separate experiments.

## Selected endpoints versus trajectories

Final subset manifests use `selected.pt`. To evaluate a step-matched checkpoint observation, use the corresponding manifest under:

```text
seed_<n>/fusion_checkpoint_grid/step_<step>/subset_<mask>.json
```

All Specialist and Joint checkpoints in one grid share the same optimizer-step index. The common Base is the validation-selected parent checkpoint used to initialize those branches.

Report both analyses when relevant:

1. **validation-selected endpoint comparison** — practical endpoint selection without test data;
2. **step-matched trajectory comparison** — how fusion changes over training time.

Do not silently substitute `final.pt` for `selected.pt`; state the endpoint policy in every result table.

## Legacy control evaluation

Surface-v3 remains an explicit identity-Base/unanchored control. Its evaluation must use its own config, output tree, and manifests. Typed-v2 likewise remains separate. Never compare a v4 Specialist with a v3 or v2 Base.
