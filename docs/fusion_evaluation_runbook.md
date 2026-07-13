# Fusion Evaluation Runbook

## Pilot reports first

The model-design pilot evaluates all four one-seed conditions automatically:

```bash
bash scripts/run_model_design_pilot.sh detach
```

Reports are written to:

```text
evaluations/model_design_pilot/<condition>_validation.json
evaluations/model_design_pilot/<condition>_test.json
evaluations/model_design_pilot/index.json
```

Use validation reports to compare or tune global design choices. Test reports are held-out confirmation and must not select a retention coefficient, alpha, or checkpoint.

## Surface-v4 all-five matched-joint comparison

After a production seed finishes:

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
  --split test \
  --examples-per-operator 64 \
  --out evaluations/surface_v4_seed_0_subset_31_test.json
```

The final subset manifest points to validation-selected base, specialist, and joint endpoints. For `subset_31`, the report includes:

- base;
- relevant specialist;
- raw bias sum;
- bias mean;
- exposure-matched all-five joint;
- gold-token NLL;
- generation correctness and exact trace validity;
- EOS accuracy and generated length;
- Jensen-Shannon divergence and next-token argmax agreement to the joint.

Always preserve the associated `experiment_contract.json` and record its fingerprint with the evaluation output.

## Intermediate subset diagnostic

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_03.json \
  --split test \
  --examples-per-operator 64 \
  --out evaluations/surface_v4_seed_0_subset_03_test.json
```

Intermediate manifests deliberately have `joint_reference_checkpoint: null`. Their results support leakage, interference, stability, and task-accuracy diagnostics, not claims of equivalence to joint training.

## Global alpha

The primary raw condition uses `--alpha 1.0`. A global alpha may be selected on validation and then frozen for test:

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v4/seed_0/fusion_subsets/subset_31.json \
  --split validation \
  --alpha 0.5 \
  --examples-per-operator 64 \
  --out evaluations/surface_v4_seed_0_alpha_0_5_validation.json
```

Do not tune alpha on test. Input-dependent alpha, routing, and learned correction are separate experiments.

## Selected endpoints versus trajectories

Final subset manifests use `selected.pt`. To evaluate a step-matched checkpoint observation, use the corresponding manifest under:

```text
seed_<n>/fusion_checkpoint_grid/step_<step>/subset_<mask>.json
```

All specialist and joint checkpoints in one grid share the same optimizer-step index. The common base is the validation-selected parent checkpoint used to initialize those branches.

Report both analyses when relevant:

1. **validation-selected endpoint comparison** — practical best endpoint without using test;
2. **step-matched trajectory comparison** — how fusion changes over training time.

Do not silently substitute `final.pt` for `selected.pt`; state the endpoint policy in every result table.

## Legacy control evaluation

Surface-v3 remains an explicit identity-base/unanchored control. Its evaluation must use its own config, output tree, and manifests. Typed-v2 likewise remains separate. Never compare a v4 specialist with a v3 or v2 base.
