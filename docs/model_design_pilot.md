# Model-Design Pilot

## Why the pilot exists

The first surface experiment used an identity common base and unrestricted full-parameter specialist fine-tuning. That construction is executable, but a failed fusion result is ambiguous:

1. every specialist may contain the same large correction that cancels the identity policy;
2. inactive operators are unconstrained and may drift far from the base;
3. the final fixed training step may be worse than an earlier checkpoint.

The pilot separates these causes before the three-seed production expense.

## 2×2 conditions

| Base | Specialist regularization | Config |
|---|---|---|
| identity | none | `model_design_pilot_identity_unanchored.yaml` |
| identity | retention | `model_design_pilot_identity_retention.yaml` |
| weak multitask | none | `model_design_pilot_weak_unanchored.yaml` |
| weak multitask | retention | `model_design_pilot_weak_retention.yaml` |

All four conditions use the same architecture, tokenizer, operator set, effective task batch, optimizer family, seed, training steps, evaluation sample counts, and checkpoint-selection rule.

## Paired-control requirement

Retention and unanchored conditions are intended to differ only in specialist regularization. The pilot therefore uses deterministic CUDA settings:

```text
deterministic_algorithms: true
allow_tf32: false
CUBLAS_WORKSPACE_CONFIG=:4096:8
flash SDPA: disabled
memory-efficient SDPA: disabled
math SDPA: enabled
```

The identity pair independently recomputes the same identity Base and Joint; the weak pair independently recomputes the same weak Base and Joint. At the end, `opfusion-audit-pilot-pairs` hashes the selected model state of `base.common` and `joint.all_five.exposure_matched` and requires exact equality within each pair. A mismatch is a scientific failure, exit status 67, and the watchdog does not retry it blindly.

The pair audit also records specialist micro-batch choices. Unequal micro-batches do not change the declared effective batch, but they do change gradient-accumulation and floating-point order, so they are reported as interpretation warnings.

## Base definitions

### Identity control

The identity base learns the shared surface protocol but not arithmetic transitions:

```text
<OP_*> expression <RESPONSE>
= expression <EOS>
```

This remains a useful control because it maximizes the amount of task behavior that must be represented in each specialist field.

### Weak multitask candidate

The weak base receives verified arithmetic traces for all five operators, but only on a restricted domain:

```text
operand magnitude <= 8
term count <= 4
```

It therefore learns shared reduction/equality/EOS behavior without receiving the full specialist domain. The full-domain specialist is expected to add capability rather than repeatedly cancel an identity policy.

## Retention-anchored specialists

The anchored condition optimizes:

```text
L = L_task
  + lambda_KL * KL(p_base || p_specialist) on inactive operators
  + lambda_param * mean((theta_specialist - theta_base)^2)
```

The base model is frozen. KL is evaluated only on response-supervised positions.

Retention prompts are sampled from the **full inactive-operator domain**, not the weak Base training domain. This matters for the weak-multitask condition: otherwise the regularizer would constrain only operands within ±8 and at most four terms while the fusion evaluation covers the full specialist range. Arithmetic labels in these batches are used only to identify response positions for teacher KL; they are not added as inactive task cross-entropy.

This is not a router and not a fusion corrector. It changes how the specialist is trained so that its bias field is more localized.

## Validation-selected endpoints

Each job retains `final.pt`, but the dependency graph and final subset manifests use `selected.pt`:

```text
selected.pt = positive-step permanent checkpoint with minimum validation token NLL
```

Selection rules:

- specialist: its own operator validation NLL;
- joint: mean validation NLL across operators;
- base: base validation NLL.

The IID test bucket is never used for checkpoint or model-design selection.

## Evaluation splits

The pilot evaluates:

```text
validation
operand_ood
length_ood
```

It intentionally does **not** evaluate the IID `test` bucket. That bucket remains reserved for the production experiment after the model-construction rule is fixed.

For each pilot split, two reports are written:

```text
<condition>_<split>.json
<condition>_<split>_units.json
```

The first compares Base, Relevant Specialist, raw sum, bias mean, and matched Joint. The second measures every specialist relative to the Base on every target operator using teacher-forced response positions:

- Base-to-unit Jensen–Shannon divergence;
- Base-to-unit KL;
- argmax agreement;
- centered bias RMS and maximum absolute magnitude;
- inactive-unit aggregate means and maxima.

The gap between Relevant Specialist and all-five fusion measures total inactive interference. Per-unit diagnostics identify which inactive fields are large or distributionally divergent.

## Experiment fingerprints

Every output root receives `experiment_contract.json`. The fingerprint includes:

- normalized run configuration;
- model-design controls;
- model and tokenizer configuration hashes;
- vocabulary hash;
- relevant training, hardened retention, diagnostics, and evaluation source hashes;
- Git commit when available.

A mismatched output directory is rejected before checkpoint reuse. Changing learning rate, base mode, retention weights, data ranges, trainer code, tokenizer, or diagnostics requires a new output directory.

## Execution

```bash
bash scripts/run_model_design_pilot.sh detach
```

Status:

```bash
bash scripts/status_model_design_pilot.sh
```

Outputs:

```text
runs/model_design_pilot/<condition>/
audits/model_design_pilot/<condition>.json
audits/model_design_pilot/pair_consistency.json
evaluations/model_design_pilot/<condition>_validation.json
evaluations/model_design_pilot/<condition>_validation_units.json
evaluations/model_design_pilot/<condition>_operand_ood.json
evaluations/model_design_pilot/<condition>_operand_ood_units.json
evaluations/model_design_pilot/<condition>_length_ood.json
evaluations/model_design_pilot/<condition>_length_ood_units.json
evaluations/model_design_pilot/index.json
```

A machine reboot stops the process. Run the same detached command again with the same checkout and configuration to resume from verified checkpoints.

## Decision rule

Choose the production construction from validation and declared OOD diagnostics only. Compare, in this order:

1. relevant-specialist validation accuracy;
2. raw-sum and bias-mean trace validity;
3. EOS stopping accuracy;
4. total all-five interference relative to the Relevant Specialist;
5. per-unit inactive JSD, KL, argmax agreement, and centered-bias magnitude;
6. Jensen–Shannon divergence and argmax agreement to the matched all-five Joint;
7. selected checkpoint step versus final step;
8. parameter displacement and retention logs;
9. pair-consistency result and micro-batch warnings.

The weak-base/retention candidate should advance only if it preserves relevant-specialist capability while reducing inactive drift or improving fusion stability. If retention suppresses specialist capability, tune its global coefficient on validation data in a separate pilot. Do not inspect the reserved IID test bucket while making that choice.

## Production gate

The production launcher requires an explicit acknowledgement:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

The environment variable is an operational safeguard, not evidence that the candidate passed the pilot. Preserve all pilot reports and the pair-consistency audit with the final experiment record.
