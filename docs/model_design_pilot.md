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

All four conditions use the same architecture, tokenizer, operator set, effective task batch, optimizer family, seed, training steps, evaluation splits, and checkpoint-selection rule.

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

The base model is frozen. KL is evaluated only on response-supervised positions. Task examples still determine the specialist's active capability; retention constrains its behavior on the other four operator families.

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

Test metrics are never used for checkpoint selection.

## Experiment fingerprints

Every output root receives `experiment_contract.json`. The fingerprint includes:

- normalized run configuration;
- model-design controls;
- model and tokenizer configuration hashes;
- vocabulary hash;
- relevant training/evaluation source hashes;
- Git commit when available.

A mismatched output directory is rejected before checkpoint reuse. Changing learning rate, base mode, retention weights, data ranges, trainer code, or tokenizer requires a new output directory.

## Execution

```bash
bash scripts/run_model_design_pilot.sh detach
```

Outputs:

```text
runs/model_design_pilot/<condition>/
audits/model_design_pilot/<condition>.json
evaluations/model_design_pilot/<condition>_validation.json
evaluations/model_design_pilot/<condition>_test.json
evaluations/model_design_pilot/index.json
```

## Decision rule

Do not select a condition from training loss alone. Compare, in this order:

1. relevant-specialist validation and test accuracy;
2. raw-sum and bias-mean trace validity;
3. EOS stopping accuracy;
4. inactive interference on each operator;
5. Jensen-Shannon divergence and argmax agreement to the matched all-five joint;
6. selected checkpoint step versus final step;
7. parameter displacement and retention logs.

The weak-base/retention candidate should advance only if it preserves specialist accuracy while reducing inactive interference or improving fusion stability. If retention suppresses specialist capability, tune its global coefficient on validation data in a separate pilot; do not tune on test.

## Production gate

The production launcher requires an explicit acknowledgement:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

The environment variable is an operational safeguard, not evidence that the candidate has passed the pilot. Preserve the pilot reports with the final experiment record.
