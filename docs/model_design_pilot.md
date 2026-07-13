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

All four conditions use the same architecture, tokenizer, operator set, effective task batch, optimizer family, seed, training steps, evaluation sample count, and checkpoint-selection rule.

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

The pair audit also records specialist runtime choices and recovery state. Unequal micro-batches do not change the declared effective batch, but they change gradient-accumulation and floating-point order. OOM reductions, non-finite restarts, or learning-rate recovery differences are reported as interpretation warnings.

## Base definitions

### Identity control

The identity Base learns the shared surface protocol but not arithmetic transitions:

```text
<OP_*> expression <RESPONSE>
= expression <EOS>
```

### Weak multitask candidate

The weak Base receives verified arithmetic traces for all five operators, but only on a restricted domain:

```text
operand magnitude <= 8
term count <= 4
```

It learns shared reduction, equality, and EOS behavior without receiving the full specialist domain.

## Retention-anchored specialists

The anchored condition optimizes:

```text
L = L_task
  + lambda_KL * KL(p_base || p_specialist) on inactive operators
  + lambda_param * mean((theta_specialist - theta_base)^2)
```

The Base is frozen. KL is evaluated only on response-supervised positions.

Retention prompts are sampled from the **full inactive-operator domain**, not the weak Base training domain. Arithmetic labels in those batches are used only to define the teacher-forcing path and response mask for KL; inactive task cross-entropy is not added.

This is not a router and not a fusion-time corrector. It constrains how the specialist field is learned.

## Validation-selected endpoints

Each job retains `final.pt`, but dependency branches and final subset manifests use `selected.pt`:

```text
selected.pt = positive-step permanent checkpoint with minimum validation token NLL
```

Selection rules:

- specialist: its own operator validation NLL;
- Joint: mean validation NLL across operators;
- Base: Base validation NLL.

## Evaluation policy

The pilot evaluates **validation only**.

```text
pilot selection split: validation
reserved final splits: iid_test, operand_ood, length_ood
```

The reserved splits are not generated or inspected while selecting the model construction. This avoids using OOD examples as development feedback and then reporting the same finite domains as final evidence.

The canonical evaluator records a synthetic-data seed in every report. Pilot model-design configs use seed `701000`; normal final evaluation defaults to `700000`. The separate namespace is provenance protection, but split reservation—not seed separation alone—is what prevents leakage for finite OOD domains.

Each condition writes:

```text
<condition>_validation.json
<condition>_validation_units.json
```

The fusion report compares Base, Relevant Specialist, raw sum, bias mean, and matched Joint. The unit report measures every specialist relative to the Base on the same validation examples:

- Base-to-unit Jensen–Shannon divergence;
- Base-to-unit KL;
- argmax agreement;
- centered bias RMS and maximum absolute magnitude;
- aggregate inactive-unit means and maxima.

The gap between Relevant Specialist and all-five fusion measures total inactive interference. Per-unit diagnostics identify which inactive fields contribute to it.

## Experiment fingerprints

Every output root receives `experiment_contract.json`. The fingerprint includes:

- normalized run configuration;
- model-design controls;
- model and tokenizer configuration hashes;
- vocabulary hash;
- relevant training, hardened retention, seeded evaluation, diagnostics, and evaluation source hashes;
- Git commit when available.

A mismatched output directory is rejected before checkpoint reuse. Changing learning rate, Base mode, retention weights, data ranges, trainer code, tokenizer, evaluation namespace, or diagnostics requires a new output directory.

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
evaluations/model_design_pilot/index.json
```

A machine reboot stops the process. Run the same detached command again with the same checkout and configuration to resume from verified checkpoints.

Artifacts created before the current experiment-contract ABI must not be adopted. Move old `runs/model_design_pilot`, `audits/model_design_pilot`, and `evaluations/model_design_pilot` trees aside before starting this corrected pilot.

## Decision rule

Select the production construction from validation only. Compare, in this order:

1. Relevant Specialist validation accuracy;
2. raw-sum and bias-mean validation trace validity;
3. validation EOS stopping accuracy;
4. total all-five validation interference relative to Relevant Specialist;
5. per-unit inactive JSD, KL, argmax agreement, and centered-bias magnitude;
6. divergence and argmax agreement to the matched all-five Joint;
7. selected checkpoint step versus final step;
8. parameter displacement and retention logs;
9. pair-consistency result and specialist runtime warnings.

The weak-Base/retention candidate should advance only if it preserves relevant-specialist capability while reducing inactive drift or improving fusion stability. Retention tuning, if required, must use validation only. Do not inspect `iid_test`, `operand_ood`, or `length_ood` until the construction is fixed.

## Production gate

The production launcher requires an explicit acknowledgement:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

The environment variable is an operational safeguard, not evidence that the candidate passed the pilot. Preserve the pilot reports, evaluation seeds, pair-consistency audit, and experiment contracts with the final record.
