# Experiment Configurations

## Real-hardware CUDA smoke

Before any unattended pilot or production run, execute:

```bash
bash scripts/run_surface_v4_cuda_smoke.sh
```

The smoke configuration is:

```text
gpt_bias_fusion_factory_surface_v4_smoke.yaml
```

It requires an actual CUDA device and exercises the seven-model dependency graph, automatic micro-batch probing, production effective batch size, deterministic CUDA settings, weak Base training, full-domain inactive retention, validation-selected checkpoints, all-five fusion evaluation, and per-unit diagnostics. It runs only two optimizer steps and writes a verified completion marker under its dedicated smoke output tree. Smoke artifacts are operational evidence only, not scientific results.

## Required model-design pilot

Run the four one-seed conditions after the CUDA smoke passes:

```text
model_design_pilot_identity_unanchored.yaml
model_design_pilot_identity_retention.yaml
model_design_pilot_weak_unanchored.yaml
model_design_pilot_weak_retention.yaml
```

Use:

```bash
bash scripts/run_model_design_pilot.sh detach
```

The 2×2 pilot separates the effect of common-base construction from inactive-operator retention. It also uses validation-selected endpoints rather than assuming that the final optimizer step is best.

## Guarded production candidate

```text
gpt_bias_fusion_factory_surface_v4.yaml
```

This candidate uses a weak multitask base, retention-anchored specialists, strict experiment fingerprints, and validation-selected dependency endpoints. It is intentionally gated until the pilot reports support the choice:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

The production launcher revalidates the standalone smoke marker for the current Git revision and reruns the smoke when necessary unless `SKIP_SMOKE=1` is explicitly supplied.

## Legacy surface-v3 condition

```text
gpt_bias_fusion_factory_surface_v3.yaml
gpt_bias_fusion_factory_surface_v3_smoke.yaml
```

Surface v3 uses the identity common base and unanchored full fine-tuning. It is retained as an explicit experimental control and requires:

```bash
OPFUSION_ALLOW_LEGACY_SURFACE_V3=1
```

## Typed-token diagnostic ablation

```text
gpt_bias_fusion_factory_v2.yaml
gpt_bias_fusion_factory_v2_smoke.yaml
```

These profiles predict `<EQ_STEP>` and `<TRACE_STOP>` output classes. Their launcher requires:

```bash
OPFUSION_ALLOW_TYPED_V2=1
```

Never combine checkpoints across output trees or tokenizer policies. Surface-v4 output roots also contain a strict `experiment_contract.json`; a changed config or code revision must use a new output directory.

The complete methodological contract is in [`docs/experiment_protocol.md`](../../docs/experiment_protocol.md).
