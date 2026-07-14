# Math Operator Units

This repository builds controlled GPT checkpoints for testing **logit-space bias fusion**. Mathematical operators are used because inputs, intermediate transformations, final values, stopping behavior, and invalid traces can be generated and verified exactly.

For one shared prefix `x`:

```text
B_k(x) = z_k(x) - z_base(x)
z_fused(x) = z_base(x) + F(B_1(x), ..., B_n(x))
```

Raw addition is a hypothesis, not an assumption. The primary factory contains no hidden input router or learned fusion corrector.

## Surface policy

The model predicts ordinary arithmetic punctuation, equality, and EOS:

```text
<OP_AGG_SUM> 1 + 2 + 3 + 4 <RESPONSE>
= 3 + 3 + 4
= 6 + 4
= 10
<EOS>
```

`<EQ_STEP>` and `<TRACE_STOP>` are implementation aliases, not separate output classes.

## Required execution order

### 1. Bootstrap and real-hardware smoke

```bash
git clone https://github.com/Unjuno/math-operator-units.git
cd math-operator-units
bash scripts/bootstrap_arch_linux.sh
bash scripts/run_surface_v4_cuda_smoke.sh
```

The smoke requires an actual CUDA device and exercises the Base, five Specialists, Joint, checkpoint selection, fusion evaluation, and unit diagnostics. It is operational evidence only.

### 2. Model-design pilot

Run the one-seed 2x2 pilot:

| Base | Specialist construction |
|---|---|
| identity | unanchored |
| identity | retention-anchored |
| weak multitask | unanchored |
| weak multitask | retention-anchored |

```bash
bash scripts/run_model_design_pilot.sh detach
bash scripts/status_model_design_pilot.sh
```

The pilot is validation-only. It reserves `iid_test`, `operand_ood`, and `length_ood`, measures inactive-unit drift, and requires exact selected Base/Joint equality across each retention/unanchored pair. If the result is ambiguous, all four conditions are repeated on one additional seed.

### 3. Guarded production

After validation supports and freezes the construction:

```bash
OPFUSION_ALLOW_V4_PRODUCTION=1 \
  bash scripts/run_bias_fusion_factory_surface_v4.sh \
    configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml \
    detach
```

Production uses seeds 0, 1, and 2. Each seed creates one Base, five Specialists, and one exposure-matched all-five Joint: 21 models total. All branches start from the validation-selected Base checkpoint.

## Current surface-v4 candidate

The implemented candidate uses:

- a weak multitask Base trained on all five operators only for operand magnitude at most 8 and at most four terms;
- five full-domain Specialists over operands -64 to 64 and three to eight terms;
- teacher-KL retention on full-domain inactive-operator prompts;
- a small parameter anchor to the frozen Base;
- deterministic CUDA semantics and strict experiment fingerprints;
- validation-selected `selected.pt` endpoints while retaining `final.pt` and checkpoint trajectories.

For Specialist `k`:

```text
L = L_task
  + lambda_KL * KL(p_base || p_k) on inactive operators
  + lambda_param * mean((theta_k - theta_base)^2)
```

The retention path is training regularization, not a router or fusion-time corrector.

## Data and optimization

IID problems are assigned by a stable hash of `(operator, initial values)`:

```text
0-69   train
70-84  validation
85-99  IID test
```

Training mixes full traces 60%, continuations 25%, and terminal-to-EOS examples 15%. SUM, MIN, and MAX use deterministic randomized valid adjacent reductions during training and canonical left-fold traces during validation/final evaluation.

The effective task batch is 128. GPU probing considers micro-batches `128, 64, 32, 16, 8, 4`; gradient accumulation preserves the effective batch. Recovery includes same-step OOM retry, bounded non-finite restart, `last.pt` resume, watchdog restart, duplicate-run locking, and optional `systemd-inhibit`.

## Active preregistered plan v2

The active plan is [`docs/experiment_plan_v2.md`](docs/experiment_plan_v2.md), with machine-readable contract [`configs/experiments/experiment_plan_v2.yaml`](configs/experiments/experiment_plan_v2.yaml).

Raw all-five sum at `alpha=1.0` remains the confirmatory condition. After all three production seeds and endpoints are frozen, but before final data are generated, the plan calibrates a complexity-ordered rescue ladder on validation:

1. global shrinkage;
2. RMS equalization or clipping;
3. static nonnegative weighted mean;
4. deterministic consensus-tempered decoding.

Selection uses three-fold leave-one-training-seed-out validation with seed `703000` and 128 examples per operator. The held-out seed is excluded from every fitted parameter and scale statistic. A learned router or logit corrector is a separate versioned study.

## Final fusion evaluation is locked

Surface-v4 final splits cannot be evaluated directly after one production seed. Calibration must first freeze:

```text
evaluations/fusion_calibration/final_authorization.json
```

The evaluator checks the active-plan hash, Git commit, experiment fingerprint, completion of seeds 0/1/2 and all calibration folds, calibration settings, and final settings. Missing or mismatched authorization fails closed.

After authorization is created by the calibration implementation, the canonical form is:

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

The primary table always retains Relevant Specialist, raw sum, bias mean, and matched Joint. A frozen rescue mixer is reported only as a secondary condition. The fallback mixer implementation and authorization generator must be completed before final IID/OOD evaluation.

## Experiment integrity

Every output root contains `experiment_contract.json`, covering normalized configuration, model design, model/tokenizer hashes, vocabulary, relevant code, and Git revision. A changed contract cannot reuse an existing output root.

Do not mix checkpoints or manifests across output trees or tokenizer policies.

## Legacy controls

Identity-Base/unanchored surface v3 requires `OPFUSION_ALLOW_LEGACY_SURFACE_V3=1`. Typed control-token v2 requires `OPFUSION_ALLOW_TYPED_V2=1`.

## Documentation

- [`docs/model_design_pilot.md`](docs/model_design_pilot.md)
- [`docs/experiment_plan_v2.md`](docs/experiment_plan_v2.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/fusion_evaluation_runbook.md`](docs/fusion_evaluation_runbook.md)
- [`docs/arch_linux_runbook.md`](docs/arch_linux_runbook.md)
- [`docs/logit_bias_semantics.md`](docs/logit_bias_semantics.md)

This is a controlled synthetic testbed, not evidence of arbitrary natural-language model composition.
