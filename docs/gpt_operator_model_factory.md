# GPT Bias-Fusion Model Factory v2

## Purpose

The factory creates the minimum controlled GPT checkpoint set needed for a first bias-fusion study. Model generation and fusion evaluation are separate stages.

## Required models

Each seed contains:

| Order | Job | Parent | Training distribution | Role |
|---:|---|---|---|---|
| 1 | `base.common` | random initialization | mixed expression-copy protocol | trained bias origin |
| 2 | `scalar.add` | `base.common/final.pt` | binary addition equality traces | specialist |
| 3 | `aggregation.sum` | `base.common/final.pt` | variable-length sum contraction traces | specialist |
| 4 | `scalar.neg` | `base.common/final.pt` | sign inversion equality traces | specialist |
| 5 | `scalar.min` | `base.common/final.pt` | minimum reduction traces | specialist |
| 6 | `scalar.max` | `base.common/final.pt` | maximum reduction traces | specialist |
| 7 | `joint.all_five.exposure_matched` | `base.common/final.pt` | exact balanced accumulation over all five families | joint reference |

The base copy objective exposes number tokens, expression structure, operator markers, `<RESPONSE>`, and `<TRACE_STOP>` without teaching operation outputs. This makes specialist-minus-base logits a better approximation to operator-specific change than specialist-minus-random logits.

## Loss definition

A record is divided into prompt and response:

```text
prompt:   <OP_AGG_SUM> expression <RESPONSE>
response: <EQ_STEP> state ... <TRACE_STOP> <EOS>
```

Labels before the first response prediction are `-100`. The task loss is:

```text
L_task = CE(response target tokens, model logits)
```

It is distinct from later fusion matching:

```text
L_match = D(p_joint, p_fused)
```

## Exposure matching

A specialist optimizer step sees 128 examples from one operator. The primary joint optimizer step sees 128 examples from each operator and averages the gradient over all 640 examples. Micro-batching and accumulation change memory use but not this exposure contract.

Checkpoint metadata records cumulative total examples and operator-specific counts. Same-step and same-exposure interpretations must not be conflated.

## VRAM adaptation

With automatic micro-batching enabled, the job probes candidate sizes on the target CUDA device. An OOM retries the same logical step after restoring RNG state and reducing only the micro-batch. Effective batch size and operator exposure remain unchanged.

## Recovery contract

Permitted automatic actions:

- reduce micro-batch after CUDA OOM;
- increase accumulation implicitly to preserve effective batch;
- resume the most recent `last.pt` after process failure;
- reduce learning-rate scale after a non-finite loss/gradient, within fixed limits.

Forbidden automatic actions:

- changing model architecture or tokenizer;
- changing data ranges or operator set;
- dropping a failed operator from the queue;
- changing effective batch or per-operator exposure;
- silently replacing raw fusion with routing or calibration.

All actions are append-logged.

## Checkpoint contract

Sixteen logical observation points are derived from configured fractions. `last.pt` is a rolling resume checkpoint and is not counted as an additional scientific observation.

Each permanent checkpoint includes model/optimizer/RNG state, ABI hashes, parent identities, losses, generation metrics, exposure counts, runtime settings, and parameter deltas from both the trained parent base and random initialization.

## Approval conditions for unattended production

1. Arch/Linux environment reports CUDA and the expected GPU.
2. All tests pass.
3. V2 tokenizer size is 2,066 and model size is 863,184 parameters.
4. A hand-inspected batch confirms prompt labels are masked.
5. CPU/unit smoke and target-GPU smoke complete.
6. Base completion is enforced before dependent jobs.
7. Joint per-operator exposure is exactly balanced.
8. Resume is tested from an interrupted job.
9. At least 15 GiB of free disk is available.
10. The resolved plan is saved before training.

## Outputs for the fusion stage

For every seed, `model_inventory.json`, `fusion_subsets/index.json`, and the checkpoint grid provide the exact base, specialist, and joint paths. The runtime experiment can then calculate:

```text
B_k(x) = z_k(x) - z_base(x)
z_fused(x) = z_base(x) + F(B_k(x) for k in subset)
```

The factory itself does not claim that `F` succeeds.
