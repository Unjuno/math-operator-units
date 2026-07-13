# GPT Operator Model Factory v1

## 1. Purpose

This document defines the executable model-generation stage for the GPT fusion experiment. The stage ends when reproducible checkpoints exist. It does not attempt to solve routing or calibration.

## 2. Models produced

For every seed, all six models start from the same `shared_initial.pt`.

| Job | Training distribution | Role |
|---|---|---|
| `scalar.add` | binary addition traces | independent unit |
| `aggregation.sum` | variable-length contraction traces | independent unit |
| `scalar.neg` | sign inversion traces | independent unit |
| `scalar.min` | pairwise minimum contraction traces | independent unit |
| `scalar.max` | pairwise maximum contraction traces | independent unit |
| `joint.all_five` | balanced union of all five families | joint reference |

Five independent units produce `2^5 = 32` runtime subsets. These are manifests, not separately trained checkpoints.

## 3. Architecture contract

The production model is `gpt_operator_1m_v1`:

| Field | Value |
|---|---:|
| Vocabulary size | 2,064 |
| Hidden width | 112 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Feed-forward width | 448 |
| Context length | 128 |
| Weight tying | enabled |
| Parameter count | 848,624 |
| Hard upper bound | 1,000,000 |

The runner constructs the model and refuses to train when the actual parameter count exceeds the bound.

## 4. Tokenizer contract

`operator_experiment_v1` is independent of the large planning tokenizer. It includes:

- four mandatory sequence tokens: PAD, BOS, EOS, UNK;
- explicit equivalence and stop tokens;
- four structural tokens used by the synthetic expressions;
- atomic integers from -1024 through 1024;
- exactly five operator-family tokens.

It contains no blank or reserved operator slots. The ordered vocabulary is hashed and written into every checkpoint.

## 5. Trace construction

The dataset generator is deterministic by `(seed, split, step, sample_index, operator_id)`. No mutable corpus is required, and a resumed run regenerates the same batch for the same step.

The main sequence format is a contractive equality trace. A resolved prefix is replaced by its value and is not copied into the next state.

```text
<OP_AGG_SUM> 1 + 2 + 3 + 4
<EQ_STEP> 3 + 3 + 4
<EQ_STEP> 6 + 4
<EQ_STEP> 10
<TRACE_STOP>
```

Minimum and maximum use the same contraction principle over bracketed lists. Binary addition and negation use a single verified step.

## 6. CUDA and numerical policy

Production configuration requires CUDA and defaults to FP32. BF16 is supported only when explicitly selected and the CUDA device reports BF16 support. The run manifest records:

- CUDA device name;
- PyTorch and Python versions;
- precision mode;
- parameter count;
- model and tokenizer metadata.

CPU is accepted only through the explicit `--allow-cpu` smoke-test switch.

## 7. Checkpoint policy

Permanent checkpoints are saved at configured early and periodic steps. Each record includes:

- model parameters;
- optimizer state;
- completed training step;
- RNG state;
- train and validation losses;
- per-operator validation loss;
- distance from the shared initialization;
- model/tokenizer ABI metadata.

`last.pt` is atomically replaced at every checkpoint so an interrupted job can resume. `checkpoint_index.jsonl` is append-only and supports later trajectory analysis.

The default permanent steps are:

```text
0, 100, 300, 1,000, 3,000, 10,000, 30,000, 100,000, 200,000
```

A checkpoint is also retained every 10,000 steps.

## 8. Comparison target

The factory produces the data needed for a later loss-matching experiment.

### Variable table

| Symbol | Meaning | SI unit | Definition | Domain/assumption | Type |
|---|---|---:|---|---|---|
| `x` | current token prefix | 1 | fixed-tokenizer sequence | length at most 128 | integer vector |
| `p_joint` | joint model next-token distribution | 1 | softmax of joint logits | same ABI | probability vector |
| `p_fused` | composed next-token distribution | 1 | output of a candidate fusion rule | same ABI | probability vector |
| `L_match` | distribution matching loss | 1 | divergence between `p_joint` and `p_fused` | nonnegative | scalar |

The later search asks whether a fusion rule can drive `L_match` toward zero on the joint data distribution while preserving operator-specific behavior. This repository does not assume existence in advance; it creates the checkpoints required to test it.

Dimensional check: both distributions and the divergence are dimensionless, so the comparison is unit-consistent.

## 9. Calibrator status

No corrector is trained by this factory. This is deliberate.

Later experiments may compare:

1. raw composition;
2. oracle applicability;
3. learned scalar reliability;
4. token-wise reliability;
5. removal-field correction.

Keeping calibration outside the model factory prevents a router or corrector from being mistaken for evidence that the underlying GPT bias fields are composable.

## 10. Minimal verification

### H

The six-job factory completes on CUDA, remains under one million parameters, and produces resumable checkpoints with deterministic data generation and comparable shared initialization.

### T

Run the unit tests, one short CUDA smoke job, then the configured three-seed batch.

### D

- PASS: all six jobs per seed complete, checkpoint metadata agree, and 32 subset manifests are emitted.
- FAIL: a parameter-limit, vocabulary-hash, CUDA, resume, or checkpoint-integrity check fails.
- UNCERTAIN: training completes but loss trajectories are unstable across seeds; this affects the later fusion experiment, not factory correctness.

### C

Likely failure modes include CUDA unavailability, disk exhaustion, incompatible PyTorch builds, corrupted resume files, numerical divergence, or a tokenizer/model vocabulary mismatch.

### U

Primary uncertainty sources are initialization seed, stochastic dropout, optimizer trajectory, CUDA kernel nondeterminism, and finite validation batches. Full checkpoints permit these effects to be measured after training.
