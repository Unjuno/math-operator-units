# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

This is the authoritative recovery checkpoint for the active research branch. Experiment computation is performed in the container; GitHub is used for persistence, review, and reproducibility.

Start/restart map:

- [`docs/research/README.md`](research/README.md)
- [`docs/research/BRANCHES.md`](research/BRANCHES.md)
- draft PR #40 on `experiment/unseen-source-generalization`

## Goal

Find a composition mechanism that can accept previously unseen computational/model specialists, infer when/how they apply without a hand-written `operator -> source` table, and compose them into unseen programs.

Current candidate stack:

`interface normalization -> behavioral semantic grounding -> source applicability -> source-specific ABI/action adapter -> common action/state-transition distribution -> sparse probability fusion -> program composition`

The main scientific shift is that **shared raw token logits are no longer treated as the universal fusion interface**.

## Established earlier results

### Shared-token arithmetic specialists

- ADD/SUM/MIN/MAX specialists were usable; NEG was consistently weak/failed.
- Oracle two-stage composition reached 191/192 E2E at strong operator conditioning.
- A learned controller trained by downstream token likelihood reached up to 191/192 without operator-label loss, but explicit operator tokens/stage boundaries remained available.
- Identity-free local logit-field statistics could recognize some unseen-source capability but caused severe irrelevant-source insertion damage.
- Independent residual/null admission improved insertion safety but did not solve stable-wrong false positives.
- Semantic-view behavioral consensus and later task-conditioned metamorphic behavior were much better relevance signals than static field morphology.

### Interface discovery / opaque semantics

- Self-normalized command specificity fixed an OOD sensitivity calibration failure and routed a newly introduced specialist reliably.
- Unlabeled prompt structure could discover a categorical control family and infer command slots without a hand-written absolute position/token list in the tested grammar.
- Random per-prompt command movement/wrapping was normalized back to native ABI with no paired composition loss.
- Completely arbitrary opaque command codes were not zero-shot identifiable. Minimal behavioral anchors resolved the permutation symmetry. This is treated as an identifiability boundary, not merely a router weakness.
- Routing stayed about 99.7% correct through depth 1–5 while dense raw-logit mixing accumulated execution errors. Sparse top-dominant fusion was substantially safer.

## Output-ABI boundary experiments

Naïve coordinate-wise interpretation fails when a source's output token meanings are permuted. An oracle adapter restores performance, so shared output coordinates are not intrinsically necessary.

Few-shot behavior cannot identify an arbitrary whole-action random codebook on unseen actions. In contrast, low-complexity/compositional output ABIs can be inferred efficiently:

- compositional foreign numeral codec: unknown radix, digit-symbol permutation, direction, variable sequence length;
- 16 random anchors: ~93.5% unseen-value decode;
- 32 random anchors: ~99.53%;
- actively selected 8 probes identified 500/500 random codecs and generalized to longer unseen sequences.

A black-box source exposing only generated token sequences also composed through the inferred common-action adapter. Internal logits/hidden states are therefore not required in principle when the external ABI has learnable structure.

## Independently trained neural specialists with private tokenizers/heads

Five neural specialists (`add`, `sub`, `min`, `max`, `xor`) were trained on a 64-state domain. Every source had:

- a private permutation of 4096 input-pair tokens;
- private output radix/digit symbols/direction;
- variable-length private output sequences;
- potentially different output-head dimension.

Across three independent pools:

- source competence: 13/15 at 4096/4096 exact, remaining 2 at 4095/4096;
- behavior-diverse codec recovery: 15/15 structurally correct;
- held-out common-action decode: 17,999/18,000;
- hard depth-1..5 composition: 4,499/4,500.

Reproduction: `scripts/experiment_neural_private_tokenizer_adapter.py`.

## Soft fusion across incompatible output heads

Direct token-logit fusion is not dimensionally defined when private head sizes differ. Each source's private sequence likelihood was therefore mapped to a normalized probability distribution over common actions.

Across three pools:

- common-action probability mixture, primary/secondary `0.6/0.4`: **560/560** E2E;
- equal probability mixture `0.5/0.5`: **168/560**;
- log-probability pooling `0.8/0.2`: **521/560**.

Therefore both the common representation and the fusion algebra matter. Arithmetic mixing of calibrated common-action probabilities was much more robust than equal mixing or product-of-experts style pooling in these screens.

Reproduction: `scripts/experiment_heterogeneous_action_space_soft_fusion.py`.

## Mixed neural architectures

A later screen mixed autoregressive GRU specialists and non-autoregressive MLP sequence specialists while retaining private input/output ABIs.

The first NAR design underfit some tasks even though ABI recovery succeeded, cleanly separating source-competence failure from adapter/fusion failure. With position-specific NAR heads, competence saturated.

Across three mixed-architecture pools:

- source exact: 14/15 at 4096/4096; 1/15 at 4095/4096;
- ABI structural recovery: 15/15;
- common-action decode: 61,439/61,440;
- hard depth-1..5: 3,000/3,000;
- common-action probability `0.6/0.4`: 600/600;
- equal `0.5/0.5`: 200/600;
- log-probability `0.8/0.2`: 537/600.

Shared decoding architecture is therefore not required in the controlled screen.

Reproduction: `scripts/experiment_mixed_architecture_common_action_fusion.py`.

## Latest result: held-out attention architecture/source insertion

A new controlled experiment added a sixth operation, `mul`, implemented **only by an attention-based Transformer specialist**. The five seen sources remained a GRU/NAR-MLP mixture.

Protocol:

1. train the five seen sources plus the Transformer source with fully private input pair vocabularies and private variable-length output ABIs;
2. infer each source ABI from 18 behavioral action anchors;
3. calibrate the source-admission threshold **only on the five seen operators/sources** (`add/sub/min/max/xor`);
4. keep the Transformer/MUL source completely absent from that threshold calibration;
5. present each opaque command through fixed behavioral `(input -> expected common action)` probes;
6. insert the Transformer using only its behavioral ABI adapter and the same command/source compatibility rule;
7. compose random depth-1..5 programs in common-action space.

Three independent pools, seeds 310/311/312:

- all 18 source instances (6 sources × 3 pools) reached exact standalone competence;
- all 18 private ABIs were structurally recovered with 100% common-action decode;
- seen-only admission threshold was `0.65` in all three pools;
- held-out MUL/Transformer compatibility score was `1.0`, second-best source `0.3`, so it was admitted without threshold retuning;
- hard routing/composition: **100% at every depth 1–5 in all three pools**;
- common-action probability fusion `0.6/0.4`: **100% at every depth 1–5 in all three pools**;
- equal `0.5/0.5` fusion again failed sharply: depth-5 E2E was approximately `0.233`, `0.275`, `0.233` across the three pools.

This removes another controlled assumption: the newly inserted useful source may use an attention architecture that was absent from admission-threshold calibration.

Reproduction: `scripts/experiment_heldout_transformer_private_abi_insertion.py`.
Aggregate: `docs/heldout_transformer_private_abi_insertion_aggregate_v1.json`.

## Current interpretation

The strongest candidate universal interface is a probability field over **actions/state transitions**, not shared token logits.

The current evidence supports separating five problems that were previously conflated:

1. **source competence** — can the source perform its computation at all?;
2. **semantic grounding/applicability** — when should an anonymous/new source apply?;
3. **ABI alignment** — how do private source outputs map to a common action representation?;
4. **fusion algebra** — how should multiple common-action distributions be combined?;
5. **program execution** — how are transitions segmented/composed through depth?

A useful source can differ in tokenizer, output head, sequence convention, GRU/MLP/attention architecture, and still participate if its behavioral ABI is inferable and its action distribution can be mapped into the common space.

## Claim boundary

These are controlled synthetic neural experiments. They do **not** establish arbitrary fusion of unrelated pretrained LLMs.

Remaining major assumptions:

- an explicit common action/state domain is still externally defined;
- source I/O ABIs have compressible/inferable structure;
- arbitrary semantic permutations require behavioral grounding;
- behavioral probes currently expose expected common actions;
- stage boundaries are externally supplied;
- position-free grammar normalization, opaque commands, heterogeneous source insertion, nuisance-source abstention, and common-action fusion have not yet all been combined in one single E2E run;
- no unrelated pretrained LLM families have yet been connected.

## Next experiments — priority order

1. **Joint integration run:** combine opaque commands + position-free interface normalization + seen-only applicability calibration + held-out Transformer/private ABI insertion + common-action probability fusion in one experiment.
2. Add one or more nuisance/decoy sources to the joint run and require exact abstention while preserving useful held-out-source gains.
3. Remove externally supplied stage boundaries by learning transition termination/segmentation.
4. Replace the explicit integer action vocabulary with a learned latent transition representation and test behavioral alignment across independently learned sources.
5. Only then move to unrelated pretrained model families through an explicit behavioral task interface.

## Repository/branch policy

- Active research branch: `experiment/unseen-source-generalization` / draft PR #40.
- Other `experiment/*` branches are historical unless explicitly reactivated.
- `*-ci-base` and `*-runner` branches are obsolete artifacts of the old Actions-based experiment workflow.
- Do not create a new branch for each screening run; use the active branch until the central scientific claim changes.

## Scientific decision

Do not return to dense raw-logit addition as the main direction unless a new experiment specifically motivates it. Current evidence favors **behaviorally grounded composition in a common action/state-transition space with source-specific ABI adapters and top-dominant probability fusion**.
