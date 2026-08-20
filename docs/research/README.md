# Operator-unit fusion research index

This directory is the entry point for the active **Paraphrase-consensus biased decoding / operator-unit fusion** research line.

## Operations / recovery

- Experiments run in the **container**. GitHub is persistence, review, and reproducibility only.
- Active branch: `experiment/unseen-source-generalization`.
- Active draft: PR #40.
- Branch status/restart rules: [`BRANCHES.md`](BRANCHES.md).
- Authoritative checkpoint: [`../container_research_checkpoint_2026-08-20.md`](../container_research_checkpoint_2026-08-20.md).
- Machine checkpoint: [`../container_research_checkpoint_2026-08-20.json`](../container_research_checkpoint_2026-08-20.json).

## Current research object

The candidate composition stack is:

`interface discovery/normalization -> behavioral semantic grounding -> source applicability -> source-specific ABI/action adapter -> common action/state-transition distribution -> sparse probability fusion -> program composition`

The central shift is that **shared token logits are not treated as the universal fusion interface**. Current evidence instead favors source-specific behavioral adapters into a common action/state-transition probability space.

## Reproduction map

Historical identity-free logit-field baseline:

- `src/opfusion/fusion_unseen_source_generalization.py`
- `tests/test_fusion_unseen_source_generalization.py`

Current controlled heterogeneous-source screens:

- `scripts/experiment_neural_private_tokenizer_adapter.py`
- `scripts/experiment_heterogeneous_action_space_soft_fusion.py`
- `scripts/experiment_mixed_architecture_common_action_fusion.py`
- `scripts/experiment_heldout_transformer_private_abi_insertion.py`
- `scripts/experiment_joint_opaque_positionfree_heldout_transformer.py`
- `scripts/experiment_joint_competent_irrelevant_plugin.py`

Key machine-readable aggregates:

- `docs/container_research_checkpoint_2026-08-20_mixed_architecture.json`
- `docs/heldout_transformer_private_abi_insertion_aggregate_v1.json`
- `docs/joint_opaque_positionfree_heldout_transformer_aggregate_v2.json`
- `docs/joint_competent_irrelevant_nuisance_aggregate_v2.json`

## Headline results

1. **Private tokenizers/output ABIs are compatible with composition when behaviorally inferable.** In the 64-state neural screen, behavior-diverse calibration recovered 15/15 private codecs; held-out decode was 17,999/18,000.
2. **Incompatible output-head dimensions can be fused after action alignment.** Common-action probability mixing `0.6/0.4` achieved 560/560 programs across three heterogeneous-head pools; equal `0.5/0.5` collapsed to 168/560.
3. **Shared neural architecture is not required in the controlled setting.** GRU + NAR-MLP pools achieved 3,000/3,000 hard depth-1..5 programs and 600/600 at common-action `0.6/0.4`.
4. **A held-out attention architecture can be inserted after seen-only admission calibration.** A private-ABI Transformer implementing new `mul`, absent from admission-threshold calibration, was correctly inserted across seeds 310/311/312; hard and `0.6/0.4` composition were perfect in the initial held-out-architecture screen.
5. **The main components now work jointly.** In the integrated screen, every operation is represented by a random opaque two-token code. The unlabeled interface corpus contains only the five seen codes, spread over seven command positions/wrappers; the held-out MUL code never occurs there. The system discovers the seen command family, infers the unknown command span from learned grammar skeletons, grounds code/source semantics from behavior, recovers private ABIs, admits the unseen Transformer under the seen-only threshold, and composes depth-1..5 programs. Hard execution was perfect in all three pools; common-action `0.6/0.4` was **359/360 = 99.72%**.
6. **A useful failure exposed the remaining interface issue.** A first integrated version used only local bigram grammaticality and achieved only ~57% held-out span accuracy because operand-adjacent spans could be locally plausible command sites. Replacing it with full command-placeholder grammar skeleton matching restored 100% span detection across all seven tested grammars.
7. **A competent but irrelevant plugin can receive exact zero sparse contribution.** A separate private-ABI GRU accurately learned `JUNK(a,b)=(7a+11b+3) mod 32`. Across three pools it scored only 0–0.2 against the six requested behaviors, never entered top-2 (0/18 command/pool combinations), and hard plus `0.6/0.4` results were paired-identical before/after insertion.
8. **Opaque semantic permutations are not zero-shot identifiable.** Behavioral grounding remains necessary when command meanings are arbitrarily permuted; this is an identifiability boundary rather than a mere routing defect.

## Claim boundary

These are controlled synthetic neural screens, not arbitrary fusion of unrelated pretrained LLMs. Important remaining assumptions:

- an explicit common action/state domain exists;
- behavioral probes provide common-action supervision for semantic/ABI grounding;
- source ABIs have inferable structure;
- all seven tested grammar skeletons are represented in the unlabeled seen-only interface corpus;
- stage boundaries are externally supplied;
- the nuisance source tested so far is behaviorally well separated, not near-confusable/adversarial.

## Next experiments

Priority order:

1. **Near-confusable nuisance:** train a competent plugin whose behavior intentionally overlaps one real operator enough to approach or enter the top-2 set, then test whether applicability calibration can abstain without sacrificing the useful secondary contribution.
2. **Unseen grammar:** with the discovery rule frozen, hold out one or more wrapper/position skeletons entirely from the unlabeled interface corpus and test structural extrapolation rather than interpolation.
3. **Stage-boundary removal:** infer transition termination/segmentation rather than supplying each program stage externally.
4. **Latent transition space:** replace explicit integer actions with a learned state-transition representation aligned from behavior.
5. Only after those controlled tests, move to unrelated pretrained model families through an explicit behavioral task interface.

## Scientific decision log

Do not return to dense raw-logit addition as the main direction unless a new experiment specifically motivates it. Current evidence supports **behaviorally grounded, top-dominant sparse probability composition in a common action/state-transition space**.
