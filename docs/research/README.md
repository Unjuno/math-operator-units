# Operator-unit fusion research index

This directory is the entry point for the current **Paraphrase-consensus biased decoding / operator-unit fusion** research line.

## Compute policy

Experiment computation is run in the local/container environment. GitHub is used for source control, persistence, review, and reproducibility. The obsolete GitHub Actions experiment runner was removed from this branch.

## Current research question

Can previously unseen computational/model specialists be inserted and composed without a hand-written `operator -> source` table, shared tokenizer, shared output head, or shared neural architecture?

The current candidate stack is:

`interface normalization -> behavioral semantic grounding -> causal source applicability -> source-specific ABI/action adapter -> common action/state-transition distribution -> sparse probability fusion -> program composition`

The key shift is that **shared raw token logits are no longer treated as the universal fusion interface**. Current evidence favors a common probability field over actions/state transitions, with source-specific interfaces inferred from behavior.

## Authoritative checkpoint

- [`../container_research_checkpoint_2026-08-20.md`](../container_research_checkpoint_2026-08-20.md) — human-readable research state, results, claim boundaries, and next experiments.
- [`../container_research_checkpoint_2026-08-20.json`](../container_research_checkpoint_2026-08-20.json) — compact machine-readable summary.
- [`../container_research_checkpoint_2026-08-20_mixed_architecture.json`](../container_research_checkpoint_2026-08-20_mixed_architecture.json) — detailed mixed-architecture aggregate.

If a chat/container session disappears, start with the Markdown checkpoint above.

## Reproduction scripts

### Historical identity-free source applicability baseline

- [`../../src/opfusion/fusion_unseen_source_generalization.py`](../../src/opfusion/fusion_unseen_source_generalization.py)
- [`../../tests/test_fusion_unseen_source_generalization.py`](../../tests/test_fusion_unseen_source_generalization.py)

This baseline is retained because it established an important negative result: source identity can be removed, but naïve local logit-field summaries cause insertion interference. It is **not** the current preferred architecture.

### Heterogeneous output ABI experiments

- [`../../scripts/experiment_neural_private_tokenizer_adapter.py`](../../scripts/experiment_neural_private_tokenizer_adapter.py) — independently trained neural specialists with private input tokenizers and variable-length output codecs; behavior-only ABI recovery.
- [`../../scripts/experiment_heterogeneous_action_space_soft_fusion.py`](../../scripts/experiment_heterogeneous_action_space_soft_fusion.py) — maps incompatible private sequence likelihoods into a common action distribution and compares fusion algebras.
- [`../../scripts/experiment_mixed_architecture_common_action_fusion.py`](../../scripts/experiment_mixed_architecture_common_action_fusion.py) — mixes autoregressive GRU and non-autoregressive MLP specialists with private tokenizers/heads.

## Current headline results

1. **Private tokenizer/output ABI can be inferred from behavior when it has compositional structure.** Across three neural pools, active behavior-diverse calibration recovered all 15 private codecs; held-out decode was 17,999/18,000.
2. **Different output-head dimensions can still be soft-fused.** After mapping each source into a common 64-action probability distribution, probability mixing at primary/secondary `0.6/0.4` achieved 560/560 depth-1..5 programs across three pools.
3. **Fusion algebra matters.** Equal `0.5/0.5` probability mixing collapsed to 168/560, while log-probability pooling at `0.8/0.2` reached 521/560.
4. **Shared neural architecture is not required in the controlled screen.** GRU + non-autoregressive MLP mixed pools recovered 15/15 codecs, achieved 3,000/3,000 hard depth-1..5 programs, and 600/600 at common-action probability mixing `0.6/0.4`.
5. **Arbitrary semantic permutations are not zero-shot identifiable.** Opaque command/source correspondences require behavioral grounding; this is treated as an identifiability boundary, not merely a router bug.

## Claim boundary

These are controlled neural/synthetic experiments. They do **not** establish arbitrary fusion of unrelated pretrained LLMs. Remaining assumptions include a learnable common state/action domain, structured source I/O ABIs, behavioral evidence for arbitrary semantic permutations, and externally supplied stage boundaries.

## Next experiments

Priority order:

1. Add an attention-based Transformer specialist as a **held-out architecture/source**: absent from source-applicability calibration, with only behavioral ABI/action-adapter calibration allowed.
2. Run one joint E2E test combining **opaque commands + position-free grammar normalization + unseen-source applicability + heterogeneous architectures + private output ABIs + common-action fusion**.
3. Remove externally supplied stage boundaries by learning transition termination/segmentation.
4. Replace the explicit integer action vocabulary with a learned latent transition representation and test whether independently learned sources align behaviorally.

## Scientific decision log

Do not return to dense raw-logit addition as the main direction unless a new experiment specifically motivates it. Current evidence supports behaviorally grounded, sparse composition in a common action/state-transition space.
