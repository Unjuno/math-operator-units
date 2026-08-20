# Operator-unit fusion research index

Active research line for **Paraphrase-consensus biased decoding / operator-unit fusion**.

- Compute: container only.
- Active branch: `experiment/unseen-source-generalization`.
- Active draft: PR #40.
- Branch/archive policy: [`BRANCHES.md`](BRANCHES.md).
- Human checkpoint: [`../container_research_checkpoint_2026-08-20.md`](../container_research_checkpoint_2026-08-20.md).
- Machine checkpoint: [`../container_research_checkpoint_2026-08-20.json`](../container_research_checkpoint_2026-08-20.json).

## Current object

`interface discovery/normalization -> active behavioral identification -> source-specific ABI/action adapter -> common or fusion-safe latent transition representation -> top-dominant sparse fusion -> learned stream segmentation/program composition`

Shared raw token logits are no longer the main candidate universal interface.

## Current headline evidence

- Private tokenizers/output codecs/output-head sizes: behaviorally aligned; held-out decode 17,999/18,000.
- Mixed GRU/NAR-MLP: hard depth1-5 3,000/3,000; common-action probability `0.6/0.4` 600/600.
- Held-out private-ABI attention Transformer/MUL, absent from seen-only admission calibration: correctly inserted across seeds 310/311/312.
- Full opaque-command + position-free + held-out-Transformer integration: hard perfect; common-action `0.6/0.4` **359/360**.
- Competent irrelevant JUNK plugin: top-2 **0/18**, exact paired zero-impact.
- SHADOW-ADD proves fixed behavioral probes can be spoofed. Forced tie-breaking is wrong about half the time.
- Bounded-proposal active identification, without enumerating all 1024 inputs: proposal budget 8 resolves **1500/1500**, wrong 0, abstain 0, mean expected-action queries ~**1.10**. Random probing with four-query cap resolves only ~68.3% and safely abstains on ~31.7%.
- Unseen wrapper/position grammar: pure structure fails on 4/7 holdouts; structure + already behavior-grounded command identity gives **11,200/11,200** span recovery and seed310 E2E **2,800/2,800**.
- Runtime stage boundaries removed in a controlled flat-stream screen: causal GRU trained only on depth1-3 delimiter-free normalized streams segments depth1-10 **3000/3000**, routing **3000/3000**, E2E **2995/3000**.
- Latent compression: learned fusion-safe **6D** code exactly matches explicit 32-way probability-fusion discrete outcomes, E2E **1799/1800**. Transition-graph spectral geometry is not automatically safe (8D 53.8%, 16D 90.7%, 24D 98.7%).
- Command-surface paraphrase: a held-out paraphrase pair never seen as a pair is recovered from compositional component alignment without a behavior query on that paraphrase; delimiter-free segmentation/routing exact, E2E **2997/3000**. Arbitrary whole-pair paraphrase remains unmappable.

## Reproduction scripts

- `scripts/experiment_neural_private_tokenizer_adapter.py`
- `scripts/experiment_heterogeneous_action_space_soft_fusion.py`
- `scripts/experiment_mixed_architecture_common_action_fusion.py`
- `scripts/experiment_heldout_transformer_private_abi_insertion.py`
- `scripts/experiment_joint_opaque_positionfree_heldout_transformer.py`
- `scripts/experiment_joint_competent_irrelevant_plugin.py`
- `scripts/experiment_shadow_add_active_probing.py`
- `scripts/experiment_active_probe_without_enumeration.py`
- `scripts/experiment_unseen_grammar_hybrid_interface.py`
- `scripts/experiment_learned_stream_stage_segmentation.py`
- `scripts/experiment_fusion_safe_latent.py`
- `scripts/experiment_spectral_transition_latent.py`
- `scripts/experiment_compositional_command_paraphrase.py`

## Key aggregates

- `docs/joint_opaque_positionfree_heldout_transformer_aggregate_v2.json`
- `docs/shadow_add_active_probe_aggregate_v2.json`
- `docs/active_probe_no_enumeration_aggregate_v1.json`
- `docs/unseen_grammar_hybrid_interface_aggregate_v1.json`
- `docs/stage_boundary_stream_aggregate_v1.json`
- `docs/fusion_safe_latent_aggregate_v1.json`
- `docs/spectral_transition_latent_aggregate_v1.json`
- `docs/paraphrase_surface_aggregate_v1.json`

## Claim boundary

Still controlled/synthetic. We still assume a common state domain, an expected-action query channel for active grounding, inferable private ABIs, and structured interface evidence. The learned 6D code uses state identity plus the desired fusion law; it is not a label-free transition representation. The paraphrase experiment is a compositional cipher with paired interface views, not natural-language paraphrasing.

## Next

1. Active identification in a non-enumerable/continuous domain, including calibrated `UNRESOLVED`.
2. Learn a fusion-safe latent transition representation from behavior **without state-identity code supervision**.
3. Stress learned stream segmentation with noisy/unseen syntax and ambiguous boundaries.
4. Move the controlled interface to unrelated pretrained model families.
