# Operator-unit fusion research index

This directory is the entry point for the active **Paraphrase-consensus biased decoding / operator-unit fusion** research line.

## Operations / recovery

- Experiments run in the **container**. GitHub is persistence, review, and reproducibility only.
- Active branch: `experiment/unseen-source-generalization`.
- Active draft: PR #40.
- Branch policy: [`BRANCHES.md`](BRANCHES.md).
- Authoritative checkpoint: [`../container_research_checkpoint_2026-08-20.md`](../container_research_checkpoint_2026-08-20.md).
- Black-box query checkpoint: [`../blackbox_query_synthesis_checkpoint_v1.json`](../blackbox_query_synthesis_checkpoint_v1.json).
- Continuous/noise/scaling checkpoint: [`../continuous_blackbox_noise_scaling_checkpoint_v1.json`](../continuous_blackbox_noise_scaling_checkpoint_v1.json).
- Stochastic plugin registry checkpoint: [`../stochastic_plugin_registry_checkpoint_v1.json`](../stochastic_plugin_registry_checkpoint_v1.json).
- Actual stochastic neural checkpoint: [`../actual_stochastic_neural_registry_checkpoint_v1.json`](../actual_stochastic_neural_registry_checkpoint_v1.json).
- Private input-gauge checkpoint: [`../private_input_gauge_checkpoint_v1.json`](../private_input_gauge_checkpoint_v1.json).
- Open-world gauge validation checkpoint: [`../open_world_input_gauge_validation_checkpoint_v1.json`](../open_world_input_gauge_validation_checkpoint_v1.json).

## Current research object

Current candidate stack:

`interface discovery -> black-box active experiment synthesis -> gauge-invariant behavioral equivalence identification -> open-world model/gauge validation -> minimal symmetry-breaking grounding -> private ABI alignment -> behavior-derived fusion-safe latent transition representation -> top-dominant sparse fusion -> learned stream segmentation`

The central shift is away from shared token logits. The current working rule is:

> Maximize gauge-invariant behavioral evidence; preserve the residual equivalence class explicitly; spend grounding only to break the remaining symmetry; then independently test whether the assumed source/gauge family can actually explain held-out behavior.

If plugins are stochastic, register a **probability distribution over behavioral relations**, not a deterministic fingerprint.

## Headline evidence

1. Private tokenizers/output ABIs, incompatible heads, GRU/NAR architectures and a held-out private-ABI Transformer can compose after behavioral alignment.
2. Full opaque-command + position-free + held-out-Transformer integration: hard depth1-5 perfect across three pools; common-action `0.6/0.4` **359/360**.
3. Learned delimiter-free stage segmentation: segmentation/routing **3000/3000** at depth1-10; E2E **2995/3000**.
4. Fixed finite behavior anchors can be spoofed by SHADOW-ADD. Active relation experiments resolve the ambiguity or safely abstain.
5. Cross-source state alignment behaves as a gauge/automorphism problem: the N=32 structured symbolic screen reduces 160 statewise correspondences to **10 symmetry-breaking bridge queries**.
6. Exact expected action labels can be removed from several calibration steps: factorized input alignment, right-identity output alignment, relation/metamorphic applicability, and observed-transition latent construction.
7. Output-label-invariant relations expose true identifiability boundaries: AND/NAND are exactly equivalent until one grounded cross-modal bit breaks the output-relabeling symmetry.
8. Generic relation contracts can be automatically mined, then reduced further to raw input-pair equality queries. In the 20-source finite black-box screen, a source committee gives **1000/1000, wrong 0** while observing only ~6.24% of the full candidate table on average.
9. The same principle works on a non-enumerable continuous domain. For 16 scalar black boxes on `[0,1]^2`, order-only queries give **1600/1600, wrong 0**; monotone-output-equivalent classes are broken only by a small grounded bridge.
10. Query complexity scales with **behavioral equivalence-class entropy**, not raw source count: 32→256 sources with 8→64 behavioral classes require mean relation queries 3.145→6.54, while a four-way residual gauge always costs 2 bridge bits.
11. `UNRESOLVED` is a meaningful safety mechanism under noisy task observations. With 10% relation-bit noise and enough query budget, **157/160 correct, wrong 0, unresolved 3**.
12. Stochastic candidate behavior is harder. A probabilistic plugin registry plus mutual-information acquisition restores 10% source-relation noise to **160/160, wrong 0** with 15 registration repeats and ~11.6 runtime relation queries. Under-registering noisy plugins can make active acquisition overconfident.
13. Actual stochastic neural decoding has now been tested directly. Independent private-ABI neural target instances remain identifiable under token sampling and inference-time representation dropout; with 20% hidden dropout the current relation screen remained **60/60, wrong 0**, while 30–40% primarily increased `UNRESOLVED` rather than wrong routing.
14. Input alignment also behaves as a gauge problem. A noiseless 2D affine private input ABI is fixed by **three non-collinear semantic-native anchors**. If the true ABI is nonlinear, adding anchors cannot repair a misspecified affine adapter; a correctly specified cubic adapter reaches 100%, and a generic RBF adapter learns a smooth non-polynomial private warp without knowing its analytic form (100% routing at 32 noiseless anchors in the current screen).
15. Exact coordinate anchors are not logically required when the input gauge lies in a known low-complexity finite family. Joint `(source, input-gauge)` active hypotheses using only output-order bits identify all eight sources with a 144-gauge library at proposal budget 512, using mean **8.48 relation queries** and no coordinate anchors.
16. Closed-world source/gauge identification is unsafe under gauge-family misspecification. Removing the true gauge produces confident wrong routes; independent held-out relation validation converts the observed deterministic errors to `UNRESOLVED`. With noisy relations the trade-off becomes statistical, and near-equivalent wrong source+gauge hypotheses can require large or targeted validation budgets.

## Key reproduction / checkpoint files

Historical and heterogeneous neural scripts remain under `scripts/`; start with PR #40 and the authoritative checkpoint for the exact file map.

Newest compact checkpoints:

- `docs/auto_relation_mining_checkpoint_v1.json`
- `docs/blackbox_query_synthesis_checkpoint_v1.json`
- `docs/continuous_blackbox_noise_scaling_checkpoint_v1.json`
- `docs/stochastic_plugin_registry_checkpoint_v1.json`
- `docs/actual_stochastic_neural_registry_checkpoint_v1.json`
- `docs/private_input_gauge_checkpoint_v1.json`
- `docs/open_world_input_gauge_validation_checkpoint_v1.json`

Newest input-gauge reproduction script:

- `scripts/experiment_anchorfree_joint_input_gauge.py`

## Claim boundary

These remain controlled synthetic experiments, not arbitrary fusion of unrelated pretrained LLMs. Important remaining assumptions include:

- known/probeable input environments;
- usable equality/order or other relational predicates;
- small grounded cross-modal predicates when residual output gauge cannot otherwise be broken;
- semantic-native coordinate anchors in the learned input-adapter screens, or a finite known gauge family in the anchor-free positive screen;
- open-world validation is not yet robustly calibrated for correlated stochastic neural errors and near-equivalent wrong gauges;
- no genuinely unrelated pretrained model families have yet been integrated end-to-end.

## Next experiments

1. Replace finite input-gauge enumeration with learned continuous gauge proposal/optimization while preserving open-world `UNRESOLVED` validation.
2. Design targeted disagreement validation for near-equivalent wrong source+gauge hypotheses instead of random held-out probes.
3. Connect the hierarchical multi-instance stochastic neural registry to the full private-input/private-output + latent-fusion program execution stack.
4. Generalize input/output gauge estimation to neural sources without a shared factorized synthetic state coordinate.
5. Move the interface to genuinely unrelated pretrained model families.

## Scientific decision log

Do not return to dense raw-logit addition as the main direction unless new evidence specifically motivates it. Current evidence supports **active black-box behavioral identification + open-world gauge validation + minimal gauge fixing + private-ABI alignment + fusion-safe transition composition**.
