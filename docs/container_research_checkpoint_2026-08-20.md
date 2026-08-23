# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

Experiments run in the **container**. GitHub is persistence/review/reproducibility only. Active branch: `experiment/unseen-source-generalization`; sole active draft: PR #40.

Restart order: `docs/research/README.md` → `docs/research/BRANCHES.md` → this file → `docs/blackbox_query_synthesis_checkpoint_v1.json` → `docs/continuous_blackbox_noise_scaling_checkpoint_v1.json` → `docs/stochastic_plugin_registry_checkpoint_v1.json` → PR #40.

## Current hypothesis

The research target is no longer shared-token logit fusion. The current object is:

`interface discovery -> black-box active experiment synthesis -> gauge-invariant behavioral equivalence identification -> minimal symmetry-breaking grounding -> private ABI alignment -> behavior-derived fusion-safe latent transition representation -> top-dominant sparse fusion -> learned stream segmentation`

Core rule:

> Use the most gauge-invariant behavioral evidence available; explicitly represent the residual equivalence/automorphism class; spend grounding only to break that remaining symmetry.

For stochastic plugins, the behavioral signature itself must be probabilistic rather than a single deterministic fingerprint.

## Established controlled results

- Private input/output ABIs, incompatible output heads, GRU/NAR architectures, and a held-out private-ABI Transformer/MUL source can be behaviorally aligned and composed.
- Full opaque-command + position-free + held-out-Transformer integration gives hard depth1-5 perfect across three pools and common-action `0.6/0.4` **359/360**.
- Delimiter-free learned stream segmentation gives segmentation/routing **3000/3000** at depth1-10 and E2E **2995/3000**.
- SHADOW-ADD proves fixed finite anchors can be spoofed; active relation experiments repair the ambiguity or safely abstain.
- Cross-source state alignment behaves as a gauge/automorphism problem. In the N=32 structured symbolic screen, ADD/SUB/MIN/MAX/XOR/MUL alignment needs only **10** symmetry-breaking bridge queries instead of 160 statewise correspondences.
- Factorized input ABI is recoverable from **32 diagonal examples** for all 1024 pairs; an arbitrary 1024-way pair codebook with the same 32 examples exposes only 3.125%.
- Output alignment can use right-identity relations rather than exact expected action labels. Routing can use relational/metamorphic evidence rather than expected answers.
- Behavior-derived 5D latent fusion can be learned without state-classification supervision; observed neural transition relations can produce 32 unique state signatures and exact `0.6/0.4` pair decode in the controlled screen.

## Automatic relation discovery and minimal bridges

Operation-specific routing contracts were removed in stages:

1. generic intervention statistics selected by seen-source variance;
2. random intervention DSL programs;
3. task-adaptive relation selection;
4. raw input-pair query synthesis using only one bit: whether outputs on two selected inputs are equal.

For the 20-source finite screen, proposal=64 gives **1999/2000**, wrong 0, without precomputing candidate behavior tables. A source committee reduces candidate-side observation further: proposal=32 / committee=4 gives **1000/1000**, wrong 0, while observing only ~**6.24%** of the full 20x1024 candidate table on average.

Output-label-invariant relations cannot distinguish output-relabeling-equivalent computations. AND/NAND are exactly indistinguishable under the full relation fingerprint. One generic cross-modal bit is enough to break that residual symmetry. A 19-operation screen using "relation first, bridge only on equivalence" gives **1900/1900**, wrong 0, mean bridge cost **0.105 query/command**.

## Continuous / non-enumerable input domain

The same query-synthesis principle was moved to `[0,1]^2`, where the input domain is not enumerated. Each query compares only the **order** of two outputs (`y1 < y2`), which is invariant to any strictly increasing output reparameterization.

Across 16 continuous black-box computations, proposal=32 / committee=4 gives **1600/1600**, wrong 0, with mean **3.95 relation queries**. The deliberately order-equivalent classes `add~avg`, `mul~geom`, and `sqsum~l2` remain unresolved until one grounded scalar-reference bit is requested; mean bridge cost is **0.375 query/command**.

## Query complexity versus source-pool size

A controlled affine-family screen uses four positive-scale/offset gauge variants per behavioral direction. Relation queries identify direction; a scalar-reference bridge fixes the four-way gauge.

With sufficient bridge proposal search:

| sources | behavioral classes | mean relation queries | mean bridge queries | result |
|---:|---:|---:|---:|---:|
| 32 | 8 | 3.145 | 2.0 | 200/200 |
| 64 | 16 | 4.37 | 2.0 | 200/200 |
| 128 | 32 | 5.40 | 2.0 | 200/200 |
| 256 | 64 | 6.54 | 2.0 | 200/200 |

This supports the working interpretation that **relation query cost tracks behavioral equivalence-class entropy**, while grounding cost tracks the residual gauge bits, rather than scaling directly with raw source count.

## Noisy task observations and UNRESOLVED

For continuous relation observations with independent bit flips, posterior confidence is used instead of forced routing.

With known 10% bit noise and up to 30 relation + 8 bridge queries: **157/160 correct, wrong 0, unresolved 3**. At 20%: **114/160 correct, wrong 0, unresolved 46**.

The noise rate can also be inferred jointly instead of supplied. A Beta-Bernoulli candidate model gives 5% noise: **320/320, wrong 0**; 10%: **300/320, wrong 0**. At 30%, threshold 0.999 produced one overconfident error; raising the threshold to 0.9999 removed observed wrong routing in the matched screen at the cost of high abstention.

Thus `UNRESOLVED` is a real safety/coverage mechanism, not just an API placeholder.

## Stochastic candidate/plugin behavior

Noise in the candidate/plugin fingerprint is substantially harder than noise in the task observation. Hard majority/mismatch scoring remains unsafe even with repeats.

A better design registers each plugin probabilistically: repeated offline relation/bridge experiments estimate `P(bit | source, query)`. Runtime then updates the task posterior against these stored distributions and does **not** re-run all candidate plugins.

Using mutual-information acquisition over the registered uncertainty:

- source relation noise 5%, registration repeats 15: **160/160**, wrong 0; mean runtime relation queries **11.08**, bridge **0.71**;
- source relation noise 10%, repeats 15: **160/160**, wrong 0; mean relation queries **11.62**, bridge **0.93**;
- source relation noise 20%, repeats 50: **154/160**, wrong 0, unresolved 6; mean relation queries **13.17**, bridge **1.60**.

Under-registering a noisy plugin can make active acquisition overconfident: at 10% noise with only 5 repeats, 18/160 routes are wrong. Therefore active querying is safe only if source-registry uncertainty itself is calibrated.

## Current claim boundary

Do **not** claim arbitrary unrelated LLM fusion yet. Remaining assumptions include:

- controlled synthetic scalar/state environments;
- known/probeable input generators;
- relation families such as equality or order that are meaningful for the environment;
- small grounded cross-modal predicates to break residual output gauge when necessary;
- factorized/probeable private ABIs in the current neural screens;
- stochastic-source tests inject relation-bit noise rather than full token-level generative stochasticity;
- no genuinely unrelated pretrained model families have yet been integrated end-to-end.

## Next experiments

1. Validate the probabilistic plugin registry with **actual stochastic neural decoding**, not injected bit flips.
2. Generalize gauge/bridge estimation to sources whose input/output state spaces do not share a factorized synthetic domain.
3. Test simultaneous task-observation noise + candidate-source stochasticity and calibrate `UNRESOLVED` without assuming independent noise.
4. Learn continuous query proposals rather than random proposal sampling while preserving black-box constraints.
5. Move the controlled interface to genuinely unrelated pretrained model families.

## Repository policy

PR #40 / `experiment/unseen-source-generalization` is the sole active research line. Historical PRs remain closed/unmerged research history. Experiment compute stays in the container; GitHub stores reproducibility scripts/checkpoints and the active scientific narrative.
