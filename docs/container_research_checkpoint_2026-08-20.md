# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

Experiments run in the **container**. GitHub is persistence/review/reproducibility only. Active branch: `experiment/unseen-source-generalization`; sole active draft: PR #40.

Restart: `docs/research/README.md` → `docs/research/BRANCHES.md` → this file → machine checkpoint/results under `docs/` → PR #40.

## Goal and current architecture

Goal: insert previously unseen computational/model specialists, infer when/how they apply without a hand-written `operator -> source` table, and compose unseen programs.

Current candidate stack:

`interface discovery/normalization -> active behavioral identification -> source-specific ABI/action adapter -> common or fusion-safe latent transition representation -> top-dominant sparse fusion -> learned stream segmentation/program composition`

The main shift is away from shared raw token logits toward **behaviorally identifiable state-transition composition**.

## Core established results

### Shared-token origin

ADD/SUM/MIN/MAX could compose near-perfectly under strong/oracle control; NEG was weak. Identity-free local logit morphology recognized some unseen capability but caused insertion interference. Self-normalized command specificity and unlabeled interface structure improved routing. Arbitrarily permuted command meanings are not zero-shot identifiable: some behavioral evidence is information-theoretically necessary. Dense/equal fusion accumulates depth interference; top-dominant sparse fusion is safer.

### Private ABIs / heterogeneous sources

Behavior can align private input tokenizers, variable-length private output codecs, and incompatible output-head sizes. Across three 64-state pools, private codec recovery was 15/15 and held-out decode 17,999/18,000. Common-action probability mixing `0.6/0.4` achieved 560/560 while equal `0.5/0.5` collapsed to 168/560.

Mixed GRU + NAR-MLP pools achieved hard depth1-5 3,000/3,000 and common-action `0.6/0.4` 600/600.

A private-ABI attention Transformer implementing new `mul`, absent from the seen-only admission calibration, was inserted across seeds 310/311/312. Seen-only threshold stayed 0.65; held-out Transformer scored 1.0 vs second-best 0.3; hard and `0.6/0.4` were perfect through depth5 in the initial held-out-architecture screen.

### Joint opaque-command integration

A full integrated screen used random two-token opaque commands, a seen-only unlabeled interface corpus over seven wrapper/position grammars, private ABIs, heterogeneous architectures, held-out Transformer insertion, and common-action fusion. The held-out MUL code never occurred in the structural corpus.

A local-bigram-only command-span detector failed (~57% held-out span accuracy). Full command-placeholder grammar-skeleton matching fixed it. With that rule frozen across three pools: held-out span/routing were exact, hard depth1-5 perfect, and `0.6/0.4` common-action fusion was **359/360 = 99.72%**.

### Irrelevant and adversarial sources

A competent private-ABI JUNK source computing `(7a+11b+3) mod 32` entered top-2 in **0/18** command/pool cases; insertion changed neither hard nor `0.6/0.4` paired outcomes.

A harder SHADOW-ADD source exactly matches the fixed ten ADD probes while agreeing with ADD on only ~75.29% of the domain. Fixed probes give both sources score 1.0. Random source ordering therefore makes forced tie-breaking wrong about half the time; worst-order depth5 falls to ~84–86%.

Two disagreement probes selected by enumerating the finite domain restored 100% depth5, motivating active identification.

## New: bounded active identification without full-domain enumeration

The controller now permits explicit `UNRESOLVED`. It does **not** scan all 1024 inputs. Each round samples only a bounded proposal set, evaluates surviving candidate sources on those proposals, chooses the proposal with maximum source disagreement, and queries the expected common action only there.

Across seeds 310/311/312, 500 trials/pool, at most four expected-action queries:

- forced fixed-probe tie-break: wrong **49.67%**;
- random probing: true source resolved **68.33%**, wrong 0, unresolved **31.67%**, mean expected-action queries **2.728**;
- active proposal budget 4: resolved **98.93%**, wrong 0, unresolved **1.07%**, mean queries **1.433**;
- active proposal budget 8: **1500/1500 resolved true, wrong 0, unresolved 0**, mean expected-action queries **1.103**.

Trade-off: active proposal-8 uses ~17.65 source evaluations/trial to save oracle labels. This removes full-domain enumeration only in the proposal-selection sense; inputs still come from a known finite generator.

See `docs/active_probe_no_enumeration_aggregate_v1.json`.

## New: unseen grammar extrapolation boundary

Holding one of seven wrapper/position grammars completely out of the unlabeled corpus shows that pure grammar structure does **not** extrapolate reliably: pure-structure held-out slot accuracies are `[1,1,0,0,0,1,0]`.

If the opaque command surface itself has already been behaviorally grounded, exact command-code membership can be combined with structure. That hybrid recovers **11,200/11,200** held-out-grammar spans. Seed310 hard execution with the omitted grammar forced at every stage gives **2,800/2,800**.

This is not zero-shot discovery of a new command phrase: only wrapper/position is unseen; command surface identity is already grounded.

## New: runtime stage-boundary removal

A small causal GRU segmenter is trained only on **delimiter-free normalized streams of depth1-3**. Runtime receives one flat token stream, not an externally supplied operation list or boundary sequence. Grounded opaque command pairs are abstracted to command markers; operands to value markers.

Across three source pools and depth1-10:

- exact segmentation: **3000/3000**;
- route accuracy: **3000/3000**;
- E2E: **2995/3000 = 99.83%**.

All five misses are seed311 source-execution residuals while segmentation/routing remain exact. This removes explicit runtime stage boundaries in the synthetic stream setting, not raw natural-language parsing.

See `docs/stage_boundary_stream_aggregate_v1.json`.

## New: latent fusion geometry

A hand-designed 5-bit state code first showed that 32-way action probabilities need not remain explicit at the fusion layer.

Two stronger controls followed:

1. **Transition-topology spectral embedding** constructed from the five seen operation graphs, with MUL excluded: E2E across three pools is 53.78% at 8D, 83.56% at 12D, 90.72% at 16D, 98.67% at 24D versus 99.67% explicit. Transition topology alone does not automatically produce fusion-safe low-dimensional geometry.
2. **Fusion-law-trained learned 6D codebook**: codebook is optimized only so `0.6 primary + 0.4 secondary` decodes back to the primary state. It decodes all 1024 one-hot state pairs correctly. On real source distributions, explicit 32-way fusion and learned6 latent fusion have identical discrete outcomes across all three pools/depth1-10: **1799/1800** each.

The learned6 result is a compression result, not label-free transition discovery: it uses externally known state identity and the desired fusion law, and source adapters still form common-action distributions before projection.

See `docs/fusion_safe_latent_aggregate_v1.json` and `docs/spectral_transition_latent_aggregate_v1.json`.

## New: command-surface paraphrase without behavior query on the paraphrase

Canonical commands are behaviorally grounded. A second command surface is a synthetic compositional cipher. For five seen commands, paired unlabeled canonical/paraphrase views reveal component-token substitutions. The held-out sixth paraphrase pair is **never seen as a pair**, but both components were individually aligned through seen commands.

Across three pools, the held-out paraphrase maps to the correct canonical command in 3/3 without querying its behavior. In delimiter-free depth1-10 streams:

- segmentation: **3000/3000**;
- routing: **3000/3000**;
- E2E: **2997/3000 = 99.9%**.

A random whole-pair paraphrase with new held-out components is unmappable, preserving the arbitrary-paraphrase identifiability boundary. This is a compositional cipher/parallel-interface experiment, not natural-language paraphrasing.

See `docs/paraphrase_surface_aggregate_v1.json`.

## Current interpretation

The system now separates seven distinct problems:

1. source competence;
2. interface/control-region discovery;
3. active semantic/applicability identification;
4. private ABI alignment;
5. fusion representation/geometry;
6. fusion sparsity/algebra;
7. stream segmentation/program execution.

Within controlled synthetic screens, useful sources may differ in tokenizer, output head, output sequence convention, GRU/MLP/attention architecture, and may be absent from admission calibration. Fixed evidence is insufficient when candidates are observationally equivalent; the controller must actively seek discriminating behavior or abstain.

## Claim boundary

Do **not** claim arbitrary-model/LLM fusion yet. Remaining assumptions include:

- a common 32/64-state environment/action domain;
- an expected-action query channel for behavioral identification;
- inferable/compressible source ABIs;
- a known input generator in the bounded active-probe screen;
- grounded command markers before learned stage segmentation;
- learned6 code supervision by state identity and fusion law;
- synthetic compositional paraphrase with paired seen views;
- no unrelated pretrained model families yet.

## Next

1. Active query synthesis in a non-enumerable/continuous domain with calibrated unresolved confidence.
2. Learn a fusion-safe latent transition representation from behavior **without state-identity code supervision**.
3. Stress stream segmentation with noisy/unseen syntax and ambiguous boundaries.
4. Move the controlled interface to unrelated pretrained model families.

## Repository policy

PR #40 / `experiment/unseen-source-generalization` is the sole active research line. Historical PRs are closed without merging; branch refs remain read-only history because GitHub MCP exposes no branch-delete operation. Old `*-ci-base` and `*-runner` refs are obsolete.
