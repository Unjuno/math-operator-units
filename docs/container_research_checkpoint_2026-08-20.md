# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

This is the authoritative recovery checkpoint for the active research branch. Experiments run in the **container**; GitHub is persistence, review, and reproducibility only.

Restart order:

1. `docs/research/README.md`
2. `docs/research/BRANCHES.md`
3. this checkpoint
4. machine-readable result JSONs under `docs/`
5. draft PR #40 on `experiment/unseen-source-generalization`

## Goal and current hypothesis

Goal: accept previously unseen computational/model specialists, infer when/how they apply without a hand-written `operator -> source` table, and compose them into unseen programs.

Current candidate stack:

`interface discovery/normalization -> behavioral semantic grounding -> source applicability -> source-specific ABI/action adapter -> common action/state-transition distribution -> sparse probability fusion -> program composition`

The main scientific shift is that **shared raw token logits are no longer treated as the universal fusion interface**.

## What the earlier shared-token work established

- ADD/SUM/MIN/MAX specialists were usable; NEG was repeatedly weak/failed.
- Oracle two-stage composition reached 191/192 E2E.
- Learned prompt controllers could approach that result, but early versions relied on explicit operator tokens and externally supplied stage boundaries.
- Identity-free local logit-field morphology could recognize some unseen capability but caused severe insertion interference.
- Semantic-view behavior and task-conditioned metamorphic behavior were stronger applicability signals than static field morphology.
- Self-normalized command specificity and unlabeled prompt structure could identify control regions without a fixed hand-written source index or absolute command position in the tested synthetic grammar.
- Arbitrarily permuted opaque command meanings were not zero-shot identifiable; minimal behavior anchors resolved the permutation symmetry. Treat this as an identifiability boundary.
- Dense/equal raw-logit fusion accumulated interference with program depth. Top-dominant sparse fusion was substantially safer.

## Output ABI and black-box boundary

Shared output token coordinates are not intrinsically necessary.

- Naïve coordinate-wise interpretation of permuted action tokens failed; an adapter restored performance.
- Arbitrary whole-action random codebooks do not generalize from a few anchors.
- Low-complexity/compositional codecs do: unknown radix, digit-symbol mapping, direction, and variable sequence length can be inferred from a small behavior set.
- An actively chosen eight-probe set identified 500/500 random compositional numeral codecs in the synthetic screen and extrapolated to longer output sequences.
- A black-box source exposing only generated token sequences could still compose through a common-action adapter when its external ABI had learnable structure.

Conclusion: **internal logit access is not the essential requirement; behaviorally identifiable source I/O structure is.**

## Neural private-tokenizer / heterogeneous-head screen

Five independently trained neural specialists (`add/sub/min/max/xor`) used private 4096-way input-pair tokenizers plus private variable-length output codecs and different output-head dimensions.

Across three 64-state pools:

- source competence: 13/15 at 4096/4096, remaining 2 at 4095/4096;
- codec structural recovery: 15/15;
- held-out common-action decode: **17,999/18,000**;
- hard depth-1..5 composition: **4,499/4,500**.

Direct raw-logit fusion was not dimensionally defined. After mapping private sequence likelihoods to a common 64-action probability distribution:

- probability mixture `0.6/0.4`: **560/560**;
- equal `0.5/0.5`: **168/560**;
- log-probability pooling `0.8/0.2`: **521/560**.

Reproduction:

- `scripts/experiment_neural_private_tokenizer_adapter.py`
- `scripts/experiment_heterogeneous_action_space_soft_fusion.py`

## Mixed neural architectures

A later screen mixed autoregressive GRU and non-autoregressive MLP sequence specialists while keeping private input/output ABIs.

Across three pools after fixing an underpowered first NAR design:

- source exact: 14/15 perfect, 1/15 with one miss;
- codec recovery: **15/15**;
- common-action decode: **61,439/61,440**;
- hard depth-1..5: **3,000/3,000**;
- probability `0.6/0.4`: **600/600**;
- equal `0.5/0.5`: **200/600**;
- log-probability `0.8/0.2`: **537/600**.

Shared decoding architecture is therefore not required in this controlled setting.

Reproduction: `scripts/experiment_mixed_architecture_common_action_fusion.py`.

## Held-out attention architecture/source

A sixth operation, `mul`, was implemented only by a private-ABI attention Transformer. Five seen GRU/NAR sources alone calibrated the source-admission threshold; the Transformer/MUL source was absent from that calibration.

Across seeds 310/311/312:

- all 18 source instances reached exact standalone competence;
- all 18 private ABIs were recovered with exact common-action decode;
- seen-only threshold = **0.65** in all pools;
- held-out MUL/Transformer compatibility = **1.0**, second-best = **0.3**;
- hard depth1-5 = **100% at every depth in every pool**;
- common-action probability `0.6/0.4` = **100% at every depth in every pool**;
- equal `0.5/0.5` again collapsed by depth 5 (~23–28%).

Reproduction: `scripts/experiment_heldout_transformer_private_abi_insertion.py`.
Aggregate: `docs/heldout_transformer_private_abi_insertion_aggregate_v1.json`.

## Integrated opaque-command + position-free + held-out Transformer run

The components were then combined rather than tested separately.

Each operation received a random **two-token opaque command**. The unlabeled interface corpus contained only the five seen commands, balanced across seven command positions/wrapper grammars. The held-out MUL code never appeared in that corpus.

Pipeline:

1. discover the seen two-token categorical command family from document frequency/exclusivity;
2. abstract values and learn full command-placeholder grammar skeletons from the unlabeled seen-only corpus;
3. locate an unknown command span by replacing each candidate two-token span with the command placeholder and scoring structural distance to the learned grammar skeletons;
4. ground opaque command semantics against anonymous source behavior probes;
5. use the seen-only threshold to admit the held-out Transformer;
6. map private outputs to common actions;
7. compose depth1-5 using hard selection or common-action probability `0.6/0.4`.

Important negative control: a first version used only local observed-bigram support for span discovery. It achieved only ~57% held-out span accuracy because operand-adjacent spans could be locally plausible command sites. This caused severe E2E collapse. The failure was isolated to interface discovery, not source competence/ABI/routing.

After replacing local-only scoring with learned **full grammar skeleton** matching and freezing that rule:

- seen command-family discovery: exact in 3/3 pools;
- held-out code absent from structural corpus: true in 3/3;
- held-out command-span detection: **100% in 3/3 pools**;
- seen-only threshold: **0.65** in 3/3;
- held-out Transformer route: top score **1.0**, second **0.3** in 3/3;
- hard depth1-5: **100% at every depth in all pools**;
- common-action `0.6/0.4`: **359/360 = 99.72%** across 3 pools × 5 depths × 24 programs.

Reproduction: `scripts/experiment_joint_opaque_positionfree_heldout_transformer.py`.
Aggregate: `docs/joint_opaque_positionfree_heldout_transformer_aggregate_v2.json`.

## Competent irrelevant plugin / exact sparse abstention

A seventh source was added: a private-ABI GRU that accurately computes

`JUNK(a,b) = (7a + 11b + 3) mod 32`.

This is not a broken/random network; it is a competent but irrelevant computational plugin.

Across seeds 310/311/312:

- JUNK standalone exact: **3/3 = 100%**;
- JUNK ABI decode: **3/3 = 100%**;
- behavior score against requested `add/sub/min/max/xor/mul`: `0–0.2`;
- JUNK enters top-2 sparse source set: **0/18 command/pool combinations**;
- hard results before/after JUNK insertion: **paired-identical in 3/3 pools**;
- common-action `0.6/0.4` results before/after JUNK: **paired-identical in 3/3 pools**.

Thus a source can be independently useful, privately encoded, and nevertheless receive **exact zero contribution** to unrelated sparse fusion by remaining outside the selected field.

Reproduction: `scripts/experiment_joint_competent_irrelevant_plugin.py`.
Aggregate: `docs/joint_competent_irrelevant_nuisance_aggregate_v2.json`.

## Current interpretation

The experimental object is now better described as **behaviorally grounded state-transition composition** than token-logit fusion.

Separate failure modes must be measured independently:

1. source competence;
2. interface/control-region discovery;
3. semantic grounding/applicability;
4. source ABI alignment;
5. fusion algebra/field sparsity;
6. program segmentation/execution.

Within the current controlled screens, a source may differ in tokenizer, output head, output sequence convention, internal GRU/MLP/attention architecture, and may be absent from admission calibration, yet still compose when its behavior can be mapped into the common action field.

## Claim boundary

Do **not** claim arbitrary-model or arbitrary-LLM fusion yet.

Remaining assumptions:

- an explicit common 32/64-state action domain is externally defined;
- behavior probes expose expected common actions for semantic/ABI grounding;
- source ABIs have inferable/compressible structure;
- the seven integrated grammar skeletons are represented in the unlabeled seen-only corpus, so the current success is structural interpolation rather than unseen-grammar extrapolation;
- stage boundaries are externally supplied;
- the nuisance source is behaviorally well separated, not near-confusable/adversarial;
- no unrelated pretrained LLM families have been integrated.

## Next experiments — priority order

1. **Near-confusable nuisance:** train a competent source intentionally close to a real operator's behavior so it approaches/enters the top-2 set. Test whether applicability can reject it without removing useful secondary signal.
2. **Unseen grammar holdout:** freeze the current interface discovery rule, remove one or more wrapper/position skeletons entirely from the unlabeled corpus, and test structural extrapolation.
3. **Remove external stage boundaries:** infer transition termination/segmentation and execute a full program stream.
4. **Learn latent transition space:** replace the explicit integer action vocabulary with a learned representation aligned from behavior.
5. After these controlled tests, move to unrelated pretrained model families through an explicit behavioral task interface.

## Repository policy

- Active research branch: `experiment/unseen-source-generalization` / draft PR #40.
- Historical PRs are closed without merging; their branch refs are retained as research history because the available GitHub MCP does not expose branch deletion.
- Other `experiment/*` branches are read-only history unless explicitly reactivated.
- `*-ci-base` and `*-runner` are obsolete old Actions artifacts.
- Do not create a new branch for each screening experiment while the central scientific object remains common-action fusion.

## Scientific decision

Do not return to dense raw-logit addition as the main direction unless new evidence specifically motivates it. Current evidence favors **behaviorally grounded, top-dominant sparse probability composition in a common action/state-transition space with source-specific ABI adapters**.
