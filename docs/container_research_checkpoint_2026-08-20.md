# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

Experiments run in the **container**. GitHub is persistence, review, and reproducibility only. Active branch: `experiment/unseen-source-generalization`; active draft: PR #40.

Restart in this order: `docs/research/README.md` → `docs/research/BRANCHES.md` → this file → newest JSON under `docs/` → PR #40.

## Goal

Accept previously unseen computational/model specialists, infer when/how they apply without a hand-written `operator -> source` table, and compose them into unseen programs.

Current candidate stack:

`interface discovery/normalization -> active behavioral grounding/applicability -> source-specific ABI/action adapter -> common action/state-transition distribution -> top-dominant sparse probability fusion -> program composition`

The main scientific shift is that **shared raw token logits are no longer treated as the universal fusion interface**.

## Established trajectory

Earlier shared-token experiments established that ADD/SUM/MIN/MAX could compose nearly perfectly under oracle/learned control, while NEG was weak. Identity-free local logit geometry could recognize some unseen capability but caused irrelevant-source insertion damage. Self-normalized causal specificity and unlabeled prompt structure improved source/interface discovery. Arbitrarily permuted command meanings were not zero-shot identifiable, so behavioral grounding is an identifiability requirement. Dense/equal fusion accumulated depth interference; top-dominant sparse fusion was safer.

Output-ABI experiments then showed that common token coordinates are not essential. Low-complexity private codecs (unknown radix, symbol mapping, sequence direction/length) can be inferred from behavior, including for black-box sources exposing only output token sequences. Arbitrary whole-action random codebooks do not few-shot generalize.

## Neural private-tokenizer / heterogeneous-source results

Across three 64-state neural pools with private input tokenizers, private variable-length output codecs, and incompatible output-head sizes:

- codec recovery: **15/15**;
- held-out decode: **17,999/18,000**;
- hard depth1-5: **4,499/4,500**;
- common-action probability `0.6/0.4`: **560/560**;
- equal `0.5/0.5`: **168/560**;
- log-probability `0.8/0.2`: **521/560**.

Mixed GRU + NAR-MLP pools then achieved codec recovery **15/15**, hard depth1-5 **3,000/3,000**, and common-action `0.6/0.4` **600/600**. Shared decoding architecture is therefore not required in this controlled setting.

## Held-out attention architecture

A new `mul` specialist using an attention Transformer, private tokenizer, and private output ABI was excluded from admission-threshold calibration. Five seen GRU/NAR sources alone set threshold `0.65`.

Across seeds 310/311/312:

- all six sources per pool reached exact standalone competence;
- all private ABIs decoded exactly;
- held-out Transformer score = **1.0**, second-best = **0.3**;
- hard depth1-5 = **100% at every depth in all pools**;
- common-action `0.6/0.4` = **100% at every depth in all pools**.

See `scripts/experiment_heldout_transformer_private_abi_insertion.py` and `docs/heldout_transformer_private_abi_insertion_aggregate_v1.json`.

## Joint opaque-command / position-free integration

Each operation received a random two-token opaque command. The unlabeled interface corpus contained only the five seen commands across seven positions/wrappers; the held-out MUL code never appeared there.

A first integrated version used local observed-bigram support to find the unknown command span and **failed**: seed310 held-out span accuracy was only ~57%, causing severe E2E collapse. Operand-adjacent spans could be locally plausible command sites.

The fix was to learn full **command-placeholder grammar skeletons** from the unlabeled seen-only corpus, then score candidate spans by structural distance; local bigrams only break ties.

With that rule frozen across seeds 310/311/312:

- seen command-family discovery: exact 3/3;
- held-out code absent from corpus: 3/3;
- held-out span detection: **100%**;
- held-out Transformer routing: **100%**;
- hard depth1-5: **100% at every depth in all pools**;
- common-action `0.6/0.4`: **359/360 = 99.72%**.

See `scripts/experiment_joint_opaque_positionfree_heldout_transformer.py` and `docs/joint_opaque_positionfree_heldout_transformer_aggregate_v2.json`.

## Competent irrelevant plugin

A private-ABI GRU was trained to perfectly compute `JUNK(a,b)=(7a+11b+3) mod 32`. It is competent, not random, but irrelevant to the six requested operations.

Across three pools it scored only 0–0.2 against requested behaviors, entered top-2 **0/18** command/pool combinations, and both hard and `0.6/0.4` results were paired-identical before/after insertion. Sparse top-2 therefore gave this clearly irrelevant plugin exact zero contribution.

See `scripts/experiment_joint_competent_irrelevant_plugin.py` and `docs/joint_competent_irrelevant_nuisance_aggregate_v2.json`.

## New identifiability boundary: SHADOW-ADD

The easy JUNK case is not sufficient. A near-confusable private-ABI GRU was therefore trained on a `SHADOW-ADD` function that:

- matches ADD **exactly on the fixed ten behavioral probes**;
- agrees with ADD on ~**75.29%** of the full 32×32 domain;
- is itself learned with 100% standalone accuracy.

On the fixed probes, true ADD and SHADOW both score **1.0**. Source order was randomized to remove identity/tie-break assumptions.

Across 3 pools × 300 order/probe trials:

| extra random probes | true ADD top rate | SHADOW top rate |
|---:|---:|---:|
| 0 | 46.44% | 53.56% |
| 1 | 62.56% | 37.44% |
| 2 | 69.33% | 30.67% |
| 4 | 85.56% | 14.44% |
| 8 | 95.11% | 4.89% |
| 16 | 99.44% | 0.56% |
| 32 | 100% | 0% |

Under worst-case source ordering, fixed-probe routing selects SHADOW for ADD and depth-5 E2E falls to **84.33%, 85.67%, 85.00%** across the three seeds.

Crucially, selecting **two inputs where the competing candidate sources disagree**, then querying the expected common action only on those two probes, restores the true ADD route and depth-5 E2E to **100%, 100%, 100%**.

Decision: a fixed finite behavioral anchor set is **not** a semantic proof. When multiple candidate sources remain observationally equivalent on calibration evidence, the controller should actively generate/select disagreement probes until the ambiguity is resolved or explicitly remain uncertain.

See `scripts/experiment_shadow_add_active_probing.py` and `docs/shadow_add_active_probe_aggregate_v2.json`.

## Current interpretation

The research object is now better described as **behaviorally grounded state-transition composition** than token-logit fusion. Distinct failure modes must remain separated:

1. source competence;
2. interface/control-region discovery;
3. semantic/applicability identification;
4. ABI alignment;
5. fusion algebra/sparsity;
6. program segmentation/execution.

A useful source may differ in tokenizer, output head, output sequence convention, internal GRU/MLP/attention architecture, and may be absent from admission calibration. What currently matters is whether its behavior can be actively identified and mapped into a common transition field.

## Claim boundary

Do **not** claim arbitrary-model or arbitrary-LLM fusion yet. Remaining assumptions include:

- an explicit common 32/64-state action domain;
- an oracle/query mechanism that can provide expected common actions on selected behavioral probes;
- structured/inferable source ABIs;
- integrated grammar success currently interpolates among seven skeletons present in the unlabeled corpus;
- stage boundaries are externally supplied;
- active disagreement search currently enumerates a small finite input domain;
- no unrelated pretrained LLM families have been connected.

## Next experiments

1. **Generalize active probing:** choose high-information disagreements without enumerating the entire input domain; quantify query/sample complexity and allow an explicit unresolved state.
2. **Unseen grammar holdout:** freeze interface discovery and remove entire wrapper/position skeletons from the unlabeled corpus to test structural extrapolation.
3. **Remove external stage boundaries:** learn transition termination/segmentation on a continuous program stream.
4. **Learn latent transition space:** replace explicit integer actions with a behaviorally aligned latent state-transition representation.
5. Only after those controlled tests, move to unrelated pretrained model families through an explicit behavioral task interface.

## Repository policy

Historical research PRs have been closed without merging; branch refs are retained because the available GitHub MCP does not expose branch deletion. `experiment/unseen-source-generalization` / PR #40 is the sole active research line. Other `experiment/*` refs, including old `*-ci-base` and `*-runner` branches, are read-only history unless explicitly reactivated.

## Scientific decision

Do not return to dense raw-logit addition as the main direction unless new evidence specifically motivates it. Current evidence favors **active behavioral identification + source-specific ABI alignment + top-dominant sparse probability composition in a common action/state-transition space**.
