# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

This is the authoritative human-readable recovery checkpoint for the current research branch. Experiment computation after the CI objection is performed in the container; GitHub is used for persistence, review, and reproducibility.

For a file map and restart instructions, see [`docs/research/README.md`](research/README.md).

## Research goal

Find a composition mechanism that can accept previously unseen computational/model specialists, infer when/how they apply without a hand-written `operator -> source` table, and compose them into unseen programs.

The current candidate stack is:

`interface normalization -> behavioral semantic grounding -> causal source applicability -> source-specific ABI/action adapter -> common action/state-transition distribution -> sparse probability fusion -> program composition`

A key scientific shift is that **shared raw token logits are no longer treated as the universal fusion interface**.

## Established results before heterogeneous-output work

- Oracle sequential composition over ADD/SUM/MIN/MAX reached 191/192 E2E at strength 4.
- Identity-free source-local logit geometry was insufficient: inserting an unseen source could improve held-out tasks while damaging unrelated tasks.
- Prompt likelihood and command-token causal sensitivity were complementary, but absolute sensitivity failed on a genuinely new DIFF source because OOD instability was misread as applicability.
- Self-normalized command specificity fixed that calibration problem and identified the new DIFF source in 160/160 prompts on an independent data seed.
- A position-free interface detector using unlabeled prompt structure recovered command slots and routing without a hand-written command position/token list.
- Random per-prompt grammar changes (prefix/suffix movement and wrappers) were normalized back to native ABI with no paired E2E loss in the tested grammar family.
- Completely arbitrary opaque command codes were not zero-shot identifiable; two behavioral anchors per command restored canonical two-stage performance. This is treated as an identifiability boundary.
- Hard routing stayed about 99.7% correct across depth 1–5. Dense 0.8/0.2 raw-logit fusion accumulated errors with depth; sparse 0.95/0.05 fusion matched paired hard performance in the original shared-token-space experiments.

## Output-ABI boundary screens

When one source's action tokens were permuted, naïve coordinate-wise interpretation failed (1.675% for an affine permutation; 0.275% for a random permutation), while an oracle adapter restored 100%.

Few-shot behavior cannot identify an arbitrary whole-action codebook on unseen actions: a random 67-action bijection with 32 anchors decoded only 48.34%. In contrast, a low-complexity affine mapping was recovered from two anchors.

For a compositional foreign numeral tokenizer with unknown radix (5–16), digit-symbol permutation, direction, and variable sequence length:

- 16 random anchors: 93.503% unseen-value decode;
- 32 random anchors: 99.5325%;
- 64 anchors: 100%;
- a fixed actively designed set of 8 probes identified 500/500 random tokenizers and decoded 1000/1000 unseen values;
- the inferred tokenizer rule extrapolated perfectly to longer values up to 999,999 even though anchor sequences were shorter.

A black-box source exposing only generated token sequences, with no logits/probabilities/hidden states, also composed through a learned common-action adapter. A random whole-value codebook negative control failed strongly. The relevant quantity is therefore not internal logit access but whether the source I/O ABI has learnable structure.

## Independently trained neural specialists with private tokenizers/output heads

A self-contained container experiment trains five neural specialists for `add`, `sub`, `min`, `max`, and `xor` on a 64-state domain. Each source has:

- a private permutation of the 4096 input-pair tokens;
- an unknown output radix in 5–12;
- a private digit-to-symbol permutation;
- a private forward/reverse sequence convention;
- therefore a potentially different output-head dimension.

The fusion layer is not given these mappings. It receives behavior anchors `(input, expected common action, observed private output sequence)` and infers the ABI.

Across three independently initialized/tokenized pools:

- 13/15 specialists reached 4096/4096 exact; the remaining 2 were 4095/4096.
- behavior-diverse 20-anchor calibration structurally recovered all 15 private codecs.
- held-out decode after calibration: **17,999/18,000 = 99.9944%**.
- hard depth-1..5 composition through learned adapters: **4,499/4,500 = 99.9778%**.

Reproduction: `scripts/experiment_neural_private_tokenizer_adapter.py`.

## Soft fusion across incompatible output-head sizes

Example private head sizes are `[14,12,12,10,13]`, `[12,10,11,10,9]`, and `[11,13,13,7,8]`. Direct token-logit fusion is therefore not even dimensionally defined across all sources.

Each source's private sequence likelihood is mapped into a normalized distribution over 64 common actions, then two sources are fused in that common space.

Across three independent pools / depth-1..5 screens:

- common-action **probability mixture 0.6/0.4**: **560/560 = 100%**;
- probability mixture 0.5/0.5: **168/560 = 30%**;
- common-action **log-probability pooling 0.8/0.2**: **521/560 = 93.04%**.

This establishes two separate points in the controlled setting:

1. common token/logit coordinates are not necessary if source outputs can be mapped into a common action distribution;
2. the algebra used in that common space matters materially.

Arithmetic mixing of calibrated action probabilities was much more robust here than product-of-experts/log-prob pooling.

Reproduction: `scripts/experiment_heterogeneous_action_space_soft_fusion.py`.

## Mixed neural architectures

The next experiment removed the shared decoding-architecture assumption by mixing:

- autoregressive GRU specialists;
- non-autoregressive MLP sequence specialists;
- private input-pair tokenizers;
- private variable-length output codecs.

The first NAR design was intentionally informative as a failure: SUB reached 42.8% and MAX 72.6%, while the codec adapter still recovered all 5 private ABIs. That separated **source competence failure** from **adapter/fusion failure**.

After using position-specific NAR output heads, source competence saturated.

Across three independent mixed-architecture pools:

- source exact: **14/15 at 4096/4096; 1/15 at 4095/4096**;
- codec structural recovery: **15/15**;
- common-action decode: **61,439/61,440**;
- hard depth-1..5 composition: **3,000/3,000**;
- common-action probability mixture 0.6/0.4: **600/600**;
- probability mixture 0.5/0.5: **200/600**;
- log-probability pooling 0.8/0.2: **537/600 = 89.5%**.

Therefore, in this controlled screen, a shared neural decoding architecture is not necessary either.

Reproduction: `scripts/experiment_mixed_architecture_common_action_fusion.py`.

## Current interpretation

The strongest candidate universal interface is no longer a shared token-logit space. It is closer to a probability field over **actions/state transitions**, with source-specific ABIs inferred from behavior.

Current working architecture:

`external prompt/program -> interface normalizer -> behavioral semantic grounding -> source applicability -> source-specific action adapter -> common action distribution -> sparse probability fusion -> next state`

This architecture naturally explains several previous observations:

- arbitrary semantic permutations require grounding because they are unidentifiable from syntax alone;
- low-complexity/compositional ABIs can be inferred from few behavior probes;
- private tokenizers and private head dimensions are acceptable if they admit an action adapter;
- dense fusion in the wrong coordinate system accumulates interference;
- source competence, routing/applicability, ABI alignment, and fusion algebra are distinct failure modes and should be measured separately.

## Claim boundaries

These heterogeneous-tokenizer/architecture experiments are controlled neural/synthetic screens. They do **not** prove arbitrary fusion of unrelated pretrained LLM architectures.

Remaining major assumptions:

- a common state/action domain is externally available or learnable;
- source I/O ABIs have enough structure to infer from feasible behavioral evidence;
- arbitrary semantic permutations still require grounding information;
- stage boundaries are externally supplied;
- opaque commands, position-free normalization, unseen-source applicability, heterogeneous architectures, and private output ABIs have not yet all been combined in one E2E experiment;
- an attention-based Transformer source has not yet been tested as a held-out architecture/source in the heterogeneous pool;
- no LLM-scale unrelated pretrained models have yet been connected by this mechanism.

## Next experiments — priority order

1. Add an attention-based Transformer specialist as a **held-out architecture/source**: absent from source-applicability calibration, with only behavioral ABI/action-adapter calibration allowed.
2. Run one joint E2E experiment combining **opaque commands + position-free grammar normalization + unseen-source applicability + heterogeneous architectures + private output ABIs + common-action probability fusion**.
3. Remove externally supplied stage boundaries by learning transition termination/segmentation.
4. Replace the explicit integer action vocabulary with a learned latent transition representation and test whether independently learned sources can be aligned behaviorally.
5. Only after the controlled pipeline is stable, move to unrelated pretrained language/model families with an explicit common behavioral task interface.

## Scientific decision

Do not return to dense raw-logit addition as the main direction unless a new experiment specifically motivates it. Current evidence favors **behaviorally grounded, sparse composition in a common action/state-transition space with source-specific ABI adapters**.
