# Container research checkpoint — 2026-08-20

Project: Paraphrase-consensus biased decoding / operator-unit fusion

This checkpoint is intended to preserve the research state if the chat/container session disappears. Experiment computation after the CI objection was performed in the container; GitHub is used here only for persistence/version control.

## Research goal

Find a model-composition mechanism that can accept previously unseen computational specialists, infer when/how they apply without a hand-written `operator -> source` table, and compose them into unseen programs. A mere operator classifier or hard-coded switch is insufficient. The strongest current hypothesis is a layered system:

1. infer/normalize the source interface;
2. ground opaque semantics from minimal behavior when symmetry makes zero-shot identification impossible;
3. estimate causal source-context applicability;
4. map source-specific outputs into a common action/state-transition space;
5. perform sparse fusion in that common space;
6. compose stagewise to arbitrary program depth.

## Established results before heterogeneous-tokenizer work

- Oracle sequential composition over ADD/SUM/MIN/MAX reached 191/192 E2E at strength 4.
- Identity-free source-local logit geometry was insufficient: adding a held-out source could improve held-out tasks while damaging unrelated tasks.
- Prompt likelihood and command-token causal sensitivity were complementary; a fixed generic compatibility score generalized across independent model/data seeds, but depended on knowing the command position.
- Command specificity (actual-command perturbation normalized by alternative-command baseline) fixed OOD-instability problems and identified a newly trained DIFF source in 160/160 prompts on an independent data seed.
- A position-free interface detector using unlabeled prompt structure (categorical-family discovery + local grammar/bigram support + rare-token abstraction) recovered command slots and routing across independent seeds.
- Random per-prompt grammar changes (prefix/suffix command movement and wrappers) were normalized back to native ABI with no paired E2E loss.
- Completely arbitrary opaque command codes were not identifiable zero-shot (near chance); two behavioral anchors per command were sufficient to restore canonical two-stage performance.
- Hard routing remained ~99.7% correct across depths 1–5. Dense 0.8/0.2 logit fusion accumulated errors with depth; sparse 0.95/0.05 fusion matched paired hard performance on the tested depth-1..5 programs.

## Synthetic output-ABI boundary screens

When one source's action tokens were permuted, naïve coordinate-wise interpretation failed (1.675% for an affine permutation; 0.275% for a random permutation), while an oracle adapter restored 100%.

Few-shot behavior cannot identify an arbitrary whole-action codebook on unseen actions: a random 67-action bijection with 32 anchors decoded only 48.34%. In contrast, a low-complexity affine mapping was recovered from two anchors.

For a compositional foreign numeral tokenizer with unknown radix (5–16), digit-symbol permutation, direction, and variable sequence length:

- 16 random anchors: 93.503% unseen-value decode;
- 32 random anchors: 99.5325%;
- 64 anchors: 100%;
- an actively designed fixed set of 8 probes identified 500/500 random tokenizers and decoded 1000/1000 unseen values;
- the learned tokenizer rule extrapolated perfectly to longer values up to 999,999 even though anchor sequences were shorter.

A black-box source exposing only generated token sequences (no logits/probabilities/hidden states) also composed through the learned common-action adapter. A random whole-value codebook negative control failed strongly, showing that learnable output-ABI structure is essential.

## New result: independently trained neural specialists with private tokenizers/output heads

A new self-contained container experiment trains five independent neural specialists for `add`, `sub`, `min`, `max`, and `xor` on a 64-state domain. Each source has:

- its own private permutation of the 4096 input-pair tokens;
- its own unknown output radix in 5–12;
- its own digit-to-symbol permutation;
- its own forward/reverse sequence convention;
- therefore a different local output vocabulary and potentially different output-head size.

The fusion layer is not given these mappings. It sees behavior anchors `(input, expected common action, observed private output sequence)` and infers the output ABI.

Across three independently initialized/tokenized pools:

- 13/15 specialists reached 4096/4096 exact; the remaining two were 4095/4096.
- The behavior-diverse 20-anchor calibration structurally recovered all 15 private codecs (15/15).
- Held-out decode after active-20 calibration was 17,999/18,000 across the three pools (99.9944%).
- Random-anchor calibration was materially worse, confirming that information content matters more than raw example count.
- Stagewise hard composition with the inferred adapters was 4,499/4,500 across 4,500 depth-1..5 random programs (99.9778%); the single miss came from a specialist's own residual prediction error, not codec inference.

## New result: soft fusion across incompatible output-head sizes

For one neural pool the private output-head sizes were `[14, 12, 12, 10, 13]`; other pools used `[12,10,11,10,9]` and `[11,13,13,7,8]`. Direct token-logit fusion is therefore not even dimensionally defined across all sources.

For each source and input, the adapter maps the source's private sequence likelihood into a normalized probability distribution over 64 common actions. Two sources are then fused in common-action space.

Paired screens over three independent pools show:

- common-action **probability mixture**, primary 0.6 / secondary 0.4: 560/560 E2E programs across depths 1–5 (100%);
- common-action probability mixture at 0.5 / 0.5: 168/560 (30%), so equal weighting is unsafe;
- common-action **log-probability pooling** at 0.8 / 0.2: 521/560 (93.04%), materially worse than probability mixing despite a larger primary weight.

This is evidence that heterogeneous-model fusion depends not only on discovering a common action space but also on choosing the correct algebra in that space. Arithmetic mixing of calibrated action probabilities is much more robust here than product-of-experts/log-prob pooling.

## Current interpretation

The strongest object is no longer "shared token logits". The candidate universal interface is closer to a probability field over **actions/state transitions**. Source-specific tokenizers and heads can be treated as latent ABIs learned from behavior when those ABIs have compressible structure.

A plausible architecture is therefore:

`external prompt/program -> interface normalizer -> behavioral semantic grounding -> source applicability -> source-specific action adapter -> common action distribution -> sparse probability fusion -> next state`

## Claim boundaries / unresolved assumptions

These heterogeneous-tokenizer experiments are controlled/synthetic and isolate ABI alignment. They do **not** yet prove arbitrary fusion of unrelated pretrained LLM architectures.

Remaining major assumptions:

- a shared externally interpretable state/action domain exists or can be learned;
- the source I/O ABI has enough compositional structure to infer from few behavioral examples;
- stage boundaries are currently supplied externally;
- source semantic grounding still requires behavioral information when labels are arbitrarily permuted;
- source applicability and ABI calibration have not yet been jointly tested with independently trained, unrelated neural architectures at LLM scale.

## Next experiments

1. Replace the private pair-token specialists with genuinely different neural architectures (e.g. Transformer vs GRU/MLP) while keeping no shared tokenizer/output head.
2. Learn the common action adapter from behavior while simultaneously leaving one architecture/source absent from router calibration.
3. Couple opaque commands + position-free grammar normalization + heterogeneous action adapters in one end-to-end experiment.
4. Remove externally supplied stage boundaries by learning state-transition termination/segmentation.
5. Test whether the common object can be a learned latent transition representation rather than an explicit integer action vocabulary.

## Scientific decision

Do not return to dense raw-logit addition as the main direction. Current evidence favors sparse, behaviorally grounded composition in a common action/state-transition space, with source-specific ABI adapters.
