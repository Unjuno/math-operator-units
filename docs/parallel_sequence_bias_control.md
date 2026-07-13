# Same-Prefix Bias Fusion

## 1. Runtime target

The runtime target is to evaluate several models or model variants on the same token prefix and combine their logit-space changes before decoding the next token.

```text
same prefix x
  -> common base logits z_base(x)
  -> specialist logits z_1(x), z_2(x), ..., z_n(x)
  -> bias fields B_k(x) = z_k(x) - z_base(x)
  -> fused field B_fused(x) = F(B_1(x), ..., B_n(x))
  -> z_fused(x) = z_base(x) + B_fused(x)
  -> p_fused(x) = softmax(z_fused(x))
  -> decode next token
```

Every field is defined over the same vocabulary, token position, and prefix.

## 2. Why same-prefix evaluation matters

Bias fields cannot be compared safely when they were produced from different tokenizers, different prefixes, or different output spaces.

The minimum ABI requirements are:

- identical ordered vocabulary;
- identical tokenizer profile and vocabulary hash;
- identical prefix token IDs;
- identical sequence position;
- compatible model output shape;
- a clearly identified common base checkpoint.

The common base is part of the experimental definition. It should represent shared behavior that is not intended to be counted repeatedly in every specialist field.

## 3. Fusion is not routing

Routing and fusion are different operations.

```text
routing:
  choose one model or a small subset

bias fusion:
  combine fields produced in the same output space
```

A router may later provide weights or select a subset, but that does not replace the need to define and evaluate the fusion rule itself.

The first experiment should not hide routing inside the specialist models or inside the fusion baseline.

## 4. Raw fusion baseline

The simplest baseline is:

```text
B_raw(x) = sum_k alpha_k B_k(x)
z_raw(x) = z_base(x) + B_raw(x)
```

The coefficients `alpha_k` must be reported explicitly.

Initial baselines should include:

1. unscaled sum;
2. arithmetic mean;
3. fixed coefficient sweeps;
4. centered-field variants;
5. single-specialist controls;
6. empty/base-only control.

Raw fusion is required even when it fails, because the failure pattern defines what a later correction method must actually solve.

## 5. Error amplification

The central failure mode is not merely that a specialist gives a wrong answer. The concern is that a specialist can produce a confident non-zero field on an irrelevant or out-of-domain prefix.

```text
irrelevant specialist field
+ another irrelevant specialist field
+ another aligned error
= amplified incorrect token probability
```

For token `v_wrong`:

```text
B_1(v_wrong | x) > 0
B_2(v_wrong | x) > 0
B_3(v_wrong | x) > 0
```

can produce:

```text
sum_k B_k(v_wrong | x) >> 0
```

The experiment must therefore measure the contribution of inactive specialists rather than assuming that they are neutral.

## 6. Mathematical operator models as a testbed

The mathematical operator models are used because same-prefix fusion can be inspected precisely.

Examples include:

- an addition specialist;
- a variable-length sum specialist;
- a negation specialist;
- minimum and maximum specialists;
- a joint model trained on the union of those data families.

The task family provides exact answers and exact intermediate transitions. This makes it possible to determine whether fusion:

- preserves the correct next transformation;
- changes the final answer correctly;
- amplifies an irrelevant operator bias;
- terminates at the correct point;
- approaches the jointly trained reference distribution.

The operator task is not the final application. It is the measurement system.

## 7. Joint-reference comparison

For each prefix `x`, compare:

```text
p_fused(. | x)
p_joint(. | x)
```

using a reported divergence:

```text
L_match(x) = D(p_joint(. | x), p_fused(. | x))
```

Aggregate this separately for:

- each specialist domain;
- mixed-domain data;
- inactive-specialist conditions;
- value OOD data;
- sequence-length OOD data;
- each saved training checkpoint.

Distribution matching must be reported alongside exact answer and trace-validity metrics.

## 8. Optional later mechanisms

After raw fusion is measured, later experiments may test:

- oracle applicability weights;
- fixed norm balancing;
- learned scalar weights;
- token-wise weights;
- routing;
- confidence or entropy weighting;
- projection removal;
- learned residual correction.

These are alternative hypotheses for reducing fusion error. None is part of the basic definition of bias fusion.

A later mechanism is useful only when it improves a clearly defined baseline without obscuring whether the specialist fields themselves contain composable information.

## 9. Required measurements

At minimum, record:

```text
single_model_task_loss
single_model_exact_accuracy
complete_trace_accuracy
stop_accuracy
inactive_bias_norm
inactive_top_probability
wrong_token_amplification
field_agreement
field_conflict
raw_fusion_task_loss
raw_fusion_exact_accuracy
joint_reference_task_loss
joint_vs_fused_kl_or_jsd
value_ood_accuracy
length_ood_accuracy
```

Measurements must be aligned by seed, base checkpoint, tokenizer ABI, and training exposure.

## 10. Short framing

```text
This project studies same-prefix bias fusion. Multiple GPT variants are evaluated on the same token prefix, their logit changes relative to a common base are combined, and the resulting field is applied to one next-token distribution. Mathematical operator models are used because the intended behavior and the amplification of incorrect contributions can be verified exactly. Routing and calibration are later comparison methods, not the definition of fusion.
```

Japanese:

```text
このプロジェクトでは、同一prefixに対する複数GPTのlogit変化を共通baseとの差分として取り出し、
それらをbias fusionして一つの次token分布を構成する。数学演算子モデルは、正しい寄与、
不要なbias、相互干渉、誤差増幅を厳密に検証できる実験系として使う。routingや補正は、
raw fusionを測定した後に比較する別の手法であり、fusionそのものの定義ではない。
```