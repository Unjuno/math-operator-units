# Generation-Path Reliability Calibrator

This document defines the model-pair structure for parallel bias composition.

The goal is not to enumerate all possible applicability rules for every sequence. Instead, each generator produces its own bias field or candidate sequence path, and a paired calibrator audits whether that generated path is reliable for that generator.

## 1. Unit structure

Each operator or control unit is a pair:

```text
U_k = (M_k, R_k)
```

where:

```text
M_k:
  sequence or bias-field generator

R_k:
  generation-path reliability calibrator
```

The generator emits a field:

```text
B_k(v | x) = M_k(x)
```

The calibrator observes the context and the generated field:

```text
R_k(x, B_k) -> reliability / attenuation / removal signal
```

The unit contribution becomes:

```text
B_tilde_k = Calibrate(B_k, R_k(x, B_k))
```

Two simple forms are:

```text
B_tilde_k(v | x) = r_k(v | x) B_k(v | x)
```

or:

```text
B_tilde_k(v | x) = B_k(v | x) - E_k(v | x)
```

where `E_k` is an estimated error or removal field.

## 2. What the calibrator checks

The calibrator does not need to know every possible valid sequence globally.

It checks whether the generator's own proposed direction looks like a valid generation path for that generator:

```text
Is this field consistent with M_k's learned operator family?
Is this path likely to be an in-domain path for M_k?
Is M_k showing an assimilation error?
Is M_k producing high confidence in a context where it is usually unreliable?
Is this field aligned with verifier, progress, or consensus signals?
```

Thus the calibrator is local to the generator's failure profile.

It is not:

```text
global parser
hard operator selector
complete applicability oracle
```

## 3. Composition after per-unit calibration

All units can be run over the same prefix:

```text
B_1, B_2, ..., B_n
```

Each field is calibrated by its own reliability model:

```text
B_tilde_1, B_tilde_2, ..., B_tilde_n
```

Then the fields are composed:

```text
F = O(B_tilde_1, B_tilde_2, ..., B_tilde_n)
```

and injected into the base logits:

```text
z_final = z_0 + lambda F
p_final = softmax(z_final)
```

The intent is that valid generation paths remain strong, while unreliable or assimilated paths are attenuated before final composition.

## 4. Training data for R_k

The generator `M_k` is trained mainly on its own operator family.

The calibrator `R_k` must see both owned and non-owned cases.

Positive examples:

```text
contexts and generated fields where M_k is in-domain
valid trajectories from M_k's operator family
fields whose softmax effect matches the reference effect
```

Negative examples:

```text
fields produced by M_k on other operator families
OOD length or depth cases
ambiguous or malformed paths
high-confidence wrong paths
operator assimilation errors
fields that conflict with verifier or progress signals
```

Labels can include:

```text
reliability score
attenuation weight
error field E_k
owned_path / non_owned_path
verifier-aligned / verifier-conflicting
```

## 5. Important distinction

The calibrator does not decide the final answer directly.

It decides how much of its paired generator's bias field should survive into composition.

```text
M_k proposes a bias field.
R_k audits that proposal.
The composition operator combines the audited fields.
Softmax chooses from the final distribution.
```

This keeps the project in bias-control space rather than parser-driven symbolic routing.

## 6. Failure mode addressed

A generator can be confidently wrong outside its training distribution.

Example:

```text
M_ADD sees a subtraction-like context
M_ADD emits an addition-like field anyway
```

Without a calibrator, the field may dominate because it is peaked.

With a reliability calibrator:

```text
R_ADD detects that the generation path is not reliable for M_ADD
R_ADD attenuates or removes the error component
B_tilde_ADD becomes small or less harmful
```

The purpose is not to prevent competition between models. The purpose is to prevent unreliable confidence from dominating the composed field.

## 7. Short framing

```text
Each generator is paired with a reliability calibrator. The generator proposes a sequence-level bias field; the calibrator audits whether that generated path is reliable for that generator and attenuates unreliable components. All calibrated fields are then fully composed and decoded through softmax.
```

Japanese:

```text
各生成モデルには、その生成経路を監査する信頼性校正器を対応させる。生成モデルは系列レベルのbias fieldを提案し、校正器はその経路がその生成モデルにとって信頼できるかを判定し、危険な成分を減衰または除去する。その後、全ての校正済みbias fieldを完全に合成し、softmaxで次token分布を得る。
```
