# Fusion Research Goals

This document defines the long-term research target for operator-unit fusion.

The project is not primarily a calculator, a router, or a replacement for symbolic computation. It is a controlled study of **logit-space semantics for model control**.

## 1. Core hypothesis

A controllable generation system can be built from typed bias operators:

```text
U_k = (M_k, C_k)
```

where:

- `M_k` is an operator-specific model that emits a bias field, logit contribution, or proposal.
- `C_k` is a corrector/gate that suppresses irrelevant, out-of-domain, or harmful output.
- fusion uses runtime-selected sets rather than assuming the whole registry is always active.
- corrected fusion combines only the semantically applicable contributions.

The fused output is:

```text
z_final = z_0 + Σ_{k in S_runtime} g_k(x) b_k(x)
```

The central question is not whether this improves arithmetic. The central question is whether human-defined logit-space bias operators can be learned, corrected, and composed while preserving measurable distributional and verifier effects.

## 2. Logit-space semantic target

Let a base model produce:

```text
z_0(v | x)
```

A control direction is a bias field:

```text
B(v | x) ∈ R^{|V|}
```

Its meaning is defined by effects such as:

```text
Δp_B = softmax(z_0 + B) - softmax(z_0)
ΔV(B) = E_{y ~ p_B}[V(y)] - E_{y ~ p_0}[V(y)]
```

Thus:

```text
meaning = distribution shift + verifier-score shift
```

This is the target semantics. Mathematical operator experiments are proxy domains for testing whether this semantics can be learned and composed under controlled conditions.

## 3. Difference from existing paradigms

This project is related to, but distinct from:

- Mixture-of-Experts: experts are routed sparsely, usually by a router.
- Adapter fusion: learned task adapters are combined inside a base model.
- Decoding-time expert methods: logits are steered by expert/anti-expert models.
- Neural operators: large models learn function-space maps, often PDE solution operators.
- Tool-using LMs: the model selects external tools and incorporates results.
- Program-search systems: LMs generate programs and external evaluators score them.

This project instead studies:

```text
logit-space bias semantics
+ typed bias operators
+ runtime-selected fusion sets
+ unit-specific applicability correctors
+ shared tokenizer/output ABI
+ inactive-leakage metrics
+ softmax/verifier effect measurement
```

Short distinction:

```text
MoE routes computation.
This project composes corrected controls.
```

## 4. What this could enable

### 4.1 Bias algebra control model

Goal:

```text
Represent generation controls as typed bias-field operations.
```

Examples:

- add or subtract a control direction
- complete a missing semantic component
- remove a projected component
- preserve useful orthogonal component
- compute agreement between two control fields
- enforce entropy/KL budget
- detect conflict between controls
- fuse multiple validators/scorers

Success criterion:

```text
Learned operators produce softmax/verifier effects close to exact or reference bias-field operators, including on unseen compositions.
```

### 4.2 Corrector-gated fusion model

Goal:

```text
Compose multiple small control fields without unrelated fields corrupting the result.
```

The corrector solves the interference problem:

```text
relevant unit -> preserve
irrelevant unit -> suppress
unknown/OOD input -> suppress
conflicting unit -> downweight or flag
```

Primary metric:

```text
inactive_leakage = Σ_{k inactive} g_k(x)
```

Additional metrics:

```text
unknown_operator_assimilation_rate
inactive_pmax
ood_entropy
ood_pmax
wrong_operator_projection_rate
```

### 4.3 Mode-switched runtime fusion

Goal:

```text
Select a small candidate fusion set for the current mode, then compose corrected bias fields in parallel.
```

Mode selection is not the main semantics. It is a runtime mechanism.

```text
mode selector:
  chooses candidate control fields

corrector:
  gates semantic applicability

fusion:
  composes corrected fields
```

This differs from routing to one expert. The target is safe composition of multiple control directions.

### 4.4 Compositional mathematical proxy

Goal:

```text
Use mathematical and bias-field operators as verifiable proxy domains.
```

Examples:

- scalar arithmetic
- vector and matrix algebra
- aggregation/statistics
- calculus operators
- probability/information operators
- bias/logit algebra
- PDE residual operators

Success criterion:

```text
Composed operator programs preserve exact or reference effects better than raw fusion and wrong-composition baselines.
```

### 4.5 Verifier-guided candidate generation

Goal:

```text
Generate candidate programs, formulas, transformations, or constructions and keep only what a verifier accepts.
```

The model should not be treated as a proof oracle. It is a candidate generator and bias composer.

Verifier classes:

- exact evaluator
- numeric tolerance evaluator
- residual evaluator
- type/shape/dimension checker
- symbolic simplifier
- theorem prover or proof assistant, later

### 4.6 Scientific operator sandbox

Goal:

```text
Move from arithmetic units to differential, integral, residual, conservation, and stability units.
```

Long-term PDE target:

```text
candidate solution / transformation / invariant
-> residual check
-> boundary and initial condition check
-> conservation/stability check
-> symbolic extraction or rejection
```

This is more realistic than claiming direct solution of arbitrary open PDEs.

## 5. Final target model classes

### Target A: Logit Bias Semantics Model

A system that treats model control as typed operations on bias fields:

```text
bias add / subtract / complete / project / remove / agree / normalize / entropy-match / KL-budget
```

Objective:

```text
Achieve controlled generation behavior whose meaning is measurable through softmax and verifier effects.
```

### Target B: Corrector-Gated Fusion Model

A fusion system that composes runtime-selected corrected units:

```text
S_runtime selected for the task or mode
correctors decide preserve/suppress
fusion remains stable as unit count grows
```

Objective:

```text
Scale from 4 units -> 16 -> 64 -> 256 without inactive leakage dominating.
```

### Target C: Operator Library Proxy Model

A model family made of small verifiable units:

```text
scalar + vector + linalg + calculus + probability + bias + control units
```

Objective:

```text
Solve unseen composed operator programs and measure whether learned composition tracks exact composition.
```

### Target D: Verifier-Guided Discovery Model

A system that proposes candidates and relies on verifiers:

```text
proposal -> exact/numeric/symbolic verifier -> mutation/rewrite -> improved proposal
```

Objective:

```text
Discover useful algorithms, transforms, invariants, or counterexamples in bounded benchmark domains.
```

### Target E: PDE Residual Fusion Model

A scientific computing system:

```text
differential units + boundary units + conservation units + residual verifier
```

Objective:

```text
Generate candidate PDE solutions or transformations whose residuals are lower than baselines.
```

## 6. What would count as genuinely new

The project should not claim novelty for generic expert routing, adapter composition, logit steering, neural operators, or arithmetic approximation.

A safer claim is:

```text
Learned bias modules are not assumed to be safely composable by raw addition. They become meaningful compositional control units only when their logit-space effects are typed, measured, and gated by applicability correctors.
```

A stronger claim, if supported by experiments, is:

```text
Runtime-selected sets of corrected bias operators can compose into unseen control transformations while limiting inactive leakage and preserving measurable softmax/verifier effects.
```

## 7. Milestone ladder

### Milestone 0: Core reproducibility

- ADD/SUM/NEG/ZERO/POS/ABS/MIN/MAX
- raw vs corrected fusion
- gate matrix
- leakage metrics
- OOD peakedness
- operator assimilation error

### Milestone 1: Bias algebra primitives

- bias add/subtract
- positive agreement
- projection removal
- completion target-current
- residual stability
- exact vs learned softmax effect comparison

### Milestone 2: Composition benchmark

- depth sweep
- length OOD
- type mismatch tests
- unseen operator programs
- wrong-composition baselines
- program-only vs distilled-unit comparison

### Milestone 3: Runtime-selected fusion sets

- mode-specific fusion sets
- 16, 64, 256 unit corrected fusion
- leakage scaling law
- conflict detection
- type-gate and pattern-gate ablations

### Milestone 4: Verifier loop

- candidate generation
- exact/numeric verifier
- mutation/rewrite
- keep only validated candidates

### Milestone 5: PDE sandbox

- heat / Poisson / Burgers residuals
- boundary and initial condition units
- residual descent and candidate generation

### Milestone 6: LLM bias-fusion prototype

- fixed base LLM
- small bias/corrector units
- mode-switched fusion sets
- softmax effect measurement
- verifier-score shift measurement
- comparison against raw steering and wrong-mode activation

## 8. Non-goals

The project should not initially claim:

- arbitrary theorem proving
- direct solution of all open PDEs
- replacement of symbolic solvers
- general AGI-style reasoning
- uncontrolled natural-language semantic correctness
- that raw LLM bias addition is safe
- that this is simply a faster MoE router

The correct initial framing is:

```text
verifiable logit-space semantics and corrector-gated compositional bias fusion
```

## 9. North-star goal

The north-star target is:

```text
A typed, runtime-selected library of corrected bias operators that can compose model-control directions in logit space, while using verifiers and leakage metrics to separate meaningful control effects from raw-fusion artifacts.
```
