# Fusion Research Goals

This document defines the long-term research target for operator-unit fusion.

## 1. Core hypothesis

A large system can be built from many operator-specific units:

```text
U_k = (M_k, C_k)
```

where:

- `M_k` is an operator-specific model.
- `C_k` is a corrector/gate that suppresses irrelevant or harmful output.
- all units can be run in an always-on manner.
- fusion combines only the surviving useful contributions.

The fused output is:

```text
z_final = z_0 + Σ_k g_k(x) b_k(x)
```

The research question is not merely whether this improves arithmetic. The central question is whether self-suppressing operator units can become a scalable substrate for compositional reasoning, mathematical search, scientific computing, and controllable generation.

## 2. Difference from existing paradigms

This project is related to, but distinct from:

- Mixture-of-Experts: experts are routed sparsely, usually by a central router.
- Adapter fusion: learned task adapters are combined inside a base model.
- Decoding-time expert methods: logits are steered by expert/anti-expert models.
- Neural operators: large models learn function-space maps, often PDE solution operators.
- Tool-using LMs: the model selects external tools and incorporates results.
- Program-search systems: LMs generate programs and external evaluators score them.

This project instead studies:

```text
always-on typed operator units
+ unit-specific correctors
+ shared tokenizer/output ABI
+ operator registry
+ fusion/leakage metrics
+ verifier-backed evaluation
```

## 3. What this could enable

### 3.1 A compositional mathematical operator model

Goal:

```text
Build many small operator models that compose into larger mathematical behavior.
```

Examples:

- scalar arithmetic
- vector and matrix algebra
- aggregation/statistics
- discrete selection
- calculus operators
- probability/information operators
- bias/logit algebra
- PDE residual operators

Success criterion:

```text
Composed operator programs solve tasks not directly trained as monolithic tasks.
```

### 3.2 Self-suppressing expert libraries

Goal:

```text
Run many units at once without unrelated units corrupting the result.
```

The corrector must solve the interference problem:

```text
irrelevant unit -> suppress
relevant unit -> preserve
conflicting unit -> downweight or flag
```

Primary metric:

```text
inactive_leakage = Σ_{k inactive} g_k(x)
```

### 3.3 Verifier-guided mathematical discovery

Goal:

```text
Generate candidate programs, formulas, transformations, or constructions and keep only what a verifier accepts.
```

The model should not be treated as a proof oracle. It is a candidate generator and bias composer.

Candidate classes:

- algorithm candidates
- formula candidates
- rewrite candidates
- invariant candidates
- PDE solution candidates
- counterexample candidates
- proof-step candidates

Verifier classes:

- exact evaluator
- numeric tolerance evaluator
- residual evaluator
- type/shape/dimension checker
- symbolic simplifier
- theorem prover or proof assistant, later

### 3.4 Scientific operator system

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

### 3.5 Controllable generation through bias algebra

Goal:

```text
Represent generation controls as typed bias-field operations.
```

Examples:

- add style or task bias
- remove unsafe/risky direction
- preserve useful orthogonal component
- enforce entropy/KL budget
- detect conflict between controls
- fuse multiple validators/scorers

This differs from ordinary prompt engineering because the control is explicit in bias/logit/operator space.

## 4. Final target model classes

### Target A: Operator Library Model

A model family made of many small units:

```text
scalar + vector + linalg + calculus + probability + bias + control units
```

Objective:

```text
Solve unseen composed operator programs better than monolithic small baselines.
```

### Target B: Self-Suppressing Fusion Model

A fusion system that can run many units simultaneously:

```text
all units active at runtime
correctors decide preserve/suppress
fusion remains stable as unit count grows
```

Objective:

```text
Scale from 4 units -> 16 -> 64 -> 256 without leakage dominating.
```

### Target C: Verifier-Guided Discovery Model

A system that proposes candidates and relies on verifiers:

```text
proposal -> exact/numeric/symbolic verifier -> mutation/rewrite -> improved proposal
```

Objective:

```text
Discover useful algorithms, transforms, invariants, or counterexamples in bounded benchmark domains.
```

### Target D: Bias Algebra Control Model

A model that treats logit/bias control as an algebra:

```text
bias add / subtract / project / remove / agree / normalize / entropy-match / KL-budget
```

Objective:

```text
Achieve controlled generation behavior that is inspectable and compositional.
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

## 5. What would count as genuinely new

The project should not claim novelty for generic expert routing, adapter composition, logit steering, or neural operators.

The stronger claim is:

```text
Typed operator-specific units can be trained independently, paired with self-suppressing correctors, and fused always-on while preserving compositional semantics and limiting inactive leakage.
```

A still stronger claim is:

```text
This architecture can serve as a substrate for verifier-guided mathematical and scientific discovery.
```

## 6. Milestone ladder

### Milestone 0: Core reproducibility

- ADD/SUM/NEG/ZERO/POS/ABS/MIN/MAX
- raw vs corrected fusion
- gate matrix
- leakage metrics

### Milestone 1: Operator expansion

- scalar numeric
- compare/sort/top-k
- aggregation/statistics
- linalg dot/proj/matmul
- bias algebra units

### Milestone 2: Composition benchmark

- depth sweep
- length OOD
- type mismatch tests
- unseen operator programs
- program-only vs distilled-unit comparison

### Milestone 3: Large unit count

- 16, 64, 256 unit always-on fusion
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

### Milestone 6: Discovery benchmarks

- matrix multiplication decompositions
- sorting/search heuristics
- combinatorial constructions
- invariant discovery
- PDE transformation candidates

## 7. Non-goals

The project should not initially claim:

- arbitrary theorem proving
- direct solution of all open PDEs
- replacement of symbolic solvers
- general AGI-style reasoning
- uncontrolled natural-language semantic correctness

The correct initial framing is:

```text
verifiable candidate generation and compositional operator fusion
```

## 8. North-star goal

The north-star target is:

```text
A typed, always-on library of self-suppressing operator units that can compose mathematical, symbolic, numerical, and control operations into larger behaviors, while using verifiers to separate valid discoveries from hallucinated outputs.
```
