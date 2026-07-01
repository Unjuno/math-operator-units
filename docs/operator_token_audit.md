# Operator Token Audit and Expansion Plan

This document audits the current operator/token design and defines what should be added before the tokenizer is treated as frozen.

## 1. Current status

Current tokenizer profile:

```text
tokenizer_core_v1
```

Current registry core includes:

```text
<ZERO>
<ID>
<NEG>
<ADD>
<SUB>          program-only
<ABS>
<POS>
<MIN>
<MAX>
<BIAS_ADD>
<BIAS_SUB>
<BIAS_CENTER>
<BIAS_POS>
<GATE>
<SUPPRESS>
<ABSTAIN>
```

Current direct canonical tokens are namespace-safe for the entries already present.

No fatal direct-token collision was found in the current `core_v0` set.

## 2. Important future collision risks

The current design is safe only if future operators use canonical namespace tokens.

### 2.1 DIV conflict

`DIV` can mean either scalar division or vector-field divergence.

Forbidden ambiguous token:

```text
<OP_DIV>
```

Required canonical split:

```text
<OP_SCALAR_DIV>
<OP_CALC_DIVERGENCE>
```

Human-facing aliases may still use `div`, but model-facing tokens must be distinct.

### 2.2 NEG / SUB symbol conflict

`-` can mean unary negation or binary subtraction.

Required canonical split:

```text
<OP_SCALAR_NEG>
<OP_SCALAR_SUB>
```

`<SUB>` may remain program-only, but if a distilled subtract unit is trained later, it must use `<OP_SCALAR_SUB>` or a reserved token assigned to `scalar.sub`.

### 2.3 MIN / MAX arity conflict

`min` and `max` can mean binary pointwise min/max or sequence reduction.

Required canonical split:

```text
<OP_SCALAR_MIN2>
<OP_SCALAR_MAX2>
<OP_AGG_MIN>
<OP_AGG_MAX>
```

The current registry entry `scalar.min` and `scalar.max` use `ScalarSeq -> Scalar`, so they are reduction-style. If binary pointwise variants are added, they must be separate.

### 2.4 ABS domain conflict

`abs` may apply to scalar, vector norm-like magnitude, complex modulus, or bias field elementwise absolute value.

Required canonical split:

```text
<OP_SCALAR_ABS>
<OP_COMPLEX_ABS>
<OP_VECTOR_NORM_L2>
<OP_BIAS_ABS>
```

### 2.5 NORM / NORMALIZE conflict

`norm` returns a scalar. `normalize` returns a value in the same space as input.

Required canonical split:

```text
<OP_AGG_NORM_L1>
<OP_AGG_NORM_L2>
<OP_AGG_NORM_INF>
<OP_AGG_NORMALIZE_L2>
```

### 2.6 LOG conflict

`log` can mean natural log, base-2 log, base-10 log, log probability, or logit-space transform.

Required canonical split:

```text
<OP_SCALAR_LOG>
<OP_SCALAR_LOG2>
<OP_SCALAR_LOG10>
<OP_PROB_LOG_PROB>
<OP_BIAS_LOG_RATIO>
```

### 2.7 SOFTMAX domain conflict

Softmax over logits is different from normalization of positive weights.

Required canonical split:

```text
<OP_PROB_SOFTMAX>
<OP_PROB_LOGSOFTMAX>
<OP_AGG_NORMALIZE_SUM>
```

### 2.8 PROJ conflict

Projection may mean vector projection, constraint projection, subspace projection, or bias projection.

Required canonical split:

```text
<OP_LINALG_PROJ_VECTOR>
<OP_LINALG_PROJ_SUBSPACE>
<OP_OPT_PROJECT_CONSTRAINT>
<OP_BIAS_PROJ>
```

### 2.9 ARGMAX vs MAX conflict

`max` returns a value. `argmax` returns an index.

Required canonical split:

```text
<OP_AGG_MAX>
<OP_COMPARE_ARGMAX>
```

### 2.10 TOOL / SEMANTIC operators must not share math tokens

Non-mathematical operators must be separated from mathematical operators even if they resemble mathematical operations.

Examples:

```text
<OP_SEM_SIMILARITY>      # semantic score, not metric distance unless specified
<OP_TOOL_RETRIEVE>       # external-state operation
<OP_CTRL_ROUTE>          # control/routing operation
```

## 3. Operators that should be added before tokenizer freeze

The current direct operator list is too small if the final system will mix many mathematical and non-mathematical units.

The following operator families should be explicitly represented either as direct tokens or planned reserved assignments.

## 4. Recommended direct token expansion

### 4.1 Scalar elementary operators

```text
<OP_SCALAR_SUB>
<OP_SCALAR_MUL>
<OP_SCALAR_DIV>
<OP_SCALAR_SAFE_DIV>
<OP_SCALAR_POW>
<OP_SCALAR_SQ>
<OP_SCALAR_SQRT>
<OP_SCALAR_EXP>
<OP_SCALAR_LOG>
<OP_SCALAR_LOG2>
<OP_SCALAR_LOG10>
<OP_SCALAR_FLOOR>
<OP_SCALAR_CEIL>
<OP_SCALAR_ROUND>
<OP_SCALAR_TRUNC>
<OP_SCALAR_MOD>
<OP_SCALAR_REM>
<OP_SCALAR_CLIP>
<OP_SCALAR_SIGN>
<OP_SCALAR_NEG_PART>
```

### 4.2 Trigonometric / hyperbolic operators

```text
<OP_SCALAR_SIN>
<OP_SCALAR_COS>
<OP_SCALAR_TAN>
<OP_SCALAR_ASIN>
<OP_SCALAR_ACOS>
<OP_SCALAR_ATAN>
<OP_SCALAR_ATAN2>
<OP_SCALAR_SINH>
<OP_SCALAR_COSH>
<OP_SCALAR_TANH>
<OP_SCALAR_ASINH>
<OP_SCALAR_ACOSH>
<OP_SCALAR_ATANH>
<OP_SCALAR_HYPOT>
```

### 4.3 Comparison operators

```text
<OP_COMPARE_EQ>
<OP_COMPARE_NEQ>
<OP_COMPARE_LT>
<OP_COMPARE_LE>
<OP_COMPARE_GT>
<OP_COMPARE_GE>
<OP_COMPARE_ARGMIN>
<OP_COMPARE_ARGMAX>
<OP_COMPARE_TOPK>
<OP_COMPARE_SORT>
<OP_COMPARE_RANK>
```

### 4.4 Aggregation / statistics operators

```text
<OP_AGG_SUM>
<OP_AGG_PROD>
<OP_AGG_MEAN>
<OP_AGG_VAR>
<OP_AGG_STD>
<OP_AGG_MIN>
<OP_AGG_MAX>
<OP_AGG_MEDIAN>
<OP_AGG_MODE>
<OP_AGG_QUANTILE>
<OP_AGG_CUMSUM>
<OP_AGG_CUMPROD>
<OP_AGG_NORM_L1>
<OP_AGG_NORM_L2>
<OP_AGG_NORM_INF>
<OP_AGG_NORMALIZE_L2>
<OP_AGG_NORMALIZE_SUM>
<OP_AGG_CENTER>
<OP_AGG_STANDARDIZE>
```

### 4.5 Boolean / set operators

```text
<OP_LOGIC_NOT>
<OP_LOGIC_AND>
<OP_LOGIC_OR>
<OP_LOGIC_XOR>
<OP_LOGIC_IMPLIES>
<OP_LOGIC_IFF>

<OP_SET_UNION>
<OP_SET_INTERSECT>
<OP_SET_DIFF>
<OP_SET_COMPLEMENT>
<OP_SET_MEMBER>
<OP_SET_CARDINALITY>
```

### 4.6 Number theory / discrete arithmetic

```text
<OP_NUMTHEORY_GCD>
<OP_NUMTHEORY_LCM>
<OP_NUMTHEORY_DIVISIBLE>
<OP_NUMTHEORY_EXACT_DIV>
<OP_NUMTHEORY_FRAC_PAIR>
<OP_NUMTHEORY_FRAC_REDUCE>
<OP_NUMTHEORY_PRIME_CHECK>
<OP_NUMTHEORY_FACTOR>
```

### 4.7 Linear algebra operators

```text
<OP_LINALG_DOT>
<OP_LINALG_OUTER>
<OP_LINALG_MATMUL>
<OP_LINALG_TRANSPOSE>
<OP_LINALG_TRACE>
<OP_LINALG_DET>
<OP_LINALG_INV>
<OP_LINALG_PINV>
<OP_LINALG_SOLVE>
<OP_LINALG_EIGEN>
<OP_LINALG_SVD>
<OP_LINALG_QR>
<OP_LINALG_CHOLESKY>
<OP_LINALG_PROJ_VECTOR>
<OP_LINALG_PROJ_SUBSPACE>
<OP_LINALG_ORTH_REMOVE>
<OP_LINALG_GRAM_SCHMIDT>
```

### 4.8 Calculus / functional operators

```text
<OP_CALC_DIFF>
<OP_CALC_DX>
<OP_CALC_DT>
<OP_CALC_DXX>
<OP_CALC_GRAD>
<OP_CALC_DIVERGENCE>
<OP_CALC_CURL>
<OP_CALC_LAPLACIAN>
<OP_CALC_JACOBIAN>
<OP_CALC_HESSIAN>
<OP_CALC_INTEGRAL>
<OP_CALC_LINE_INTEGRAL>
<OP_CALC_CONVOLUTION>
<OP_CALC_FOURIER>
<OP_CALC_INV_FOURIER>
<OP_CALC_LAPLACE_TRANSFORM>
<OP_CALC_TAYLOR>
<OP_CALC_LIMIT>
```

### 4.9 Probability / information operators

```text
<OP_PROB_SIGMOID>
<OP_PROB_SOFTMAX>
<OP_PROB_LOGSOFTMAX>
<OP_PROB_ENTROPY>
<OP_PROB_KL>
<OP_PROB_JSD>
<OP_PROB_CROSS_ENTROPY>
<OP_PROB_EXPECTATION>
<OP_PROB_VARIANCE>
<OP_PROB_SAMPLE>
<OP_PROB_BAYES_UPDATE>
<OP_PROB_LOGIT>
<OP_PROB_TEMPERATURE>
```

### 4.10 Bias algebra operators

```text
<OP_BIAS_ABS>
<OP_BIAS_NEG_PART>
<OP_BIAS_NORM>
<OP_BIAS_NORMALIZE>
<OP_BIAS_MIN>
<OP_BIAS_MAX>
<OP_BIAS_AGREE>
<OP_BIAS_PROJ>
<OP_BIAS_REMOVE>
<OP_BIAS_ORTH_ADD>
<OP_BIAS_MEDIAN>
<OP_BIAS_RANK_FUSE>
<OP_BIAS_SNR>
<OP_BIAS_KL_BUDGET>
<OP_BIAS_ENTROPY_MATCH>
<OP_BIAS_CLIP>
<OP_BIAS_MASK>
<OP_BIAS_LOG_RATIO>
<OP_BIAS_CAUSAL_DELTA>
<OP_BIAS_INTERACTION_RESIDUAL>
```

### 4.11 Optimization / solver operators

```text
<OP_OPT_GRAD_STEP>
<OP_OPT_NEWTON_STEP>
<OP_OPT_PROX_STEP>
<OP_OPT_PROJECT_CONSTRAINT>
<OP_OPT_LAGRANGE_STEP>
<OP_OPT_FIXED_POINT>
<OP_OPT_ARGMIN_CONT>
<OP_OPT_ARGMAX_CONT>
<OP_OPT_LINE_SEARCH>
<OP_OPT_RESIDUAL_DESCENT>
```

### 4.12 PDE / residual operators

```text
<OP_PDE_ODE_RESIDUAL>
<OP_PDE_RESIDUAL>
<OP_PDE_HEAT_RESIDUAL>
<OP_PDE_BURGERS_RESIDUAL>
<OP_PDE_POISSON_RESIDUAL>
<OP_PDE_WAVE_RESIDUAL>
<OP_PDE_NS_RESIDUAL>
<OP_PDE_BC_ENFORCE>
<OP_PDE_IC_ENFORCE>
<OP_PDE_CONSERVATION_CHECK>
<OP_PDE_ENERGY>
<OP_PDE_LYAPUNOV>
<OP_PDE_STABILITY_CHECK>
```

### 4.13 Tensor / shape / program operators

These are not pure math operators, but they are necessary for real computation graphs.

```text
<OP_TENSOR_RESHAPE>
<OP_TENSOR_TRANSPOSE>
<OP_TENSOR_SLICE>
<OP_TENSOR_CONCAT>
<OP_TENSOR_STACK>
<OP_TENSOR_GATHER>
<OP_TENSOR_SCATTER>
<OP_TENSOR_MASK>
<OP_TENSOR_BROADCAST>
<OP_TENSOR_EINSUM>

<OP_CTRL_COMPOSE>
<OP_CTRL_PIPE>
<OP_CTRL_BRANCH>
<OP_CTRL_SELECT>
<OP_CTRL_ROUTE>
<OP_CTRL_HALT>
<OP_CTRL_LOOP>
<OP_CTRL_SCAN>
<OP_CTRL_REDUCE>
<OP_CTRL_MAP>
<OP_CTRL_APPLY>
```

### 4.14 Non-mathematical semantic/tool operators

```text
<OP_SEM_TOKENIZE>
<OP_SEM_DETOKENIZE>
<OP_SEM_PARSE>
<OP_SEM_SEMANTIC_SIM>
<OP_SEM_ENTAILS>
<OP_SEM_CONTRADICTS>
<OP_SEM_CLASSIFY>
<OP_SEM_SUMMARIZE>
<OP_SEM_TRANSLATE>
<OP_SEM_STYLE_SHIFT>

<OP_TOOL_RETRIEVE>
<OP_TOOL_RERANK>
<OP_TOOL_FETCH>
<OP_TOOL_CALL>
<OP_TOOL_READ_FILE>
<OP_TOOL_WRITE_FILE>
<OP_TOOL_EXEC_CODE>
<OP_TOOL_VERIFY_EXTERNAL>
```

## 5. Project-specific operators worth inventing

These operators are not standard mathematical primitives, but are useful for this architecture.

### 5.1 Applicability / suppression operators

```text
<OP_CTRL_APPLICABILITY>
<OP_CTRL_INHIBIT>
<OP_CTRL_EXCITE>
<OP_CTRL_LEAKAGE_PENALTY>
<OP_CTRL_TYPE_GATE>
<OP_CTRL_PATTERN_GATE>
<OP_CTRL_CONFIDENCE_GATE>
<OP_CTRL_UNCERTAINTY_GATE>
```

### 5.2 Fusion operators

```text
<OP_FUSION_SUM>
<OP_FUSION_WEIGHTED_SUM>
<OP_FUSION_PRODUCT_OF_EXPERTS>
<OP_FUSION_MIXTURE>
<OP_FUSION_VOTE>
<OP_FUSION_MEDIAN>
<OP_FUSION_TRIMMED_MEAN>
<OP_FUSION_CONFLICT_DETECT>
<OP_FUSION_CONSENSUS>
<OP_FUSION_DISAGREEMENT>
```

### 5.3 Verifier operators

```text
<OP_VERIFY_EXACT>
<OP_VERIFY_NUMERIC_TOL>
<OP_VERIFY_TYPE>
<OP_VERIFY_SHAPE>
<OP_VERIFY_DIMENSION>
<OP_VERIFY_RESIDUAL>
<OP_VERIFY_INVARIANT>
<OP_VERIFY_COUNTEREXAMPLE>
```

### 5.4 Candidate-generation operators

```text
<OP_CANDIDATE_PERTURB>
<OP_CANDIDATE_MUTATE_PROGRAM>
<OP_CANDIDATE_SIMPLIFY_PROGRAM>
<OP_CANDIDATE_REWRITE>
<OP_CANDIDATE_DISTILL>
```

These should be classified as `non_math_control`, `hybrid`, or `verifier`, not as `math_exact`.

## 6. Recommendation

Before freezing tokenizer v1, add direct tokens for the full recommended expansion above, or explicitly place each one into a reserved-token assignment table.

Do not rely only on generic reserved tokens if an operator is clearly expected to become central. Central operators deserve stable human-readable canonical tokens.

Recommended rule:

```text
Core and expected operators: direct named tokens.
Rare or speculative operators: reserved slots.
Unstable semantic/tool operators: fallback spelling or reserved slots.
```

## 7. Next action

Create one machine-readable file:

```text
configs/operators/operator_expansion_backlog.yaml
```

It should include every recommended operator with:

```yaml
opcode:
canonical_token:
kind:
domain:
type_signature:
status: planned | active | reserved | program_only | speculative
priority: S0 | S1 | S2 | S3 | S4 | S5
notes:
```
