# Math Operator Units

This repository builds controlled GPT checkpoints for testing **logit-space bias fusion**. Mathematical operators are used because inputs, intermediate transformations, final values, stopping behavior, and invalid traces can be generated and verified exactly.

For one shared prefix `x`:

```text
bias_i(x) = logits_specialist_i(x) - logits_base(x)
```

The active research question is how several simultaneously evaluated bias fields can be composed without an external operator router.

## Current experimental result

Memoryless raw addition, normalized pooling, learned static coefficients, coordinate-wise gates, and candidate-token evidence mixtures do not preserve reliable autoregressive trajectories. The strongest validation result so far is a **continuous stateful source mixture**:

```text
state_t = memory * state_(t-1) + (1 - memory) * log(instantaneous_weights_t)
weights_t = softmax(state_t / temperature)
mixture_t = sum_i weights_t[i] * source_probability_t[i]
state_t += feedback * centered_log_support_i(generated_token_t)
```

All source weights remain strictly positive; every model is evaluated at every token. No operator id, task label, subset mask, external classifier, or discrete source switch is used.

A 48-condition fine search selected `memory=0.95`, `feedback=0.35`, and `temperature=1.0`. Independent validation over three model seeds and 36 examples per operator produced final-value/valid-trace macro accuracy `0.4222`, compared with `0.1000` for bias mean. Detailed results are in [`docs/stateful_fusion_search_results.md`](docs/stateful_fusion_search_results.md).

The result is not production-ready. NEG remains a failed unit, and the next experiment separates persistent reliability for numeric, structural, and stopping-token coordinates before moving to held-out compound programs.
