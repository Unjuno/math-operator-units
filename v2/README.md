# Operator Fusion V2

Clean-room reimplementation of the behaviorally identified operator/model composition stack.

## Design rule

V2 keeps the experimental principles and discards accumulated experiment-specific helpers. The core must not know operation names, state cardinality, tokenizer structure, neural architecture, or analytic operator definitions.

Current core pipeline:

`register -> active invariant relation identification -> UNRESOLVED on residual equivalence -> minimal grounded bridge when available -> common-interface execution -> composition`

`core.py` contains only the registry, Bayesian/MI query selection, residual bridge step, and sequential composer.

## First clean-room screen

Checkpoint: `../docs/v2_clean_room_checkpoint_v1.json`

The same `core.py` was exercised in three independently defined environments:

- discrete finite-state operators with private output relabeling;
- continuous scalar functions with positive affine output gauge;
- independently trained stochastic MLP/GRU specialists with independent private input/output permutations and actual multinomial token sampling.

Results:

- discrete routing: 60/60, wrong 0, unresolved 0; depth1-5 programs 50/50;
- continuous routing: 60/60, wrong 0, unresolved 0; depth1-5 programs 50/50;
- stochastic neural seeds 15-17: 96/96 routes, wrong 0, unresolved 0; depth1-5 programs 150/150; mean relation queries 2.99.

Exact relation-equivalent pairs were also tested. With invariant relations only, both discrete and continuous screens returned `UNRESOLVED` 40/40. Adding one grounded bridge resolved 40/40 in each environment with wrong 0.

## Output-ABI ablation

Script: `experiment_output_adapter.py`

Checkpoint: `../docs/v2_output_adapter_ablation_v1.json`

Correct routing does not imply composability. Across three stochastic-neural seeds, routing was 108/108, but interpreting private output tokens as common states gave only 36/360 correct programs. A lazy output adapter recovered from the source-local right-identity contract `f(a,0)=a` restored 360/360, matching the oracle adapter.

This is a deliberate separation of failure modes:

`source identity != output ABI alignment`.

## Current boundaries

V2 is not yet the final generic system.

- The environment wrapper still supplies a shared semantic intervention/probe domain.
- Input adapter discovery is not yet first-class.
- The first three-environment screen supplied `execute_common` adapters from the environment; the output-ABI ablation begins removing that assumption.
- The right-identity adapter currently uses one grounded contract observation per common state.
- Open-world gauge-family misspecification and persistent model-instance uncertainty are not wired into the clean-room core yet.
- Composition is currently hard source selection; fusion-safe transition latents and sparse probability fusion will be reintroduced only after ABI discovery is explicit.
- Neural sources are small synthetic specialists, not unrelated pretrained model families.

## Next V2 experiments

1. Make input adapter discovery explicit and remove the hidden shared semantic-input transport.
2. Add open-world predictive validation without changing the registry interface.
3. Add persistent model-instance uncertainty to stochastic registration.
4. Reintroduce behavior-derived transition latents and top-dominant sparse fusion.
5. Run the unchanged V2 interface against unrelated pretrained model families.
