# Research branch map

This repository accumulated many one-experiment branches while the operator-unit fusion hypothesis was evolving. The branch names are retained as immutable research history unless an explicit cleanup tool is available; they should not be treated as simultaneously active work.

## Active research line

- `experiment/unseen-source-generalization` — **ACTIVE**. Draft PR #40 is the canonical continuation point for the current generic-composition work.

Current hypothesis: source-specific interfaces/ABIs are behaviorally aligned into a common action or state-transition probability space, followed by sparse probability fusion and program composition.

## Historical experiment branches

All other `experiment/*` branches should be treated as **ARCHIVE / READ-ONLY** unless explicitly reactivated. Their PRs are being closed rather than merged so the exact code/result lineage remains inspectable.

The important historical phases are:

1. stateful continuous probability/logit fusion;
2. oracle and learned operator-state composition;
3. boundary/reset and recursive-depth studies;
4. identity-free unseen-source field scoring;
5. residual/null admission and behavioral/semantic-view admission;
6. metamorphic task-contract admission;
7. current common-action / heterogeneous-ABI work consolidated in PR #40.

## Obsolete branch suffixes

- `*-ci-base` — historical helper bases created when experiments were run through Actions. Do not use for new work.
- `*-runner` — historical Actions runner branches. Do not use for new work.

Experiments are now computed in the container. GitHub is used for persistence, review, and reproducibility.

## New branch policy

Do not create a new branch for every screening experiment. Continue on `experiment/unseen-source-generalization` while the scientific object remains common-action fusion. Create a new branch only when one of these changes:

- the central scientific claim changes materially;
- a clean mergeable implementation is being separated from research notebooks/screens;
- the work needs independent review/release lifecycle.

If a new line is required, prefer `research/<short-topic>` for long-lived research integration and `experiment/<short-topic>` only for isolated, disposable ablations.

## Restart rule

After a session/container loss, read in order:

1. `docs/research/README.md`
2. `docs/container_research_checkpoint_2026-08-20.md`
3. the newest compact JSON checkpoint/result under `docs/`
4. PR #40 description and latest commits

Do not infer the current direction from the large set of historical branch names alone.
