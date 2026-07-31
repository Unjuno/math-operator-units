from __future__ import annotations

import torch

from opfusion.fusion_stateful_sparse_bias import candidate_grid, sparse_bias_logits


def test_sparse_bias_renormalizes_surviving_specialists() -> None:
    sources = torch.tensor(
        [
            [0.0, 1.0, -1.0],
            [2.0, 0.0, -1.0],
            [-1.0, 2.0, 0.0],
            [0.0, -1.0, 2.0],
        ]
    )
    weights = torch.tensor([0.5, 0.30, 0.15, 0.05])
    fused, sparse = sparse_bias_logits(sources, weights, alpha=1.0, threshold=0.10)
    assert fused.shape == sources[0].shape
    assert torch.isclose(sparse.sum(), torch.tensor(1.0))
    assert sparse[-1] == 0
    assert bool(torch.isfinite(fused).all())


def test_sparse_candidate_grid_is_unique() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 72
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert all(candidate.threshold > 0 for candidate in candidates)
    assert all(candidate.alpha >= 1.0 for candidate in candidates)
