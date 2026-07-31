from __future__ import annotations

import torch

from opfusion.fusion_stateful_sparse_probability import candidate_grid, sparse_probability_weights


def test_sparse_probability_weights_are_positive_and_normalized() -> None:
    dense = torch.tensor([0.50, 0.25, 0.15, 0.07, 0.02, 0.01])
    sparse = sparse_probability_weights(dense, threshold=0.05, power=2.0)
    assert sparse.shape == dense.shape
    assert bool((sparse > 0).all())
    assert torch.isclose(sparse.sum(), torch.tensor(1.0))
    assert sparse[0] > dense[0]
    assert sparse[-1] < dense[-1]


def test_zero_threshold_power_one_preserves_weights_up_to_floor() -> None:
    dense = torch.softmax(torch.randn(6), dim=0)
    sparse = sparse_probability_weights(dense, threshold=0.0, power=1.0, floor=1e-9)
    assert torch.allclose(sparse, dense, atol=1e-6)


def test_candidate_grid_is_unique() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 96
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
