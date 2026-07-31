from __future__ import annotations

import torch

from opfusion.fusion_stateful_geometric import candidate_grid, geometric_logits


def test_geometric_logits_matches_base_at_zero_alpha() -> None:
    sources = torch.tensor(
        [
            [0.0, 1.0, -1.0],
            [2.0, 0.0, -1.0],
            [-1.0, 2.0, 0.0],
        ]
    )
    weights = torch.tensor([0.4, 0.35, 0.25])
    fused = geometric_logits(sources, weights, alpha=0.0)
    assert torch.allclose(fused, torch.log_softmax(sources[0], dim=-1))


def test_geometric_logits_is_finite() -> None:
    sources = torch.randn(6, 17)
    weights = torch.softmax(torch.randn(6), dim=0)
    fused = geometric_logits(sources, weights, alpha=1.5)
    assert fused.shape == (17,)
    assert bool(torch.isfinite(fused).all())


def test_candidate_grid_is_unique() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 72
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
