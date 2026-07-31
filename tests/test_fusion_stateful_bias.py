from __future__ import annotations

import torch

from opfusion.fusion_stateful_bias import candidate_grid, stateful_bias_logits


def test_stateful_bias_logits_is_finite_and_base_relative() -> None:
    sources = torch.tensor(
        [
            [0.0, 1.0, -1.0, 0.5],
            [1.0, 0.0, -0.5, 0.5],
            [-0.5, 1.5, 0.0, -1.0],
        ]
    )
    weights = torch.tensor([0.4, 0.35, 0.25])
    fused = stateful_bias_logits(sources, weights, alpha=0.5)
    assert fused.shape == sources[0].shape
    assert bool(torch.isfinite(fused).all())
    zero_specialists = weights.clone()
    zero_specialists[1:] = 0.0
    assert torch.allclose(stateful_bias_logits(sources, zero_specialists, alpha=1.0), sources[0])


def test_candidate_grid_is_unique_and_positive() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 72
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert all(candidate.memory > 0 for candidate in candidates)
    assert all(candidate.feedback > 0 for candidate in candidates)
    assert all(candidate.alpha > 0 for candidate in candidates)
