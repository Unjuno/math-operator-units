from __future__ import annotations

import torch

from opfusion.fusion_coordinate_pool import (
    PoolCandidate,
    candidate_grid,
    compose_logits,
    coordinate_residual,
)


def test_candidate_grid_covers_robust_pooling_families() -> None:
    modes = {candidate.mode for candidate in candidate_grid()}
    assert {
        "positive_max",
        "positive_top2",
        "positive_logmeanexp",
        "max_veto",
        "signed_absmax",
        "median",
        "trimmed_mean",
        "sparse_sum",
    } <= modes


def test_coordinate_pooling_preserves_shape_for_batched_and_unbatched_inputs() -> None:
    torch.manual_seed(2)
    candidate = PoolCandidate("test", "positive_max", alpha=0.75, threshold=0.25)
    base = torch.randn(13)
    units = torch.randn(5, 13)
    assert compose_logits(base, units, candidate).shape == base.shape

    batched_base = torch.randn(4, 13)
    batched_units = torch.randn(4, 5, 13)
    assert compose_logits(batched_base, batched_units, candidate).shape == batched_base.shape


def test_positive_max_keeps_the_strongest_coordinate_support_without_unit_switching() -> None:
    base = torch.zeros(7)
    units = torch.zeros(5, 7)
    units[0, 2] = 3.0
    units[1, 4] = 2.0
    candidate = PoolCandidate("test", "positive_max", alpha=1.0, threshold=0.0)
    residual = coordinate_residual(base, units, candidate)
    assert residual[2] > residual[0]
    assert residual[4] > residual[0]
    assert torch.isfinite(residual).all()


def test_veto_penalizes_coordinates_with_strong_opposing_evidence() -> None:
    base = torch.zeros(5)
    units = torch.zeros(5, 5)
    units[0, 1] = 3.0
    units[1, 1] = -4.0
    no_veto = PoolCandidate("a", "max_veto", alpha=1.0, veto=0.0)
    with_veto = PoolCandidate("b", "max_veto", alpha=1.0, veto=1.0)
    assert coordinate_residual(base, units, with_veto)[1] < coordinate_residual(base, units, no_veto)[1]


def test_all_pooling_modes_are_finite() -> None:
    torch.manual_seed(5)
    base = torch.randn(3, 17)
    units = torch.randn(3, 5, 17)
    representatives = {
        "positive_max": PoolCandidate("a", "positive_max", 1.0, 0.25),
        "positive_top2": PoolCandidate("b", "positive_top2", 1.0, 0.25),
        "positive_logmeanexp": PoolCandidate("c", "positive_logmeanexp", 1.0, 0.25, 0.5),
        "max_veto": PoolCandidate("d", "max_veto", 1.0, 0.25, veto=0.5),
        "signed_absmax": PoolCandidate("e", "signed_absmax", 0.5, 0.25),
        "median": PoolCandidate("f", "median", 1.0),
        "trimmed_mean": PoolCandidate("g", "trimmed_mean", 1.0),
        "sparse_sum": PoolCandidate("h", "sparse_sum", 0.25, 0.5),
    }
    for candidate in representatives.values():
        output = compose_logits(base, units, candidate)
        assert output.shape == base.shape
        assert torch.isfinite(output).all()
