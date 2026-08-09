from __future__ import annotations

import pytest
import torch

from opfusion import fusion_controller_state_boundary_carry as boundary


def test_blend_boundary_state_full_reset_matches_instant() -> None:
    initial = torch.tensor([1.0, 2.0, 3.0])
    instant = torch.tensor([4.0, 5.0, 6.0])
    actual = boundary.blend_boundary_state(initial, instant, reset_fraction=1.0)
    assert torch.equal(actual, instant)


def test_blend_boundary_state_zero_reset_carries_initial() -> None:
    initial = torch.tensor([1.0, 2.0, 3.0])
    instant = torch.tensor([4.0, 5.0, 6.0])
    actual = boundary.blend_boundary_state(initial, instant, reset_fraction=0.0)
    assert torch.equal(actual, initial)


def test_blend_boundary_state_half_reset_is_midpoint() -> None:
    initial = torch.tensor([1.0, 2.0, 3.0])
    instant = torch.tensor([5.0, 6.0, 7.0])
    actual = boundary.blend_boundary_state(initial, instant, reset_fraction=0.5)
    assert torch.allclose(actual, torch.tensor([3.0, 4.0, 5.0]))


def test_blend_boundary_state_rejects_invalid_reset() -> None:
    initial = torch.ones(3)
    instant = torch.zeros(3)
    with pytest.raises(ValueError):
        boundary.blend_boundary_state(initial, instant, reset_fraction=-0.1)
    with pytest.raises(ValueError):
        boundary.blend_boundary_state(initial, instant, reset_fraction=1.1)


def test_blend_boundary_state_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        boundary.blend_boundary_state(
            torch.ones(2), torch.ones(3), reset_fraction=0.5
        )
