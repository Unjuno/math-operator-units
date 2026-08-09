from __future__ import annotations

import torch

from opfusion import fusion_timescale_specific_boundary_reset as specific


def test_blend_timescale_full_reset() -> None:
    initial = torch.tensor([1.0, 2.0])
    instant = torch.tensor([5.0, 6.0])
    actual = specific.blend_timescale_state(
        initial, instant, reset_fraction=1.0
    )
    assert torch.equal(actual, instant)


def test_blend_timescale_full_carry() -> None:
    initial = torch.tensor([1.0, 2.0])
    instant = torch.tensor([5.0, 6.0])
    actual = specific.blend_timescale_state(
        initial, instant, reset_fraction=0.0
    )
    assert torch.equal(actual, initial)


def test_blend_timescale_partial_reset() -> None:
    initial = torch.tensor([1.0, 3.0])
    instant = torch.tensor([5.0, 7.0])
    actual = specific.blend_timescale_state(
        initial, instant, reset_fraction=0.5
    )
    assert torch.allclose(actual, torch.tensor([3.0, 5.0]))
