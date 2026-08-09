from __future__ import annotations

import pytest
import torch

from opfusion import fusion_posterior_transition_boundary_gate as gate


def test_dynamic_gate_is_zero_for_identical_one_hot_posterior() -> None:
    posterior = torch.tensor([0.0, 1.0, 0.0, 0.0])
    actual = gate.posterior_transition_gate(posterior, posterior)
    assert actual.item() == 0.0


def test_dynamic_gate_is_one_for_orthogonal_one_hot_posteriors() -> None:
    previous = torch.tensor([1.0, 0.0, 0.0, 0.0])
    current = torch.tensor([0.0, 0.0, 1.0, 0.0])
    actual = gate.posterior_transition_gate(previous, current)
    assert actual.item() == 1.0


def test_dynamic_gate_is_soft_for_uncertain_transition() -> None:
    previous = torch.tensor([0.9, 0.1, 0.0, 0.0])
    current = torch.tensor([0.1, 0.9, 0.0, 0.0])
    actual = gate.posterior_transition_gate(previous, current)
    assert torch.isclose(actual, torch.tensor(0.82))


def test_gate_scale_clips_at_one() -> None:
    previous = torch.tensor([1.0, 0.0, 0.0, 0.0])
    current = torch.tensor([0.0, 1.0, 0.0, 0.0])
    actual = gate.posterior_transition_gate(previous, current, scale=1.2)
    assert actual.item() == 1.0


def test_control_modes_ignore_similarity() -> None:
    posterior = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert gate.posterior_transition_gate(
        posterior, posterior, mode="always_reset"
    ).item() == 1.0
    assert gate.posterior_transition_gate(
        posterior, posterior, mode="always_carry"
    ).item() == 0.0


def test_invalid_mode_and_scale_rejected() -> None:
    posterior = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        gate.posterior_transition_gate(posterior, posterior, mode="unknown")
    with pytest.raises(ValueError):
        gate.posterior_transition_gate(posterior, posterior, scale=-0.1)
