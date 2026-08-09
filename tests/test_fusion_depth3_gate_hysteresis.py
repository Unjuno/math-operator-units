from __future__ import annotations

import pytest
import torch

from opfusion import fusion_depth3_gate_hysteresis as hysteresis


def test_zero_hysteresis_reproduces_raw_gate() -> None:
    assert hysteresis.gate_with_hysteresis(0.2, 0.9, rho=0.0) == 0.2


def test_full_hysteresis_carries_previous_activation() -> None:
    assert hysteresis.gate_with_hysteresis(0.0, 1.0, rho=1.0) == 1.0
    assert hysteresis.gate_with_hysteresis(1.0, 0.0, rho=1.0) == 1.0


def test_soft_hysteresis_applies_gate_floor() -> None:
    actual = hysteresis.gate_with_hysteresis(0.1, 1.0, rho=0.875)
    assert actual == 0.875


def test_effective_previous_produces_requested_dot_gate() -> None:
    current = torch.tensor([0.9, 0.1, 0.0, 0.0])
    target = 0.875
    previous = hysteresis._effective_previous_for_gate(
        current, target_gate=target
    )
    dot = torch.sum(previous.float() * current.float())
    gate = 1.0 - dot
    assert torch.isclose(gate, torch.tensor(target), atol=1e-6)


def test_invalid_hysteresis_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        hysteresis.gate_with_hysteresis(-0.1, 0.0, rho=0.5)
    with pytest.raises(ValueError):
        hysteresis.gate_with_hysteresis(0.1, 1.1, rho=0.5)
    with pytest.raises(ValueError):
        hysteresis.gate_with_hysteresis(0.1, 0.2, rho=1.1)
