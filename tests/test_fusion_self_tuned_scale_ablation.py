from __future__ import annotations

import pytest

from opfusion import fusion_self_tuned_operator_controller as self_tuned
from opfusion.fusion_self_tuned_scale_ablation import run_with_constant_evaluation_scale


def test_constant_scale_wrapper_restores_controller_method_on_error(monkeypatch) -> None:
    original = self_tuned.SelfTunedPromptController.predicted_scale

    def fail(**kwargs):
        raise RuntimeError("sentinel")

    monkeypatch.setattr(self_tuned, "run_experiment", fail)
    with pytest.raises(RuntimeError, match="sentinel"):
        run_with_constant_evaluation_scale(scale=6.5)
    assert self_tuned.SelfTunedPromptController.predicted_scale is original


def test_constant_scale_rejects_nonpositive_value() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_with_constant_evaluation_scale(scale=0.0)
