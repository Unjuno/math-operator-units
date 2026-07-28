from __future__ import annotations

import torch

from opfusion.fusion_compose import CalibrationBatch
from opfusion.fusion_onpolicy import (
    BiasGeometryCompositor,
    export_parameters,
    fit_geometry_compositor,
    load_parameters,
    parse_probabilities,
)


VOCAB = 13
OPERATORS = 5


def test_geometry_compositor_uses_positive_all_unit_weights() -> None:
    torch.manual_seed(3)
    base = torch.randn(4, VOCAB)
    units = torch.randn(4, OPERATORS, VOCAB)
    model = BiasGeometryCompositor(hidden_size=6, weight_floor=0.03)
    fused, weights, disagreement = model.compose(base, units)
    assert fused.shape == base.shape
    assert weights.shape == (4, OPERATORS)
    assert disagreement.shape == base.shape
    assert torch.all(weights >= 0.03 - 1e-7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.isfinite(fused).all()


def test_geometry_parameter_round_trip() -> None:
    first = BiasGeometryCompositor(hidden_size=4, weight_floor=0.02)
    with torch.no_grad():
        first.unit_prior.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5]))
    values = export_parameters(first)
    second = BiasGeometryCompositor(hidden_size=4, weight_floor=0.02)
    load_parameters(second, values)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.allclose(left, right)


def test_fit_geometry_compositor_is_finite() -> None:
    torch.manual_seed(5)
    positions = 20
    batch = CalibrationBatch(
        base_logits=torch.randn(positions, VOCAB),
        unit_logits=torch.randn(positions, OPERATORS, VOCAB),
        gold=torch.randint(0, VOCAB, (positions,)),
    )
    model = BiasGeometryCompositor(hidden_size=4, weight_floor=0.03)
    report = fit_geometry_compositor(
        model,
        batch=batch,
        steps=3,
        batch_positions=8,
        learning_rate=0.01,
        l2_weight=0.001,
        entropy_penalty=0.02,
        entropy_target_ratio=0.7,
        seed=9,
    )
    assert report["calibration_positions"] == positions
    assert 0.0 <= report["calibration_token_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(report["calibration_gold_nll"]))


def test_parse_probabilities() -> None:
    assert parse_probabilities("0.25,0.5,1") == (0.25, 0.5, 1.0)
