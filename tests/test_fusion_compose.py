from __future__ import annotations

import torch

from opfusion.fusion_compose import (
    AlgebraicCompositor,
    CalibrationBatch,
    export_parameters,
    fit_compositor,
    fixed_compose,
    load_parameters,
)


VOCAB = 11
OPERATORS = 5


def _classes() -> torch.Tensor:
    return torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0], dtype=torch.long)


def test_global_linear_uses_all_five_fields_without_task_labels() -> None:
    base = torch.randn(3, VOCAB)
    units = torch.randn(3, OPERATORS, VOCAB)
    model = AlgebraicCompositor("global_linear", token_class_ids=_classes())
    fused = model(base, units)
    assert fused.shape == base.shape
    assert model.weights.shape == (OPERATORS,)


def test_token_class_linear_applies_static_per_vocab_class_weights() -> None:
    base = torch.zeros(VOCAB)
    units = torch.zeros(OPERATORS, VOCAB)
    units[0, 8] = 2.0
    model = AlgebraicCompositor("token_class_linear", token_class_ids=_classes())
    with torch.no_grad():
        model.class_weights.zero_()
        model.class_weights[0, 0] = 1.0
    fused = model(base, units)
    assert fused.shape == base.shape
    assert float(fused[8]) != 0.0
    assert torch.allclose(fused[:8], torch.zeros(8))


def test_pairwise_zero_interactions_is_finite() -> None:
    base = torch.randn(4, VOCAB)
    units = torch.randn(4, OPERATORS, VOCAB)
    model = AlgebraicCompositor("pairwise_polynomial", token_class_ids=_classes())
    with torch.no_grad():
        model.interactions.zero_()
    fused = model(base, units)
    assert torch.isfinite(fused).all()
    assert fused.shape == base.shape


def test_fixed_composition_modes_are_shape_preserving() -> None:
    base = torch.randn(2, VOCAB)
    units = torch.randn(2, OPERATORS, VOCAB)
    for mode in ("raw_sum", "bias_mean", "rms_mean"):
        fused = fixed_compose(base, units, mode=mode)
        assert fused.shape == base.shape
        assert torch.isfinite(fused).all()


def test_parameter_export_round_trip() -> None:
    first = AlgebraicCompositor("pairwise_polynomial", token_class_ids=_classes())
    with torch.no_grad():
        first.weights.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
        first.interactions.copy_(torch.arange(first.interactions.numel(), dtype=torch.float32) / 10.0)
    values = export_parameters(first)
    second = AlgebraicCompositor("pairwise_polynomial", token_class_ids=_classes())
    load_parameters(second, values)
    assert torch.allclose(first.weights, second.weights)
    assert torch.allclose(first.interactions, second.interactions)


def test_fit_compositor_updates_only_small_composition_parameters() -> None:
    torch.manual_seed(7)
    positions = 24
    base = torch.randn(positions, VOCAB)
    units = torch.randn(positions, OPERATORS, VOCAB)
    gold = torch.randint(0, VOCAB, (positions,))
    batch = CalibrationBatch(base_logits=base, unit_logits=units, gold=gold)
    model, report = fit_compositor(
        "global_linear",
        batch=batch,
        token_classes=_classes(),
        steps=4,
        batch_positions=8,
        learning_rate=0.01,
        l2_weight=0.001,
        seed=11,
    )
    assert report["calibration_positions"] == positions
    assert len(list(model.parameters())) == 1
    assert torch.isfinite(model.weights).all()
