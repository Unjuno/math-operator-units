from __future__ import annotations

import torch

from opfusion.fusion_coordinate_gate import CoordinateGateCompositor, batch_metrics
from opfusion.fusion_sparse_valid import SparseValidBatch


def _token_features(vocabulary_size: int) -> torch.Tensor:
    features = torch.zeros(vocabulary_size, 6)
    features[:, 0] = torch.arange(vocabulary_size) % 2
    return features


def test_coordinate_gate_is_permutation_equivariant_over_specialists() -> None:
    generator = torch.Generator().manual_seed(23)
    base = torch.randn(3, 19, generator=generator)
    units = torch.randn(3, 5, 19, generator=generator)
    model = CoordinateGateCompositor(token_features=_token_features(19), hidden_size=8)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    fused, gate, global_gate, _ = model.compose(base, units)
    permuted_fused, permuted_gate, permuted_global_gate, _ = model.compose(base, units.index_select(1, permutation))

    assert torch.allclose(fused, permuted_fused, atol=1e-6)
    assert torch.allclose(permuted_gate, gate.index_select(1, permutation), atol=1e-6)
    assert torch.allclose(permuted_global_gate, global_gate.index_select(1, permutation), atol=1e-6)


def test_coordinate_gate_outputs_continuous_bounded_gates() -> None:
    generator = torch.Generator().manual_seed(29)
    base = torch.randn(2, 13, generator=generator)
    units = torch.randn(2, 5, 13, generator=generator)
    model = CoordinateGateCompositor(token_features=_token_features(13), hidden_size=7)

    fused, gate, global_gate, threshold = model.compose(base, units)

    assert fused.shape == base.shape
    assert gate.shape == units.shape
    assert global_gate.shape == units.shape[:-1]
    assert bool(((gate >= 0) & (gate <= 1)).all())
    assert bool(((global_gate >= 0) & (global_gate <= 1)).all())
    assert float(threshold) >= 0.0


def test_coordinate_gate_metrics_support_set_valued_targets() -> None:
    generator = torch.Generator().manual_seed(31)
    base = torch.randn(5, 11, generator=generator)
    units = torch.randn(5, 5, 11, generator=generator)
    valid = torch.zeros(5, 11, dtype=torch.bool)
    valid[0, [1, 2]] = True
    valid[1, [3]] = True
    valid[2, [4, 5]] = True
    valid[3, [6]] = True
    valid[4, [7, 8, 9]] = True
    batch = SparseValidBatch(base_logits=base, unit_logits=units, valid_mask=valid)
    model = CoordinateGateCompositor(token_features=_token_features(11), hidden_size=6)

    metrics = batch_metrics(model, batch, chunk_size=2)

    assert 0.0 <= metrics["valid_top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["valid_probability_mass"] <= 1.0
    assert 0.0 <= metrics["mean_coordinate_gate"] <= 1.0
