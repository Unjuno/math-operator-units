from __future__ import annotations

import torch

from opfusion.fusion_sparse_valid import SparseValidBatch
from opfusion.fusion_validity_mixture import (
    SourceValidityMixer,
    export_parameters,
    load_parameters,
    mixer_metrics,
    validity_targets,
)


def test_source_validity_mixer_is_a_continuous_probability_mixture() -> None:
    generator = torch.Generator().manual_seed(7)
    source_logits = torch.randn(5, 6, 17, generator=generator)
    model = SourceValidityMixer(vocabulary_size=17, hidden_size=8, sketch_size=5)

    fused, weights = model.compose(source_logits)

    assert fused.shape == (5, 17)
    assert weights.shape == (5, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert bool((weights > 0).all())

    expected = (weights.unsqueeze(-1) * torch.softmax(source_logits.float(), dim=-1)).sum(dim=-2)
    assert torch.allclose(torch.softmax(fused, dim=-1), expected, atol=1e-6)


def test_validity_targets_favor_sources_with_more_valid_probability_mass() -> None:
    source_logits = torch.tensor(
        [
            [
                [0.0, 8.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
                [0.0, 7.0, 0.0, 0.0],
                [0.0, 0.0, 7.0, 0.0],
            ]
        ]
    )
    valid_mask = torch.tensor([[False, True, False, False]])

    target, oracle, valid_mass = validity_targets(source_logits, valid_mask, target_temperature=0.25)

    assert int(oracle.item()) == 0
    assert int(target.argmax(dim=-1).item()) == 0
    assert float(valid_mass[0, 0]) > float(valid_mass[0, 1])


def test_parameter_export_and_reload_preserve_outputs() -> None:
    generator = torch.Generator().manual_seed(11)
    source_logits = torch.randn(3, 6, 13, generator=generator)
    first = SourceValidityMixer(vocabulary_size=13, hidden_size=7, sketch_size=4, temperature=0.75)
    second = SourceValidityMixer(vocabulary_size=13, hidden_size=7, sketch_size=4, temperature=0.75)

    load_parameters(second, export_parameters(first))
    first_logits, first_weights = first.compose(source_logits)
    second_logits, second_weights = second.compose(source_logits)

    assert torch.allclose(first_logits, second_logits)
    assert torch.allclose(first_weights, second_weights)


def test_mixer_metrics_accept_set_valued_targets() -> None:
    generator = torch.Generator().manual_seed(19)
    base_logits = torch.randn(4, 11, generator=generator)
    unit_logits = torch.randn(4, 5, 11, generator=generator)
    valid_mask = torch.zeros(4, 11, dtype=torch.bool)
    valid_mask[0, [1, 2]] = True
    valid_mask[1, [3]] = True
    valid_mask[2, [4, 5, 6]] = True
    valid_mask[3, [7]] = True
    batch = SparseValidBatch(base_logits=base_logits, unit_logits=unit_logits, valid_mask=valid_mask)
    model = SourceValidityMixer(vocabulary_size=11, hidden_size=6, sketch_size=3)

    metrics = mixer_metrics(model, batch, target_temperature=0.5)

    assert 0.0 <= metrics["valid_top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["valid_probability_mass"] <= 1.0
    assert metrics["effective_source_count"] >= 1.0
