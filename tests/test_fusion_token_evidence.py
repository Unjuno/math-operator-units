from __future__ import annotations

import torch

from opfusion.fusion_sparse_valid import SparseValidBatch
from opfusion.fusion_token_evidence import TokenEvidenceCompositor, batch_metrics, source_stack


def _token_features(vocabulary_size: int) -> torch.Tensor:
    features = torch.zeros(vocabulary_size, 6)
    features[:, 0] = (torch.arange(vocabulary_size) % 2).float()
    return features


def test_token_evidence_is_equivariant_to_specialist_permutation() -> None:
    generator = torch.Generator().manual_seed(37)
    base = torch.randn(3, 17, generator=generator)
    units = torch.randn(3, 5, 17, generator=generator)
    model = TokenEvidenceCompositor(token_features=_token_features(17), hidden_size=8)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    first_logits, first_gates, _ = model.compose(source_stack(base, units))
    second_logits, second_gates, _ = model.compose(source_stack(base, units.index_select(1, permutation)))

    assert torch.allclose(first_logits, second_logits, atol=1e-6)
    assert torch.allclose(second_gates[:, 0], first_gates[:, 0], atol=1e-6)
    assert torch.allclose(second_gates[:, 1:], first_gates[:, 1:].index_select(1, permutation), atol=1e-6)


def test_token_evidence_outputs_normalized_distribution_and_continuous_gates() -> None:
    generator = torch.Generator().manual_seed(41)
    sources = torch.randn(2, 6, 13, generator=generator)
    model = TokenEvidenceCompositor(token_features=_token_features(13), hidden_size=7)

    logits, gates, totals = model.compose(sources)

    assert logits.shape == (2, 13)
    assert gates.shape == sources.shape
    assert totals.shape == (2, 13)
    assert torch.allclose(logits.exp().sum(dim=-1), torch.ones(2), atol=1e-6)
    assert bool(((gates > 0) & (gates <= 1)).all())


def test_token_evidence_metrics_accept_set_valued_targets() -> None:
    generator = torch.Generator().manual_seed(43)
    base = torch.randn(5, 11, generator=generator)
    units = torch.randn(5, 5, 11, generator=generator)
    valid = torch.zeros(5, 11, dtype=torch.bool)
    valid[0, [1, 2]] = True
    valid[1, [3]] = True
    valid[2, [4, 5]] = True
    valid[3, [6]] = True
    valid[4, [7, 8, 9]] = True
    batch = SparseValidBatch(base_logits=base, unit_logits=units, valid_mask=valid)
    model = TokenEvidenceCompositor(token_features=_token_features(11), hidden_size=6)

    metrics = batch_metrics(model, batch, chunk_size=2)

    assert 0.0 <= metrics["valid_top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["valid_probability_mass"] <= 1.0
    assert 0.0 < metrics["mean_coordinate_gate"] <= 1.0
    assert 1.0 <= metrics["mean_effective_sources_per_token"] <= 6.0
