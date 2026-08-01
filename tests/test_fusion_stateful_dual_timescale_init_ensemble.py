from __future__ import annotations

import torch

from opfusion.fusion_stateful_dual_timescale_init_ensemble import SourceValidityEnsemble
from opfusion.fusion_validity_mixture import SourceValidityMixer


def _members() -> tuple[SourceValidityMixer, ...]:
    members = []
    for seed in (21, 22, 23):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            members.append(SourceValidityMixer(vocabulary_size=13, hidden_size=7, sketch_size=4))
    return tuple(members)


def test_arithmetic_ensemble_is_positive_continuous_mixture() -> None:
    source_logits = torch.randn(5, 6, 13, generator=torch.Generator().manual_seed(50))
    ensemble = SourceValidityEnsemble(_members(), mode="arithmetic")
    fused, weights = ensemble.compose(source_logits)
    assert fused.shape == (5, 13)
    assert weights.shape == (5, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert bool((weights > 0).all())


def test_arithmetic_weights_equal_member_mean() -> None:
    source_logits = torch.randn(4, 6, 13, generator=torch.Generator().manual_seed(51))
    members = _members()
    ensemble = SourceValidityEnsemble(members, mode="arithmetic")
    _, weights = ensemble.compose(source_logits)
    expected = torch.stack([member.compose(source_logits)[1] for member in members]).mean(dim=0)
    assert torch.allclose(weights, expected)


def test_geometric_weights_equal_normalized_log_mean() -> None:
    source_logits = torch.randn(4, 6, 13, generator=torch.Generator().manual_seed(52))
    members = _members()
    ensemble = SourceValidityEnsemble(members, mode="geometric")
    _, weights = ensemble.compose(source_logits)
    raw = torch.stack([member.compose(source_logits)[1] for member in members])
    expected = torch.softmax(raw.clamp_min(1e-12).log().mean(dim=0), dim=-1)
    assert torch.allclose(weights, expected)


def test_bad_mode_fails() -> None:
    try:
        SourceValidityEnsemble(_members(), mode="router")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported mode must fail")
