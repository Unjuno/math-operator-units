from __future__ import annotations

import torch

from opfusion.fusion_stateful_mixture import StateCandidate, candidate_grid
from opfusion.fusion_validity_mixture import SourceValidityMixer


def test_state_candidate_grid_contains_memoryless_and_persistent_conditions() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 36
    assert any(row.memory == 0.0 and row.feedback == 0.0 and row.temperature == 1.0 for row in candidates)
    assert any(row.memory == 0.95 and row.feedback == 0.35 for row in candidates)
    assert len({row.candidate_id for row in candidates}) == len(candidates)


def test_continuous_state_update_keeps_all_source_weights_positive() -> None:
    generator = torch.Generator().manual_seed(47)
    source_logits = torch.randn(6, 13, generator=generator)
    mixer = SourceValidityMixer(vocabulary_size=13, hidden_size=7, sketch_size=4)
    candidate = StateCandidate("test", memory=0.8, feedback=0.15, temperature=0.75)
    state = None

    for _ in range(4):
        _, instant_weights = mixer.compose(source_logits)
        instant_state = instant_weights.clamp_min(1e-9).log()
        state = instant_state if state is None else candidate.memory * state + (1.0 - candidate.memory) * instant_state
        weights = torch.softmax(state / candidate.temperature, dim=-1)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert bool((weights > 0).all())
        selected = int(torch.argmax((weights.unsqueeze(-1) * torch.softmax(source_logits, dim=-1)).sum(dim=-2)).item())
        support = torch.log_softmax(source_logits, dim=-1)[:, selected]
        state = state + candidate.feedback * (support - support.mean())
        source_logits = source_logits + 0.01
