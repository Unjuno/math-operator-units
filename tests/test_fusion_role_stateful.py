from __future__ import annotations

import torch

from opfusion.fusion_role_stateful import (
    ROLE_NAMES,
    build_role_index,
    candidate_grid,
    role_masks,
    role_weighted_mixture,
)
from opfusion.tokenizer import FixedVocabTokenizer


def test_role_index_separates_numeric_structure_and_eos() -> None:
    tokenizer = FixedVocabTokenizer(
        ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "-2", "0", "=", "+"]
    )
    index = build_role_index(tokenizer)
    assert index.tolist() == [1, 1, 2, 1, 0, 0, 1, 1]
    masks = role_masks(index)
    assert masks.shape == (len(ROLE_NAMES), tokenizer.vocab_size)
    assert torch.allclose(masks.sum(dim=0), torch.ones(tokenizer.vocab_size))


def test_role_weighted_mixture_is_positive_and_normalized() -> None:
    source_logits = torch.tensor(
        [
            [0.0, 1.0, -1.0, 0.5],
            [1.0, -1.0, 0.0, 0.5],
            [-0.5, 0.0, 1.5, -1.0],
        ]
    )
    role_index = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    role_weights = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    mixture = role_weighted_mixture(source_logits, role_weights, role_index)
    assert mixture.shape == (4,)
    assert bool((mixture > 0).all())
    assert torch.isclose(mixture.sum(), torch.tensor(1.0))


def test_candidate_grid_is_unique_and_nonrouting() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 48
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert all(candidate.memory > 0 for candidate in candidates)
    assert all(candidate.role_mass_scale > 0 for candidate in candidates)
