from __future__ import annotations

import torch

from opfusion.fusion_sparse_valid import SparseValidBatch
from opfusion.fusion_stateful_dual_timescale_confirmatory_deterministic import (
    fit_mixer_deterministic,
)
from opfusion.fusion_validity_mixture import export_parameters


def _batch() -> SparseValidBatch:
    generator = torch.Generator().manual_seed(123)
    base_logits = torch.randn(12, 9, generator=generator)
    unit_logits = torch.randn(12, 5, 9, generator=generator)
    valid_mask = torch.zeros(12, 9, dtype=torch.bool)
    for index in range(12):
        valid_mask[index, index % 9] = True
        valid_mask[index, (index + 2) % 9] = True
    return SparseValidBatch(
        base_logits=base_logits,
        unit_logits=unit_logits,
        valid_mask=valid_mask,
    )


def _fit(seed: int):
    return fit_mixer_deterministic(
        batch=_batch(),
        vocabulary_size=9,
        hidden_size=6,
        sketch_size=3,
        temperature=0.75,
        target_temperature=0.35,
        source_supervision_weight=1.0,
        entropy_target=1.0,
        entropy_penalty=0.2,
        l2_weight=0.001,
        learning_rate=0.01,
        steps=8,
        batch_positions=4,
        seed=seed,
    )[0]


def test_same_seed_produces_identical_parameters() -> None:
    first = export_parameters(_fit(77))
    second = export_parameters(_fit(77))
    assert first == second


def test_different_seed_changes_parameters() -> None:
    first = export_parameters(_fit(77))
    second = export_parameters(_fit(78))
    assert first != second
