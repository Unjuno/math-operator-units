from __future__ import annotations

import os

import torch

from opfusion.fusion_sparse_valid import SparseValidBatch
from opfusion.fusion_stateful_dual_timescale_init_sensitivity import (
    ENV_INIT_SEED,
    fit_mixer_with_independent_init_seed,
)
from opfusion.fusion_validity_mixture import export_parameters


def _batch() -> SparseValidBatch:
    generator = torch.Generator().manual_seed(440)
    base_logits = torch.randn(16, 9, generator=generator)
    unit_logits = torch.randn(16, 5, 9, generator=generator)
    valid_mask = torch.zeros(16, 9, dtype=torch.bool)
    for index in range(16):
        valid_mask[index, index % 9] = True
        valid_mask[index, (index + 1) % 9] = True
    return SparseValidBatch(base_logits, unit_logits, valid_mask)


def _fit(init_seed: int):
    os.environ[ENV_INIT_SEED] = str(init_seed)
    return fit_mixer_with_independent_init_seed(
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
        seed=731000,
    )[0]


def test_same_init_seed_is_exactly_reproducible() -> None:
    assert export_parameters(_fit(91)) == export_parameters(_fit(91))


def test_init_seed_changes_parameters_with_data_order_fixed() -> None:
    assert export_parameters(_fit(91)) != export_parameters(_fit(92))


def test_missing_init_seed_fails() -> None:
    os.environ.pop(ENV_INIT_SEED, None)
    try:
        fit_mixer_with_independent_init_seed(
            batch=_batch(), vocabulary_size=9, hidden_size=6, sketch_size=3,
            temperature=0.75, target_temperature=0.35,
            source_supervision_weight=1.0, entropy_target=1.0,
            entropy_penalty=0.2, l2_weight=0.001, learning_rate=0.01,
            steps=1, batch_positions=4, seed=731000,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing initialization seed must fail")
