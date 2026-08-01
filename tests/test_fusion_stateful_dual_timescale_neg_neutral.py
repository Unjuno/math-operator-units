from __future__ import annotations

import torch

from opfusion.fusion_stateful_dual_timescale_neg_neutral import neutralize_neg_source
from opfusion.training.data import EXPERIMENT_OPERATORS


def test_neutralize_neg_source_replaces_only_neg_with_base() -> None:
    source_count = 1 + len(EXPERIMENT_OPERATORS)
    sources = torch.arange(source_count * 7, dtype=torch.float32).reshape(source_count, 7)
    result = neutralize_neg_source(sources)
    neg_index = 1 + tuple(EXPERIMENT_OPERATORS).index("scalar.neg")

    assert torch.equal(result[neg_index], result[0])
    for index in range(source_count):
        if index != neg_index:
            assert torch.equal(result[index], sources[index])
    assert not torch.equal(sources[neg_index], sources[0])


def test_neutralize_neg_source_does_not_mutate_input() -> None:
    source_count = 1 + len(EXPERIMENT_OPERATORS)
    sources = torch.randn(source_count, 11)
    original = sources.clone()
    neutralize_neg_source(sources)
    assert torch.equal(sources, original)


def test_neutralize_neg_source_rejects_wrong_source_axis() -> None:
    sources = torch.zeros(len(EXPERIMENT_OPERATORS), 5)
    try:
        neutralize_neg_source(sources)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong source axis must fail")
