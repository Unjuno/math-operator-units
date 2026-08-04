from __future__ import annotations

import pytest
import torch

from opfusion.fusion_stateful_dual_timescale import DualTimescaleCandidate
from opfusion.fusion_stateful_oracle_operator import (
    combine_oracle_operator_states,
    oracle_operator_prior,
)
from opfusion.training.data import EXPERIMENT_OPERATORS


def _candidate() -> DualTimescaleCandidate:
    return DualTimescaleCandidate(
        candidate_id="test",
        fast_memory=0.50,
        slow_memory=0.98,
        slow_mix=0.25,
        feedback=0.35,
        temperature=1.00,
    )


def test_oracle_prior_targets_matching_specialist() -> None:
    operator = EXPERIMENT_OPERATORS[0]
    prior = oracle_operator_prior(
        operator,
        source_count=len(EXPERIMENT_OPERATORS) + 1,
        strength=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert prior.shape == (len(EXPERIMENT_OPERATORS) + 1,)
    assert float(prior[0]) == 0.0
    assert float(prior[1]) == 2.0
    assert int(torch.count_nonzero(prior)) == 1


def test_oracle_strength_increases_matching_source_weight_without_masking() -> None:
    source_count = len(EXPERIMENT_OPERATORS) + 1
    fast = torch.zeros(source_count)
    slow = torch.zeros(source_count)
    operator = EXPERIMENT_OPERATORS[2]

    baseline = combine_oracle_operator_states(
        fast,
        slow,
        candidate=_candidate(),
        operator=operator,
        strength=0.0,
    )
    conditioned = combine_oracle_operator_states(
        fast,
        slow,
        candidate=_candidate(),
        operator=operator,
        strength=4.0,
    )
    target = 1 + EXPERIMENT_OPERATORS.index(operator)

    assert torch.isclose(baseline.sum(), torch.tensor(1.0))
    assert torch.isclose(conditioned.sum(), torch.tensor(1.0))
    assert bool(torch.all(conditioned > 0.0))
    assert float(conditioned[target]) > float(baseline[target])
    assert int(torch.argmax(conditioned)) == target


def test_oracle_prior_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unknown operator"):
        oracle_operator_prior(
            "UNKNOWN",
            source_count=len(EXPERIMENT_OPERATORS) + 1,
            strength=1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
