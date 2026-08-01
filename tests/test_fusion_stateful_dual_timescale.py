from __future__ import annotations

import torch

from opfusion.fusion_stateful_dual_timescale import candidate_grid, combine_states


def test_candidate_grid_is_unique_and_ordered() -> None:
    candidates = candidate_grid()
    assert len(candidates) == 48
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert all(candidate.fast_memory < candidate.slow_memory for candidate in candidates)


def test_combine_states_is_positive_and_normalized() -> None:
    fast = torch.tensor([0.0, -1.0, -2.0])
    slow = torch.tensor([-2.0, -1.0, 0.0])
    weights = combine_states(fast, slow, slow_mix=0.5, temperature=1.0)
    assert bool((weights > 0).all())
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_slow_mix_endpoints_recover_each_state() -> None:
    fast = torch.tensor([2.0, 0.0])
    slow = torch.tensor([0.0, 2.0])
    fast_weights = combine_states(fast, slow, slow_mix=0.0, temperature=1.0)
    slow_weights = combine_states(fast, slow, slow_mix=1.0, temperature=1.0)
    assert fast_weights[0] > fast_weights[1]
    assert slow_weights[1] > slow_weights[0]


def test_invalid_combine_arguments_fail() -> None:
    state = torch.zeros(3)
    for slow_mix in (-0.1, 1.1):
        try:
            combine_states(state, state, slow_mix=slow_mix, temperature=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid slow_mix must fail")

    try:
        combine_states(state, state, slow_mix=0.5, temperature=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive temperature must fail")
