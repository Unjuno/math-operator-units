from __future__ import annotations

from opfusion.fusion_stateful_dual_timescale_confirmatory import candidate_grid


def test_confirmatory_grid_is_fixed_and_unique() -> None:
    candidates = candidate_grid()
    assert [candidate.candidate_id for candidate in candidates] == [
        "confirm_exploratory_best",
        "confirm_robust_all_best",
        "confirm_robust_neutral_best",
    ]
    assert len({candidate.candidate_id for candidate in candidates}) == 3
    assert all(candidate.fast_memory < candidate.slow_memory for candidate in candidates)
