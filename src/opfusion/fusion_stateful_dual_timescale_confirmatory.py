from __future__ import annotations

from opfusion import fusion_stateful_dual_timescale as dual_timescale
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_stateful_dual_timescale import DualTimescaleCandidate


def candidate_grid() -> tuple[DualTimescaleCandidate, ...]:
    """Pre-registered candidates from the exploratory and robustness runs."""
    return (
        DualTimescaleCandidate(
            candidate_id="confirm_exploratory_best",
            fast_memory=0.50,
            slow_memory=0.95,
            slow_mix=0.25,
            feedback=0.35,
            temperature=0.75,
        ),
        DualTimescaleCandidate(
            candidate_id="confirm_robust_all_best",
            fast_memory=0.50,
            slow_memory=0.98,
            slow_mix=0.25,
            feedback=0.35,
            temperature=1.00,
        ),
        DualTimescaleCandidate(
            candidate_id="confirm_robust_neutral_best",
            fast_memory=0.50,
            slow_memory=0.98,
            slow_mix=0.75,
            feedback=0.35,
            temperature=1.00,
        ),
    )


implementation.candidate_grid = candidate_grid


def main() -> int:
    return dual_timescale.main()


if __name__ == "__main__":
    raise SystemExit(main())
