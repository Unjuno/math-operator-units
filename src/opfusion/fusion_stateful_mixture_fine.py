from __future__ import annotations

from opfusion import fusion_stateful_mixture as implementation


def _fine_candidate_grid() -> tuple[implementation.StateCandidate, ...]:
    rows: list[implementation.StateCandidate] = []
    for memory in (0.9, 0.95, 0.98, 0.99):
        for feedback in (0.0, 0.1, 0.2, 0.35):
            for temperature in (0.5, 0.75, 1.0):
                rows.append(
                    implementation.StateCandidate(
                        candidate_id=f"fine_m{memory}_f{feedback}_t{temperature}",
                        memory=memory,
                        feedback=feedback,
                        temperature=temperature,
                    )
                )
    return tuple(rows)


implementation.candidate_grid = _fine_candidate_grid


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
