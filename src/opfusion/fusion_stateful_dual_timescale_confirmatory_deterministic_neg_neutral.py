from __future__ import annotations

# Install the fixed NEG-to-Base transform before entering deterministic confirmation.
from opfusion import fusion_stateful_dual_timescale_neg_neutral as neg_neutral  # noqa: F401
from opfusion import fusion_stateful_dual_timescale_confirmatory_deterministic as deterministic


def main() -> int:
    return deterministic.main()


if __name__ == "__main__":
    raise SystemExit(main())
