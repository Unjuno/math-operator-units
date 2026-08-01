from __future__ import annotations

# Importing the neutralization module installs the fixed NEG-to-Base source transform.
from opfusion import fusion_stateful_dual_timescale_neg_neutral as neg_neutral  # noqa: F401
from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory


def main() -> int:
    return confirmatory.main()


if __name__ == "__main__":
    raise SystemExit(main())
