from __future__ import annotations

from typing import Iterable

from opfusion import fusion_unseen_source_residual_gating as residual


_original_generate_residual = residual._generate_residual


def _generate_residual_compatible(**kwargs):
    output, diagnostics = _original_generate_residual(**kwargs)
    diagnostics = dict(diagnostics)
    diagnostics["mean_oracle_source_weight"] = diagnostics.get(
        "mean_oracle_source_gate", 0.0
    )
    return output, diagnostics


def main(argv: Iterable[str] | None = None) -> int:
    residual._generate_residual = _generate_residual_compatible
    return residual.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
