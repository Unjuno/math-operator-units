from __future__ import annotations

from typing import Mapping

import torch

from opfusion import fusion_stateful_dual_timescale as dual_timescale
from opfusion import fusion_stateful_mixture as implementation
from opfusion.training.data import EXPERIMENT_OPERATORS


NEG_OPERATOR = "scalar.neg"


def neutralize_neg_source(sources: torch.Tensor) -> torch.Tensor:
    """Replace the failed NEG specialist distribution with the Base distribution.

    This is a fixed inventory ablation, not prompt-conditioned routing. The source
    remains present in the six-way probability mixture, but contributes zero
    Base-relative information at every token.
    """
    expected_sources = 1 + len(EXPERIMENT_OPERATORS)
    if sources.ndim < 2 or int(sources.shape[0]) != expected_sources:
        raise ValueError(
            f"expected source axis of size {expected_sources}, got {tuple(sources.shape)}"
        )
    neg_index = 1 + tuple(EXPERIMENT_OPERATORS).index(NEG_OPERATOR)
    neutralized = sources.clone()
    neutralized[neg_index] = neutralized[0]
    return neutralized


_original_source_logits = implementation._source_logits


def _source_logits_neg_neutral(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    ids: torch.Tensor,
) -> torch.Tensor:
    sources = _original_source_logits(base=base, units=units, ids=ids)
    return neutralize_neg_source(sources)


implementation._source_logits = _source_logits_neg_neutral


def main() -> int:
    return dual_timescale.main()


if __name__ == "__main__":
    raise SystemExit(main())
