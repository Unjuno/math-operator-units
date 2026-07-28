import pytest
import torch

from opfusion.fusion_search import (
    aggregate_fusion_factory,
    enumerate_nonempty_subsets,
    parse_alpha_grid,
    rms_equalize_biases,
    select_aggregate_winners,
    select_winners,
    subset_operators,
)


def test_enumerates_all_nonempty_five_operator_subsets() -> None:
    masks = enumerate_nonempty_subsets(5)
    assert masks == tuple(range(1, 32))
    assert len(masks) == 31


def test_subset_operator_mapping_uses_bit_order() -> None:
    operators = ("a", "b", "c", "d", "e")
    assert subset_operators(0b10101, operators) == ("a", "c", "e")


def test_alpha_grid_parsing_deduplicates_in_order() -> None:
    assert parse_alpha_grid("0.25, 1, 0.25, 2") == (0.25, 1.0, 2.0)


@pytest.mark.parametrize("value", ["", "-0.1,1", "nan", "inf"])
def test_alpha_grid_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_alpha_grid(value)


def test_rms_equalization_centers_and_equalizes_nonzero_units() -> None:
    torch.manual_seed(0)
    first = torch.randn(3, 11)
    second = 8.0 * torch.randn(3, 11) + 20.0
    zero = torch.zeros(3, 11)
    result = rms_equalize_biases(torch.stack([first, second, zero]))

    assert torch.allclose(result.mean(dim=-1), torch.zeros(3, 3), atol=1e-6)
    rms = result.float().pow(2).mean(dim=(1, 2)).sqrt()
    assert torch.allclose(rms[0], rms[1], atol=1e-6)
    assert torch.equal(result[2], torch.zeros_like(result[2]))


def _row(*, cohort: str, mask: int, mode: str, alpha: float, accuracy: float, nll: float, inactive: float = 0.0) -> dict:
    return {
        "cohort_id": cohort,
        "source": "fusion_factory",
        "subset_mask": mask,
        "subset_id": f"subset_{mask:02d}",
        "operators": ["scalar.add"],
        "cardinality": 1,
        "mode": mode,
        "alpha": alpha,
        "active_token_accuracy": accuracy,
        "active_token_nll": nll,
        "inactive_base_argmax_agreement": 1.0,
        "inactive_centered_delta_rms": inactive,
        "all_five_joint_jsd": None,
    }


def test_cohort_winner_prefers_accuracy_then_nll_then_inactive_drift() -> None:
    rows = [
        _row(cohort="seed0", mask=1, mode="raw_sum", alpha=0.5, accuracy=0.8, nll=0.4),
        _row(cohort="seed0", mask=1, mode="raw_sum", alpha=1.0, accuracy=0.9, nll=0.7),
        _row(cohort="seed0", mask=1, mode="rms_equalized_sum", alpha=1.0, accuracy=0.9, nll=0.3),
    ]
    winner = select_winners(rows)[0]
    assert winner["mode"] == "rms_equalized_sum"
    assert winner["alpha"] == 1.0


def test_cross_seed_aggregation_and_stability_adjusted_winner() -> None:
    rows = [
        _row(cohort="seed0", mask=1, mode="raw_sum", alpha=0.5, accuracy=0.9, nll=0.2),
        _row(cohort="seed1", mask=1, mode="raw_sum", alpha=0.5, accuracy=0.9, nll=0.2),
        _row(cohort="seed0", mask=1, mode="raw_sum", alpha=1.0, accuracy=1.0, nll=0.1),
        _row(cohort="seed1", mask=1, mode="raw_sum", alpha=1.0, accuracy=0.6, nll=0.1),
    ]
    aggregate = aggregate_fusion_factory(rows)
    assert len(aggregate) == 2
    winner = select_aggregate_winners(aggregate)[0]
    assert winner["alpha"] == 0.5
    assert winner["active_token_accuracy_std"] == pytest.approx(0.0)
