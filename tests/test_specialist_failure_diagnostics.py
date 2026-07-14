from opfusion.specialist_failure_diagnostics import DEFAULT_OPERATORS, _first_divergence


def test_default_diagnostic_operators_target_failed_specialists() -> None:
    assert DEFAULT_OPERATORS == ("aggregation.sum", "scalar.neg")


def test_first_divergence_reports_token_or_length_difference() -> None:
    assert _first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert _first_divergence([1, 2], [1, 2, 3]) == 2
    assert _first_divergence([1, 2], [1, 2]) is None
