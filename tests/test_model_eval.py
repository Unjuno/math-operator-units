from __future__ import annotations

import torch

from opfusion.model_eval import (
    MetricCounter,
    _correct_prefix,
    finalize_counter,
    minimum_passing_snapshots,
    passes_metrics,
)


def test_correct_prefix_stops_at_first_mismatch() -> None:
    assert _correct_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert _correct_prefix([1, 2], [1, 2, 3]) == 2
    assert _correct_prefix([1, 2, 3], [1, 2, 3]) == 3


def test_finalize_counter_reports_teacher_and_prefix_metrics() -> None:
    counter = MetricCounter(
        examples=2,
        exact=1,
        token_correct=5,
        token_count=6,
        final_correct=1,
        final_count=2,
        trace_valid=1,
        stop_correct=2,
        generated_tokens=7,
        teacher_correct=7,
        teacher_tokens=8,
        teacher_nll_sum=4.0,
        correct_prefix_tokens=4,
        expected_tokens=6,
    )
    result = finalize_counter(counter)
    assert result["response_exact_accuracy"] == 0.5
    assert result["final_value_accuracy"] == 0.5
    assert result["teacher_forced_token_accuracy"] == 0.875
    assert result["teacher_forced_nll"] == 0.5
    assert result["mean_correct_prefix_fraction"] == 4 / 6


def test_quality_gate_distinguishes_identity_and_arithmetic() -> None:
    arithmetic = {
        "response_exact_accuracy": 0.7,
        "final_value_accuracy": 0.85,
        "trace_validity_accuracy": 0.85,
        "stop_accuracy": 1.0,
        "teacher_forced_token_accuracy": 0.9,
    }
    identity = {
        "response_exact_accuracy": 0.96,
        "final_value_accuracy": None,
        "trace_validity_accuracy": 0.96,
        "stop_accuracy": 1.0,
        "teacher_forced_token_accuracy": 0.98,
    }
    assert passes_metrics(arithmetic)
    assert passes_metrics(identity, identity=True)
    assert not passes_metrics(identity)


def test_minimum_passing_snapshot_uses_declared_data_order() -> None:
    rows = [
        {
            "parameter_scale": "1m",
            "target_operator": "scalar.add",
            "role": "specialist",
            "training_examples_label": "262K",
            "checkpoint_step": 2559,
            "quality_gate_passed": True,
        },
        {
            "parameter_scale": "1m",
            "target_operator": "scalar.add",
            "role": "specialist",
            "training_examples_label": "16K",
            "checkpoint_step": 156,
            "quality_gate_passed": True,
        },
        {
            "parameter_scale": "1m",
            "target_operator": "scalar.neg",
            "role": "specialist",
            "training_examples_label": "1M",
            "checkpoint_step": 9800,
            "quality_gate_passed": False,
        },
    ]
    result = minimum_passing_snapshots(rows)
    by_operator = {row["target_operator"]: row for row in result}
    assert by_operator["scalar.add"]["minimum_passing_training_examples_label"] == "16K"
    assert by_operator["scalar.neg"]["minimum_passing_training_examples_label"] is None


def test_torch_available_for_evaluator() -> None:
    assert torch.tensor([1.0]).item() == 1.0
