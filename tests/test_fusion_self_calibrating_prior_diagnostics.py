from __future__ import annotations

import torch

from opfusion.fusion_self_calibrating_operator_controller import (
    FUNCTIONAL_OPERATORS,
    SelfCalibratingController,
    TeacherForcedTrace,
)
from opfusion.fusion_self_calibrating_prior_diagnostics import (
    SOURCE_NAMES,
    teacher_forced_prompt_diagnostics,
)


def test_prior_diagnostics_report_all_sources_and_operators() -> None:
    controller = SelfCalibratingController(
        vocabulary_size=16,
        embedding_size=4,
        hidden_size=5,
    )
    traces = [
        TeacherForcedTrace(
            prompt=(1, 3 + operator_index, 9, 2),
            operator_index=operator_index,
            combined_states=torch.zeros(1, len(SOURCE_NAMES)),
            target_source_probabilities=torch.full((1, len(SOURCE_NAMES)), 0.5),
        )
        for operator_index in range(len(FUNCTIONAL_OPERATORS))
    ]
    report = teacher_forced_prompt_diagnostics(
        controller,
        traces,
        device=torch.device("cpu"),
    )
    assert report["source_order"] == list(SOURCE_NAMES)
    assert set(report["mean_log_prior_by_operator"]) == set(FUNCTIONAL_OPERATORS)
    for operator in FUNCTIONAL_OPERATORS:
        assert set(report["mean_log_prior_by_operator"][operator]) == set(SOURCE_NAMES)
        soft = report["mean_prior_softmax_by_operator"][operator]
        assert abs(sum(soft.values()) - 1.0) < 1e-6
        argmax = report["argmax_source_distribution_by_operator"][operator]
        assert abs(sum(argmax.values()) - 1.0) < 1e-6
