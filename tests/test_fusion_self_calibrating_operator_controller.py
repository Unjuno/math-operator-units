from __future__ import annotations

from types import SimpleNamespace

import torch

from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion.fusion_self_calibrating_operator_controller import (
    FUNCTIONAL_OPERATORS,
    SOURCE_COUNT,
    SelfCalibratingController,
    TeacherForcedTrace,
    combine_self_calibrating_states,
    fit_self_calibrating_controller,
    teacher_forced_metrics,
)
from opfusion.training.data import EXPERIMENT_OPERATORS


def test_controller_prior_is_gauge_fixed() -> None:
    torch.manual_seed(1)
    controller = SelfCalibratingController(
        vocabulary_size=12,
        embedding_size=4,
        hidden_size=5,
    )
    with torch.no_grad():
        controller.source_head.weight.normal_()
        controller.source_head.bias.normal_()
    prior = controller.source_prior([1, 3, 5, 7], device=torch.device("cpu"))
    assert prior.shape == (SOURCE_COUNT,)
    torch.testing.assert_close(prior.mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_zero_prior_matches_oracle_zero_prior_control() -> None:
    fast = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    slow = torch.tensor([-1.5, -1.0, -2.0, -3.0, -4.0, -5.0])
    candidate = SimpleNamespace(slow_mix=0.25, temperature=0.75)
    learned = combine_self_calibrating_states(
        fast,
        slow,
        candidate=candidate,
        source_prior=torch.zeros_like(fast),
    )
    expected = oracle.combine_oracle_operator_states(
        fast,
        slow,
        candidate=candidate,
        operator="scalar.add",
        strength=0.0,
    )
    torch.testing.assert_close(learned, expected)


def _toy_traces() -> list[TeacherForcedTrace]:
    rows: list[TeacherForcedTrace] = []
    for label, operator_token in enumerate((4, 5, 6, 7)):
        operator = FUNCTIONAL_OPERATORS[label]
        matching_source = 1 + EXPERIMENT_OPERATORS.index(operator)
        for value_token in (8, 9, 10, 11):
            target_probabilities = torch.full((3, SOURCE_COUNT), 0.08)
            target_probabilities[:, matching_source] = 0.90
            rows.append(
                TeacherForcedTrace(
                    prompt=(1, operator_token, value_token, 2),
                    operator_index=label,
                    combined_states=torch.zeros(3, SOURCE_COUNT),
                    target_source_probabilities=target_probabilities,
                )
            )
    return rows


def test_self_calibration_is_deterministic_and_reduces_token_nll() -> None:
    traces = _toy_traces()
    device = torch.device("cpu")
    torch.manual_seed(737)
    initial = SelfCalibratingController(
        vocabulary_size=12,
        embedding_size=8,
        hidden_size=8,
    )
    initial_metrics = teacher_forced_metrics(
        initial,
        traces,
        pad_id=0,
        device=device,
    )
    kwargs = dict(
        traces=traces,
        holdout_traces=traces,
        vocabulary_size=12,
        pad_id=0,
        embedding_size=8,
        hidden_size=8,
        learning_rate=0.05,
        steps=160,
        prior_l2_weight=0.001,
        seed=739000,
        device=device,
    )
    first, first_report = fit_self_calibrating_controller(**kwargs)
    second, second_report = fit_self_calibrating_controller(**kwargs)
    assert first_report == second_report
    assert first_report["train_metrics"]["token_nll"] < initial_metrics["token_nll"]
    assert first_report["train_metrics"]["prior_argmax_matching_accuracy"] == 1.0
    assert first_report["holdout_metrics"]["prior_argmax_matching_accuracy"] == 1.0
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0.0, atol=0.0)
