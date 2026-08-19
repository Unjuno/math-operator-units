from __future__ import annotations

from types import SimpleNamespace

import torch

from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion.fusion_learned_operator_controller import (
    FUNCTIONAL_OPERATORS,
    PromptOperatorController,
    combine_learned_operator_states,
    continuous_operator_prior,
    fit_prompt_controller,
)
from opfusion.training.data import EXPERIMENT_OPERATORS


def test_continuous_operator_prior_maps_only_functional_specialists() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
    prior = continuous_operator_prior(probabilities, source_count=6, strength=2.0)
    assert prior[0].item() == 0.0
    assert prior[1 + EXPERIMENT_OPERATORS.index("scalar.neg")].item() == 0.0
    for index, operator in enumerate(FUNCTIONAL_OPERATORS):
        source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
        assert torch.isclose(prior[source_index], 2.0 * probabilities[index])


def test_zero_strength_matches_oracle_zero_prior_control() -> None:
    fast = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    slow = torch.tensor([-1.5, -1.0, -2.0, -3.0, -4.0, -5.0])
    candidate = SimpleNamespace(slow_mix=0.25, temperature=0.75)
    controller_probabilities = torch.tensor([0.05, 0.15, 0.30, 0.50])
    learned = combine_learned_operator_states(
        fast,
        slow,
        candidate=candidate,
        controller_probabilities=controller_probabilities,
        strength=0.0,
    )
    expected = oracle.combine_oracle_operator_states(
        fast,
        slow,
        candidate=candidate,
        operator="scalar.add",
        strength=0.0,
    )
    torch.testing.assert_close(learned, expected)


def test_prompt_controller_probabilities_are_normalized() -> None:
    torch.manual_seed(1)
    controller = PromptOperatorController(vocabulary_size=12, embedding_size=4, hidden_size=5)
    probabilities = controller.probabilities([1, 3, 5, 7], device=torch.device("cpu"))
    assert probabilities.shape == (4,)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert bool((probabilities > 0).all())


def _toy_examples() -> list[tuple[list[int], int]]:
    rows: list[tuple[list[int], int]] = []
    for label, operator_token in enumerate((4, 5, 6, 7)):
        for value_token in (8, 9, 10, 11):
            rows.append(([1, operator_token, value_token, 2], label))
    return rows


def test_controller_fit_is_deterministic_and_learns_operator_token_signal() -> None:
    examples = _toy_examples()
    kwargs = dict(
        examples=examples,
        holdout_examples=examples,
        vocabulary_size=12,
        pad_id=0,
        embedding_size=8,
        hidden_size=8,
        learning_rate=0.05,
        steps=120,
        seed=737000,
        device=torch.device("cpu"),
    )
    first, first_report = fit_prompt_controller(**kwargs)
    second, second_report = fit_prompt_controller(**kwargs)
    assert first_report == second_report
    assert first_report["train_metrics"]["accuracy"] == 1.0
    assert first_report["holdout_metrics"]["accuracy"] == 1.0
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0.0, atol=0.0)
