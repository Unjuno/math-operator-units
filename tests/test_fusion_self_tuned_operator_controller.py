from __future__ import annotations

import torch

from opfusion.fusion_self_tuned_operator_controller import (
    FUNCTIONAL_OPERATORS,
    SelfTunedPromptController,
    TeacherForcedControlBatch,
    fit_self_tuned_controller,
    source_prior_from_control,
)
from opfusion.training.data import EXPERIMENT_OPERATORS


def test_source_prior_maps_probability_times_learned_scale() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4])
    scale = torch.tensor(3.0)
    prior = source_prior_from_control(probabilities, scale, source_count=6)
    assert prior.shape == (6,)
    assert prior[0].item() == 0.0
    assert prior[1 + EXPERIMENT_OPERATORS.index("scalar.neg")].item() == 0.0
    for index, operator in enumerate(FUNCTIONAL_OPERATORS):
        source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
        assert torch.isclose(prior[source_index], 3.0 * probabilities[index])


def test_self_tuned_controller_outputs_positive_scale_and_normalized_direction() -> None:
    torch.manual_seed(7)
    controller = SelfTunedPromptController(vocabulary_size=12, embedding_size=4, hidden_size=5)
    probabilities, scale = controller.prompt_control([1, 4, 8, 2], device=torch.device("cpu"))
    assert probabilities.shape == (4,)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    assert bool((probabilities > 0).all())
    assert float(scale) > 0.0


def _toy_batch() -> tuple[TeacherForcedControlBatch, list[tuple[list[int], int]]]:
    prompts: list[list[int]] = []
    labels: list[int] = []
    base_scores: list[torch.Tensor] = []
    target_probabilities: list[torch.Tensor] = []
    holdout: list[tuple[list[int], int]] = []
    source_count = 6
    for label, operator_token in enumerate((4, 5, 6, 7)):
        prompt = [1, operator_token, 8 + label, 2]
        holdout.append((prompt, label))
        for _ in range(8):
            row = torch.full((source_count,), 0.05)
            source_index = 1 + EXPERIMENT_OPERATORS.index(FUNCTIONAL_OPERATORS[label])
            row[source_index] = 0.90
            prompts.append(prompt)
            labels.append(label)
            base_scores.append(torch.zeros(source_count))
            target_probabilities.append(row)
    ids = torch.tensor(prompts, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    batch = TeacherForcedControlBatch(
        prompt_ids=ids,
        prompt_mask=mask,
        operator_labels=torch.tensor(labels, dtype=torch.long),
        base_scores=torch.stack(base_scores),
        target_source_probabilities=torch.stack(target_probabilities),
    )
    return batch, holdout


def test_self_tuned_fit_is_deterministic_and_learns_control_magnitude() -> None:
    batch, holdout = _toy_batch()
    kwargs = dict(
        batch=batch,
        holdout_examples=holdout,
        vocabulary_size=12,
        pad_id=0,
        embedding_size=8,
        hidden_size=8,
        learning_rate=0.05,
        steps=140,
        batch_positions=64,
        auxiliary_operator_weight=0.25,
        scale_l2_weight=0.0001,
        seed=739000,
        device=torch.device("cpu"),
    )
    first, first_report = fit_self_tuned_controller(**kwargs)
    second, second_report = fit_self_tuned_controller(**kwargs)
    assert first_report == second_report
    assert first_report["holdout_metrics"]["accuracy"] == 1.0
    assert first_report["optimization_last"] < first_report["optimization_first"]
    assert first_report["holdout_scale_mean"] > 1.0
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0.0, atol=0.0)
