from __future__ import annotations

import torch

from opfusion import fusion_controller_objective_ablation as objective
from opfusion import fusion_self_tuned_operator_controller as self_tuned


def _toy_batch() -> self_tuned.TeacherForcedControlBatch:
    prompt_ids = torch.tensor(
        [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=torch.long
    )
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    base_scores = torch.zeros((4, 6), dtype=torch.float32)
    target_source_probabilities = torch.tensor(
        [
            [0.10, 0.80, 0.20, 0.20, 0.20, 0.05],
            [0.10, 0.20, 0.80, 0.20, 0.20, 0.05],
            [0.10, 0.20, 0.20, 0.80, 0.20, 0.05],
            [0.10, 0.20, 0.20, 0.20, 0.80, 0.05],
        ],
        dtype=torch.float32,
    )
    return self_tuned.TeacherForcedControlBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        operator_labels=labels,
        base_scores=base_scores,
        target_source_probabilities=target_source_probabilities,
    )


def _controller() -> self_tuned.SelfTunedPromptController:
    torch.manual_seed(123)
    return self_tuned.SelfTunedPromptController(
        vocabulary_size=16,
        embedding_size=8,
        hidden_size=8,
    )


def test_direction_loss_selects_requested_objective() -> None:
    batch = _toy_batch()
    indices = torch.arange(batch.positions)
    controller = _controller()

    ce_loss, ce_metrics = objective._direction_loss(
        controller,
        batch,
        indices,
        objective="ce_only",
        fixed_strength=4.0,
        auxiliary_operator_weight=0.25,
    )
    nll_loss, nll_metrics = objective._direction_loss(
        controller,
        batch,
        indices,
        objective="nll_only",
        fixed_strength=4.0,
        auxiliary_operator_weight=0.25,
    )
    mixed_loss, mixed_metrics = objective._direction_loss(
        controller,
        batch,
        indices,
        objective="nll_ce",
        fixed_strength=4.0,
        auxiliary_operator_weight=0.25,
    )

    assert torch.isclose(ce_loss, torch.tensor(ce_metrics["operator_ce"]), atol=1e-6)
    assert torch.isclose(nll_loss, torch.tensor(nll_metrics["nll"]), atol=1e-6)
    expected = mixed_metrics["nll"] + 0.25 * mixed_metrics["operator_ce"]
    assert abs(float(mixed_loss.detach()) - expected) < 1e-6


def test_direction_loss_is_independent_of_predicted_scale_head() -> None:
    batch = _toy_batch()
    indices = torch.arange(batch.positions)
    first = _controller()
    second = _controller()
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        second.scale_head.weight.fill_(100.0)
        second.scale_head.bias.fill_(100.0)

    first_loss, first_metrics = objective._direction_loss(
        first,
        batch,
        indices,
        objective="nll_only",
        fixed_strength=4.0,
        auxiliary_operator_weight=0.25,
    )
    second_loss, second_metrics = objective._direction_loss(
        second,
        batch,
        indices,
        objective="nll_only",
        fixed_strength=4.0,
        auxiliary_operator_weight=0.25,
    )

    assert torch.allclose(first_loss, second_loss, atol=1e-7)
    assert first_metrics == second_metrics


def test_unknown_objective_is_rejected() -> None:
    batch = _toy_batch()
    controller = _controller()
    try:
        objective._direction_loss(
            controller,
            batch,
            torch.arange(batch.positions),
            objective="unknown",
            fixed_strength=4.0,
            auxiliary_operator_weight=0.25,
        )
    except ValueError as exc:
        assert "unknown objective" in str(exc)
    else:
        raise AssertionError("unknown objective should raise")
