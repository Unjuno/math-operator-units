from __future__ import annotations

import torch

from opfusion import fusion_learned_operator_controller as learned
from opfusion.training.data import EXPERIMENT_OPERATORS


def test_prompt_operator_controller_shapes() -> None:
    controller = learned.PromptOperatorController(32, embedding_size=4, hidden_size=8)
    ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    logits = controller(ids, mask)
    assert logits.shape == (2, len(learned.FUNCTIONAL_OPERATORS))
    assert torch.isfinite(logits).all()


def test_feature_mode_masks_only_operator_token() -> None:
    prompt = [7, 11, 13, 17, 19]
    assert learned.apply_feature_mode_to_prompt(prompt, bos_id=7, mode="full") == prompt
    assert learned.apply_feature_mode_to_prompt(prompt, bos_id=7, mode="masked") == [7, 7, 13, 17, 19]


def test_controller_prior_maps_soft_posterior_to_functional_sources() -> None:
    posterior = torch.tensor([0.1, 0.2, 0.3, 0.4])
    strength = 2.0
    prior = learned.controller_operator_prior(
        posterior,
        source_count=len(EXPERIMENT_OPERATORS) + 1,
        strength=strength,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert prior.shape == (len(EXPERIMENT_OPERATORS) + 1,)
    assert prior[0].item() == 0.0
    assert prior[1 + EXPERIMENT_OPERATORS.index("scalar.neg")].item() == 0.0
    for index, operator in enumerate(learned.FUNCTIONAL_OPERATORS):
        source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
        assert torch.isclose(prior[source_index], posterior[index] * strength)


def test_padding_mask_excludes_padding_positions() -> None:
    ids, mask = learned._pad_prompts(
        [[1, 2, 3], [4, 5]], pad_id=0, device=torch.device("cpu")
    )
    assert ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert mask.tolist() == [[True, True, True], [True, True, False]]
