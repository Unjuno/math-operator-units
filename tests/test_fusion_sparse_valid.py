from __future__ import annotations

from types import SimpleNamespace

import torch

from opfusion.fusion_sparse_valid import (
    SparseEvidenceCompositor,
    build_valid_prefix_records,
    valid_set_loss,
    valid_state_paths,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import SyntheticDataConfig, SyntheticTraceFactory, TrainingExample


def _tokenizer() -> FixedVocabTokenizer:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "<OP_SCALAR_ADD>",
        "<OP_AGG_SUM>",
        "<OP_SCALAR_NEG>",
        "<OP_SCALAR_MIN>",
        "<OP_SCALAR_MAX>",
        "<PLUS>",
        "<COMMA>",
        "<LBRACK>",
        "<RBRACK>",
        "=",
        "<RESPONSE>",
        "<TASK_COPY>",
        *[f"<N_{value}>" for value in range(-16, 17)],
    ]
    config = SimpleNamespace(tokens=tokens, aliases={})
    return FixedVocabTokenizer.from_config(config)


def _factory() -> tuple[FixedVocabTokenizer, SyntheticTraceFactory]:
    tokenizer = _tokenizer()
    config = SyntheticDataConfig(
        operand_min=-4,
        operand_max=4,
        min_terms=3,
        max_terms=3,
        numeric_token_min=-16,
        numeric_token_max=16,
        value_ood_abs_min=5,
        value_ood_abs_max=5,
        length_ood_min_terms=4,
        length_ood_max_terms=4,
        randomized_train_reduction=False,
    )
    return tokenizer, SyntheticTraceFactory(tokenizer, config)


def test_sum_enumerates_both_adjacent_reduction_orders() -> None:
    example = TrainingExample(
        job_id="aggregation.sum",
        operator_id="aggregation.sum",
        prompt_tokens=(),
        response_tokens=(),
        final_value=6,
        split="validation",
        task="full_trace",
        initial_values=(1, 2, 3),
        prompt_state_values=(1, 2, 3),
        trace_states=((1, 2, 3), (3, 3), (6,)),
    )
    paths = valid_state_paths(example)
    assert len(paths) == 2
    assert ((1, 2, 3), (3, 3), (6,)) in paths
    assert ((1, 2, 3), (1, 5), (6,)) in paths


def test_valid_prefix_trie_exposes_set_valued_next_token() -> None:
    tokenizer, factory = _factory()
    example = TrainingExample(
        job_id="aggregation.sum",
        operator_id="aggregation.sum",
        prompt_tokens=(),
        response_tokens=(),
        final_value=6,
        split="validation",
        task="full_trace",
        initial_values=(1, 2, 3),
        prompt_state_values=(1, 2, 3),
        trace_states=((1, 2, 3), (3, 3), (6,)),
    )
    records = build_valid_prefix_records(factory, tokenizer, example)
    eq_id = tokenizer.token_to_id["="]
    root = next(row for row in records if row.prefix == ())
    after_eq = next(row for row in records if row.prefix == (eq_id,))
    assert root.valid_next == (eq_id,)
    assert set(after_eq.valid_next) == {
        tokenizer.token_to_id["<N_1>"],
        tokenizer.token_to_id["<N_3>"],
    }


def test_valid_set_loss_rewards_probability_mass_on_any_valid_token() -> None:
    valid = torch.tensor([[False, True, True, False]])
    bad = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    good = torch.tensor([[0.0, 4.0, 3.0, 0.0]])
    assert valid_set_loss(good, valid) < valid_set_loss(bad, valid)


def test_sparse_compositor_has_no_positive_confidence_floor() -> None:
    torch.manual_seed(3)
    model = SparseEvidenceCompositor(hidden_size=4, use_confidence=True, allow_threshold=True)
    assert model.score_network is not None
    with torch.no_grad():
        model.score_network[-1].weight.zero_()
        model.score_network[-1].bias.fill_(-20.0)
    base = torch.randn(2, 19)
    units = torch.randn(2, 5, 19)
    fused, confidence, active, threshold = model.compose(base, units)
    assert fused.shape == base.shape
    assert confidence.shape == (2, 5)
    assert float(confidence.max()) < 1e-6
    assert active.shape == (2, 5)
    assert float(threshold) >= 0.0
    assert torch.isfinite(fused).all()


def test_uniform_sparse_uses_all_five_raw_fields() -> None:
    torch.manual_seed(4)
    model = SparseEvidenceCompositor(hidden_size=4, use_confidence=False, allow_threshold=True)
    base = torch.randn(3, 17)
    units = torch.randn(3, 5, 17)
    fused, confidence, _, _ = model.compose(base, units)
    assert fused.shape == base.shape
    assert torch.equal(confidence, torch.ones_like(confidence))
    assert torch.isfinite(fused).all()
