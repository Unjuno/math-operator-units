from __future__ import annotations

import pytest

from opfusion import fusion_nested_expression_structural_planner as nested
from opfusion.tokenizer import FixedVocabTokenizer


def _tokenizer() -> FixedVocabTokenizer:
    # Minimal ABI-compatible vocabulary for structural parser unit tests.
    tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>", "<SEP>",
        "<LPAREN>", "<RPAREN>", "<COMMA>", "<RESPONSE>",
        "<OP_SCALAR_ADD>", "<OP_AGG_SUM>", "<OP_SCALAR_MIN>", "<OP_SCALAR_MAX>",
        *[f"<N_{value}>" for value in range(-20, 21)],
    ]
    return FixedVocabTokenizer(tokens=tokens)


def test_nested_expression_round_trip() -> None:
    tokenizer = _tokenizer()
    ids = nested.nested_expression_ids(
        tokenizer=tokenizer,
        outer_operator="scalar.max",
        inner_operator="scalar.add",
        inner_values=(3, -2),
        outer_extras=(7, 1),
    )
    plan = nested.parse_two_stage_plan(ids, tokenizer=tokenizer)
    assert plan == nested.TwoStagePlan(
        outer_operator="scalar.max",
        inner_operator="scalar.add",
        inner_values=(3, -2),
        outer_extras=(7, 1),
    )


def test_nested_expression_sum_to_add_round_trip() -> None:
    tokenizer = _tokenizer()
    ids = nested.nested_expression_ids(
        tokenizer=tokenizer,
        outer_operator="scalar.add",
        inner_operator="aggregation.sum",
        inner_values=(1, 2, 3),
        outer_extras=(4,),
    )
    plan = nested.parse_two_stage_plan(ids, tokenizer=tokenizer)
    assert plan.inner_operator == "aggregation.sum"
    assert plan.outer_operator == "scalar.add"
    assert plan.inner_values == (1, 2, 3)
    assert plan.outer_extras == (4,)


def test_parser_rejects_missing_outer_separator() -> None:
    tokenizer = _tokenizer()
    tokens = nested.nested_expression_tokens(
        outer_operator="scalar.add",
        inner_operator="scalar.add",
        inner_values=(1, 2),
        outer_extras=(3,),
    )
    inner_close = tokens.index("<RPAREN>")
    del tokens[inner_close + 1]
    ids = tokenizer.encode_tokens(tokens, add_bos=True, add_eos=False)
    with pytest.raises(ValueError):
        nested.parse_two_stage_plan(ids, tokenizer=tokenizer)


def test_parser_rejects_invalid_outer_arity() -> None:
    tokenizer = _tokenizer()
    tokens = nested.nested_expression_tokens(
        outer_operator="scalar.add",
        inner_operator="scalar.add",
        inner_values=(1, 2),
        outer_extras=(3,),
    )
    # Add a second outer extra, which would make scalar.add receive three values.
    outer_close = len(tokens) - 2
    tokens[outer_close:outer_close] = ["<COMMA>", "<N_4>"]
    ids = tokenizer.encode_tokens(tokens, add_bos=True, add_eos=False)
    with pytest.raises(ValueError):
        nested.parse_two_stage_plan(ids, tokenizer=tokenizer)
