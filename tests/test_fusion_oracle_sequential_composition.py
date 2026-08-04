from __future__ import annotations

import pytest

from opfusion.fusion_oracle_sequential_composition import (
    apply_operator,
    composition_operands,
    parse_final_numeric_token,
    prompt_ids_for_values,
)
from opfusion.tokenizer import FixedVocabTokenizer


def _tokenizer() -> FixedVocabTokenizer:
    return FixedVocabTokenizer(
        [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>",
            "<OP_SCALAR_ADD>",
            "<RESPONSE>",
            "<PLUS>",
            "<N_2>",
            "<N_3>",
            "<N_5>",
            "=",
        ]
    )


class _Factory:
    @staticmethod
    def _state_tokens(operator: str, values):
        assert operator == "scalar.add"
        return ["<N_2>", "<PLUS>", "<N_3>"]


def test_apply_operator_definitions() -> None:
    assert apply_operator("scalar.add", (2, 3)) == 5
    assert apply_operator("aggregation.sum", (2, -3, 4)) == 3
    assert apply_operator("scalar.min", (2, -3, 4)) == -3
    assert apply_operator("scalar.max", (2, -3, 4)) == 4
    with pytest.raises(ValueError):
        apply_operator("scalar.add", (1, 2, 3))


def test_parse_final_numeric_token_uses_last_number_before_eos() -> None:
    tokenizer = _tokenizer()
    ids = tokenizer.encode_tokens(
        ["=", "<N_2>", "=", "<N_5>"], add_bos=False, add_eos=True
    )
    assert parse_final_numeric_token(ids, tokenizer) == 5
    assert parse_final_numeric_token([tokenizer.eos_id], tokenizer) is None


def test_prompt_ids_encode_operator_state_and_response_boundary() -> None:
    tokenizer = _tokenizer()
    ids = prompt_ids_for_values(
        factory=_Factory(),
        tokenizer=tokenizer,
        operator="scalar.add",
        values=(2, 3),
    )
    assert tokenizer.decode(ids) == (
        "<BOS> <OP_SCALAR_ADD> <N_2> <PLUS> <N_3> <RESPONSE>"
    )


def test_composition_operands_are_deterministic_and_well_shaped() -> None:
    first = composition_operands(
        inner_operator="aggregation.sum",
        outer_operator="scalar.add",
        seed=733000,
        sample_index=2,
    )
    second = composition_operands(
        inner_operator="aggregation.sum",
        outer_operator="scalar.add",
        seed=733000,
        sample_index=2,
    )
    assert first == second
    inner, extras = first
    assert len(inner) == 3
    assert len(extras) == 1
    assert all(-16 <= value <= 16 for value in (*inner, *extras))
