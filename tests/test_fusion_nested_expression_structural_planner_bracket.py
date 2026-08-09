from __future__ import annotations

from opfusion import fusion_nested_expression_structural_planner as base
from opfusion import fusion_nested_expression_structural_planner_bracket as bracket
from opfusion.tokenizer import FixedVocabTokenizer


def _tokenizer() -> FixedVocabTokenizer:
    # Mirror the active surface tokenizer ABI: structural generator-facing
    # names are aliases, while decoded canonical tokens are ordinary symbols.
    tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<RESPONSE>",
        "+", ",", "[", "]", "=",
        "<OP_SCALAR_ADD>", "<OP_AGG_SUM>", "<OP_SCALAR_MIN>", "<OP_SCALAR_MAX>",
        *[f"<N_{value}>" for value in range(-20, 21)],
    ]
    aliases = {
        "<PLUS>": "+",
        "<COMMA>": ",",
        "<LBRACK>": "[",
        "<RBRACK>": "]",
    }
    return FixedVocabTokenizer(tokens=tokens, aliases=aliases)


def test_bracket_nested_expression_round_trip_through_aliases() -> None:
    tokenizer = _tokenizer()
    tokens = bracket.nested_expression_tokens(
        outer_operator="scalar.max",
        inner_operator="aggregation.sum",
        inner_values=(1, -2, 3),
        outer_extras=(4, 5),
    )
    ids = tokenizer.encode_tokens(tokens, add_bos=True, add_eos=False)
    assert tokenizer.tokens[ids[2]] == "["
    plan = bracket.parse_two_stage_plan(ids, tokenizer=tokenizer)
    assert plan == base.TwoStagePlan(
        outer_operator="scalar.max",
        inner_operator="aggregation.sum",
        inner_values=(1, -2, 3),
        outer_extras=(4, 5),
    )


def test_bracket_surface_uses_only_checkpoint_supported_aliases() -> None:
    tokens = bracket.nested_expression_tokens(
        outer_operator="scalar.add",
        inner_operator="scalar.add",
        inner_values=(1, 2),
        outer_extras=(3,),
    )
    assert "<LPAREN>" not in tokens
    assert "<RPAREN>" not in tokens
    assert tokens.count("<LBRACK>") == 2
    assert tokens.count("<RBRACK>") == 2
