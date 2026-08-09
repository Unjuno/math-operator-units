from __future__ import annotations

from typing import Sequence

from opfusion import fusion_nested_expression_structural_planner as base
from opfusion import fusion_oracle_sequential_composition as seq
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import OPERATOR_TOKENS


FUNCTIONAL_OPERATORS = base.FUNCTIONAL_OPERATORS
TOKEN_TO_OPERATOR = base.TOKEN_TO_OPERATOR


def nested_expression_tokens(
    *,
    outer_operator: str,
    inner_operator: str,
    inner_values: Sequence[int],
    outer_extras: Sequence[int],
) -> list[str]:
    if outer_operator not in FUNCTIONAL_OPERATORS:
        raise KeyError(outer_operator)
    if inner_operator not in FUNCTIONAL_OPERATORS:
        raise KeyError(inner_operator)
    # The active fusion-factory checkpoint vocabulary is narrower than the
    # core tokenizer design.  LBRACK/RBRACK/COMMA are guaranteed by
    # SyntheticTraceFactory, so use them as the depth-2 call delimiters.
    return [
        OPERATOR_TOKENS[outer_operator],
        "<LBRACK>",
        OPERATOR_TOKENS[inner_operator],
        "<LBRACK>",
        *base._comma_list(inner_values),
        "<RBRACK>",
        "<COMMA>",
        *base._comma_list(outer_extras),
        "<RBRACK>",
        "<RESPONSE>",
    ]


def _parse_numeric_list(
    tokens: Sequence[str], start: int
) -> tuple[tuple[int, ...], int]:
    values: list[int] = []
    index = start
    expect_number = True
    while index < len(tokens):
        token = tokens[index]
        if token == "<RBRACK>":
            if expect_number and values:
                raise ValueError("trailing comma before closing bracket")
            if not values:
                raise ValueError("empty operand list")
            return tuple(values), index + 1
        if expect_number:
            values.append(base._parse_number_token(token))
            expect_number = False
        else:
            if token != "<COMMA>":
                raise ValueError(f"expected comma, got {token}")
            expect_number = True
        index += 1
    raise ValueError("unterminated operand list")


def parse_two_stage_plan(
    prompt_ids: Sequence[int], *, tokenizer: FixedVocabTokenizer
) -> base.TwoStagePlan:
    tokens = [tokenizer.tokens[int(token_id)] for token_id in prompt_ids]
    index = 0
    if tokens and tokens[0] == tokenizer.tokens[tokenizer.bos_id]:
        index += 1
    if index >= len(tokens) or tokens[index] not in TOKEN_TO_OPERATOR:
        raise ValueError("missing outer operator")
    outer_operator = TOKEN_TO_OPERATOR[tokens[index]]
    index += 1
    if index >= len(tokens) or tokens[index] != "<LBRACK>":
        raise ValueError("missing outer left bracket")
    index += 1
    if index >= len(tokens) or tokens[index] not in TOKEN_TO_OPERATOR:
        raise ValueError("missing inner operator")
    inner_operator = TOKEN_TO_OPERATOR[tokens[index]]
    index += 1
    if index >= len(tokens) or tokens[index] != "<LBRACK>":
        raise ValueError("missing inner left bracket")
    index += 1

    inner_values, index = _parse_numeric_list(tokens, index)
    if index >= len(tokens) or tokens[index] != "<COMMA>":
        raise ValueError("missing separator between inner expression and outer extras")
    index += 1
    outer_extras, index = _parse_numeric_list(tokens, index)
    if index >= len(tokens) or tokens[index] != "<RESPONSE>":
        raise ValueError("missing response marker")
    index += 1
    if index != len(tokens):
        raise ValueError("unexpected tokens after response marker")

    seq.apply_operator(inner_operator, inner_values)
    seq.apply_operator(outer_operator, (0, *outer_extras))
    return base.TwoStagePlan(
        outer_operator=outer_operator,
        inner_operator=inner_operator,
        inner_values=inner_values,
        outer_extras=outer_extras,
    )


def main() -> int:
    # Reuse the full deterministic evaluation pipeline, changing only the
    # nested surface/parser to tokens guaranteed by the checkpoint ABI.
    base.nested_expression_tokens = nested_expression_tokens
    base.parse_two_stage_plan = parse_two_stage_plan
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
