from __future__ import annotations

from typing import Sequence

from opfusion import fusion_nested_expression_structural_planner as base
from opfusion import fusion_oracle_sequential_composition as seq
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import OPERATOR_TOKENS


FUNCTIONAL_OPERATORS = base.FUNCTIONAL_OPERATORS


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
    # Generator-facing aliases are intentional here. FixedVocabTokenizer maps
    # them onto the checkpoint's canonical surface IDs ("[", "]", ",").
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


def _parse_numeric_id_list(
    ids: Sequence[int],
    start: int,
    *,
    tokenizer: FixedVocabTokenizer,
    comma_id: int,
    right_id: int,
) -> tuple[tuple[int, ...], int]:
    values: list[int] = []
    index = start
    expect_number = True
    while index < len(ids):
        token_id = int(ids[index])
        if token_id == right_id:
            if expect_number and values:
                raise ValueError("trailing comma before closing bracket")
            if not values:
                raise ValueError("empty operand list")
            return tuple(values), index + 1
        if expect_number:
            if token_id < 0 or token_id >= tokenizer.vocab_size:
                raise ValueError("numeric token id out of range")
            values.append(base._parse_number_token(tokenizer.tokens[token_id]))
            expect_number = False
        else:
            if token_id != comma_id:
                raise ValueError(
                    f"expected comma id {comma_id}, got token id {token_id}"
                )
            expect_number = True
        index += 1
    raise ValueError("unterminated operand list")


def parse_two_stage_plan(
    prompt_ids: Sequence[int], *, tokenizer: FixedVocabTokenizer
) -> base.TwoStagePlan:
    ids = [int(token_id) for token_id in prompt_ids]
    operator_by_id = {
        tokenizer.token_to_id[OPERATOR_TOKENS[operator]]: operator
        for operator in FUNCTIONAL_OPERATORS
    }
    left_id = tokenizer.token_to_id["<LBRACK>"]
    right_id = tokenizer.token_to_id["<RBRACK>"]
    comma_id = tokenizer.token_to_id["<COMMA>"]
    response_id = tokenizer.token_to_id["<RESPONSE>"]

    index = 0
    if ids and ids[0] == tokenizer.bos_id:
        index += 1
    if index >= len(ids) or ids[index] not in operator_by_id:
        raise ValueError("missing outer operator")
    outer_operator = operator_by_id[ids[index]]
    index += 1
    if index >= len(ids) or ids[index] != left_id:
        raise ValueError("missing outer left bracket")
    index += 1
    if index >= len(ids) or ids[index] not in operator_by_id:
        raise ValueError("missing inner operator")
    inner_operator = operator_by_id[ids[index]]
    index += 1
    if index >= len(ids) or ids[index] != left_id:
        raise ValueError("missing inner left bracket")
    index += 1

    inner_values, index = _parse_numeric_id_list(
        ids,
        index,
        tokenizer=tokenizer,
        comma_id=comma_id,
        right_id=right_id,
    )
    if index >= len(ids) or ids[index] != comma_id:
        raise ValueError("missing separator between inner expression and outer extras")
    index += 1
    outer_extras, index = _parse_numeric_id_list(
        ids,
        index,
        tokenizer=tokenizer,
        comma_id=comma_id,
        right_id=right_id,
    )
    if index >= len(ids) or ids[index] != response_id:
        raise ValueError("missing response marker")
    index += 1
    if index != len(ids):
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
    # Reuse the deterministic evaluation pipeline, changing only the nested
    # surface/parser to the checkpoint ABI and matching structural IDs rather
    # than decoded canonical strings, so aliases are handled correctly.
    base.nested_expression_tokens = nested_expression_tokens
    base.parse_two_stage_plan = parse_two_stage_plan
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
