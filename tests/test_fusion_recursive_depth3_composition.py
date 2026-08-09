from __future__ import annotations

from opfusion import fusion_recursive_depth3_composition as depth3
from opfusion.tokenizer import FixedVocabTokenizer


def _surface_tokenizer() -> FixedVocabTokenizer:
    tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<RESPONSE>",
        "+", ",", "[", "]", "=",
        "<OP_SCALAR_ADD>", "<OP_AGG_SUM>", "<OP_SCALAR_MIN>", "<OP_SCALAR_MAX>",
        *[f"<N_{value}>" for value in range(-128, 129)],
    ]
    aliases = {
        "<PLUS>": "+",
        "<COMMA>": ",",
        "<LBRACK>": "[",
        "<RBRACK>": "]",
    }
    return FixedVocabTokenizer(tokens=tokens, aliases=aliases)


def test_depth3_plan_round_trip_through_surface_aliases() -> None:
    tokenizer = _surface_tokenizer()
    expected = depth3.NestedPlan(
        operator="scalar.max",
        child=depth3.NestedPlan(
            operator="aggregation.sum",
            child=depth3.NestedPlan(
                operator="scalar.add",
                leaf_values=(3, -2),
            ),
            extras=(4, 5),
        ),
        extras=(-1, 7),
    )
    ids = depth3.prompt_ids_for_plan(expected, tokenizer=tokenizer)
    parsed = depth3.parse_nested_plan(ids, tokenizer=tokenizer)
    assert parsed == expected
    assert depth3.plan_depth(parsed) == 3
    assert depth3.operators_leaf_to_root(parsed) == (
        "scalar.add",
        "aggregation.sum",
        "scalar.max",
    )


def test_depth3_true_value_matches_manual_composition() -> None:
    plan = depth3.NestedPlan(
        operator="scalar.min",
        child=depth3.NestedPlan(
            operator="scalar.max",
            child=depth3.NestedPlan(
                operator="aggregation.sum",
                leaf_values=(2, 3, -1),
            ),
            extras=(9, 4),
        ),
        extras=(8, 6),
    )
    # sum=4; max(4,9,4)=9; min(9,8,6)=6
    assert depth3.true_plan_value(plan) == 6


def test_transition_count_distinguishes_zero_one_two_switches() -> None:
    same = depth3.NestedPlan(
        operator="scalar.add",
        child=depth3.NestedPlan(
            operator="scalar.add",
            child=depth3.NestedPlan(
                operator="scalar.add", leaf_values=(1, 2)
            ),
            extras=(3,),
        ),
        extras=(4,),
    )
    one = depth3.NestedPlan(
        operator="scalar.max",
        child=depth3.NestedPlan(
            operator="scalar.max",
            child=depth3.NestedPlan(
                operator="scalar.add", leaf_values=(1, 2)
            ),
            extras=(3, 4),
        ),
        extras=(5, 6),
    )
    two = depth3.NestedPlan(
        operator="scalar.min",
        child=depth3.NestedPlan(
            operator="scalar.max",
            child=depth3.NestedPlan(
                operator="scalar.add", leaf_values=(1, 2)
            ),
            extras=(3, 4),
        ),
        extras=(5, 6),
    )
    assert depth3.transition_count(same) == 0
    assert depth3.transition_count(one) == 1
    assert depth3.transition_count(two) == 2


def test_generated_depth3_plan_has_valid_shape_and_range() -> None:
    plan = depth3.make_depth3_plan(
        inner_operator="aggregation.sum",
        middle_operator="scalar.add",
        outer_operator="scalar.max",
        seed=735000,
        sample_index=0,
    )
    assert depth3.plan_depth(plan) == 3
    assert plan.child is not None and plan.child.child is not None
    assert len(plan.child.child.leaf_values) == 3
    assert len(plan.child.extras) == 1
    assert len(plan.extras) == 2
    assert -128 <= depth3.true_plan_value(plan) <= 128


def test_counter_finalizes_end_to_end_as_stage3_accuracy() -> None:
    counter = depth3._empty_counter()
    counter["cases"] = 2
    counter["plan_parse"] = 2
    counter["plan_exact"] = 2
    counter["stage1_attempted"] = 2
    counter["stage1_parse"] = 2
    counter["stage1_correct"] = 2
    counter["stage1_local_correct"] = 2
    counter["stage2_attempted"] = 2
    counter["stage2_parse"] = 2
    counter["stage2_correct"] = 2
    counter["stage2_local_correct"] = 2
    counter["stage3_attempted"] = 2
    counter["stage3_parse"] = 2
    counter["stage3_correct"] = 1
    counter["stage3_local_correct"] = 1
    counter["stage1_correct_cases"] = 2
    counter["stage2_correct_cases"] = 2
    counter["stage2_correct_given_stage1_correct"] = 2
    counter["stage3_correct_given_stage2_correct"] = 1
    result = depth3._finalize_counter(counter)
    assert result["end_to_end_accuracy"] == 0.5
    assert result["stage2_accuracy_given_stage1_correct"] == 1.0
    assert result["stage3_accuracy_given_stage2_correct"] == 0.5
