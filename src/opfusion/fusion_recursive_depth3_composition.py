from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_learned_operator_controller as learned
from opfusion import fusion_oracle_sequential_composition as seq
from opfusion import fusion_posterior_transition_boundary_gate as transition
from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import OPERATOR_TOKENS, SyntheticTraceFactory


FUNCTIONAL_OPERATORS = seq.FUNCTIONAL_OPERATORS
DEFAULT_DATA_SEED = 735_000


@dataclass(frozen=True)
class NestedPlan:
    operator: str
    child: NestedPlan | None = None
    leaf_values: tuple[int, ...] = ()
    extras: tuple[int, ...] = ()


@dataclass
class StageExecution:
    operator: str
    true_value: int
    local_target: int | None
    generated_value: int | None
    attempted: bool
    parsed: bool
    correct: bool
    local_correct: bool
    gate_value: float | None
    controller_correct: bool | None


@dataclass
class ExecutionState:
    value: int | None
    fast_state: torch.Tensor | None
    slow_state: torch.Tensor | None
    posterior: torch.Tensor | None
    stages: list[StageExecution]


def _number_token(value: int) -> str:
    return f"<N_{int(value)}>"


def _parse_number_token(token: str) -> int:
    if not token.startswith("<N_") or not token.endswith(">"):
        raise ValueError(f"expected atomic integer token, got {token}")
    return int(token[3:-1])


def _comma_list(values: Sequence[int]) -> list[str]:
    output: list[str] = []
    for index, value in enumerate(values):
        if index:
            output.append("<COMMA>")
        output.append(_number_token(int(value)))
    return output


def plan_tokens(plan: NestedPlan) -> list[str]:
    if plan.operator not in FUNCTIONAL_OPERATORS:
        raise KeyError(plan.operator)
    output = [OPERATOR_TOKENS[plan.operator], "<LBRACK>"]
    if plan.child is None:
        if not plan.leaf_values or plan.extras:
            raise ValueError("leaf plan must contain leaf_values and no extras")
        output.extend(_comma_list(plan.leaf_values))
    else:
        if plan.leaf_values or not plan.extras:
            raise ValueError("internal plan must contain child/extras and no leaf_values")
        output.extend(plan_tokens(plan.child))
        output.append("<COMMA>")
        output.extend(_comma_list(plan.extras))
    output.append("<RBRACK>")
    return output


def prompt_ids_for_plan(
    plan: NestedPlan, *, tokenizer: FixedVocabTokenizer
) -> list[int]:
    return tokenizer.encode_tokens(
        [*plan_tokens(plan), "<RESPONSE>"],
        add_bos=True,
        add_eos=False,
    )


def _parse_numeric_ids(
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
                raise ValueError("trailing comma")
            if not values:
                raise ValueError("empty numeric list")
            return tuple(values), index + 1
        if expect_number:
            if token_id < 0 or token_id >= tokenizer.vocab_size:
                raise ValueError("numeric token id out of range")
            values.append(_parse_number_token(tokenizer.tokens[token_id]))
            expect_number = False
        else:
            if token_id != comma_id:
                raise ValueError("expected comma")
            expect_number = True
        index += 1
    raise ValueError("unterminated numeric list")


def _parse_expr(
    ids: Sequence[int],
    start: int,
    *,
    tokenizer: FixedVocabTokenizer,
    operator_by_id: Mapping[int, str],
    left_id: int,
    right_id: int,
    comma_id: int,
) -> tuple[NestedPlan, int]:
    index = start
    if index >= len(ids) or int(ids[index]) not in operator_by_id:
        raise ValueError("missing operator")
    operator = operator_by_id[int(ids[index])]
    index += 1
    if index >= len(ids) or int(ids[index]) != left_id:
        raise ValueError("missing left bracket")
    index += 1

    if index < len(ids) and int(ids[index]) in operator_by_id:
        child, index = _parse_expr(
            ids,
            index,
            tokenizer=tokenizer,
            operator_by_id=operator_by_id,
            left_id=left_id,
            right_id=right_id,
            comma_id=comma_id,
        )
        if index >= len(ids) or int(ids[index]) != comma_id:
            raise ValueError("missing child/extras separator")
        extras, index = _parse_numeric_ids(
            ids,
            index + 1,
            tokenizer=tokenizer,
            comma_id=comma_id,
            right_id=right_id,
        )
        # Validate only operator arity/shape here; no target answer is used.
        seq.apply_operator(operator, (0, *extras))
        return NestedPlan(operator=operator, child=child, extras=extras), index

    leaf_values, index = _parse_numeric_ids(
        ids,
        index,
        tokenizer=tokenizer,
        comma_id=comma_id,
        right_id=right_id,
    )
    seq.apply_operator(operator, leaf_values)
    return NestedPlan(operator=operator, leaf_values=leaf_values), index


def parse_nested_plan(
    prompt_ids: Sequence[int], *, tokenizer: FixedVocabTokenizer
) -> NestedPlan:
    ids = [int(token_id) for token_id in prompt_ids]
    operator_by_id = {
        tokenizer.token_to_id[OPERATOR_TOKENS[operator]]: operator
        for operator in FUNCTIONAL_OPERATORS
    }
    left_id = tokenizer.token_to_id["<LBRACK>"]
    right_id = tokenizer.token_to_id["<RBRACK>"]
    comma_id = tokenizer.token_to_id["<COMMA>"]
    response_id = tokenizer.token_to_id["<RESPONSE>"]
    index = 1 if ids and ids[0] == tokenizer.bos_id else 0
    plan, index = _parse_expr(
        ids,
        index,
        tokenizer=tokenizer,
        operator_by_id=operator_by_id,
        left_id=left_id,
        right_id=right_id,
        comma_id=comma_id,
    )
    if index >= len(ids) or ids[index] != response_id:
        raise ValueError("missing response marker")
    index += 1
    if index != len(ids):
        raise ValueError("unexpected tokens after response marker")
    return plan


def plan_depth(plan: NestedPlan) -> int:
    return 1 if plan.child is None else 1 + plan_depth(plan.child)


def operators_leaf_to_root(plan: NestedPlan) -> tuple[str, ...]:
    if plan.child is None:
        return (plan.operator,)
    return (*operators_leaf_to_root(plan.child), plan.operator)


def transition_count(plan: NestedPlan) -> int:
    operators = operators_leaf_to_root(plan)
    return sum(left != right for left, right in zip(operators, operators[1:]))


def true_plan_value(plan: NestedPlan) -> int:
    if plan.child is None:
        return seq.apply_operator(plan.operator, plan.leaf_values)
    child_value = true_plan_value(plan.child)
    return seq.apply_operator(plan.operator, (child_value, *plan.extras))


def _operand_count(operator: str, *, leaf: bool) -> int:
    if leaf:
        return 2 if operator == "scalar.add" else 3
    return 1 if operator == "scalar.add" else 2


def make_depth3_plan(
    *,
    inner_operator: str,
    middle_operator: str,
    outer_operator: str,
    seed: int,
    sample_index: int,
) -> NestedPlan:
    rng = random.Random(
        seq._stable_seed(
            "recursive-depth3-composition-v1",
            seed,
            sample_index,
            inner_operator,
            middle_operator,
            outer_operator,
        )
    )
    leaf_values = tuple(
        rng.randint(-16, 16)
        for _ in range(_operand_count(inner_operator, leaf=True))
    )
    middle_extras = tuple(
        rng.randint(-16, 16)
        for _ in range(_operand_count(middle_operator, leaf=False))
    )
    outer_extras = tuple(
        rng.randint(-16, 16)
        for _ in range(_operand_count(outer_operator, leaf=False))
    )
    leaf = NestedPlan(operator=inner_operator, leaf_values=leaf_values)
    middle = NestedPlan(
        operator=middle_operator,
        child=leaf,
        extras=middle_extras,
    )
    return NestedPlan(
        operator=outer_operator,
        child=middle,
        extras=outer_extras,
    )


def execute_recursive(
    plan: NestedPlan,
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    controller: learned.PromptOperatorController,
    candidate,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> ExecutionState:
    true_value = true_plan_value(plan)
    if plan.child is None:
        generated, diagnostics, fast, slow, posterior = transition._generate_value(
            base=base,
            units=units,
            mixer=mixer,
            controller=controller,
            candidate=candidate,
            operator=plan.operator,
            values=plan.leaf_values,
            factory=factory,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        stage = StageExecution(
            operator=plan.operator,
            true_value=true_value,
            local_target=true_value,
            generated_value=generated,
            attempted=True,
            parsed=generated is not None,
            correct=generated == true_value,
            local_correct=generated == true_value,
            gate_value=None,
            controller_correct=bool(diagnostics["controller_correct"]),
        )
        return ExecutionState(generated, fast, slow, posterior, [stage])

    child = execute_recursive(
        plan.child,
        base=base,
        units=units,
        mixer=mixer,
        controller=controller,
        candidate=candidate,
        factory=factory,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    if (
        child.value is None
        or child.fast_state is None
        or child.slow_state is None
        or child.posterior is None
    ):
        skipped = StageExecution(
            operator=plan.operator,
            true_value=true_value,
            local_target=None,
            generated_value=None,
            attempted=False,
            parsed=False,
            correct=False,
            local_correct=False,
            gate_value=None,
            controller_correct=None,
        )
        return ExecutionState(None, None, None, None, [*child.stages, skipped])

    local_target = seq.apply_operator(plan.operator, (child.value, *plan.extras))
    generated, diagnostics, fast, slow, posterior = transition._generate_value(
        base=base,
        units=units,
        mixer=mixer,
        controller=controller,
        candidate=candidate,
        operator=plan.operator,
        values=(child.value, *plan.extras),
        factory=factory,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
        initial_fast_state=child.fast_state,
        initial_slow_state=child.slow_state,
        previous_posterior=child.posterior,
    )
    stage = StageExecution(
        operator=plan.operator,
        true_value=true_value,
        local_target=local_target,
        generated_value=generated,
        attempted=True,
        parsed=generated is not None,
        correct=generated == true_value,
        local_correct=generated == local_target,
        gate_value=float(diagnostics["gate_value"]),
        controller_correct=bool(diagnostics["controller_correct"]),
    )
    return ExecutionState(generated, fast, slow, posterior, [*child.stages, stage])


def _empty_counter() -> dict[str, int]:
    counter = {
        "cases": 0,
        "plan_parse": 0,
        "plan_exact": 0,
        "all_stages_correct": 0,
        "stage2_correct_given_stage1_correct": 0,
        "stage1_correct_cases": 0,
        "stage3_correct_given_stage2_correct": 0,
        "stage2_correct_cases": 0,
    }
    for stage in (1, 2, 3):
        counter[f"stage{stage}_attempted"] = 0
        counter[f"stage{stage}_parse"] = 0
        counter[f"stage{stage}_correct"] = 0
        counter[f"stage{stage}_local_correct"] = 0
    return counter


def _merge_counter(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def _finalize_counter(counter: Mapping[str, int]) -> dict[str, float | int | None]:
    cases = int(counter["cases"])
    result: dict[str, float | int | None] = {
        **{key: int(value) for key, value in counter.items()},
        "plan_parse_rate": counter["plan_parse"] / max(1, cases),
        "plan_exact_rate": counter["plan_exact"] / max(1, cases),
        "all_stages_accuracy": counter["all_stages_correct"] / max(1, cases),
        "stage2_accuracy_given_stage1_correct": (
            counter["stage2_correct_given_stage1_correct"]
            / counter["stage1_correct_cases"]
            if counter["stage1_correct_cases"]
            else None
        ),
        "stage3_accuracy_given_stage2_correct": (
            counter["stage3_correct_given_stage2_correct"]
            / counter["stage2_correct_cases"]
            if counter["stage2_correct_cases"]
            else None
        ),
    }
    for stage in (1, 2, 3):
        attempted = int(counter[f"stage{stage}_attempted"])
        result[f"stage{stage}_attempt_rate"] = attempted / max(1, cases)
        result[f"stage{stage}_parse_rate"] = counter[f"stage{stage}_parse"] / max(1, cases)
        result[f"stage{stage}_accuracy"] = counter[f"stage{stage}_correct"] / max(1, cases)
        result[f"stage{stage}_local_accuracy_given_attempt"] = (
            counter[f"stage{stage}_local_correct"] / attempted if attempted else None
        )
    result["end_to_end_accuracy"] = result["stage3_accuracy"]
    return result


def _update_counter(counter: dict[str, int], stages: Sequence[StageExecution]) -> None:
    if len(stages) != 3:
        raise ValueError(f"expected three stages, got {len(stages)}")
    for index, stage in enumerate(stages, start=1):
        counter[f"stage{index}_attempted"] += int(stage.attempted)
        counter[f"stage{index}_parse"] += int(stage.parsed)
        counter[f"stage{index}_correct"] += int(stage.correct)
        counter[f"stage{index}_local_correct"] += int(stage.local_correct)
    counter["stage1_correct_cases"] += int(stages[0].correct)
    counter["stage2_correct_cases"] += int(stages[1].correct)
    if stages[0].correct:
        counter["stage2_correct_given_stage1_correct"] += int(stages[1].correct)
    if stages[1].correct:
        counter["stage3_correct_given_stage2_correct"] += int(stages[2].correct)
    counter["all_stages_correct"] += int(all(stage.correct for stage in stages))


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    mixer: torch.nn.Module,
    controller: learned.PromptOperatorController,
    examples_per_triple: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    candidate = confirmatory.candidate_grid()[1]
    aggregate = _empty_counter()
    triple_rows: dict[str, dict[str, float | int | None]] = {}
    transition_rows = {count: _empty_counter() for count in (0, 1, 2)}
    gate_sum = 0.0
    same_gate_sum = 0.0
    switch_gate_sum = 0.0
    gate_calls = 0
    same_gate_calls = 0
    switch_gate_calls = 0

    for inner in FUNCTIONAL_OPERATORS:
        for middle in FUNCTIONAL_OPERATORS:
            for outer in FUNCTIONAL_OPERATORS:
                triple_id = f"{inner}->{middle}->{outer}"
                counter = _empty_counter()
                for sample_index in range(examples_per_triple):
                    expected = make_depth3_plan(
                        inner_operator=inner,
                        middle_operator=middle,
                        outer_operator=outer,
                        seed=data_seed,
                        sample_index=sample_index,
                    )
                    prompt = prompt_ids_for_plan(expected, tokenizer=tokenizer)
                    counter["cases"] += 1
                    aggregate["cases"] += 1
                    transitions = transition_count(expected)
                    transition_rows[transitions]["cases"] += 1
                    try:
                        parsed = parse_nested_plan(prompt, tokenizer=tokenizer)
                    except (ValueError, KeyError, IndexError):
                        continue
                    counter["plan_parse"] += 1
                    aggregate["plan_parse"] += 1
                    transition_rows[transitions]["plan_parse"] += 1
                    exact = parsed == expected
                    counter["plan_exact"] += int(exact)
                    aggregate["plan_exact"] += int(exact)
                    transition_rows[transitions]["plan_exact"] += int(exact)

                    execution = execute_recursive(
                        parsed,
                        base=base,
                        units=units,
                        mixer=mixer,
                        controller=controller,
                        candidate=candidate,
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )
                    _update_counter(counter, execution.stages)
                    _update_counter(aggregate, execution.stages)
                    _update_counter(transition_rows[transitions], execution.stages)

                    operators = operators_leaf_to_root(parsed)
                    for stage_index in (1, 2):
                        stage = execution.stages[stage_index]
                        if stage.gate_value is None:
                            continue
                        gate = float(stage.gate_value)
                        gate_sum += gate
                        gate_calls += 1
                        if operators[stage_index - 1] == operators[stage_index]:
                            same_gate_sum += gate
                            same_gate_calls += 1
                        else:
                            switch_gate_sum += gate
                            switch_gate_calls += 1
                triple_rows[triple_id] = _finalize_counter(counter)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "aggregate": _finalize_counter(aggregate),
        "triples": triple_rows,
        "by_transition_count": {
            str(count): _finalize_counter(counter)
            for count, counter in transition_rows.items()
        },
        "mean_gate": gate_sum / max(1, gate_calls),
        "mean_same_operator_gate": same_gate_sum / max(1, same_gate_calls),
        "mean_switch_operator_gate": switch_gate_sum / max(1, switch_gate_calls),
    }


def run_experiment(
    *,
    root: Path,
    examples_per_triple: int,
    data_seed: int,
    max_new_tokens: int,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    calibration_seed: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    hidden_size: int,
    sketch_size: int,
    controller_seed: int,
    controller_examples_per_operator: int,
    controller_validation_examples_per_operator: int,
    controller_steps: int,
    controller_learning_rate: float,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    mixer, fit_report = seq.fit_ensemble(
        cohorts,
        root=root,
        calibration_examples_per_operator=calibration_examples_per_operator,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions_per_cohort=max_positions_per_cohort,
        calibration_seed=calibration_seed,
        fit_steps=fit_steps,
        fit_batch_positions=fit_batch_positions,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        sketch_size=sketch_size,
        device=device,
    )

    controller_run = load_run_config(cohorts[0].config_path)
    controller_tokenizer = FixedVocabTokenizer.from_config(
        root / controller_run.tokenizer_config
    )
    controller_factory = SyntheticTraceFactory(controller_tokenizer, controller_run.data)
    controller, controller_fit = learned.fit_prompt_controller(
        factory=controller_factory,
        tokenizer=controller_tokenizer,
        device=device,
        seed=controller_seed,
        examples_per_operator=controller_examples_per_operator,
        validation_examples_per_operator=controller_validation_examples_per_operator,
        steps=controller_steps,
        learning_rate=controller_learning_rate,
        feature_mode="full",
    )

    cohort_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            mixer=mixer,
            controller=controller,
            examples_per_triple=examples_per_triple,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]

    aggregate = _empty_counter()
    triple_aggregate = {
        f"{inner}->{middle}->{outer}": _empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for middle in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    transition_aggregate = {count: _empty_counter() for count in (0, 1, 2)}
    mean_gates: list[float] = []
    same_gates: list[float] = []
    switch_gates: list[float] = []
    for report in cohort_reports:
        raw_aggregate = {
            key: int(report["aggregate"][key]) for key in _empty_counter()
        }
        _merge_counter(aggregate, raw_aggregate)
        for triple_id, row in report["triples"].items():
            raw = {key: int(row[key]) for key in _empty_counter()}
            _merge_counter(triple_aggregate[triple_id], raw)
        for count in (0, 1, 2):
            row = report["by_transition_count"][str(count)]
            raw = {key: int(row[key]) for key in _empty_counter()}
            _merge_counter(transition_aggregate[count], raw)
        mean_gates.append(float(report["mean_gate"]))
        same_gates.append(float(report["mean_same_operator_gate"]))
        switch_gates.append(float(report["mean_switch_operator_gate"]))

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_recursive_depth3_composition",
        "claim_boundary": (
            "each case is supplied as one depth-3 nested expression; a deterministic recursive parser/executor "
            "derives the three-stage plan and scalar handoffs; each specialist stage uses the isolated-prompt "
            "learned operator controller and posterior-transition gate; this is not a learned parser, not single-"
            "pass language-model execution, and does not test NEG, final IID, OOD, branches, or loops"
        ),
        "depth": 3,
        "operator_triple_count": 64,
        "examples_per_triple_per_cohort": examples_per_triple,
        "model_cohort_count": len(cohort_reports),
        "data_seed": data_seed,
        "controller_operator_strength": float(
            os.environ.get(learned.ENV_CONTROLLER_STRENGTH, "1.0")
        ),
        "posterior_gate_mode": os.environ.get(
            transition.ENV_GATE_MODE, "dynamic"
        ),
        "posterior_gate_scale": float(
            os.environ.get(transition.ENV_GATE_SCALE, "1.0")
        ),
        "fixed_candidate": confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": fit_report,
        "controller_fit": controller_fit,
        "cohorts": cohort_reports,
        "aggregate": _finalize_counter(aggregate),
        "triples": {
            triple_id: _finalize_counter(counter)
            for triple_id, counter in triple_aggregate.items()
        },
        "by_transition_count": {
            str(count): _finalize_counter(counter)
            for count, counter in transition_aggregate.items()
        },
        "mean_gate": sum(mean_gates) / max(1, len(mean_gates)),
        "mean_same_operator_gate": sum(same_gates) / max(1, len(same_gates)),
        "mean_switch_operator_gate": sum(switch_gates) / max(1, len(switch_gates)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples-per-triple", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--calibration-examples-per-operator", type=int, default=8)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1536)
    parser.add_argument("--calibration-seed", type=int, default=seq.DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sketch-size", type=int, default=8)
    parser.add_argument("--controller-seed", type=int, default=learned.DEFAULT_CONTROLLER_SEED)
    parser.add_argument("--controller-examples-per-operator", type=int, default=64)
    parser.add_argument("--controller-validation-examples-per-operator", type=int, default=32)
    parser.add_argument("--controller-steps", type=int, default=300)
    parser.add_argument("--controller-learning-rate", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_experiment(
        root=args.root.resolve(),
        examples_per_triple=args.examples_per_triple,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        max_prefixes_per_example=args.max_prefixes_per_example,
        max_positions_per_cohort=args.max_positions_per_cohort,
        calibration_seed=args.calibration_seed,
        fit_steps=args.fit_steps,
        fit_batch_positions=args.fit_batch_positions,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        sketch_size=args.sketch_size,
        controller_seed=args.controller_seed,
        controller_examples_per_operator=args.controller_examples_per_operator,
        controller_validation_examples_per_operator=args.controller_validation_examples_per_operator,
        controller_steps=args.controller_steps,
        controller_learning_rate=args.controller_learning_rate,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(
        json.dumps(
            {
                "aggregate": report["aggregate"],
                "by_transition_count": report["by_transition_count"],
                "mean_same_operator_gate": report["mean_same_operator_gate"],
                "mean_switch_operator_gate": report["mean_switch_operator_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
