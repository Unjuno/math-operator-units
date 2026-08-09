from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
TOKEN_TO_OPERATOR = {
    token: operator
    for operator, token in OPERATOR_TOKENS.items()
    if operator in FUNCTIONAL_OPERATORS
}


@dataclass(frozen=True)
class TwoStagePlan:
    outer_operator: str
    inner_operator: str
    inner_values: tuple[int, ...]
    outer_extras: tuple[int, ...]


def _number_token(value: int) -> str:
    return f"<N_{int(value)}>"


def _parse_number_token(token: str) -> int:
    if not token.startswith("<N_") or not token.endswith(">"):
        raise ValueError(f"expected atomic numeric token, got {token}")
    return int(token[3:-1])


def _comma_list(values: Sequence[int]) -> list[str]:
    output: list[str] = []
    for index, value in enumerate(values):
        if index:
            output.append("<COMMA>")
        output.append(_number_token(int(value)))
    return output


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
    # Uniform function-call surface, independent of the specialist's native
    # single-stage state representation:
    # OUTER( INNER(a,b,...), c,d,... ) <RESPONSE>
    return [
        OPERATOR_TOKENS[outer_operator],
        "<LPAREN>",
        OPERATOR_TOKENS[inner_operator],
        "<LPAREN>",
        *_comma_list(inner_values),
        "<RPAREN>",
        "<COMMA>",
        *_comma_list(outer_extras),
        "<RPAREN>",
        "<RESPONSE>",
    ]


def nested_expression_ids(
    *,
    tokenizer: FixedVocabTokenizer,
    outer_operator: str,
    inner_operator: str,
    inner_values: Sequence[int],
    outer_extras: Sequence[int],
) -> list[int]:
    return tokenizer.encode_tokens(
        nested_expression_tokens(
            outer_operator=outer_operator,
            inner_operator=inner_operator,
            inner_values=inner_values,
            outer_extras=outer_extras,
        ),
        add_bos=True,
        add_eos=False,
    )


def _parse_numeric_list(tokens: Sequence[str], start: int) -> tuple[tuple[int, ...], int]:
    values: list[int] = []
    index = start
    expect_number = True
    while index < len(tokens):
        token = tokens[index]
        if token == "<RPAREN>":
            if expect_number and values:
                raise ValueError("trailing comma before closing parenthesis")
            if not values:
                raise ValueError("empty operand list")
            return tuple(values), index + 1
        if expect_number:
            values.append(_parse_number_token(token))
            expect_number = False
        else:
            if token != "<COMMA>":
                raise ValueError(f"expected comma, got {token}")
            expect_number = True
        index += 1
    raise ValueError("unterminated operand list")


def parse_two_stage_plan(
    prompt_ids: Sequence[int], *, tokenizer: FixedVocabTokenizer
) -> TwoStagePlan:
    tokens = [tokenizer.tokens[int(token_id)] for token_id in prompt_ids]
    index = 0
    if tokens and tokens[0] == tokenizer.tokens[tokenizer.bos_id]:
        index += 1
    if index >= len(tokens) or tokens[index] not in TOKEN_TO_OPERATOR:
        raise ValueError("missing outer operator")
    outer_operator = TOKEN_TO_OPERATOR[tokens[index]]
    index += 1
    if index >= len(tokens) or tokens[index] != "<LPAREN>":
        raise ValueError("missing outer left parenthesis")
    index += 1
    if index >= len(tokens) or tokens[index] not in TOKEN_TO_OPERATOR:
        raise ValueError("missing inner operator")
    inner_operator = TOKEN_TO_OPERATOR[tokens[index]]
    index += 1
    if index >= len(tokens) or tokens[index] != "<LPAREN>":
        raise ValueError("missing inner left parenthesis")
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

    # Structural validation of the plan.  No target result is used here.
    seq.apply_operator(inner_operator, inner_values)
    outer_arity_values = (0, *outer_extras)
    seq.apply_operator(outer_operator, outer_arity_values)
    return TwoStagePlan(
        outer_operator=outer_operator,
        inner_operator=inner_operator,
        inner_values=inner_values,
        outer_extras=outer_extras,
    )


def execute_plan(
    plan: TwoStagePlan,
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
) -> tuple[int | None, int | None, dict[str, Any]]:
    generated_inner, inner_diag, inner_fast, inner_slow, inner_posterior = transition._generate_value(
        base=base,
        units=units,
        mixer=mixer,
        controller=controller,
        candidate=candidate,
        operator=plan.inner_operator,
        values=plan.inner_values,
        factory=factory,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    generated_outer: int | None = None
    outer_diag: dict[str, Any] | None = None
    if generated_inner is not None:
        generated_outer, outer_diag, _, _, _ = transition._generate_value(
            base=base,
            units=units,
            mixer=mixer,
            controller=controller,
            candidate=candidate,
            operator=plan.outer_operator,
            values=(generated_inner, *plan.outer_extras),
            factory=factory,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            device=device,
            initial_fast_state=inner_fast,
            initial_slow_state=inner_slow,
            previous_posterior=inner_posterior,
        )
    return generated_inner, generated_outer, {
        "inner": inner_diag,
        "outer": outer_diag,
    }


def _empty_counter() -> dict[str, int]:
    return {
        "cases": 0,
        "plan_parse": 0,
        "plan_exact": 0,
        "inner_parse": 0,
        "inner_correct": 0,
        "outer_parse": 0,
        "end_to_end_correct": 0,
        "inner_correct_cases": 0,
        "outer_correct_given_inner_correct": 0,
    }


def _finalize(counter: Mapping[str, int]) -> dict[str, float | int | None]:
    cases = int(counter["cases"])
    inner_correct_cases = int(counter["inner_correct_cases"])
    return {
        **{key: int(value) for key, value in counter.items()},
        "plan_parse_rate": counter["plan_parse"] / max(1, cases),
        "plan_exact_rate": counter["plan_exact"] / max(1, cases),
        "inner_parse_rate": counter["inner_parse"] / max(1, cases),
        "inner_accuracy": counter["inner_correct"] / max(1, cases),
        "outer_parse_rate": counter["outer_parse"] / max(1, cases),
        "end_to_end_accuracy": counter["end_to_end_correct"] / max(1, cases),
        "outer_accuracy_given_inner_correct": (
            counter["outer_correct_given_inner_correct"] / inner_correct_cases
            if inner_correct_cases
            else None
        ),
    }


def _merge(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    mixer: torch.nn.Module,
    controller: learned.PromptOperatorController,
    examples_per_pair: int,
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
    pair_rows: dict[str, dict[str, float | int | None]] = {}
    gate_sum = 0.0
    same_gate_sum = 0.0
    switch_gate_sum = 0.0
    outer_diag_calls = 0
    same_gate_calls = 0
    switch_gate_calls = 0

    for inner_operator in FUNCTIONAL_OPERATORS:
        for outer_operator in FUNCTIONAL_OPERATORS:
            pair_id = f"{inner_operator}->{outer_operator}"
            same_operator = inner_operator == outer_operator
            counter = _empty_counter()
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = seq.composition_operands(
                    inner_operator=inner_operator,
                    outer_operator=outer_operator,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = seq.apply_operator(inner_operator, inner_values)
                true_final = seq.apply_operator(
                    outer_operator, (true_inner, *outer_extras)
                )
                prompt = nested_expression_ids(
                    tokenizer=tokenizer,
                    outer_operator=outer_operator,
                    inner_operator=inner_operator,
                    inner_values=inner_values,
                    outer_extras=outer_extras,
                )

                counter["cases"] += 1
                try:
                    plan = parse_two_stage_plan(prompt, tokenizer=tokenizer)
                except (ValueError, KeyError, IndexError):
                    continue
                counter["plan_parse"] += 1
                expected_plan = TwoStagePlan(
                    outer_operator=outer_operator,
                    inner_operator=inner_operator,
                    inner_values=tuple(inner_values),
                    outer_extras=tuple(outer_extras),
                )
                counter["plan_exact"] += int(plan == expected_plan)

                generated_inner, generated_outer, diagnostics = execute_plan(
                    plan,
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
                inner_correct = generated_inner == true_inner
                counter["inner_parse"] += int(generated_inner is not None)
                counter["inner_correct"] += int(inner_correct)
                counter["outer_parse"] += int(generated_outer is not None)
                counter["end_to_end_correct"] += int(generated_outer == true_final)
                if inner_correct:
                    counter["inner_correct_cases"] += 1
                    counter["outer_correct_given_inner_correct"] += int(
                        generated_outer == true_final
                    )

                outer_diag = diagnostics["outer"]
                if outer_diag is not None:
                    gate_value = float(outer_diag["gate_value"])
                    gate_sum += gate_value
                    outer_diag_calls += 1
                    if same_operator:
                        same_gate_sum += gate_value
                        same_gate_calls += 1
                    else:
                        switch_gate_sum += gate_value
                        switch_gate_calls += 1

            _merge(aggregate, counter)
            pair_rows[pair_id] = _finalize(counter)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "candidate": candidate.__dict__,
        "aggregate": _finalize(aggregate),
        "pairs": pair_rows,
        "mean_gate": gate_sum / max(1, outer_diag_calls),
        "mean_same_operator_gate": same_gate_sum / max(1, same_gate_calls),
        "mean_switch_operator_gate": switch_gate_sum / max(1, switch_gate_calls),
    }


def run_experiment(
    *,
    root: Path,
    examples_per_pair: int,
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
    controller_factory = SyntheticTraceFactory(
        controller_tokenizer, controller_run.data
    )
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
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]

    aggregate = _empty_counter()
    pair_aggregate = {
        f"{inner}->{outer}": _empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    mean_gates: list[float] = []
    same_gates: list[float] = []
    switch_gates: list[float] = []
    for report in cohort_reports:
        raw_aggregate = {
            key: int(report["aggregate"][key]) for key in _empty_counter()
        }
        _merge(aggregate, raw_aggregate)
        mean_gates.append(float(report["mean_gate"]))
        same_gates.append(float(report["mean_same_operator_gate"]))
        switch_gates.append(float(report["mean_switch_operator_gate"]))
        for pair_id, row in report["pairs"].items():
            raw_pair = {key: int(row[key]) for key in _empty_counter()}
            _merge(pair_aggregate[pair_id], raw_pair)

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_nested_expression_structural_planner",
        "claim_boundary": (
            "each evaluation case is supplied to the executor as one depth-2 nested-expression token prompt; "
            "a deterministic structural parser derives inner/outer operators, operands, execution order, and "
            "scalar handoff from that prompt; learned operator control and posterior-transition reset are then "
            "used for the two specialist generation stages; this is not a learned parser or single-pass language-"
            "model execution, and does not test NEG, final IID, or OOD generalization"
        ),
        "nested_surface": "OUTER(INNER(values...), extras...) <RESPONSE>",
        "operators": list(FUNCTIONAL_OPERATORS),
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "controller_operator_strength": float(
            __import__("os").environ.get(learned.ENV_CONTROLLER_STRENGTH, "1.0")
        ),
        "posterior_gate_mode": __import__("os").environ.get(
            transition.ENV_GATE_MODE, "dynamic"
        ),
        "posterior_gate_scale": float(
            __import__("os").environ.get(transition.ENV_GATE_SCALE, "1.0")
        ),
        "fixed_candidate": confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": fit_report,
        "controller_fit": controller_fit,
        "cohorts": cohort_reports,
        "aggregate": _finalize(aggregate),
        "pairs": {
            pair_id: _finalize(counter)
            for pair_id, counter in pair_aggregate.items()
        },
        "mean_gate": sum(mean_gates) / max(1, len(mean_gates)),
        "mean_same_operator_gate": sum(same_gates) / max(1, len(same_gates)),
        "mean_switch_operator_gate": sum(switch_gates) / max(1, len(switch_gates)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples-per-pair", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=seq.DEFAULT_DATA_SEED)
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
        examples_per_pair=args.examples_per_pair,
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
                "mean_gate": report["mean_gate"],
                "mean_same_operator_gate": report["mean_same_operator_gate"],
                "mean_switch_operator_gate": report["mean_switch_operator_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
