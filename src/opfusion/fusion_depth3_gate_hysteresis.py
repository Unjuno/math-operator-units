from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_learned_operator_controller as learned
from opfusion import fusion_oracle_sequential_composition as seq
from opfusion import fusion_posterior_transition_boundary_gate as transition
from opfusion import fusion_recursive_depth3_composition as depth3
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.tokenizer import FixedVocabTokenizer


ENV_GATE_HYSTERESIS = "OPFUSION_GATE_HYSTERESIS"


def gate_with_hysteresis(
    raw_gate: float,
    previous_gate: float,
    *,
    rho: float,
) -> float:
    if not 0.0 <= raw_gate <= 1.0:
        raise ValueError("raw_gate must be in [0, 1]")
    if not 0.0 <= previous_gate <= 1.0:
        raise ValueError("previous_gate must be in [0, 1]")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return max(float(raw_gate), float(rho) * float(previous_gate))


def _rho() -> float:
    value = float(os.environ.get(ENV_GATE_HYSTERESIS, "0.0"))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{ENV_GATE_HYSTERESIS} must be in [0, 1]")
    return value


def _effective_previous_for_gate(
    current_posterior: torch.Tensor,
    *,
    target_gate: float,
) -> torch.Tensor:
    """Construct a vector that makes PR31's dot-product gate equal target_gate.

    The vector is only a control input to the gate calculation; current operator
    weights still use the real controller posterior computed inside the stage.
    """
    if not 0.0 <= target_gate <= 1.0:
        raise ValueError("target_gate must be in [0, 1]")
    current = current_posterior.float()
    norm_sq = torch.sum(current * current).clamp_min(1e-12)
    desired_dot = current.new_tensor(1.0 - float(target_gate))
    effective = (desired_dot / norm_sq) * current
    return effective.to(dtype=current_posterior.dtype)


def _generate_value_hysteresis(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    controller: learned.PromptOperatorController,
    candidate,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
    initial_fast_state: torch.Tensor | None = None,
    initial_slow_state: torch.Tensor | None = None,
    previous_posterior: torch.Tensor | None = None,
    previous_gate: float = 0.0,
) -> tuple[
    int | None,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    prompt = seq.prompt_ids_for_values(
        factory=factory,
        tokenizer=tokenizer,
        operator=operator,
        values=values,
    )
    effective_previous = previous_posterior
    raw_gate = 0.0
    target_gate = 0.0

    if initial_fast_state is not None or initial_slow_state is not None:
        if (
            initial_fast_state is None
            or initial_slow_state is None
            or previous_posterior is None
        ):
            raise ValueError("boundary state/posterior must be supplied together")
        feature_mode = os.environ.get(
            learned.ENV_CONTROLLER_FEATURE_MODE, "full"
        ).strip().lower()
        current_posterior = learned._controller_posterior(
            controller,
            prompt=prompt,
            tokenizer=tokenizer,
            feature_mode=feature_mode,
            device=device,
        )
        raw_gate_tensor = transition.posterior_transition_gate(
            previous_posterior.to(device=current_posterior.device),
            current_posterior,
            mode=transition._gate_mode(),
            scale=transition._gate_scale(),
        )
        raw_gate = float(raw_gate_tensor.detach().cpu())
        target_gate = gate_with_hysteresis(
            raw_gate,
            previous_gate,
            rho=_rho(),
        )
        if target_gate > raw_gate + 1e-8:
            effective_previous = _effective_previous_for_gate(
                current_posterior,
                target_gate=target_gate,
            )

    generated, diagnostics, fast, slow, posterior = transition._generate_stage(
        base=base,
        units=units,
        mixer=mixer,
        controller=controller,
        candidate=candidate,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        device=device,
        initial_fast_state=initial_fast_state,
        initial_slow_state=initial_slow_state,
        previous_posterior=effective_previous,
    )
    source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
    diagnostics["mean_matching_source_weight"] = diagnostics["mean_source_weights"][
        source_index
    ]
    diagnostics["controller_correct"] = float(
        diagnostics["controller_predicted_operator"] == operator
    )
    diagnostics["raw_gate_value"] = raw_gate
    diagnostics["previous_boundary_gate"] = float(previous_gate)
    diagnostics["hysteresis_rho"] = _rho()
    diagnostics["target_gate_value"] = target_gate
    if initial_fast_state is not None:
        diagnostics["gate_value"] = target_gate

    return (
        seq.parse_final_numeric_token(generated, tokenizer),
        diagnostics,
        fast,
        slow,
        posterior,
    )


def execute_recursive_hysteresis(
    plan: depth3.NestedPlan,
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
) -> depth3.ExecutionState:
    true_value = depth3.true_plan_value(plan)
    if plan.child is None:
        generated, diagnostics, fast, slow, posterior = _generate_value_hysteresis(
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
        stage = depth3.StageExecution(
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
        state = depth3.ExecutionState(generated, fast, slow, posterior, [stage])
        setattr(state, "last_gate", 0.0)
        return state

    child = execute_recursive_hysteresis(
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
        skipped = depth3.StageExecution(
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
        state = depth3.ExecutionState(
            None, None, None, None, [*child.stages, skipped]
        )
        setattr(state, "last_gate", float(getattr(child, "last_gate", 0.0)))
        return state

    previous_gate = float(getattr(child, "last_gate", 0.0))
    local_target = seq.apply_operator(plan.operator, (child.value, *plan.extras))
    generated, diagnostics, fast, slow, posterior = _generate_value_hysteresis(
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
        previous_gate=previous_gate,
    )
    target_gate = float(diagnostics["target_gate_value"])
    stage = depth3.StageExecution(
        operator=plan.operator,
        true_value=true_value,
        local_target=local_target,
        generated_value=generated,
        attempted=True,
        parsed=generated is not None,
        correct=generated == true_value,
        local_correct=generated == local_target,
        gate_value=target_gate,
        controller_correct=bool(diagnostics["controller_correct"]),
    )
    state = depth3.ExecutionState(
        generated,
        fast,
        slow,
        posterior,
        [*child.stages, stage],
    )
    setattr(state, "last_gate", target_gate)
    return state


def main() -> int:
    # Reuse PR33's parser, dataset, calibration, aggregation, and reporting;
    # replace only the recursive stage execution law.
    depth3.execute_recursive = execute_recursive_hysteresis
    args = depth3._parser().parse_args()
    report = depth3.run_experiment(
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
        controller_validation_examples_per_operator=(
            args.controller_validation_examples_per_operator
        ),
        controller_steps=args.controller_steps,
        controller_learning_rate=args.controller_learning_rate,
        device_name=args.device,
    )
    report["evaluation_role"] = "validation_only_depth3_gate_hysteresis"
    report["gate_hysteresis_rho"] = _rho()
    report["claim_boundary"] = (
        "depth-3 recursive execution with deterministic parsing and scalar handoff; "
        "the posterior-transition reset gate additionally retains a decayed previous-boundary "
        "activation via g_t=max(delta_t,rho*g_{t-1}); no gold operator identity is used by "
        "the gate; this does not test learned parsing, arbitrary-depth single-pass execution, "
        "NEG, final IID, OOD, branches, or loops"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(
        json.dumps(
            {
                "rho": report["gate_hysteresis_rho"],
                "gate_mode": report["posterior_gate_mode"],
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
