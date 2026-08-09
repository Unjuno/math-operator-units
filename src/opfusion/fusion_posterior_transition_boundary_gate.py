from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_controller_state_boundary_carry as boundary
from opfusion import fusion_learned_operator_controller as learned
from opfusion import fusion_oracle_sequential_composition as seq
from opfusion import fusion_stateful_dual_timescale as dual_timescale
from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


ENV_GATE_MODE = "OPFUSION_POSTERIOR_GATE_MODE"
ENV_GATE_SCALE = "OPFUSION_POSTERIOR_GATE_SCALE"
FUNCTIONAL_OPERATORS = seq.FUNCTIONAL_OPERATORS


def posterior_transition_gate(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    mode: str = "dynamic",
    scale: float = 1.0,
) -> torch.Tensor:
    """Return a soft reset gate from controller-posterior transition only.

    `dynamic` uses 1 - dot(previous, current), scaled then clipped to [0, 1].
    The two other modes are validation controls and do not inspect labels.
    """
    if previous.ndim != 1 or current.ndim != 1 or previous.shape != current.shape:
        raise ValueError("posterior vectors must have matching one-dimensional shape")
    normalized_mode = mode.strip().lower()
    if normalized_mode == "always_reset":
        return previous.new_tensor(1.0)
    if normalized_mode == "always_carry":
        return previous.new_tensor(0.0)
    if normalized_mode != "dynamic":
        raise ValueError(f"unsupported gate mode: {mode}")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    similarity = torch.sum(previous.float() * current.float())
    return (float(scale) * (1.0 - similarity)).clamp(0.0, 1.0).to(previous.dtype)


def _gate_mode() -> str:
    return os.environ.get(ENV_GATE_MODE, "dynamic").strip().lower()


def _gate_scale() -> float:
    return float(os.environ.get(ENV_GATE_SCALE, "1.0"))


def _generate_stage(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    controller: learned.PromptOperatorController,
    candidate: dual_timescale.DualTimescaleCandidate,
    prompt: Sequence[int],
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
    initial_fast_state: torch.Tensor | None = None,
    initial_slow_state: torch.Tensor | None = None,
    previous_posterior: torch.Tensor | None = None,
) -> tuple[list[int], dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    if (initial_fast_state is None) != (initial_slow_state is None):
        raise ValueError("initial fast and slow states must be supplied together")
    if initial_fast_state is not None and previous_posterior is None:
        raise ValueError("previous posterior is required when carrying boundary state")

    strength = float(os.environ.get(learned.ENV_CONTROLLER_STRENGTH, "1.0"))
    feature_mode = os.environ.get(
        learned.ENV_CONTROLLER_FEATURE_MODE, "full"
    ).strip().lower()
    posterior = learned._controller_posterior(
        controller,
        prompt=prompt,
        tokenizer=tokenizer,
        feature_mode=feature_mode,
        device=device,
    )
    predicted_class = int(posterior.argmax().item())

    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    fast_state: torch.Tensor | None = None
    slow_state: torch.Tensor | None = None
    source_weight_sum: torch.Tensor | None = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    timescale_gap_sum = 0.0
    positions = 0
    gate_value = 0.0
    boundary_shift = 0.0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = implementation._source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()

            if fast_state is None or slow_state is None:
                if initial_fast_state is None or initial_slow_state is None:
                    fast_state = instant_state
                    slow_state = instant_state
                else:
                    assert previous_posterior is not None
                    gate = posterior_transition_gate(
                        previous_posterior.to(device=posterior.device),
                        posterior,
                        mode=_gate_mode(),
                        scale=_gate_scale(),
                    )
                    gate_value = float(gate.detach().cpu())
                    carried_fast = initial_fast_state.to(
                        device=instant_state.device, dtype=instant_state.dtype
                    )
                    carried_slow = initial_slow_state.to(
                        device=instant_state.device, dtype=instant_state.dtype
                    )
                    fast_state = boundary.blend_boundary_state(
                        carried_fast,
                        instant_state,
                        reset_fraction=gate_value,
                    )
                    slow_state = boundary.blend_boundary_state(
                        carried_slow,
                        instant_state,
                        reset_fraction=gate_value,
                    )
                    boundary_shift = float(
                        (
                            (1.0 - float(candidate.slow_mix))
                            * (fast_state - carried_fast).abs().mean()
                            + float(candidate.slow_mix)
                            * (slow_state - carried_slow).abs().mean()
                        )
                        .detach()
                        .cpu()
                    )
                state_change = instant_state.new_tensor(0.0)
            else:
                previous_combined = (
                    (1.0 - float(candidate.slow_mix)) * fast_state
                    + float(candidate.slow_mix) * slow_state
                )
                fast_state = (
                    float(candidate.fast_memory) * fast_state
                    + (1.0 - float(candidate.fast_memory)) * instant_state
                )
                slow_state = (
                    float(candidate.slow_memory) * slow_state
                    + (1.0 - float(candidate.slow_memory)) * instant_state
                )
                current_combined = (
                    (1.0 - float(candidate.slow_mix)) * fast_state
                    + float(candidate.slow_mix) * slow_state
                )
                state_change = (current_combined - previous_combined).abs().mean()

            combined = (
                (1.0 - float(candidate.slow_mix)) * fast_state
                + float(candidate.slow_mix) * slow_state
            ) / float(candidate.temperature)
            prior = learned.controller_operator_prior(
                posterior,
                source_count=int(combined.shape[-1]),
                strength=strength,
                device=combined.device,
                dtype=combined.dtype,
            )
            weights = torch.softmax(combined + prior, dim=-1)
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0.0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                token_support = token_support - token_support.mean()
                fast_state = fast_state + float(candidate.feedback) * token_support
                slow_state = slow_state + 0.25 * float(candidate.feedback) * token_support

            entropy_sum += float(
                (-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu()
            )
            max_weight_sum += float(weights.max().detach().cpu())
            state_change_sum += float(state_change.detach().cpu())
            timescale_gap_sum += float((fast_state - slow_state).abs().mean().detach().cpu())
            source_weight_sum = (
                weights.detach().clone()
                if source_weight_sum is None
                else source_weight_sum + weights.detach()
            )
            positions += 1
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == tokenizer.eos_id:
                break

    if fast_state is None or slow_state is None:
        raise RuntimeError("generation produced no fusion state")
    mean_source_weights = (
        source_weight_sum / max(1, positions)
        if source_weight_sum is not None
        else torch.zeros(len(EXPERIMENT_OPERATORS) + 1, device=device)
    )
    diagnostics = {
        "controller_posterior": [float(value) for value in posterior.detach().cpu()],
        "controller_predicted_operator": FUNCTIONAL_OPERATORS[predicted_class],
        "controller_confidence": float(posterior.max().detach().cpu()),
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
        "mean_timescale_gap": timescale_gap_sum / max(1, positions),
        "mean_source_weights": [float(value) for value in mean_source_weights.detach().cpu()],
        "gate_value": gate_value,
        "boundary_state_shift": boundary_shift,
        "positions": positions,
    }
    return (
        output,
        diagnostics,
        fast_state.detach().clone(),
        slow_state.detach().clone(),
        posterior.detach().clone(),
    )


def _generate_value(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    controller: learned.PromptOperatorController,
    candidate: dual_timescale.DualTimescaleCandidate,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
    initial_fast_state: torch.Tensor | None = None,
    initial_slow_state: torch.Tensor | None = None,
    previous_posterior: torch.Tensor | None = None,
) -> tuple[int | None, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = seq.prompt_ids_for_values(
        factory=factory,
        tokenizer=tokenizer,
        operator=operator,
        values=values,
    )
    generated, diagnostics, fast_state, slow_state, posterior = _generate_stage(
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
        previous_posterior=previous_posterior,
    )
    source_index = 1 + EXPERIMENT_OPERATORS.index(operator)
    diagnostics["mean_matching_source_weight"] = diagnostics["mean_source_weights"][source_index]
    diagnostics["controller_correct"] = float(
        diagnostics["controller_predicted_operator"] == operator
    )
    return (
        seq.parse_final_numeric_token(generated, tokenizer),
        diagnostics,
        fast_state,
        slow_state,
        posterior,
    )


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
    aggregate = seq._empty_counter()
    pair_rows: dict[str, dict[str, float | int | None]] = {}
    matching_weight_sum = 0.0
    controller_correct_sum = 0.0
    controller_confidence_sum = 0.0
    gate_sum = 0.0
    same_gate_sum = 0.0
    switch_gate_sum = 0.0
    same_gate_calls = 0
    switch_gate_calls = 0
    diagnostic_calls = 0
    outer_calls = 0

    for inner_operator in FUNCTIONAL_OPERATORS:
        for outer_operator in FUNCTIONAL_OPERATORS:
            pair_id = f"{inner_operator}->{outer_operator}"
            counter = seq._empty_counter()
            same_operator = inner_operator == outer_operator
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = seq.composition_operands(
                    inner_operator=inner_operator,
                    outer_operator=outer_operator,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = seq.apply_operator(inner_operator, inner_values)
                true_outer_values = (true_inner, *outer_extras)
                true_final = seq.apply_operator(outer_operator, true_outer_values)

                generated_inner, inner_diag, inner_fast, inner_slow, inner_posterior = _generate_value(
                    base=base,
                    units=units,
                    mixer=mixer,
                    controller=controller,
                    candidate=candidate,
                    operator=inner_operator,
                    values=inner_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                oracle_outer, oracle_diag, _, _, _ = _generate_value(
                    base=base,
                    units=units,
                    mixer=mixer,
                    controller=controller,
                    candidate=candidate,
                    operator=outer_operator,
                    values=true_outer_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    initial_fast_state=inner_fast,
                    initial_slow_state=inner_slow,
                    previous_posterior=inner_posterior,
                )

                chained_outer: int | None = None
                chained_diag: dict[str, Any] | None = None
                if generated_inner is not None:
                    chained_outer, chained_diag, _, _, _ = _generate_value(
                        base=base,
                        units=units,
                        mixer=mixer,
                        controller=controller,
                        candidate=candidate,
                        operator=outer_operator,
                        values=(generated_inner, *outer_extras),
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        initial_fast_state=inner_fast,
                        initial_slow_state=inner_slow,
                        previous_posterior=inner_posterior,
                    )

                inner_correct = generated_inner == true_inner
                counter["cases"] += 1
                counter["inner_parse"] += int(generated_inner is not None)
                counter["inner_correct"] += int(inner_correct)
                counter["oracle_intermediate_outer_parse"] += int(oracle_outer is not None)
                counter["oracle_intermediate_outer_correct"] += int(oracle_outer == true_final)
                counter["end_to_end_outer_parse"] += int(chained_outer is not None)
                counter["end_to_end_correct"] += int(chained_outer == true_final)
                if inner_correct:
                    counter["inner_correct_cases"] += 1
                    counter["end_to_end_correct_given_inner_correct"] += int(chained_outer == true_final)

                for row in (inner_diag, oracle_diag, chained_diag):
                    if row is not None:
                        matching_weight_sum += float(row["mean_matching_source_weight"])
                        controller_correct_sum += float(row["controller_correct"])
                        controller_confidence_sum += float(row["controller_confidence"])
                        diagnostic_calls += 1
                for row in (oracle_diag, chained_diag):
                    if row is not None:
                        gate = float(row["gate_value"])
                        gate_sum += gate
                        outer_calls += 1
                        if same_operator:
                            same_gate_sum += gate
                            same_gate_calls += 1
                        else:
                            switch_gate_sum += gate
                            switch_gate_calls += 1

            seq._merge_counter(aggregate, counter)
            pair_rows[pair_id] = seq._finalize_counter(counter)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "candidate": candidate.__dict__,
        "aggregate": seq._finalize_counter(aggregate),
        "pairs": pair_rows,
        "mean_matching_source_weight": matching_weight_sum / max(1, diagnostic_calls),
        "controller_stage_accuracy": controller_correct_sum / max(1, diagnostic_calls),
        "controller_mean_confidence": controller_confidence_sum / max(1, diagnostic_calls),
        "mean_gate": gate_sum / max(1, outer_calls),
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
    controller_tokenizer = FixedVocabTokenizer.from_config(root / controller_run.tokenizer_config)
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
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts[:3]
    ]

    aggregate = seq._empty_counter()
    pair_aggregate = {
        f"{inner}->{outer}": seq._empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    mean_weights: list[float] = []
    controller_stage_accuracies: list[float] = []
    controller_confidences: list[float] = []
    mean_gates: list[float] = []
    same_gates: list[float] = []
    switch_gates: list[float] = []
    for report in cohort_reports:
        raw_aggregate = {key: int(report["aggregate"][key]) for key in seq._empty_counter()}
        seq._merge_counter(aggregate, raw_aggregate)
        mean_weights.append(float(report["mean_matching_source_weight"]))
        controller_stage_accuracies.append(float(report["controller_stage_accuracy"]))
        controller_confidences.append(float(report["controller_mean_confidence"]))
        mean_gates.append(float(report["mean_gate"]))
        same_gates.append(float(report["mean_same_operator_gate"]))
        switch_gates.append(float(report["mean_switch_operator_gate"]))
        for pair_id, row in report["pairs"].items():
            raw_pair = {key: int(row[key]) for key in seq._empty_counter()}
            seq._merge_counter(pair_aggregate[pair_id], raw_pair)

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_posterior_transition_boundary_gate",
        "claim_boundary": (
            "operator identity and boundary-reset magnitude are derived from learned controller posteriors; "
            "the stage boundary itself and scalar handoff remain externally supplied, and model-token history "
            "is reset between stages; no gold operator identity is used by the gate; no NEG, single-pass nested "
            "execution, final IID test, or OOD split is tested"
        ),
        "gate_mode": _gate_mode(),
        "gate_scale": _gate_scale(),
        "controller_operator_strength": float(
            os.environ.get(learned.ENV_CONTROLLER_STRENGTH, "1.0")
        ),
        "operators": list(FUNCTIONAL_OPERATORS),
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "fixed_candidate": confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": fit_report,
        "controller_fit": controller_fit,
        "cohorts": cohort_reports,
        "aggregate": seq._finalize_counter(aggregate),
        "pairs": {
            pair_id: seq._finalize_counter(counter)
            for pair_id, counter in pair_aggregate.items()
        },
        "mean_matching_source_weight": sum(mean_weights) / max(1, len(mean_weights)),
        "controller_stage_accuracy": sum(controller_stage_accuracies) / max(1, len(controller_stage_accuracies)),
        "controller_mean_confidence": sum(controller_confidences) / max(1, len(controller_confidences)),
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
                "gate_mode": report["gate_mode"],
                "mean_gate": report["mean_gate"],
                "mean_same_operator_gate": report["mean_same_operator_gate"],
                "mean_switch_operator_gate": report["mean_switch_operator_gate"],
                "controller_stage_accuracy": report["controller_stage_accuracy"],
                "mean_matching_source_weight": report["mean_matching_source_weight"],
                "aggregate": report["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
