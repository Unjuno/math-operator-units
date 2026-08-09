from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_controller_state_boundary_carry as boundary
from opfusion import fusion_learned_operator_controller as learned
from opfusion import fusion_stateful_dual_timescale as dual_timescale
from opfusion import fusion_stateful_mixture as implementation
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import EXPERIMENT_OPERATORS


ENV_FAST_BOUNDARY_RESET = "OPFUSION_FAST_BOUNDARY_RESET"
ENV_SLOW_BOUNDARY_RESET = "OPFUSION_SLOW_BOUNDARY_RESET"


def _reset_value(name: str) -> float:
    value = float(os.environ.get(name, "1.0"))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def blend_timescale_state(
    initial_state: torch.Tensor,
    instant_state: torch.Tensor,
    *,
    reset_fraction: float,
) -> torch.Tensor:
    return boundary.blend_boundary_state(
        initial_state,
        instant_state,
        reset_fraction=reset_fraction,
    )


def _generate_timescale_conditioned(
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
    boundary_reset: float,
    initial_fast_state: torch.Tensor | None = None,
    initial_slow_state: torch.Tensor | None = None,
) -> tuple[list[int], dict[str, Any], torch.Tensor, torch.Tensor]:
    del boundary_reset  # Kept only for signature compatibility with PR28.
    if (initial_fast_state is None) != (initial_slow_state is None):
        raise ValueError("initial fast and slow states must be supplied together")

    fast_reset = _reset_value(ENV_FAST_BOUNDARY_RESET)
    slow_reset = _reset_value(ENV_SLOW_BOUNDARY_RESET)
    strength = float(os.environ.get(learned.ENV_CONTROLLER_STRENGTH, "0.0"))
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
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    timescale_gap_sum = 0.0
    source_weight_sum: torch.Tensor | None = None
    fast_boundary_shift = 0.0
    slow_boundary_shift = 0.0
    positions = 0

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
                    carried_fast = initial_fast_state.to(
                        device=instant_state.device, dtype=instant_state.dtype
                    )
                    carried_slow = initial_slow_state.to(
                        device=instant_state.device, dtype=instant_state.dtype
                    )
                    fast_state = blend_timescale_state(
                        carried_fast,
                        instant_state,
                        reset_fraction=fast_reset,
                    )
                    slow_state = blend_timescale_state(
                        carried_slow,
                        instant_state,
                        reset_fraction=slow_reset,
                    )
                    fast_boundary_shift = float(
                        (fast_state - carried_fast).abs().mean().detach().cpu()
                    )
                    slow_boundary_shift = float(
                        (slow_state - carried_slow).abs().mean().detach().cpu()
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
            mixture = (
                weights.unsqueeze(-1) * probabilities
            ).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0.0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                token_support = token_support - token_support.mean()
                fast_state = fast_state + float(candidate.feedback) * token_support
                slow_state = (
                    slow_state
                    + 0.25 * float(candidate.feedback) * token_support
                )

            entropy_sum += float(
                (-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu()
            )
            max_weight_sum += float(weights.max().detach().cpu())
            state_change_sum += float(state_change.detach().cpu())
            timescale_gap_sum += float(
                (fast_state - slow_state).abs().mean().detach().cpu()
            )
            source_weight_sum = (
                weights.detach().clone()
                if source_weight_sum is None
                else source_weight_sum + weights.detach()
            )
            positions += 1

            ids = torch.cat(
                [
                    ids,
                    torch.tensor([[next_id]], dtype=torch.long, device=device),
                ],
                dim=1,
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
    mean_boundary_shift = (
        (1.0 - float(candidate.slow_mix)) * fast_boundary_shift
        + float(candidate.slow_mix) * slow_boundary_shift
    )
    return (
        output,
        {
            "controller_posterior": [
                float(value) for value in posterior.detach().cpu()
            ],
            "controller_predicted_operator": learned.FUNCTIONAL_OPERATORS[
                predicted_class
            ],
            "controller_confidence": float(posterior.max().detach().cpu()),
            "mean_weight_entropy": entropy_sum / max(1, positions),
            "mean_max_weight": max_weight_sum / max(1, positions),
            "mean_state_change": state_change_sum / max(1, positions),
            "mean_timescale_gap": timescale_gap_sum / max(1, positions),
            "mean_source_weights": [
                float(value) for value in mean_source_weights.detach().cpu()
            ],
            "boundary_reset": None,
            "fast_boundary_reset": fast_reset,
            "slow_boundary_reset": slow_reset,
            "fast_boundary_state_shift": fast_boundary_shift,
            "slow_boundary_state_shift": slow_boundary_shift,
            "boundary_state_shift": mean_boundary_shift,
            "positions": positions,
        },
        fast_state.detach().clone(),
        slow_state.detach().clone(),
    )


def main() -> int:
    # PR28's evaluation pipeline already performs the exact dataset/model/controller
    # setup needed here. Replacing only its generation kernel isolates reset law.
    boundary._generate_with_boundary_state = _generate_timescale_conditioned
    args = boundary._parser().parse_args()
    report = boundary.run_experiment(
        root=args.root.resolve(),
        boundary_reset=1.0,
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
        controller_validation_examples_per_operator=(
            args.controller_validation_examples_per_operator
        ),
        controller_steps=args.controller_steps,
        controller_learning_rate=args.controller_learning_rate,
        device_name=args.device,
    )
    fast_reset = _reset_value(ENV_FAST_BOUNDARY_RESET)
    slow_reset = _reset_value(ENV_SLOW_BOUNDARY_RESET)
    report["evaluation_role"] = (
        "validation_only_timescale_specific_boundary_reset"
    )
    report["claim_boundary"] = (
        "operator identity is predicted from each externally delimited stage prompt; "
        "scalar handoff and the stage boundary remain externally supplied; fast and "
        "slow fusion memories are reset independently at that boundary; no NEG, "
        "single-pass nested execution, final IID test, or OOD split is tested"
    )
    report["boundary_reset"] = None
    report["fast_boundary_reset"] = fast_reset
    report["slow_boundary_reset"] = slow_reset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(
        json.dumps(
            {
                "fast_boundary_reset": fast_reset,
                "slow_boundary_reset": slow_reset,
                "strength": report["controller_operator_strength"],
                "controller_stage_accuracy": report["controller_stage_accuracy"],
                "mean_matching_source_weight": report["mean_matching_source_weight"],
                "mean_boundary_state_shift": report["mean_boundary_state_shift"],
                "aggregate": report["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
