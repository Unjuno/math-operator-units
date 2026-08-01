from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_validity_mixture import SourceValidityMixer


@dataclass(frozen=True)
class DualTimescaleCandidate:
    candidate_id: str
    fast_memory: float
    slow_memory: float
    slow_mix: float
    feedback: float
    temperature: float


def candidate_grid() -> tuple[DualTimescaleCandidate, ...]:
    """A compact 48-condition grid around the successful persistent mixture.

    The fast state can react to a local change in evidence. The slow state preserves
    evidence accumulated over the trajectory. Their convex combination is continuous;
    every source retains strictly positive probability mass.
    """
    rows: list[DualTimescaleCandidate] = []
    for fast_memory in (0.50, 0.80):
        for slow_memory in (0.95, 0.98):
            for slow_mix in (0.25, 0.50, 0.75):
                for feedback in (0.10, 0.35):
                    for temperature in (0.75, 1.00):
                        rows.append(
                            DualTimescaleCandidate(
                                candidate_id=(
                                    f"dual_fm{fast_memory}_sm{slow_memory}"
                                    f"_x{slow_mix}_fb{feedback}_t{temperature}"
                                ),
                                fast_memory=fast_memory,
                                slow_memory=slow_memory,
                                slow_mix=slow_mix,
                                feedback=feedback,
                                temperature=temperature,
                            )
                        )
    return tuple(rows)


def combine_states(
    fast_state: torch.Tensor,
    slow_state: torch.Tensor,
    *,
    slow_mix: float,
    temperature: float,
) -> torch.Tensor:
    if fast_state.shape != slow_state.shape:
        raise ValueError("fast_state and slow_state must have the same shape")
    if not 0.0 <= slow_mix <= 1.0:
        raise ValueError("slow_mix must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    combined = (1.0 - float(slow_mix)) * fast_state + float(slow_mix) * slow_state
    return torch.softmax(combined / float(temperature), dim=-1)


def _generate_dual_timescale(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: SourceValidityMixer,
    candidate: DualTimescaleCandidate,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    fast_state: torch.Tensor | None = None
    slow_state: torch.Tensor | None = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    timescale_gap_sum = 0.0
    positions = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = implementation._source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()

            if fast_state is None or slow_state is None:
                fast_state = instant_state
                slow_state = instant_state
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

            weights = combine_states(
                fast_state,
                slow_state,
                slow_mix=candidate.slow_mix,
                temperature=candidate.temperature,
            )
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0.0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                token_support = token_support - token_support.mean()
                fast_state = fast_state + float(candidate.feedback) * token_support
                # The slow channel receives a reduced update so one accidental token
                # cannot permanently lock the source mixture.
                slow_state = slow_state + 0.25 * float(candidate.feedback) * token_support

            entropy_sum += float(
                (-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu()
            )
            max_weight_sum += float(weights.max().detach().cpu())
            state_change_sum += float(state_change.detach().cpu())
            timescale_gap_sum += float((fast_state - slow_state).abs().mean().detach().cpu())
            positions += 1

            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == eos_id:
                break

    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
        "mean_timescale_gap": timescale_gap_sum / max(1, positions),
    }


_original_search = implementation.search_stateful_mixtures


def _search_dual_timescale(**kwargs: Any) -> dict[str, Any]:
    report = _original_search(**kwargs)
    report["evaluation_role"] = "validation_only_dual_timescale_stateful_probability_fusion"
    report["algebra"] = (
        "continuous arithmetic probability pooling with fast and slow persistent "
        "source-reliability states"
    )
    report["claim_boundary"] = (
        "Base and all five specialists are evaluated at every token; all source "
        "weights remain strictly positive; no operator labels, subset masks, or "
        "discrete switching; final splits unopened"
    )
    return report


implementation.candidate_grid = candidate_grid
implementation._generate_stateful = _generate_dual_timescale
implementation.search_stateful_mixtures = _search_dual_timescale


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
