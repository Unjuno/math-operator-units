from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_validity_mixture import SourceValidityMixer


@dataclass(frozen=True)
class SparseProbabilityCandidate:
    candidate_id: str
    memory: float
    feedback: float
    temperature: float
    threshold: float
    power: float


def candidate_grid() -> tuple[SparseProbabilityCandidate, ...]:
    rows: list[SparseProbabilityCandidate] = []
    for memory in (0.90, 0.95, 0.98):
        for feedback in (0.15, 0.35):
            for temperature in (0.75, 1.00):
                for threshold in (0.00, 0.02, 0.05, 0.10):
                    for power in (1.0, 2.0):
                        rows.append(
                            SparseProbabilityCandidate(
                                candidate_id=(
                                    f"sprob_m{memory}_f{feedback}_t{temperature}"
                                    f"_q{threshold}_p{power}"
                                ),
                                memory=memory,
                                feedback=feedback,
                                temperature=temperature,
                                threshold=threshold,
                                power=power,
                            )
                        )
    return tuple(rows)


def sparse_probability_weights(
    weights: torch.Tensor,
    *,
    threshold: float,
    power: float,
    floor: float = 1e-5,
) -> torch.Tensor:
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if threshold < 0 or power <= 0 or floor <= 0:
        raise ValueError("invalid sparse-weight parameters")
    shifted = F.relu(weights - float(threshold))
    if float(shifted.sum()) <= 1e-12:
        shifted = weights
    sharpened = shifted.clamp_min(0).pow(float(power))
    positive = sharpened + float(floor)
    return positive / positive.sum().clamp_min(1e-12)


def _generate_sparse_probability(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: SourceValidityMixer,
    candidate: SparseProbabilityCandidate,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    state: torch.Tensor | None = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = implementation._source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()
            if state is None:
                state = instant_state
                state_change = instant_state.new_tensor(0.0)
            else:
                previous = state
                state = float(candidate.memory) * state + (1.0 - float(candidate.memory)) * instant_state
                state_change = (state - previous).abs().mean()
            dense_weights = torch.softmax(state / float(candidate.temperature), dim=-1)
            weights = sparse_probability_weights(
                dense_weights,
                threshold=candidate.threshold,
                power=candidate.power,
            )
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                state = state + float(candidate.feedback) * (token_support - token_support.mean())

            entropy_sum += float((-(weights * weights.clamp_min(1e-9).log()).sum()).cpu())
            max_weight_sum += float(weights.max().cpu())
            state_change_sum += float(state_change.cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
    }


_original_search = implementation.search_stateful_mixtures


def _search_sparse_probability(**kwargs: Any) -> dict[str, Any]:
    report = _original_search(**kwargs)
    report["evaluation_role"] = "validation_only_sparse_stateful_probability_fusion"
    report["algebra"] = "persistent source reliability with continuous soft-thresholding, positive floor, renormalization, and arithmetic probability pooling"
    report["claim_boundary"] = "Base and all five specialists are evaluated at every token; weights are continuous and retain a positive floor; no operator labels or discrete switching; final splits unopened"
    return report


implementation.candidate_grid = candidate_grid
implementation._generate_stateful = _generate_sparse_probability
implementation.search_stateful_mixtures = _search_sparse_probability


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
