from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_stateful_bias as implementation
from opfusion.fusion_validity_mixture import SourceValidityMixer


@dataclass(frozen=True)
class GeometricCandidate:
    candidate_id: str
    memory: float
    feedback: float
    temperature: float
    alpha: float


def candidate_grid() -> tuple[GeometricCandidate, ...]:
    rows: list[GeometricCandidate] = []
    for memory in (0.90, 0.95, 0.98):
        for feedback in (0.00, 0.15, 0.35):
            for temperature in (0.75, 1.00):
                for alpha in (0.5, 1.0, 1.5, 2.0):
                    rows.append(
                        GeometricCandidate(
                            candidate_id=f"geo_m{memory}_f{feedback}_t{temperature}_a{alpha}",
                            memory=memory,
                            feedback=feedback,
                            temperature=temperature,
                            alpha=alpha,
                        )
                    )
    return tuple(rows)


def geometric_logits(
    source_logits: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if source_logits.ndim != 2:
        raise ValueError("source_logits must be [sources, vocabulary]")
    if source_weights.shape != (source_logits.shape[0],):
        raise ValueError("source_weights must match source count")
    log_probabilities = torch.log_softmax(source_logits.float(), dim=-1)
    geometric = (source_weights.unsqueeze(-1) * log_probabilities).sum(dim=0)
    base = log_probabilities[0]
    return base + float(alpha) * (geometric - base)


def _generate_stateful_geometric(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: SourceValidityMixer,
    candidate: GeometricCandidate,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    state: torch.Tensor | None = None
    entropy_sum = 0.0
    base_weight_sum = 0.0
    max_weight_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = implementation._source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()
            if state is None:
                state = instant_state
            else:
                state = float(candidate.memory) * state + (1.0 - float(candidate.memory)) * instant_state
            weights = torch.softmax(state / float(candidate.temperature), dim=-1)
            fused = geometric_logits(sources, weights, alpha=candidate.alpha)
            next_id = int(torch.argmax(fused).item())
            output.append(next_id)
            if candidate.feedback > 0:
                support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                state = state + float(candidate.feedback) * (support - support.mean())
            entropy_sum += float((-(weights * weights.clamp_min(1e-9).log()).sum()).cpu())
            base_weight_sum += float(weights[0].cpu())
            max_weight_sum += float(weights.max().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_base_weight": base_weight_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
    }


_original_search = implementation.search_stateful_biases


def _search_stateful_geometric(**kwargs: Any) -> dict[str, Any]:
    report = _original_search(**kwargs)
    report["evaluation_role"] = "validation_only_stateful_geometric_logit_fusion"
    report["algebra"] = "persistent reliability weighted geometric pool of all source distributions, interpolated from Base without clipping"
    report["claim_boundary"] = "Base and all five specialists retain strictly positive continuous weights; no operator labels or discrete switching; final splits unopened"
    return report


implementation.candidate_grid = candidate_grid
implementation._generate_stateful_bias = _generate_stateful_geometric
implementation.search_stateful_biases = _search_stateful_geometric


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
