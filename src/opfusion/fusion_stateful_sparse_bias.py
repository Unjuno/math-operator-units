from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from opfusion import fusion_stateful_bias as implementation
from opfusion.fusion_validity_mixture import SourceValidityMixer


@dataclass(frozen=True)
class SparseBiasCandidate:
    candidate_id: str
    memory: float
    feedback: float
    temperature: float
    alpha: float
    threshold: float


def candidate_grid() -> tuple[SparseBiasCandidate, ...]:
    rows: list[SparseBiasCandidate] = []
    for memory in (0.90, 0.95):
        for feedback in (0.15, 0.35):
            for temperature in (0.75, 1.00):
                for threshold in (0.05, 0.10, 0.15):
                    for alpha in (1.0, 2.0, 4.0):
                        rows.append(
                            SparseBiasCandidate(
                                candidate_id=f"ssb_m{memory}_f{feedback}_t{temperature}_q{threshold}_a{alpha}",
                                memory=memory,
                                feedback=feedback,
                                temperature=temperature,
                                alpha=alpha,
                                threshold=threshold,
                            )
                        )
    return tuple(rows)


def sparse_bias_logits(
    source_logits: torch.Tensor,
    source_weights: torch.Tensor,
    *,
    alpha: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_logits.ndim != 2:
        raise ValueError("source_logits must be [sources, vocabulary]")
    if source_weights.shape != (source_logits.shape[0],):
        raise ValueError("source_weights must match source count")
    base = source_logits[0]
    biases = source_logits[1:] - base.unsqueeze(0)
    biases = biases - biases.mean(dim=-1, keepdim=True)
    specialist_weights = source_weights[1:]
    sparse = F.relu(specialist_weights - float(threshold))
    if float(sparse.sum()) <= 1e-9:
        sparse = specialist_weights
    sparse = sparse / sparse.sum().clamp_min(1e-9)
    residual = (sparse.unsqueeze(-1).to(biases.dtype) * biases).sum(dim=0)
    rms = biases.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    cap = (4.0 * rms.median()).clamp_min(1e-4).to(residual.dtype)
    bounded = cap * torch.tanh(float(alpha) * residual / cap)
    return base + bounded, sparse


def _generate_stateful_sparse_bias(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: SourceValidityMixer,
    candidate: SparseBiasCandidate,
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
            fused, sparse = sparse_bias_logits(
                sources,
                weights,
                alpha=candidate.alpha,
                threshold=candidate.threshold,
            )
            next_id = int(torch.argmax(fused).item())
            output.append(next_id)
            if candidate.feedback > 0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                state = state + float(candidate.feedback) * (token_support - token_support.mean())
            entropy_sum += float((-(sparse * sparse.clamp_min(1e-9).log()).sum()).cpu())
            max_weight_sum += float(sparse.max().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_base_weight": 0.0,
        "mean_max_weight": max_weight_sum / max(1, positions),
    }


implementation.candidate_grid = candidate_grid
implementation._generate_stateful_bias = _generate_stateful_sparse_bias


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
