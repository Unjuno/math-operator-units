from __future__ import annotations

import torch

from opfusion import fusion_sparse_valid as implementation


def _corrected_evidence_features(
    self: implementation.SparseEvidenceCompositor,
    base_logits: torch.Tensor,
    unit_logits: torch.Tensor,
    centered: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rms = centered.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    normalized = centered / rms.unsqueeze(-1).to(centered.dtype)
    consensus = normalized.mean(dim=-2, keepdim=True)
    consensus_rms = consensus.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    cosine = (normalized * consensus).float().mean(dim=-1) / consensus_rms

    base_prob = torch.softmax(base_logits.float(), dim=-1)
    unit_prob = torch.softmax(unit_logits.float(), dim=-1)
    base_entropy = -(base_prob * base_prob.clamp_min(1e-9).log()).sum(dim=-1)
    unit_entropy = -(unit_prob * unit_prob.clamp_min(1e-9).log()).sum(dim=-1)
    entropy_gain = base_entropy.unsqueeze(-1) - unit_entropy
    margin_gain = self._margin(unit_logits) - self._margin(base_logits).unsqueeze(-1)

    features = torch.stack(
        [
            rms.log(),
            normalized.amax(dim=-1),
            -normalized.amin(dim=-1),
            entropy_gain,
            margin_gain,
            cosine,
        ],
        dim=-1,
    )
    return features, rms


implementation.SparseEvidenceCompositor.evidence_features = _corrected_evidence_features


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
