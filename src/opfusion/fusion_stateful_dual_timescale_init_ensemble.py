from __future__ import annotations

import os
from typing import Any, Iterable

import torch
from torch import nn

from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_dual_timescale_confirmatory_deterministic as deterministic
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_validity_mixture import SourceValidityMixer


ENV_ENSEMBLE_MODE = "OPFUSION_MIXER_ENSEMBLE_MODE"
INIT_SEEDS = (731000, 731001, 731002, 731003, 731004)
_original_fit_mixer = deterministic._original_fit_mixer


class SourceValidityEnsemble(nn.Module):
    """Continuous ensemble over source-validity weights.

    Every member scores all six sources. The ensemble averages the continuous
    source-weight distributions; it never selects a member or a specialist.
    """

    def __init__(self, members: Iterable[SourceValidityMixer], *, mode: str) -> None:
        super().__init__()
        self.members = nn.ModuleList(tuple(members))
        if not self.members:
            raise ValueError("at least one mixer is required")
        if mode not in {"arithmetic", "geometric"}:
            raise ValueError(f"unsupported ensemble mode: {mode}")
        self.mode = mode

    def compose(self, source_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        member_weights = torch.stack(
            [member.compose(source_logits)[1] for member in self.members], dim=0
        )
        if self.mode == "arithmetic":
            weights = member_weights.mean(dim=0)
        else:
            mean_log = member_weights.clamp_min(1e-12).log().mean(dim=0)
            weights = torch.softmax(mean_log, dim=-1)
        probabilities = torch.softmax(source_logits.float(), dim=-1)
        mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
        return mixture.log(), weights

    def forward(self, source_logits: torch.Tensor) -> torch.Tensor:
        return self.compose(source_logits)[0]


def fit_mixer_ensemble(**kwargs: Any):
    mode = os.environ.get(ENV_ENSEMBLE_MODE, "arithmetic")
    batch = kwargs["batch"]
    cuda_devices: list[int] = []
    if batch.base_logits.is_cuda:
        device_index = batch.base_logits.device.index
        cuda_devices = [0 if device_index is None else int(device_index)]

    members: list[SourceValidityMixer] = []
    reports: list[dict[str, Any]] = []
    for init_seed in INIT_SEEDS:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(init_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(init_seed)
            member, report = _original_fit_mixer(**kwargs)
        members.append(member)
        reports.append({"init_seed": init_seed, "fit": report})

    ensemble = SourceValidityEnsemble(members, mode=mode)
    return ensemble, {
        "ensemble_mode": mode,
        "member_count": len(members),
        "initialization_seeds": list(INIT_SEEDS),
        "member_fit_reports": reports,
    }


implementation.fit_mixer = fit_mixer_ensemble


def main() -> int:
    return deterministic.main()


if __name__ == "__main__":
    raise SystemExit(main())
