from __future__ import annotations

import os
from typing import Any

import torch

from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_dual_timescale_confirmatory_deterministic as deterministic
from opfusion import fusion_stateful_mixture as implementation


ENV_INIT_SEED = "OPFUSION_MIXER_INIT_SEED"
_original_fit_mixer = deterministic._original_fit_mixer


def fit_mixer_with_independent_init_seed(**kwargs: Any):
    """Keep calibration data/order fixed while varying only mixer initialization."""
    if ENV_INIT_SEED not in os.environ:
        raise RuntimeError(f"{ENV_INIT_SEED} must be set")
    init_seed = int(os.environ[ENV_INIT_SEED])
    batch = kwargs["batch"]
    cuda_devices: list[int] = []
    if batch.base_logits.is_cuda:
        device_index = batch.base_logits.device.index
        cuda_devices = [0 if device_index is None else int(device_index)]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(init_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(init_seed)
        return _original_fit_mixer(**kwargs)


implementation.fit_mixer = fit_mixer_with_independent_init_seed


def main() -> int:
    # deterministic.main applies one-thread deterministic inference to collection,
    # calibration, and verification. The fit hook above changes initialization only.
    return deterministic.main()


if __name__ == "__main__":
    raise SystemExit(main())
