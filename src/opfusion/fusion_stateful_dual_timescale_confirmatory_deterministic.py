from __future__ import annotations

from typing import Any

import torch

from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_mixture as implementation


_original_fit_mixer = implementation.fit_mixer


def fit_mixer_deterministic(**kwargs: Any):
    """Run the existing mixer fit with deterministic initialization and reductions."""
    seed = int(kwargs["seed"])
    batch = kwargs["batch"]
    cuda_devices: list[int] = []
    if batch.base_logits.is_cuda:
        device_index = batch.base_logits.device.index
        cuda_devices = [0 if device_index is None else int(device_index)]

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            return _original_fit_mixer(**kwargs)
    finally:
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


implementation.fit_mixer = fit_mixer_deterministic


def main() -> int:
    """Run data collection, calibration, and verification deterministically."""
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        return confirmatory.main()
    finally:
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(main())
