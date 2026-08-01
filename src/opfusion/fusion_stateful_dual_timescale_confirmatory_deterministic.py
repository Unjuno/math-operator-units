from __future__ import annotations

from typing import Any

import torch

from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_mixture as implementation


_original_fit_mixer = implementation.fit_mixer


def fit_mixer_deterministic(**kwargs: Any):
    """Run the existing mixer fit with deterministic initialization and reductions.

    The original fit function uses its seed for minibatch permutations but creates
    the neural mixer before seeding PyTorch's global RNG. CPU parallel reductions
    can also produce small cross-run floating-point differences. This wrapper forks
    RNG state, seeds initialization, enables deterministic algorithms, and performs
    calibration with one CPU thread before restoring caller settings.
    """
    seed = int(kwargs["seed"])
    batch = kwargs["batch"]
    cuda_devices: list[int] = []
    if batch.base_logits.is_cuda:
        device_index = batch.base_logits.device.index
        cuda_devices = [0 if device_index is None else int(device_index)]

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            return _original_fit_mixer(**kwargs)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


implementation.fit_mixer = fit_mixer_deterministic


def main() -> int:
    return confirmatory.main()


if __name__ == "__main__":
    raise SystemExit(main())
