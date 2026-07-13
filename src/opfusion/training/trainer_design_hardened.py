from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from . import trainer_design as original
from .config import RunConfig
from .data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from .design_config import load_design_run_config, model_design
from .trainer import _find_repo_root


_ORIGINAL_BATCH = SyntheticTraceFactory.batch


def _configure_deterministic_pilot(config: RunConfig) -> None:
    """Force deterministic attention/math backends when the run requests them."""

    if not config.deterministic_algorithms:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


@contextmanager
def _full_domain_inactive_retention(job_id: str, config: RunConfig) -> Iterator[None]:
    """Use full-domain inactive prompts for teacher-KL retention.

    The original implementation requested ``base.common`` batches. Under the
    weak-multitask design that silently restricted retention prompts to the
    base's ±8 / four-term training domain. The labels are used only as a
    response-position mask for KL, so redirecting those calls to the inactive
    operator job preserves the same model-facing prefix while covering the
    full specialist domain.
    """

    design = model_design(config)
    active = job_id in EXPERIMENT_OPERATORS and design.specialist_retention_kl_weight > 0.0
    if not active:
        yield
        return

    def patched_batch(self: SyntheticTraceFactory, operator_id: str, *args: Any, **kwargs: Any):
        forced_operator = kwargs.get("forced_operator")
        if operator_id == "base.common" and forced_operator in EXPERIMENT_OPERATORS:
            return _ORIGINAL_BATCH(self, str(forced_operator), *args, **kwargs)
        return _ORIGINAL_BATCH(self, operator_id, *args, **kwargs)

    SyntheticTraceFactory.batch = patched_batch  # type: ignore[method-assign]
    try:
        yield
    finally:
        SyntheticTraceFactory.batch = _ORIGINAL_BATCH  # type: ignore[method-assign]


def train_job(
    *,
    repo_root: Path,
    config: RunConfig,
    job_id: str,
    seed: int,
    allow_cpu: bool = False,
) -> Path:
    _configure_deterministic_pilot(config)
    with _full_domain_inactive_retention(job_id, config):
        return original.train_job(
            repo_root=repo_root,
            config=config,
            job_id=job_id,
            seed=seed,
            allow_cpu=allow_cpu,
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train one design-controlled model with deterministic pilot backends, "
            "full-domain inactive retention, strict fingerprints, and validation selection"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke/pilot only")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).resolve()
    repo_root = _find_repo_root(config_path.parent)
    config = load_design_run_config(config_path)
    final = train_job(
        repo_root=repo_root,
        config=config,
        job_id=args.job,
        seed=args.seed,
        allow_cpu=args.allow_cpu,
    )
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
