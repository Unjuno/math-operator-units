from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from . import batch as core_batch
from .trainer_surface import train_job


def run_batch(config_path: Path, *, allow_cpu: bool = False) -> int:
    # batch.py resolves train_job from its module globals. Replace that binding
    # so the existing dependency queue, manifests, recovery, and plan logic use
    # the fixed-sample/verifier-backed surface evaluator.
    core_batch.train_job = train_job
    return core_batch.run_batch(config_path, allow_cpu=allow_cpu)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the surface-form bias-fusion model factory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke-test only; production config requires CUDA")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config)
    if args.plan_only:
        return core_batch.main(["--config", str(config_path), "--plan-only"])
    return run_batch(config_path, allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    raise SystemExit(main())
