from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from opfusion import fusion_eval as core
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


DEFAULT_FINAL_EVALUATION_SEED = 700_000
DEFAULT_PILOT_EVALUATION_SEED = 701_000
_ORIGINAL_TRAINING_EXAMPLE = SyntheticTraceFactory.training_example


def _default_evaluation_seed(config_path: Path) -> int:
    config = load_run_config(config_path)
    return (
        DEFAULT_PILOT_EVALUATION_SEED
        if config.experiment_id.startswith("model_design_pilot_")
        else DEFAULT_FINAL_EVALUATION_SEED
    )


@contextmanager
def _evaluation_seed_override(evaluation_seed: int) -> Iterator[None]:
    if evaluation_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")

    def patched(self: SyntheticTraceFactory, job_id: str, *args: Any, **kwargs: Any):
        # The core evaluator uses a private fixed seed. Override only that
        # evaluation namespace and leave all other factory calls untouched.
        if int(kwargs.get("seed", -1)) == DEFAULT_FINAL_EVALUATION_SEED:
            kwargs["seed"] = evaluation_seed
        return _ORIGINAL_TRAINING_EXAMPLE(self, job_id, *args, **kwargs)

    SyntheticTraceFactory.training_example = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        SyntheticTraceFactory.training_example = _ORIGINAL_TRAINING_EXAMPLE  # type: ignore[method-assign]


def evaluate_manifest_seeded(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    split: str = "test",
    examples_per_operator: int = 64,
    max_new_tokens: int = 256,
    alpha: float = 1.0,
    device_name: str = "auto",
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    resolved_seed = _default_evaluation_seed(config_path) if evaluation_seed is None else evaluation_seed
    with _evaluation_seed_override(resolved_seed):
        report = core.evaluate_manifest(
            config_path=config_path,
            manifest_path=manifest_path,
            split=split,
            examples_per_operator=examples_per_operator,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            device_name=device_name,
        )
    report["evaluation_seed"] = resolved_seed
    report["evaluation_role"] = (
        "model_design_development"
        if resolved_seed == DEFAULT_PILOT_EVALUATION_SEED
        else "final_or_user_selected"
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate base-relative logit fusion with an explicit, recorded data-generation seed"
    )
    parser.add_argument("--config", default="configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples-per-operator", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate_manifest_seeded(
        config_path=args.config,
        manifest_path=args.manifest,
        split=args.split,
        examples_per_operator=args.examples_per_operator,
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        device_name=args.device,
        evaluation_seed=args.evaluation_seed,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
