from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from opfusion import fusion_eval as core
from opfusion.final_eval_policy import FINAL_SPLITS, validate_evaluation_policy
from opfusion.training.config import load_run_config


DEFAULT_FINAL_EVALUATION_SEED = core.DEFAULT_EVALUATION_SEED
DEFAULT_PILOT_EVALUATION_SEED = 701_000


def _default_evaluation_seed(config_path: Path) -> int:
    config = load_run_config(config_path)
    return (
        DEFAULT_PILOT_EVALUATION_SEED
        if config.experiment_id.startswith("model_design_pilot_")
        else DEFAULT_FINAL_EVALUATION_SEED
    )


def _evaluation_role(*, experiment_id: str, split: str, authorization: dict[str, Any] | None) -> str:
    if experiment_id.startswith("model_design_pilot_"):
        return "model_design_development"
    if authorization is not None and split in FINAL_SPLITS:
        return "preregistered_final"
    if split == "validation":
        return "validation_development_or_calibration"
    return "legacy_or_user_selected"


def evaluate_manifest_seeded(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    split: str = "iid_test",
    examples_per_operator: int = 64,
    max_new_tokens: int = 256,
    alpha: float = 1.0,
    device_name: str = "auto",
    evaluation_seed: int | None = None,
    final_authorization_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    run = load_run_config(config_path)
    resolved_seed = _default_evaluation_seed(config_path) if evaluation_seed is None else evaluation_seed
    if resolved_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authorization = validate_evaluation_policy(
        repo_root=config_path.parents[2],
        config=run,
        manifest=manifest,
        split=split,
        evaluation_seed=resolved_seed,
        examples_per_operator=examples_per_operator,
        final_authorization_path=final_authorization_path,
    )
    report = core.evaluate_manifest(
        config_path=config_path,
        manifest_path=manifest_path,
        split=split,
        examples_per_operator=examples_per_operator,
        max_new_tokens=max_new_tokens,
        alpha=alpha,
        device_name=device_name,
        evaluation_seed=resolved_seed,
    )
    report["experiment_fingerprint"] = manifest.get("experiment_fingerprint")
    report["evaluation_role"] = _evaluation_role(
        experiment_id=run.experiment_id,
        split=split,
        authorization=authorization,
    )
    if authorization is not None:
        report["final_authorization"] = authorization
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate base-relative logit fusion with an explicit, recorded data-generation seed"
    )
    parser.add_argument("--config", default="configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--examples-per-operator", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--final-authorization")
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
        final_authorization_path=args.final_authorization,
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
