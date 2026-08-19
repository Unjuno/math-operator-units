from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from opfusion import fusion_controller_objective_ablation as objective
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_self_tuned_operator_controller as self_tuned


def run_with_fixed_training_data_seed(*, controller_data_seed: int, **kwargs):
    """Vary controller initialization while keeping teacher-forced training rows fixed."""
    original_collect = self_tuned.collect_teacher_forced_control_batch

    def fixed_collect(*args, **collect_kwargs):
        collect_kwargs["data_seed"] = int(controller_data_seed)
        return original_collect(*args, **collect_kwargs)

    self_tuned.collect_teacher_forced_control_batch = fixed_collect
    try:
        report = objective.run_experiment(**kwargs)
    finally:
        self_tuned.collect_teacher_forced_control_batch = original_collect
    report["controller_data_seed"] = int(controller_data_seed)
    report["replication_axis"] = "controller_initialization_seed_only"
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replicate fixed-strength controller-objective ablation across initialization seeds"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--examples-per-pair", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=sequential.DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--calibration-examples-per-operator", type=int, default=8)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1536)
    parser.add_argument("--calibration-seed", type=int, default=sequential.DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sketch-size", type=int, default=8)
    parser.add_argument("--controller-train-examples-per-operator", type=int, default=24)
    parser.add_argument("--controller-holdout-examples-per-operator", type=int, default=32)
    parser.add_argument("--controller-max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--controller-data-seed", type=int, default=739000)
    parser.add_argument("--controller-seed", type=int, required=True)
    parser.add_argument("--controller-holdout-seed", type=int, default=739500)
    parser.add_argument("--controller-embedding-size", type=int, default=16)
    parser.add_argument("--controller-hidden-size", type=int, default=16)
    parser.add_argument("--controller-learning-rate", type=float, default=0.02)
    parser.add_argument("--controller-steps", type=int, default=400)
    parser.add_argument("--controller-batch-positions", type=int, default=256)
    parser.add_argument("--auxiliary-operator-weight", type=float, default=0.25)
    parser.add_argument("--fixed-strength", type=float, default=objective.DEFAULT_FIXED_STRENGTH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        report = run_with_fixed_training_data_seed(
            controller_data_seed=args.controller_data_seed,
            root=args.root,
            examples_per_pair=args.examples_per_pair,
            data_seed=args.data_seed,
            max_new_tokens=args.max_new_tokens,
            calibration_examples_per_operator=args.calibration_examples_per_operator,
            max_prefixes_per_example=args.max_prefixes_per_example,
            max_positions_per_cohort=args.max_positions_per_cohort,
            calibration_seed=args.calibration_seed,
            fit_steps=args.fit_steps,
            fit_batch_positions=args.fit_batch_positions,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            sketch_size=args.sketch_size,
            controller_train_examples_per_operator=args.controller_train_examples_per_operator,
            controller_holdout_examples_per_operator=args.controller_holdout_examples_per_operator,
            controller_max_positions_per_cohort=args.controller_max_positions_per_cohort,
            controller_seed=args.controller_seed,
            controller_holdout_seed=args.controller_holdout_seed,
            controller_embedding_size=args.controller_embedding_size,
            controller_hidden_size=args.controller_hidden_size,
            controller_learning_rate=args.controller_learning_rate,
            controller_steps=args.controller_steps,
            controller_batch_positions=args.controller_batch_positions,
            auxiliary_operator_weight=args.auxiliary_operator_weight,
            fixed_strength=args.fixed_strength,
            device_name=args.device,
        )
    finally:
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output.resolve())
    print(json.dumps({
        name: {
            "end_to_end": row["aggregate"]["end_to_end_accuracy"],
            "end_to_end_correct": row["aggregate"]["end_to_end_correct"],
            "composition_direction_accuracy": row["composition_prompt_direction"]["accuracy"],
            "holdout_accuracy": row["controller_fit"]["holdout_metrics"]["accuracy"],
            "mean_matching_source_weight": row["mean_matching_source_weight"],
        }
        for name, row in report["conditions"].items()
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
