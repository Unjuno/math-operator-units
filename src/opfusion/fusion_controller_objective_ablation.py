from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from opfusion import fusion_learned_operator_controller as learned
from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_self_tuned_operator_controller as self_tuned
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


OBJECTIVES = ("ce_only", "nll_ce", "nll_only")
DEFAULT_FIXED_STRENGTH = 4.0


def _direction_loss(
    controller: self_tuned.SelfTunedPromptController,
    batch: self_tuned.TeacherForcedControlBatch,
    indices: torch.Tensor,
    *,
    objective: str,
    fixed_strength: float,
    auxiliary_operator_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    ids = batch.prompt_ids.index_select(0, indices)
    mask = batch.prompt_mask.index_select(0, indices)
    labels = batch.operator_labels.index_select(0, indices)
    base_scores = batch.base_scores.index_select(0, indices)
    target_source_probabilities = batch.target_source_probabilities.index_select(0, indices)

    operator_logits, _ = controller.control(ids, mask)
    operator_probabilities = torch.softmax(operator_logits, dim=-1)
    scale = torch.full(
        operator_probabilities.shape[:-1],
        float(fixed_strength),
        dtype=operator_probabilities.dtype,
        device=operator_probabilities.device,
    )
    prior = self_tuned.source_prior_from_control(
        operator_probabilities,
        scale,
        source_count=int(base_scores.shape[-1]),
    )
    weights = torch.softmax(base_scores + prior, dim=-1)
    target_probability = (weights * target_source_probabilities).sum(dim=-1).clamp_min(1e-12)
    nll = -target_probability.log().mean()
    operator_ce = torch.nn.functional.cross_entropy(operator_logits, labels)

    if objective == "ce_only":
        loss = operator_ce
    elif objective == "nll_only":
        loss = nll
    else:
        loss = nll + float(auxiliary_operator_weight) * operator_ce

    entropy = -(operator_probabilities.clamp_min(1e-12).log() * operator_probabilities).sum(dim=-1).mean()
    return loss, {
        "nll": float(nll.detach().cpu()),
        "operator_ce": float(operator_ce.detach().cpu()),
        "direction_entropy": float(entropy.detach().cpu()),
    }


def fit_direction_controller(
    *,
    batch: self_tuned.TeacherForcedControlBatch,
    holdout_examples: Sequence[tuple[Sequence[int], int]],
    vocabulary_size: int,
    pad_id: int,
    embedding_size: int,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    objective: str,
    fixed_strength: float,
    auxiliary_operator_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[self_tuned.SelfTunedPromptController, dict[str, Any]]:
    if batch.positions <= 0:
        raise ValueError("teacher-forced batch is empty")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    controller = self_tuned.SelfTunedPromptController(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate, weight_decay=1e-4)

    first_loss: float | None = None
    last_loss: float | None = None
    first_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] | None = None
    controller.train()
    for _ in range(steps):
        if batch_positions >= batch.positions:
            indices = torch.arange(batch.positions, device=device)
        else:
            indices = torch.randint(0, batch.positions, (batch_positions,), device=device)
        loss, metrics = _direction_loss(
            controller,
            batch,
            indices,
            objective=objective,
            fixed_strength=fixed_strength,
            auxiliary_operator_weight=auxiliary_operator_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
            first_metrics = metrics
        last_loss = float(loss.detach().cpu())
        last_metrics = metrics

    controller.eval()
    holdout_metrics = learned.controller_metrics(
        controller,
        holdout_examples,
        pad_id=pad_id,
        device=device,
    )
    return controller, {
        "objective": objective,
        "fixed_strength_train": fixed_strength,
        "fixed_strength_eval": fixed_strength,
        "auxiliary_operator_weight": auxiliary_operator_weight,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_positions": batch_positions,
        "teacher_forced_positions": batch.positions,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "optimization_first": first_loss,
        "optimization_last": last_loss,
        "first_step_metrics": first_metrics,
        "last_step_metrics": last_metrics,
        "holdout_metrics": holdout_metrics,
    }


def _composition_direction_metrics(
    controller: self_tuned.SelfTunedPromptController,
    cohort: Cohort,
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    correct = 0
    count = 0
    matching_probability = 0.0
    per_operator = {
        operator: {"correct": 0, "count": 0, "matching_probability": 0.0}
        for operator in sequential.FUNCTIONAL_OPERATORS
    }

    for inner in sequential.FUNCTIONAL_OPERATORS:
        for outer in sequential.FUNCTIONAL_OPERATORS:
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = sequential.composition_operands(
                    inner_operator=inner,
                    outer_operator=outer,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = sequential.apply_operator(inner, inner_values)
                for operator, values in (
                    (inner, inner_values),
                    (outer, (true_inner, *outer_extras)),
                ):
                    prompt = sequential.prompt_ids_for_values(
                        factory=factory,
                        tokenizer=tokenizer,
                        operator=operator,
                        values=values,
                    )
                    probabilities = controller.probabilities(prompt, device=device)
                    expected = sequential.FUNCTIONAL_OPERATORS.index(operator)
                    predicted = int(probabilities.argmax().item())
                    matched = int(predicted == expected)
                    probability = float(probabilities[expected].detach().cpu())
                    correct += matched
                    count += 1
                    matching_probability += probability
                    row = per_operator[operator]
                    row["correct"] += matched
                    row["count"] += 1
                    row["matching_probability"] += probability

    return {
        "examples": count,
        "accuracy": correct / max(1, count),
        "mean_matching_probability": matching_probability / max(1, count),
        "per_operator": {
            operator: {
                "accuracy": row["correct"] / max(1, row["count"]),
                "mean_matching_probability": row["matching_probability"] / max(1, row["count"]),
                "examples": row["count"],
            }
            for operator, row in per_operator.items()
        },
    }


def _aggregate_cohort_reports(
    cohort_reports: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    aggregate = sequential._empty_counter()
    pair_aggregate = {
        f"{inner}->{outer}": sequential._empty_counter()
        for inner in sequential.FUNCTIONAL_OPERATORS
        for outer in sequential.FUNCTIONAL_OPERATORS
    }
    mean_weights: list[float] = []
    for report in cohort_reports:
        sequential._merge_counter(
            aggregate,
            {key: int(report["aggregate"][key]) for key in sequential._empty_counter()},
        )
        mean_weights.append(float(report["mean_matching_source_weight"]))
        for pair_id, row in report["pairs"].items():
            sequential._merge_counter(
                pair_aggregate[pair_id],
                {key: int(row[key]) for key in sequential._empty_counter()},
            )
    return (
        sequential._finalize_counter(aggregate),
        {
            pair_id: sequential._finalize_counter(counter)
            for pair_id, counter in pair_aggregate.items()
        },
        sum(mean_weights) / max(1, len(mean_weights)),
    )


def _evaluate_controller(
    controller: self_tuned.SelfTunedPromptController,
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    fixed_strength: float,
    mixer: torch.nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], float]:
    previous_controller = learned._ACTIVE_CONTROLLER
    previous_strength = os.environ.get(learned.ENV_CONTROLLER_STRENGTH)
    original_generator = oracle._generate_oracle_operator
    learned._ACTIVE_CONTROLLER = controller
    os.environ[learned.ENV_CONTROLLER_STRENGTH] = repr(float(fixed_strength))
    oracle._generate_oracle_operator = learned._generate_learned_operator
    try:
        reports = [
            sequential.evaluate_cohort(
                cohort,
                root=root,
                mixer=mixer,
                examples_per_pair=examples_per_pair,
                data_seed=data_seed,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            for cohort in cohorts[:3]
        ]
    finally:
        oracle._generate_oracle_operator = original_generator
        learned._ACTIVE_CONTROLLER = previous_controller
        if previous_strength is None:
            os.environ.pop(learned.ENV_CONTROLLER_STRENGTH, None)
        else:
            os.environ[learned.ENV_CONTROLLER_STRENGTH] = previous_strength
    aggregate, pairs, mean_weight = _aggregate_cohort_reports(reports)
    return reports, aggregate, pairs, mean_weight


def run_experiment(
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    calibration_seed: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    hidden_size: int,
    sketch_size: int,
    controller_train_examples_per_operator: int,
    controller_holdout_examples_per_operator: int,
    controller_max_positions_per_cohort: int,
    controller_seed: int,
    controller_holdout_seed: int,
    controller_embedding_size: int,
    controller_hidden_size: int,
    controller_learning_rate: float,
    controller_steps: int,
    controller_batch_positions: int,
    auxiliary_operator_weight: float,
    fixed_strength: float,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    mixer, mixer_fit = sequential.fit_ensemble(
        cohorts,
        root=root,
        calibration_examples_per_operator=calibration_examples_per_operator,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions_per_cohort=max_positions_per_cohort,
        calibration_seed=calibration_seed,
        fit_steps=fit_steps,
        fit_batch_positions=fit_batch_positions,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        sketch_size=sketch_size,
        device=device,
    )

    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    training_batch = self_tuned.collect_teacher_forced_control_batch(
        cohorts,
        root=root,
        mixer=mixer,
        examples_per_operator=controller_train_examples_per_operator,
        data_seed=controller_seed,
        max_positions_per_cohort=controller_max_positions_per_cohort,
        device=device,
    )
    holdout_examples = self_tuned._standalone_controller_examples(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=controller_holdout_examples_per_operator,
        seed=controller_holdout_seed,
        split="validation",
    )

    conditions: dict[str, Any] = {}
    for objective in OBJECTIVES:
        controller, fit_report = fit_direction_controller(
            batch=training_batch,
            holdout_examples=holdout_examples,
            vocabulary_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            embedding_size=controller_embedding_size,
            hidden_size=controller_hidden_size,
            learning_rate=controller_learning_rate,
            steps=controller_steps,
            batch_positions=controller_batch_positions,
            objective=objective,
            fixed_strength=fixed_strength,
            auxiliary_operator_weight=auxiliary_operator_weight,
            seed=controller_seed,
            device=device,
        )
        composition_direction = _composition_direction_metrics(
            controller,
            cohorts[0],
            root=root,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            device=device,
        )
        cohort_reports, aggregate, pairs, mean_weight = _evaluate_controller(
            controller,
            cohorts,
            root=root,
            examples_per_pair=examples_per_pair,
            data_seed=data_seed,
            max_new_tokens=max_new_tokens,
            fixed_strength=fixed_strength,
            mixer=mixer,
            device=device,
        )
        conditions[objective] = {
            "controller_fit": fit_report,
            "composition_prompt_direction": composition_direction,
            "cohort_reports": cohort_reports,
            "aggregate": aggregate,
            "pairs": pairs,
            "mean_matching_source_weight": mean_weight,
        }

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_controller_objective_ablation",
        "claim_boundary": (
            "same controller architecture, teacher-forced standalone data, seed, optimizer, minibatch schedule, "
            "fixed train/eval prior strength, mixer, and 192-case sequential composition protocol across objectives; "
            "explicit operator tokens remain visible and stage boundaries remain external"
        ),
        "objectives": list(OBJECTIVES),
        "fixed_strength": fixed_strength,
        "controller_seed": controller_seed,
        "controller_holdout_seed": controller_holdout_seed,
        "teacher_forced_positions": training_batch.positions,
        "mixer_fit": mixer_fit,
        "conditions": conditions,
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablate controller training objective at fixed source-prior strength"
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
    parser.add_argument("--controller-seed", type=int, default=self_tuned.DEFAULT_CONTROLLER_SEED)
    parser.add_argument(
        "--controller-holdout-seed", type=int, default=self_tuned.DEFAULT_CONTROLLER_HOLDOUT_SEED
    )
    parser.add_argument("--controller-embedding-size", type=int, default=16)
    parser.add_argument("--controller-hidden-size", type=int, default=16)
    parser.add_argument("--controller-learning-rate", type=float, default=0.02)
    parser.add_argument("--controller-steps", type=int, default=400)
    parser.add_argument("--controller-batch-positions", type=int, default=256)
    parser.add_argument("--auxiliary-operator-weight", type=float, default=0.25)
    parser.add_argument("--fixed-strength", type=float, default=DEFAULT_FIXED_STRENGTH)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="evaluations/controller_objective_ablation/summary.json"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        report = run_experiment(
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
    for objective in OBJECTIVES:
        row = report["conditions"][objective]
        print(
            objective,
            json.dumps(
                {
                    "holdout": row["controller_fit"]["holdout_metrics"],
                    "composition_direction": row["composition_prompt_direction"],
                    "aggregate": row["aggregate"],
                    "mean_matching_source_weight": row["mean_matching_source_weight"],
                },
                sort_keys=True,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
