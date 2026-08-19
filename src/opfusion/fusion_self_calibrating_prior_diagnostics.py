from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_self_calibrating_operator_controller as selfcal
from opfusion.fusion_search import discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


SOURCE_NAMES = ("base", *EXPERIMENT_OPERATORS)


def _prior_rows_for_prompts(
    controller: selfcal.SelfCalibratingController,
    rows: Sequence[tuple[str, Sequence[int]]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    grouped: dict[str, list[torch.Tensor]] = {
        operator: [] for operator in selfcal.FUNCTIONAL_OPERATORS
    }
    for operator, prompt in rows:
        grouped[operator].append(controller.source_prior(prompt, device=device).detach().cpu())

    mean_prior_by_operator: dict[str, dict[str, float]] = {}
    mean_softmax_by_operator: dict[str, dict[str, float]] = {}
    argmax_distribution_by_operator: dict[str, dict[str, float]] = {}
    for operator, vectors in grouped.items():
        if not vectors:
            continue
        matrix = torch.stack(vectors, dim=0)
        mean_prior = matrix.mean(dim=0)
        mean_softmax = torch.softmax(matrix, dim=-1).mean(dim=0)
        argmax = matrix.argmax(dim=-1)
        mean_prior_by_operator[operator] = {
            name: float(mean_prior[index]) for index, name in enumerate(SOURCE_NAMES)
        }
        mean_softmax_by_operator[operator] = {
            name: float(mean_softmax[index]) for index, name in enumerate(SOURCE_NAMES)
        }
        argmax_distribution_by_operator[operator] = {
            name: float((argmax == index).float().mean())
            for index, name in enumerate(SOURCE_NAMES)
        }
    return {
        "source_order": list(SOURCE_NAMES),
        "mean_log_prior_by_operator": mean_prior_by_operator,
        "mean_prior_softmax_by_operator": mean_softmax_by_operator,
        "argmax_source_distribution_by_operator": argmax_distribution_by_operator,
    }


def teacher_forced_prompt_diagnostics(
    controller: selfcal.SelfCalibratingController,
    traces: Sequence[selfcal.TeacherForcedTrace],
    *,
    device: torch.device,
) -> dict[str, Any]:
    rows = [
        (selfcal.FUNCTIONAL_OPERATORS[trace.operator_index], trace.prompt)
        for trace in traces
    ]
    return _prior_rows_for_prompts(controller, rows, device=device)


def composition_prompt_diagnostics(
    controller: selfcal.SelfCalibratingController,
    cohort,
    *,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    rows: list[tuple[str, Sequence[int]]] = []
    for inner in selfcal.FUNCTIONAL_OPERATORS:
        for outer in selfcal.FUNCTIONAL_OPERATORS:
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
                    rows.append(
                        (
                            operator,
                            sequential.prompt_ids_for_values(
                                factory=factory,
                                tokenizer=tokenizer,
                                operator=operator,
                                values=values,
                            ),
                        )
                    )
    return _prior_rows_for_prompts(controller, rows, device=device)


def run_diagnostics(
    *,
    root: Path,
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
    controller_seed: int,
    controller_holdout_seed: int,
    controller_embedding_size: int,
    controller_hidden_size: int,
    controller_learning_rate: float,
    controller_steps: int,
    controller_prior_l2_weight: float,
    composition_examples_per_pair: int,
    data_seed: int,
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
    train_trace_sets = []
    tokenizers = []
    for cohort in cohorts[:2]:
        traces, tokenizer = selfcal.collect_teacher_forced_traces(
            cohort,
            root=root,
            mixer=mixer,
            examples_per_operator=controller_train_examples_per_operator,
            seed=controller_seed,
            split="train",
            device=device,
        )
        train_trace_sets.append(traces)
        tokenizers.append(tokenizer)
    holdout_traces, holdout_tokenizer = selfcal.collect_teacher_forced_traces(
        cohorts[2],
        root=root,
        mixer=mixer,
        examples_per_operator=controller_holdout_examples_per_operator,
        seed=controller_holdout_seed,
        split="validation",
        device=device,
    )
    tokenizers.append(holdout_tokenizer)
    if len({tokenizer.vocab_hash for tokenizer in tokenizers}) != 1:
        raise RuntimeError("prior diagnostics require identical tokenizers")
    tokenizer = tokenizers[0]
    train_traces = [trace for traces in train_trace_sets for trace in traces]
    controller, controller_fit = selfcal.fit_self_calibrating_controller(
        traces=train_traces,
        holdout_traces=holdout_traces,
        vocabulary_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embedding_size=controller_embedding_size,
        hidden_size=controller_hidden_size,
        learning_rate=controller_learning_rate,
        steps=controller_steps,
        prior_l2_weight=controller_prior_l2_weight,
        seed=controller_seed,
        device=device,
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "source_order": list(SOURCE_NAMES),
        "controller_fit": controller_fit,
        "mixer_fit": mixer_fit,
        "train_prompt_prior_diagnostics": teacher_forced_prompt_diagnostics(
            controller, train_traces, device=device
        ),
        "holdout_prompt_prior_diagnostics": teacher_forced_prompt_diagnostics(
            controller, holdout_traces, device=device
        ),
        "composition_prompt_prior_diagnostics": composition_prompt_diagnostics(
            controller,
            cohorts[0],
            root=root,
            examples_per_pair=composition_examples_per_pair,
            data_seed=data_seed,
            device=device,
        ),
        "claim_boundary": (
            "diagnostic only: mean-centered additive log-priors and softmax(prior) are not "
            "the final fusion weights because the learned prior is added to the dynamic mixer state"
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect source offsets learned by the self-calibrating controller")
    parser.add_argument("--root", type=Path, default=Path("."))
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
    parser.add_argument("--controller-holdout-examples-per-operator", type=int, default=16)
    parser.add_argument("--controller-seed", type=int, default=selfcal.DEFAULT_CONTROLLER_SEED)
    parser.add_argument("--controller-holdout-seed", type=int, default=selfcal.DEFAULT_CONTROLLER_HOLDOUT_SEED)
    parser.add_argument("--controller-embedding-size", type=int, default=16)
    parser.add_argument("--controller-hidden-size", type=int, default=16)
    parser.add_argument("--controller-learning-rate", type=float, default=0.03)
    parser.add_argument("--controller-steps", type=int, default=300)
    parser.add_argument("--controller-prior-l2-weight", type=float, default=0.001)
    parser.add_argument("--composition-examples-per-pair", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=sequential.DEFAULT_DATA_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/self_calibrating_operator_controller/prior-diagnostics.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        report = run_diagnostics(
            root=args.root,
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
            controller_seed=args.controller_seed,
            controller_holdout_seed=args.controller_holdout_seed,
            controller_embedding_size=args.controller_embedding_size,
            controller_hidden_size=args.controller_hidden_size,
            controller_learning_rate=args.controller_learning_rate,
            controller_steps=args.controller_steps,
            controller_prior_l2_weight=args.controller_prior_l2_weight,
            composition_examples_per_pair=args.composition_examples_per_pair,
            data_seed=args.data_seed,
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
    print(json.dumps(report["holdout_prompt_prior_diagnostics"], sort_keys=True))
    print(json.dumps(report["composition_prompt_prior_diagnostics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
