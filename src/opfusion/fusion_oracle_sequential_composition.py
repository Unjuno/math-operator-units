from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_dual_timescale_init_ensemble as ensemble
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_sparse_valid import SparseValidBatch, _collect_valid_batch
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import OPERATOR_TOKENS, SyntheticTraceFactory


FUNCTIONAL_OPERATORS: tuple[str, ...] = (
    "scalar.add",
    "aggregation.sum",
    "scalar.min",
    "scalar.max",
)
DEFAULT_CALIBRATION_SEED = 731_000
DEFAULT_DATA_SEED = 733_000


def apply_operator(operator: str, values: Sequence[int]) -> int:
    if operator == "scalar.add":
        if len(values) != 2:
            raise ValueError("scalar.add requires exactly two values")
        return int(values[0] + values[1])
    if operator == "aggregation.sum":
        if len(values) < 2:
            raise ValueError("aggregation.sum requires at least two values")
        return int(sum(values))
    if operator == "scalar.min":
        if len(values) < 2:
            raise ValueError("scalar.min requires at least two values")
        return int(min(values))
    if operator == "scalar.max":
        if len(values) < 2:
            raise ValueError("scalar.max requires at least two values")
        return int(max(values))
    raise KeyError(operator)


def parse_final_numeric_token(
    generated: Sequence[int], tokenizer: FixedVocabTokenizer
) -> int | None:
    """Return the final atomic integer emitted before EOS, if one exists."""
    final: int | None = None
    for token_id in generated:
        if token_id == tokenizer.eos_id:
            break
        if token_id < 0 or token_id >= tokenizer.vocab_size:
            return None
        token = tokenizer.tokens[token_id]
        if token.startswith("<N_") and token.endswith(">"):
            try:
                final = int(token[3:-1])
            except ValueError:
                return None
    return final


def prompt_ids_for_values(
    *,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    operator: str,
    values: Sequence[int],
) -> list[int]:
    tokens = [
        OPERATOR_TOKENS[operator],
        *factory._state_tokens(operator, values),
        "<RESPONSE>",
    ]
    return tokenizer.encode_tokens(tokens, add_bos=True, add_eos=False)


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def composition_operands(
    *, inner_operator: str, outer_operator: str, seed: int, sample_index: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rng = random.Random(
        _stable_seed(
            "oracle-sequential-composition-v1",
            seed,
            sample_index,
            inner_operator,
            outer_operator,
        )
    )
    inner_count = 2 if inner_operator == "scalar.add" else 3
    outer_extra_count = 1 if outer_operator == "scalar.add" else 2
    inner_values = tuple(rng.randint(-16, 16) for _ in range(inner_count))
    outer_extras = tuple(rng.randint(-16, 16) for _ in range(outer_extra_count))
    return inner_values, outer_extras


def _empty_counter() -> dict[str, int]:
    return {
        "cases": 0,
        "inner_parse": 0,
        "inner_correct": 0,
        "oracle_intermediate_outer_parse": 0,
        "oracle_intermediate_outer_correct": 0,
        "end_to_end_outer_parse": 0,
        "end_to_end_correct": 0,
        "end_to_end_correct_given_inner_correct": 0,
        "inner_correct_cases": 0,
    }


def _merge_counter(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def _finalize_counter(counter: Mapping[str, int]) -> dict[str, float | int | None]:
    cases = int(counter["cases"])
    inner_correct_cases = int(counter["inner_correct_cases"])
    return {
        **{key: int(value) for key, value in counter.items()},
        "inner_parse_rate": counter["inner_parse"] / max(1, cases),
        "inner_accuracy": counter["inner_correct"] / max(1, cases),
        "oracle_intermediate_outer_parse_rate": counter[
            "oracle_intermediate_outer_parse"
        ]
        / max(1, cases),
        "oracle_intermediate_outer_accuracy": counter[
            "oracle_intermediate_outer_correct"
        ]
        / max(1, cases),
        "end_to_end_outer_parse_rate": counter["end_to_end_outer_parse"]
        / max(1, cases),
        "end_to_end_accuracy": counter["end_to_end_correct"] / max(1, cases),
        "end_to_end_accuracy_given_inner_correct": (
            counter["end_to_end_correct_given_inner_correct"] / inner_correct_cases
            if inner_correct_cases
            else None
        ),
    }


def _generate_value(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    candidate,
    operator: str,
    values: Sequence[int],
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[int | None, dict[str, float]]:
    prompt = prompt_ids_for_values(
        factory=factory,
        tokenizer=tokenizer,
        operator=operator,
        values=values,
    )
    generated, diagnostics = oracle._generate_oracle_operator(
        base=base,
        units=units,
        mixer=mixer,
        candidate=candidate,
        operator=operator,
        prompt=prompt,
        eos_id=tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return parse_final_numeric_token(generated, tokenizer), diagnostics


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    mixer: torch.nn.Module,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    candidate = confirmatory.candidate_grid()[1]
    aggregate = _empty_counter()
    pair_rows: dict[str, dict[str, float | int | None]] = {}
    matching_weight_sum = 0.0
    matching_weight_positions = 0

    for inner_operator in FUNCTIONAL_OPERATORS:
        for outer_operator in FUNCTIONAL_OPERATORS:
            pair_id = f"{inner_operator}->{outer_operator}"
            counter = _empty_counter()
            for sample_index in range(examples_per_pair):
                inner_values, outer_extras = composition_operands(
                    inner_operator=inner_operator,
                    outer_operator=outer_operator,
                    seed=data_seed,
                    sample_index=sample_index,
                )
                true_inner = apply_operator(inner_operator, inner_values)
                true_outer_values = (true_inner, *outer_extras)
                true_final = apply_operator(outer_operator, true_outer_values)

                generated_inner, inner_diag = _generate_value(
                    base=base,
                    units=units,
                    mixer=mixer,
                    candidate=candidate,
                    operator=inner_operator,
                    values=inner_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                oracle_outer, oracle_outer_diag = _generate_value(
                    base=base,
                    units=units,
                    mixer=mixer,
                    candidate=candidate,
                    operator=outer_operator,
                    values=true_outer_values,
                    factory=factory,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )

                chained_outer: int | None = None
                chained_diag: dict[str, float] | None = None
                if generated_inner is not None:
                    chained_outer, chained_diag = _generate_value(
                        base=base,
                        units=units,
                        mixer=mixer,
                        candidate=candidate,
                        operator=outer_operator,
                        values=(generated_inner, *outer_extras),
                        factory=factory,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        device=device,
                    )

                inner_correct = generated_inner == true_inner
                counter["cases"] += 1
                counter["inner_parse"] += int(generated_inner is not None)
                counter["inner_correct"] += int(inner_correct)
                counter["oracle_intermediate_outer_parse"] += int(
                    oracle_outer is not None
                )
                counter["oracle_intermediate_outer_correct"] += int(
                    oracle_outer == true_final
                )
                counter["end_to_end_outer_parse"] += int(chained_outer is not None)
                counter["end_to_end_correct"] += int(chained_outer == true_final)
                if inner_correct:
                    counter["inner_correct_cases"] += 1
                    counter["end_to_end_correct_given_inner_correct"] += int(
                        chained_outer == true_final
                    )

                for row in (inner_diag, oracle_outer_diag, chained_diag):
                    if row is not None:
                        matching_weight_sum += float(row["mean_oracle_source_weight"])
                        matching_weight_positions += 1

            _merge_counter(aggregate, counter)
            pair_rows[pair_id] = _finalize_counter(counter)

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "candidate": candidate.__dict__,
        "aggregate": _finalize_counter(aggregate),
        "pairs": pair_rows,
        "mean_matching_source_weight": matching_weight_sum
        / max(1, matching_weight_positions),
    }


def fit_ensemble(
    cohorts: Sequence[Cohort],
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
    device: torch.device,
):
    batches = [
        _collect_valid_batch(
            cohort,
            root=root,
            examples_per_operator=calibration_examples_per_operator,
            data_seed=calibration_seed,
            max_prefixes_per_example=max_prefixes_per_example,
            max_positions=max_positions_per_cohort,
            device=device,
        )[0]
        for cohort in cohorts[:2]
    ]
    calibration = SparseValidBatch(
        base_logits=torch.cat([batch.base_logits for batch in batches], dim=0),
        unit_logits=torch.cat([batch.unit_logits for batch in batches], dim=0),
        valid_mask=torch.cat([batch.valid_mask for batch in batches], dim=0),
    )
    mixer, fit_report = ensemble.fit_mixer_ensemble(
        batch=calibration,
        vocabulary_size=int(calibration.base_logits.shape[-1]),
        hidden_size=hidden_size,
        sketch_size=sketch_size,
        temperature=0.75,
        target_temperature=0.35,
        source_supervision_weight=1.0,
        entropy_target=math.log(2.0),
        entropy_penalty=0.2,
        l2_weight=0.001,
        learning_rate=learning_rate,
        steps=fit_steps,
        batch_positions=fit_batch_positions,
        seed=calibration_seed,
    )
    mixer.eval()
    return mixer, fit_report


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

    mixer, fit_report = fit_ensemble(
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
    cohort_reports = [
        evaluate_cohort(
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

    aggregate = _empty_counter()
    pair_aggregate = {
        f"{inner}->{outer}": _empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    mean_weights: list[float] = []
    for report in cohort_reports:
        raw_aggregate = {
            key: int(report["aggregate"][key]) for key in _empty_counter()
        }
        _merge_counter(aggregate, raw_aggregate)
        mean_weights.append(float(report["mean_matching_source_weight"]))
        for pair_id, row in report["pairs"].items():
            raw_pair = {key: int(row[key]) for key in _empty_counter()}
            _merge_counter(pair_aggregate[pair_id], raw_pair)

    strength = float(os.environ.get(oracle.ENV_ORACLE_STRENGTH, "0.0"))
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_oracle_sequential_composition",
        "claim_boundary": (
            "two separately generated stages with an externally supplied operator at "
            "each stage; stage state is reset at the boundary; no nested single-pass "
            "execution, learned controller, NEG, final IID test, or OOD split is tested"
        ),
        "operators": list(FUNCTIONAL_OPERATORS),
        "ordered_pair_count": len(pair_aggregate),
        "examples_per_pair_per_cohort": examples_per_pair,
        "model_cohort_count": len(cohort_reports),
        "oracle_operator_strength": strength,
        "data_seed": data_seed,
        "calibration_seed": calibration_seed,
        "fixed_candidate": confirmatory.candidate_grid()[1].__dict__,
        "mixer_fit": fit_report,
        "cohort_reports": cohort_reports,
        "aggregate": _finalize_counter(aggregate),
        "pairs": {
            pair_id: _finalize_counter(counter)
            for pair_id, counter in pair_aggregate.items()
        },
        "mean_matching_source_weight": sum(mean_weights) / max(1, len(mean_weights)),
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage oracle-controlled operator composition"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--examples-per-pair", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--calibration-examples-per-operator", type=int, default=8)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1536)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sketch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="evaluations/oracle_sequential_composition/summary.json"
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
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
