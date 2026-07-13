from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from opfusion.model import load_config
from opfusion.tokenizer import FixedVocabTokenizer
from .config import load_run_config
from .data import EXPERIMENT_OPERATORS, OPERAND_OOD_SPLITS, SyntheticTraceFactory


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("could not locate repository root containing pyproject.toml")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _canonical_ids(tokenizer: FixedVocabTokenizer, tokens: Iterable[str]) -> list[int]:
    return tokenizer.encode_tokens(tokens, add_bos=False, add_eos=False)


def audit_data(config_path: str | Path, *, samples_per_operator: int = 512) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = _find_repo_root(config_path.parent)
    run = load_run_config(config_path)
    tokenizer = FixedVocabTokenizer.from_config(_resolve(root, run.tokenizer_config))
    model_config = load_config(_resolve(root, run.model_config))
    factory = SyntheticTraceFactory(tokenizer, run.data)

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    max_sequence_length = 0
    split_keys: dict[str, set[tuple[str, tuple[int, ...]]]] = defaultdict(set)
    task_counts: Counter[str] = Counter()
    non_left_reductions = 0
    generated_examples = 0

    def fail(kind: str, **payload: Any) -> None:
        failures.append({"kind": kind, **payload})

    splits = ("train", "validation", "test", "operand_ood", "length_ood")
    for operator_index, operator_id in enumerate(EXPERIMENT_OPERATORS):
        for split_index, split in enumerate(splits):
            for sample_index in range(samples_per_operator):
                kwargs = dict(
                    job_id=operator_id,
                    seed=91,
                    split=split,
                    step=operator_index * 10_000_000 + split_index * 1_000_000 + sample_index,
                    sample_index=sample_index,
                )
                first = factory.training_example(**kwargs)
                second = factory.training_example(**kwargs)
                generated_examples += 1
                if first != second:
                    fail("nondeterministic_example", operator=operator_id, split=split, sample=sample_index)
                    continue
                task_counts[first.task] += int(split == "train")
                if split in {"train", "validation", "test"}:
                    if first.partition_bucket is None:
                        fail("missing_partition_bucket", operator=operator_id, split=split)
                    elif not factory._bucket_matches(split, first.partition_bucket):
                        fail(
                            "wrong_partition_bucket",
                            operator=operator_id,
                            split=split,
                            bucket=first.partition_bucket,
                        )
                    split_keys[split].add((operator_id, first.initial_values))
                if split in OPERAND_OOD_SPLITS:
                    lower = run.data.value_ood_abs_min
                    if any(abs(value) < lower for value in first.initial_values):
                        fail("operand_ood_inside_train_range", operator=operator_id, values=first.initial_values)
                if split == "length_ood" and operator_id not in {"scalar.add", "scalar.neg"}:
                    if len(first.initial_values) < run.data.length_ood_min_terms:
                        fail("length_ood_too_short", operator=operator_id, values=first.initial_values)

                unknown = [token for token in first.all_tokens if token not in tokenizer.token_to_id]
                if unknown:
                    fail("unknown_tokens", operator=operator_id, split=split, tokens=unknown)
                encoded = factory.encode_training_example(first, response_only=run.response_only_loss)
                max_sequence_length = max(max_sequence_length, len(encoded.input_ids) + 1)
                if len(encoded.input_ids) > model_config.max_seq_len:
                    fail(
                        "context_overflow",
                        operator=operator_id,
                        split=split,
                        length=len(encoded.input_ids),
                        context=model_config.max_seq_len,
                    )
                if run.response_only_loss:
                    first_supervised = encoded.prompt_length - 1
                    if any(label != -100 for label in encoded.labels[:first_supervised]):
                        fail("prompt_label_leakage", operator=operator_id, split=split)
                    if encoded.labels[first_supervised] == -100:
                        fail("first_response_token_not_supervised", operator=operator_id, split=split)

                expected_ids = tokenizer.encode_tokens(first.response_tokens, add_bos=False, add_eos=True)
                verification = factory.verify_generated_ids(first, expected_ids)
                if not verification.get("valid"):
                    fail(
                        "expected_trace_failed_verifier",
                        operator=operator_id,
                        split=split,
                        task=first.task,
                        verification=verification,
                    )

                if split == "train" and operator_id in {"aggregation.sum", "scalar.min", "scalar.max"}:
                    states = first.trace_states
                    if len(states) >= 2 and len(states[0]) > 2:
                        before, after = states[0], states[1]
                        left_candidate = None
                        if operator_id == "aggregation.sum":
                            left_candidate = (before[0] + before[1], *before[2:])
                        elif operator_id == "scalar.min":
                            left_candidate = (min(before[0], before[1]), *before[2:])
                        else:
                            left_candidate = (max(before[0], before[1]), *before[2:])
                        non_left_reductions += int(tuple(after) != tuple(left_candidate))

    overlaps = {
        "train_validation": len(split_keys["train"] & split_keys["validation"]),
        "train_test": len(split_keys["train"] & split_keys["test"]),
        "validation_test": len(split_keys["validation"] & split_keys["test"]),
    }
    for name, count in overlaps.items():
        if count:
            fail("iid_split_overlap", pair=name, count=count)

    surface_mode = factory.eq_canonical == "=" and not factory.explicit_stop
    if "surface" in tokenizer.profile:
        if not surface_mode:
            fail(
                "surface_policy_not_active",
                equality_token=factory.eq_canonical,
                explicit_stop=factory.explicit_stop,
            )
        if "<EQ_STEP>" in tokenizer.tokens or "<TRACE_STOP>" in tokenizer.tokens:
            fail("typed_control_tokens_in_surface_vocab")
        if tokenizer.aliases.get("<TRACE_STOP>") != "<EOS>":
            fail("surface_stop_alias_not_eos")

    train_total = sum(task_counts.values())
    observed_ratios = {key: value / max(1, train_total) for key, value in sorted(task_counts.items())}
    expected_raw = {
        "full_trace": run.data.full_trace_weight,
        "continuation": run.data.continuation_weight,
        "terminal_stop": run.data.terminal_weight,
    }
    expected_total = sum(expected_raw.values())
    expected_ratios = {key: value / expected_total for key, value in expected_raw.items()}
    # Scalar jobs cannot produce a continuation view; audit the aggregate of all
    # operators with a deliberately broad tolerance and report exact counts.
    for key, expected in expected_ratios.items():
        observed = observed_ratios.get(key, 0.0)
        if abs(observed - expected) > 0.15:
            warnings.append(
                {
                    "kind": "trace_view_ratio_deviation",
                    "task": key,
                    "expected": expected,
                    "observed": observed,
                }
            )
    if run.data.randomized_train_reduction and non_left_reductions == 0:
        fail("no_non_left_training_reductions_observed")

    # Base protocol audit: it must teach ordinary equality/EOS syntax without an
    # arithmetic answer transition.
    for sample_index in range(min(samples_per_operator, 128)):
        example = factory.training_example(
            "base.common",
            seed=17,
            split="train",
            step=sample_index,
            sample_index=sample_index,
        )
        if example.task != "identity_equivalence":
            fail("base_wrong_task", task=example.task)
        if example.final_value is not None:
            fail("base_contains_arithmetic_target", value=example.final_value)
        expected_ids = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
        verification = factory.verify_generated_ids(example, expected_ids)
        if not verification.get("valid"):
            fail("base_protocol_failed_verifier", verification=verification)

    report = {
        "status": "passed" if not failures else "failed",
        "experiment_id": run.experiment_id,
        "tokenizer_profile": tokenizer.profile,
        "vocab_size": tokenizer.vocab_size,
        "vocab_hash": tokenizer.vocab_hash,
        "model_context": model_config.max_seq_len,
        "samples_per_operator_per_split": samples_per_operator,
        "generated_examples": generated_examples,
        "max_sequence_length": max_sequence_length,
        "surface_policy": {
            "active": surface_mode,
            "canonical_equality_token": factory.eq_canonical,
            "explicit_trace_stop": factory.explicit_stop,
            "termination_token": "<EOS>" if not factory.explicit_stop else "<TRACE_STOP>",
        },
        "iid_split_unique_counts": {key: len(value) for key, value in split_keys.items()},
        "iid_split_overlaps": overlaps,
        "train_task_counts": dict(sorted(task_counts.items())),
        "train_task_ratios": observed_ratios,
        "non_left_reductions_observed": non_left_reductions,
        "failures": failures,
        "warnings": warnings,
    }
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deterministic generated data before a long operator-model run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-operator", type=int, default=512)
    parser.add_argument("--out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.samples_per_operator <= 0:
        parser.error("--samples-per-operator must be positive")
    report = audit_data(args.config, samples_per_operator=args.samples_per_operator)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
