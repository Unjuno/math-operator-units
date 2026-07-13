from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from opfusion.tokenizer import FixedVocabTokenizer

from .audit_data import _find_repo_root, _resolve, audit_data as audit_core_data
from .data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from .design_config import load_design_run_config, model_design


def _same_prompt_schema(base: Any, specialist: Any) -> bool:
    return bool(
        base.prompt_tokens
        and specialist.prompt_tokens
        and base.prompt_tokens[0] == specialist.prompt_tokens[0]
        and base.prompt_tokens[-1] == "<RESPONSE>"
        and specialist.prompt_tokens[-1] == "<RESPONSE>"
        and "<TASK_COPY>" not in base.prompt_tokens
        and "<TASK_COPY>" not in specialist.prompt_tokens
    )


def audit_design_data(config_path: str | Path, *, samples_per_operator: int = 512) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    report = audit_core_data(config_path, samples_per_operator=samples_per_operator)
    root = _find_repo_root(config_path.parent)
    run = load_design_run_config(config_path)
    design = model_design(run)
    tokenizer = FixedVocabTokenizer.from_config(_resolve(root, run.tokenizer_config))
    factory = SyntheticTraceFactory(tokenizer, run.data)
    failures = list(report.get("failures", []))
    base_counts: dict[str, int] = {}

    def fail(kind: str, **payload: Any) -> None:
        failures.append({"kind": kind, **payload})

    for operator_index, operator_id in enumerate(EXPERIMENT_OPERATORS):
        for sample_index in range(min(samples_per_operator, 256)):
            kwargs = dict(
                job_id="base.common",
                seed=17,
                split="train",
                step=operator_index * 1_000_000 + sample_index,
                sample_index=sample_index,
                forced_operator=operator_id,
            )
            first = factory.training_example(**kwargs)
            second = factory.training_example(**kwargs)
            if first != second:
                fail("nondeterministic_model_design_base", operator=operator_id, sample=sample_index)
                continue
            base_counts[first.task] = base_counts.get(first.task, 0) + 1
            specialist = factory.training_example(
                operator_id,
                seed=17,
                split="validation",
                step=operator_index * 1_000_000 + sample_index,
                sample_index=sample_index,
                forced_operator=operator_id,
            )
            base_validation = factory.training_example(
                "base.common",
                seed=17,
                split="validation",
                step=operator_index * 1_000_000 + sample_index,
                sample_index=sample_index,
                forced_operator=operator_id,
            )
            if not _same_prompt_schema(base_validation, specialist):
                fail("base_specialist_prefix_schema_mismatch", operator=operator_id)
            expected_ids = tokenizer.encode_tokens(first.response_tokens, add_bos=False, add_eos=True)
            verification = factory.verify_generated_ids(first, expected_ids)
            if not verification.get("valid"):
                fail(
                    "model_design_base_failed_verifier",
                    operator=operator_id,
                    task=first.task,
                    verification=verification,
                )
            if design.base_target_mode == "identity":
                if first.task != "identity_equivalence" or first.final_value is not None:
                    fail("identity_base_contract_violation", operator=operator_id, task=first.task)
            else:
                if first.job_id != "base.common" or first.task not in {"full_trace", "continuation", "terminal_stop"}:
                    fail("weak_base_task_semantics_invalid", operator=operator_id, task=first.task)
                if len(first.initial_values) > design.base_weak_max_terms:
                    fail("weak_base_too_long", operator=operator_id, values=first.initial_values)
                if any(abs(value) > design.base_weak_operand_abs_max for value in first.initial_values):
                    fail("weak_base_operand_out_of_range", operator=operator_id, values=first.initial_values)

    report["model_design"] = design.to_dict()
    report["model_design_base_task_counts"] = dict(sorted(base_counts.items()))
    report["failures"] = failures
    report["status"] = "passed" if not failures else "failed"
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit generated specialist data and the actual configured common-base design"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-operator", type=int, default=512)
    parser.add_argument("--out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.samples_per_operator <= 0:
        parser.error("--samples-per-operator must be positive")
    report = audit_design_data(args.config, samples_per_operator=args.samples_per_operator)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
