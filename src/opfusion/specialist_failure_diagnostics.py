from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from opfusion.fusion_eval import _load_model, _next_logits, _teacher_forced_logits
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory

DEFAULT_OPERATORS = ("aggregation.sum", "scalar.neg")


def _first_divergence(generated: Sequence[int], expected: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(generated, expected)):
        if left != right:
            return index
    return None if len(generated) == len(expected) else min(len(generated), len(expected))


def _decode(tokenizer: FixedVocabTokenizer, ids: Sequence[int]) -> list[str]:
    return [tokenizer.tokens[int(token_id)] for token_id in ids]


def _generate(model: torch.nn.Module, prompt: list[int], *, eos_id: int, max_new_tokens: int, device: torch.device) -> list[int]:
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _next_logits(model, ids)
            next_id = int(torch.argmax(logits, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def diagnose_manifest(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    operators: Sequence[str] = DEFAULT_OPERATORS,
    split: str = "validation",
    evaluation_seed: int = 704_000,
    examples_per_operator: int = 64,
    max_new_tokens: int = 256,
    device_name: str = "auto",
    retain_examples: int = 20,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    run = load_run_config(config_path)
    root = config_path.parents[2]
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active = manifest.get("unit_checkpoints", {})

    unknown = [operator for operator in operators if operator not in EXPERIMENT_OPERATORS]
    if unknown:
        raise ValueError(f"unsupported operators: {unknown}")
    missing = [operator for operator in operators if operator not in active]
    if missing:
        raise RuntimeError(f"manifest has no relevant specialist for: {missing}")
    if examples_per_operator <= 0 or retain_examples < 0:
        raise ValueError("example counts must be nonnegative and examples_per_operator positive")

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    report: dict[str, Any] = {
        "config": str(config_path),
        "manifest": str(manifest_path),
        "experiment_fingerprint": manifest.get("experiment_fingerprint"),
        "split": split,
        "evaluation_seed": evaluation_seed,
        "examples_per_operator": examples_per_operator,
        "max_new_tokens": max_new_tokens,
        "device": str(device),
        "operators": {},
    }

    for operator_index, operator_id in enumerate(operators):
        model = _load_model(active[operator_id], device=device, tokenizer=tokenizer)
        counters = Counter()
        reason_counts: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        for sample_index in range(examples_per_operator):
            example = factory.training_example(
                operator_id,
                seed=evaluation_seed,
                split=split,
                step=operator_index,
                sample_index=sample_index,
            )
            prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
            expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
            generated = _generate(
                model,
                prompt,
                eos_id=tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            verification = factory.verify_generated_ids(example, generated)
            reason = str(verification.get("reason") or "valid")
            reason_counts[reason] += 1
            counters["generated_valid"] += int(bool(verification.get("valid")))
            counters["generated_stop_correct"] += int(bool(verification.get("stop_correct")))
            counters["generated_final_correct"] += int(bool(verification.get("final_correct")))
            counters["generated_exact"] += int(generated == expected)

            sequence = prompt + expected
            input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
            response_start = len(prompt) - 1
            targets = torch.tensor(sequence[1:], dtype=torch.long, device=device)[response_start:]
            with torch.no_grad():
                logits = _teacher_forced_logits(model, input_ids, response_start).squeeze(0)
            predictions = logits.argmax(dim=-1)
            counters["teacher_token_correct"] += int((predictions == targets).sum().item())
            counters["teacher_token_count"] += int(targets.numel())
            counters["teacher_first_correct"] += int(predictions[0].item() == targets[0].item())
            counters["teacher_sequence_exact"] += int(torch.equal(predictions, targets))

            if len(examples) < retain_examples and not verification.get("valid"):
                divergence = _first_divergence(generated, expected)
                examples.append(
                    {
                        "sample_index": sample_index,
                        "task": example.task,
                        "initial_values": list(example.initial_values),
                        "prompt_tokens": list(example.prompt_tokens),
                        "gold_tokens": _decode(tokenizer, expected),
                        "generated_tokens": _decode(tokenizer, generated),
                        "failure_reason": reason,
                        "first_divergence_index": divergence,
                        "generated_length": len(generated),
                        "gold_length": len(expected),
                        "stop_correct": bool(verification.get("stop_correct")),
                        "final_correct": verification.get("final_correct"),
                    }
                )

        n = examples_per_operator
        report["operators"][operator_id] = {
            "checkpoint": active[operator_id],
            "generation": {
                "trace_validity": counters["generated_valid"] / n,
                "stop_accuracy": counters["generated_stop_correct"] / n,
                "final_value_accuracy": counters["generated_final_correct"] / n,
                "exact_response_accuracy": counters["generated_exact"] / n,
                "failure_reasons": dict(sorted(reason_counts.items())),
            },
            "teacher_forced": {
                "token_accuracy": counters["teacher_token_correct"] / max(1, counters["teacher_token_count"]),
                "first_token_accuracy": counters["teacher_first_correct"] / n,
                "sequence_argmax_accuracy": counters["teacher_sequence_exact"] / n,
            },
            "failed_examples": examples,
        }
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose specialist teacher-forcing versus autoregressive failures")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--operators", nargs="+", default=list(DEFAULT_OPERATORS))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--evaluation-seed", type=int, default=704_000)
    parser.add_argument("--examples-per-operator", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--retain-examples", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = diagnose_manifest(
        config_path=args.config,
        manifest_path=args.manifest,
        operators=args.operators,
        split=args.split,
        evaluation_seed=args.evaluation_seed,
        examples_per_operator=args.examples_per_operator,
        max_new_tokens=args.max_new_tokens,
        retain_examples=args.retain_examples,
        device_name=args.device,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
