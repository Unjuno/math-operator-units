from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opfusion.fusion_eval import (
    Jensen_shannon_divergence,
    _load_model,
    _teacher_forced_logits,
    center_logit_field,
)
from opfusion.fusion_eval_seeded import (
    DEFAULT_FINAL_EVALUATION_SEED,
    DEFAULT_PILOT_EVALUATION_SEED,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


def _base_to_unit_kl(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
    base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
    unit_log_probs = F.log_softmax(unit_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    return (base_probs * (base_log_probs - unit_log_probs)).sum(dim=-1).mean()


def evaluate_unit_diagnostics(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    split: str = "validation",
    examples_per_operator: int = 64,
    device_name: str = "auto",
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[2]
    run = load_run_config(config_path)
    resolved_seed = (
        DEFAULT_PILOT_EVALUATION_SEED
        if evaluation_seed is None and run.experiment_id.startswith("model_design_pilot_")
        else DEFAULT_FINAL_EVALUATION_SEED
        if evaluation_seed is None
        else evaluation_seed
    )
    if resolved_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")
    tokenizer = FixedVocabTokenizer.from_config(repo_root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("vocab_hash") != tokenizer.vocab_hash:
        raise RuntimeError("manifest vocabulary hash does not match the configured tokenizer")
    if examples_per_operator <= 0:
        raise ValueError("examples_per_operator must be positive")
    if split not in {"validation", "test", "iid_test", "operand_ood", "length_ood"}:
        raise ValueError(f"unsupported diagnostic split: {split}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    base = _load_model(manifest["base_checkpoint"], device=device, tokenizer=tokenizer)
    unit_paths = manifest.get("unit_checkpoints", {})
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in unit_paths.items()
    }
    missing = [operator for operator in EXPERIMENT_OPERATORS if operator not in units]
    if missing:
        raise RuntimeError(f"all-five diagnostics require every specialist; missing={missing}")

    results: dict[str, Any] = {}
    with torch.no_grad():
        for target_index, target_operator in enumerate(EXPERIMENT_OPERATORS):
            counters = {
                unit_operator: {
                    "examples": 0,
                    "response_positions": 0,
                    "jsd_sum": 0.0,
                    "base_to_unit_kl_sum": 0.0,
                    "argmax_agreement": 0,
                    "centered_bias_sq_sum": 0.0,
                    "centered_bias_elements": 0,
                    "centered_bias_max_abs": 0.0,
                }
                for unit_operator in EXPERIMENT_OPERATORS
            }
            for sample_index in range(examples_per_operator):
                example = factory.training_example(
                    target_operator,
                    seed=resolved_seed,
                    split=split,
                    step=target_index,
                    sample_index=sample_index,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                sequence = prompt + expected
                input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                response_start = len(prompt) - 1
                base_logits = _teacher_forced_logits(base, input_ids, response_start)
                for unit_operator, model in units.items():
                    unit_logits = _teacher_forced_logits(model, input_ids, response_start)
                    centered = center_logit_field(unit_logits - base_logits)
                    row = counters[unit_operator]
                    row["examples"] += 1
                    positions = int(unit_logits.shape[-2])
                    row["response_positions"] += positions
                    row["jsd_sum"] += float(
                        Jensen_shannon_divergence(base_logits, unit_logits).detach().cpu()
                    )
                    row["base_to_unit_kl_sum"] += float(
                        _base_to_unit_kl(base_logits, unit_logits).detach().cpu()
                    )
                    row["argmax_agreement"] += int(
                        (base_logits.argmax(dim=-1) == unit_logits.argmax(dim=-1)).sum().item()
                    )
                    row["centered_bias_sq_sum"] += float(centered.float().pow(2).sum().detach().cpu())
                    row["centered_bias_elements"] += int(centered.numel())
                    row["centered_bias_max_abs"] = max(
                        float(row["centered_bias_max_abs"]),
                        float(centered.float().abs().max().detach().cpu()),
                    )

            operator_result: dict[str, Any] = {}
            inactive_jsd: list[float] = []
            inactive_bias_rms: list[float] = []
            for unit_operator, row in counters.items():
                n = max(1, int(row["examples"]))
                positions = max(1, int(row["response_positions"]))
                elements = max(1, int(row["centered_bias_elements"]))
                jsd = float(row["jsd_sum"]) / n
                bias_rms = math.sqrt(float(row["centered_bias_sq_sum"]) / elements)
                relation = "relevant" if unit_operator == target_operator else "inactive"
                operator_result[unit_operator] = {
                    "relation": relation,
                    "mean_base_unit_jsd": jsd,
                    "mean_base_to_unit_kl": float(row["base_to_unit_kl_sum"]) / n,
                    "base_unit_argmax_agreement": int(row["argmax_agreement"]) / positions,
                    "centered_bias_rms": bias_rms,
                    "centered_bias_max_abs": float(row["centered_bias_max_abs"]),
                    "examples": int(row["examples"]),
                    "response_positions": int(row["response_positions"]),
                }
                if relation == "inactive":
                    inactive_jsd.append(jsd)
                    inactive_bias_rms.append(bias_rms)
            operator_result["inactive_summary"] = {
                "mean_jsd": sum(inactive_jsd) / max(1, len(inactive_jsd)),
                "max_jsd": max(inactive_jsd, default=0.0),
                "mean_centered_bias_rms": sum(inactive_bias_rms) / max(1, len(inactive_bias_rms)),
                "max_centered_bias_rms": max(inactive_bias_rms, default=0.0),
            }
            results[target_operator] = operator_result

    return {
        "experiment_id": run.experiment_id,
        "manifest": str(manifest_path),
        "subset_id": manifest.get("subset_id"),
        "experiment_fingerprint": manifest.get("experiment_fingerprint"),
        "split": split,
        "examples_per_operator": examples_per_operator,
        "evaluation_seed": resolved_seed,
        "evaluation_role": (
            "model_design_development"
            if resolved_seed == DEFAULT_PILOT_EVALUATION_SEED
            else "final_or_user_selected"
        ),
        "device": str(device),
        "metric_scope": "teacher-forced response positions on canonical target traces",
        "results": results,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure relevant and inactive specialist drift relative to the common base"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--examples-per-operator", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate_unit_diagnostics(
        config_path=args.config,
        manifest_path=args.manifest,
        split=args.split,
        examples_per_operator=args.examples_per_operator,
        evaluation_seed=args.evaluation_seed,
        device_name=args.device,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
