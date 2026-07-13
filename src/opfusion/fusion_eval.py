from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from opfusion.model import GPTConfig, GPTModel
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


DEFAULT_EVALUATION_SEED = 700_000


def fuse_logits(
    base_logits: torch.Tensor,
    specialist_logits: Sequence[torch.Tensor],
    *,
    mode: str = "raw_sum",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compose specialist logit fields relative to one common base.

    ``raw_sum`` implements ``z_base + alpha * sum(z_k - z_base)``.
    ``bias_mean`` replaces the sum with the mean. An empty subset is exactly
    the base, and a singleton raw sum is exactly the specialist logits.
    """

    if mode not in {"raw_sum", "bias_mean"}:
        raise ValueError(f"unsupported fusion mode: {mode}")
    if not specialist_logits:
        return base_logits
    for logits in specialist_logits:
        if logits.shape != base_logits.shape:
            raise ValueError("all logits must have the same shape")
    biases = torch.stack([logits - base_logits for logits in specialist_logits], dim=0)
    combined = biases.sum(dim=0) if mode == "raw_sum" else biases.mean(dim=0)
    return base_logits + float(alpha) * combined


def center_logit_field(field: torch.Tensor) -> torch.Tensor:
    """Remove the vocabulary-wise additive constant without changing softmax."""

    return field - field.mean(dim=-1, keepdim=True)


def Jensen_shannon_divergence(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    """Return mean Jensen-Shannon divergence over all non-vocabulary axes."""

    if left_logits.shape != right_logits.shape:
        raise ValueError("JSD logits must have identical shapes")
    left_log_p = F.log_softmax(left_logits.float(), dim=-1)
    right_log_p = F.log_softmax(right_logits.float(), dim=-1)
    left_p = left_log_p.exp()
    right_p = right_log_p.exp()
    mean_p = 0.5 * (left_p + right_p)
    mean_log_p = mean_p.clamp_min(torch.finfo(mean_p.dtype).tiny).log()
    left_kl = torch.sum(left_p * (left_log_p - mean_log_p), dim=-1)
    right_kl = torch.sum(right_p * (right_log_p - mean_log_p), dim=-1)
    return (0.5 * (left_kl + right_kl)).mean()


def _load_model(path: str | Path, *, device: torch.device, tokenizer: FixedVocabTokenizer) -> GPTModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("vocab_hash") != tokenizer.vocab_hash:
        raise RuntimeError(f"checkpoint vocabulary hash mismatch: {path}")
    if int(payload.get("vocab_size", -1)) != tokenizer.vocab_size:
        raise RuntimeError(f"checkpoint vocabulary size mismatch: {path}")
    config = GPTConfig.from_dict(payload["model_config"])
    model = GPTModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _next_logits(model: GPTModel, ids: torch.Tensor) -> torch.Tensor:
    condition = ids[:, -model.config.max_seq_len :]
    return model(condition)[:, -1, :]


def _generate_condition(
    *,
    condition: str,
    prompt: list[int],
    base: GPTModel,
    specialists: Sequence[GPTModel],
    relevant_specialist: GPTModel | None,
    joint: GPTModel | None,
    eos_id: int,
    max_new_tokens: int,
    mode: str,
    alpha: float,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if condition == "base":
                logits = _next_logits(base, ids)
            elif condition == "relevant_specialist":
                if relevant_specialist is None:
                    raise ValueError("relevant specialist is not active for this target")
                logits = _next_logits(relevant_specialist, ids)
            elif condition == "joint_reference":
                if joint is None:
                    raise ValueError("joint reference is not available")
                logits = _next_logits(joint, ids)
            elif condition in {"raw_sum", "bias_mean"}:
                base_logits = _next_logits(base, ids)
                unit_logits = [_next_logits(model, ids) for model in specialists]
                logits = fuse_logits(base_logits, unit_logits, mode=condition, alpha=alpha)
            else:
                raise ValueError(condition)
            next_id = int(torch.argmax(logits, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def _teacher_forced_logits(model: GPTModel, input_ids: torch.Tensor, start: int) -> torch.Tensor:
    return model(input_ids)[:, start:, :]


def _token_accuracy(generated: Sequence[int], expected: Sequence[int]) -> tuple[int, int]:
    width = max(len(generated), len(expected))
    correct = sum(
        int(index < len(generated) and index < len(expected) and generated[index] == expected[index])
        for index in range(width)
    )
    return correct, width


def evaluate_manifest(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    split: str = "test",
    examples_per_operator: int = 64,
    max_new_tokens: int = 256,
    alpha: float = 1.0,
    device_name: str = "auto",
    evaluation_seed: int = DEFAULT_EVALUATION_SEED,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    run = load_run_config(config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    if manifest.get("vocab_hash") != tokenizer.vocab_hash:
        raise RuntimeError("manifest vocabulary hash does not match the configured tokenizer")
    if examples_per_operator <= 0 or max_new_tokens <= 0:
        raise ValueError("evaluation sizes must be positive")
    if evaluation_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")
    if split not in {"validation", "test", "iid_test", "operand_ood", "length_ood"}:
        raise ValueError(f"unsupported evaluation split: {split}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    base = _load_model(manifest["base_checkpoint"], device=device, tokenizer=tokenizer)
    active_paths = manifest.get("unit_checkpoints", {})
    active_models = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in active_paths.items()
    }
    matched_joint_path = manifest.get("joint_reference_checkpoint")
    joint = _load_model(matched_joint_path, device=device, tokenizer=tokenizer) if matched_joint_path else None

    conditions = ["base", "raw_sum", "bias_mean"]
    if joint is not None:
        conditions.append("joint_reference")

    results: dict[str, Any] = {}
    for operator_index, operator_id in enumerate(EXPERIMENT_OPERATORS):
        operator_conditions = list(conditions)
        if operator_id in active_models:
            operator_conditions.insert(1, "relevant_specialist")
        counters = {
            condition: {
                "exact": 0,
                "token_correct": 0,
                "token_count": 0,
                "final_correct": 0,
                "final_count": 0,
                "stop_correct": 0,
                "trace_valid": 0,
                "generated_tokens": 0,
                "gold_nll_sum": 0.0,
                "gold_token_count": 0,
                "joint_jsd_sum": 0.0,
                "joint_argmax_agreement": 0,
                "joint_position_count": 0,
            }
            for condition in operator_conditions
        }

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
            sequence = prompt + expected
            input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor(sequence[1:], dtype=torch.long, device=device)
            response_start = len(prompt) - 1

            with torch.no_grad():
                base_tf = _teacher_forced_logits(base, input_ids, response_start)
                specialist_tf = {
                    name: _teacher_forced_logits(model, input_ids, response_start)
                    for name, model in active_models.items()
                }
                joint_tf = _teacher_forced_logits(joint, input_ids, response_start) if joint is not None else None
                fused_tf = {
                    mode: fuse_logits(base_tf, list(specialist_tf.values()), mode=mode, alpha=alpha)
                    for mode in ("raw_sum", "bias_mean")
                }
                condition_tf: dict[str, torch.Tensor] = {
                    "base": base_tf,
                    "raw_sum": fused_tf["raw_sum"],
                    "bias_mean": fused_tf["bias_mean"],
                }
                if operator_id in specialist_tf:
                    condition_tf["relevant_specialist"] = specialist_tf[operator_id]
                if joint_tf is not None:
                    condition_tf["joint_reference"] = joint_tf

            gold = targets[response_start:]
            for condition in operator_conditions:
                generated = _generate_condition(
                    condition=condition,
                    prompt=prompt,
                    base=base,
                    specialists=list(active_models.values()),
                    relevant_specialist=active_models.get(operator_id),
                    joint=joint,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    mode=condition if condition in {"raw_sum", "bias_mean"} else "raw_sum",
                    alpha=alpha,
                    device=device,
                )
                stats = counters[condition]
                stats["exact"] += int(generated == expected)
                correct, count = _token_accuracy(generated, expected)
                stats["token_correct"] += correct
                stats["token_count"] += count
                stats["generated_tokens"] += len(generated)
                verification = factory.verify_generated_ids(example, generated)
                stats["stop_correct"] += int(bool(verification.get("stop_correct")))
                stats["trace_valid"] += int(bool(verification.get("valid")))
                if example.final_value is not None:
                    stats["final_count"] += 1
                    stats["final_correct"] += int(bool(verification.get("final_correct")))

                logits = condition_tf[condition]
                nll = F.cross_entropy(logits.squeeze(0), gold, reduction="sum")
                stats["gold_nll_sum"] += float(nll.detach().cpu())
                stats["gold_token_count"] += int(gold.numel())
                if joint_tf is not None and condition != "joint_reference":
                    stats["joint_jsd_sum"] += float(Jensen_shannon_divergence(logits, joint_tf).detach().cpu())
                    stats["joint_argmax_agreement"] += int((logits.argmax(dim=-1) == joint_tf.argmax(dim=-1)).sum().item())
                    stats["joint_position_count"] += int(gold.numel())

        operator_result: dict[str, Any] = {}
        for condition, stats in counters.items():
            n = examples_per_operator
            operator_result[condition] = {
                "response_exact_accuracy": stats["exact"] / n,
                "response_token_accuracy": stats["token_correct"] / max(1, stats["token_count"]),
                "final_value_accuracy": stats["final_correct"] / max(1, stats["final_count"]) if stats["final_count"] else None,
                "stop_accuracy": stats["stop_correct"] / n,
                "trace_validity_accuracy": stats["trace_valid"] / n,
                "mean_generated_tokens": stats["generated_tokens"] / n,
                "gold_token_nll": stats["gold_nll_sum"] / max(1, stats["gold_token_count"]),
                "joint_jsd": stats["joint_jsd_sum"] / n if stats["joint_position_count"] else None,
                "joint_argmax_agreement": stats["joint_argmax_agreement"] / max(1, stats["joint_position_count"]) if stats["joint_position_count"] else None,
            }
        results[operator_id] = operator_result

    return {
        "experiment_id": run.experiment_id,
        "manifest": str(Path(manifest_path).resolve()),
        "subset_id": manifest.get("subset_id"),
        "active_operators": list(active_models),
        "joint_reference_status": manifest.get("joint_reference_status"),
        "split": split,
        "examples_per_operator": examples_per_operator,
        "evaluation_seed": evaluation_seed,
        "alpha": alpha,
        "device": str(device),
        "results": results,
        "claim_boundary": manifest.get("claim_boundary"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate base-relative logit fusion for one generated subset manifest")
    parser.add_argument("--config", default="configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples-per-operator", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = evaluate_manifest(
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
