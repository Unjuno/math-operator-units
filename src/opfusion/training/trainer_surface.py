from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opfusion.model import GPTModel
from opfusion.tokenizer import FixedVocabTokenizer
from .config import RunConfig, load_run_config
from .data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from . import trainer as core


def _evaluate_loss_fixed(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    *,
    job_id: str,
    seed: int,
    config: RunConfig,
    device: torch.device,
    precision: str,
    micro_batch_size: int,
    split: str = "validation",
) -> dict[str, float]:
    """Evaluate exactly ``config.eval_examples`` examples per target.

    The original implementation used ``eval_batches * selected_micro_batch``
    examples, which made metrics depend on GPU memory. This implementation
    accumulates summed token NLL over a fixed sample count.
    """
    model.eval()
    losses: dict[str, float] = {}
    targets: tuple[str | None, ...] = (None,) if job_id == "base.common" else tuple(EXPERIMENT_OPERATORS)
    with torch.no_grad():
        for target_index, forced_operator in enumerate(targets):
            key = forced_operator or job_id
            consumed = 0
            nll_sum = 0.0
            supervised_tokens = 0
            while consumed < config.eval_examples:
                chunk = min(micro_batch_size, config.eval_examples - consumed)
                input_ids, labels = factory.batch(
                    job_id,
                    seed=seed + 100_000,
                    split=split,
                    step=target_index * 1_000_000,
                    batch_size=chunk,
                    device=device,
                    response_only=config.response_only_loss,
                    sample_offset=consumed,
                    forced_operator=forced_operator,
                )
                with core._autocast(device, precision):
                    logits = model(input_ids)
                    value = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                count = int((labels != -100).sum().item())
                nll_sum += float(value.detach().cpu())
                supervised_tokens += count
                consumed += chunk
            losses[key] = nll_sum / max(1, supervised_tokens)
    model.train()
    losses["mean"] = sum(losses.values()) / max(1, len(losses))
    return losses


def _greedy_until_eos(
    model: GPTModel,
    prompt: list[int],
    *,
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            condition = ids[:, -model.config.max_seq_len :]
            next_id = int(torch.argmax(model(condition)[:, -1, :], dim=-1).item())
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return generated


def _token_accuracy(generated: list[int], expected: list[int]) -> tuple[int, int]:
    width = max(len(generated), len(expected))
    correct = sum(
        int(index < len(generated) and index < len(expected) and generated[index] == expected[index])
        for index in range(width)
    )
    return correct, width


def _evaluate_generation_verified(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    tokenizer: FixedVocabTokenizer,
    *,
    job_id: str,
    seed: int,
    config: RunConfig,
    device: torch.device,
) -> dict[str, Any]:
    if config.generation_eval_examples <= 0 or "<RESPONSE>" not in tokenizer.token_to_id:
        return {}
    model.eval()
    splits = ("validation", "operand_ood", "length_ood")
    targets: tuple[str | None, ...] = (None,) if job_id == "base.common" else tuple(EXPERIMENT_OPERATORS)
    output: dict[str, Any] = {}
    for split in splits:
        split_result: dict[str, Any] = {}
        for target_index, forced_operator in enumerate(targets):
            exact = 0
            final_correct = 0
            final_count = 0
            stop_correct = 0
            trace_valid = 0
            token_correct = 0
            token_count = 0
            generated_lengths = 0
            for sample_index in range(config.generation_eval_examples):
                example = factory.training_example(
                    job_id,
                    seed=seed + 200_000,
                    split=split,
                    step=target_index,
                    sample_index=sample_index,
                    forced_operator=forced_operator,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                if len(prompt) + len(expected) > model.config.max_seq_len:
                    raise ValueError(
                        f"evaluation sequence length {len(prompt) + len(expected)} exceeds context {model.config.max_seq_len}"
                    )
                generated = _greedy_until_eos(
                    model,
                    prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=config.generation_max_new_tokens,
                    device=device,
                )
                generated_lengths += len(generated)
                exact += int(generated == expected)
                correct, count = _token_accuracy(generated, expected)
                token_correct += correct
                token_count += count
                verification = factory.verify_generated_ids(example, generated)
                stop_correct += int(bool(verification.get("stop_correct")))
                trace_valid += int(bool(verification.get("valid")))
                if example.final_value is not None:
                    final_count += 1
                    final_correct += int(bool(verification.get("final_correct")))
            key = forced_operator or job_id
            n = config.generation_eval_examples
            split_result[key] = {
                "response_exact_accuracy": exact / n,
                "response_token_accuracy": token_correct / max(1, token_count),
                "final_value_accuracy": final_correct / max(1, final_count) if final_count else None,
                "stop_accuracy": stop_correct / n,
                "trace_validity_accuracy": trace_valid / n,
                "mean_generated_tokens": generated_lengths / n,
            }
        output[split] = split_result
    model.train()
    return output


def _install_surface_evaluators() -> None:
    core._evaluate_loss = _evaluate_loss_fixed
    core._evaluate_generation = _evaluate_generation_verified


def train_job(**kwargs: Any) -> Path:
    _install_surface_evaluators()
    return core.train_job(**kwargs)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one surface-form GPT operator model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke-test only; production config requires CUDA")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).resolve()
    repo_root = core._find_repo_root(config_path.parent)
    config = load_run_config(config_path)
    final = train_job(
        repo_root=repo_root,
        config=config,
        job_id=args.job,
        seed=args.seed,
        allow_cpu=args.allow_cpu,
    )
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
