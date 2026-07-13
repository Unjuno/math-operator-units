from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer
from .config import RunConfig, load_run_config
from .data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


class NonFiniteTrainingError(RuntimeError):
    """Raised after preserving the last good checkpoint for scheduler recovery."""


@dataclass
class RuntimeState:
    micro_batch_size: int
    lr_scale: float = 1.0
    oom_reductions: int = 0
    non_finite_restarts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_size": self.micro_batch_size,
            "lr_scale": self.lr_scale,
            "oom_reductions": self.oom_reductions,
            "non_finite_restarts": self.non_finite_restarts,
        }


@dataclass(frozen=True)
class StepResult:
    loss: float
    grad_norm: float
    micro_batch_size: int
    supervised_examples: int
    per_operator_examples: dict[str, int]


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _jsonl_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("could not locate repository root containing pyproject.toml")


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _set_seed(seed: int, deterministic_algorithms: bool) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=not deterministic_algorithms)


def _device(config: RunConfig, allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if config.require_cuda and not allow_cpu:
        raise RuntimeError("CUDA is required by this experiment configuration, but torch.cuda.is_available() is false")
    return torch.device("cpu")


def _resolve_precision(config: RunConfig, device: torch.device) -> str:
    if config.precision != "auto":
        value = config.precision
    else:
        value = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if value == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 was requested but the CUDA device does not report BF16 support")
    return value


def _configure_cuda(config: RunConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def _learning_rate(step: int, config: RunConfig, lr_scale: float = 1.0) -> float:
    opt = config.optimizer
    if step < opt.warmup_steps:
        base = opt.learning_rate * float(step + 1) / float(max(1, opt.warmup_steps))
    else:
        progress = (step - opt.warmup_steps) / float(max(1, config.max_steps - opt.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        base = opt.min_learning_rate + cosine * (opt.learning_rate - opt.min_learning_rate)
    return base * lr_scale


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _delta_summary(initial: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> dict[str, Any]:
    diff_sq = 0.0
    init_sq = 0.0
    current_sq = 0.0
    dot = 0.0
    groups: dict[str, dict[str, float]] = {}
    for name, initial_tensor in initial.items():
        current_tensor = current[name]
        if not torch.is_floating_point(initial_tensor):
            continue
        if (
            name == "lm_head.weight"
            and "token_embedding.weight" in initial
            and torch.equal(initial_tensor, initial["token_embedding.weight"])
            and torch.equal(current_tensor, current["token_embedding.weight"])
        ):
            continue
        a = initial_tensor.double().reshape(-1)
        b = current_tensor.double().reshape(-1)
        d = b - a
        diff_sq += float(torch.dot(d, d))
        init_sq += float(torch.dot(a, a))
        current_sq += float(torch.dot(b, b))
        dot += float(torch.dot(a, b))
        prefix = name.split(".", 1)[0]
        group = groups.setdefault(prefix, {"delta_sq": 0.0, "initial_sq": 0.0})
        group["delta_sq"] += float(torch.dot(d, d))
        group["initial_sq"] += float(torch.dot(a, a))
    initial_norm = math.sqrt(init_sq)
    current_norm = math.sqrt(current_sq)
    delta_norm = math.sqrt(diff_sq)
    cosine = dot / max(initial_norm * current_norm, 1e-30)
    return {
        "initial_to_current_l2": delta_norm,
        "relative_initial_to_current_l2": delta_norm / max(initial_norm, 1e-30),
        "initial_current_cosine": cosine,
        "parameter_groups": {
            name: {
                "delta_l2": math.sqrt(values["delta_sq"]),
                "relative_delta_l2": math.sqrt(values["delta_sq"]) / max(math.sqrt(values["initial_sq"]), 1e-30),
            }
            for name, values in sorted(groups.items())
        },
    }


def _stable_job_seed(seed: int, job_id: str) -> int:
    digest = hashlib.blake2b(job_id.encode("utf-8"), digest_size=4).digest()
    return seed + 1_000_000 + int.from_bytes(digest, "big")


def _read_runtime_state(path: Path, config: RunConfig) -> RuntimeState:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeState(
            micro_batch_size=int(raw.get("micro_batch_size", 0)),
            lr_scale=float(raw.get("lr_scale", 1.0)),
            oom_reductions=int(raw.get("oom_reductions", 0)),
            non_finite_restarts=int(raw.get("non_finite_restarts", 0)),
        )
    return RuntimeState(micro_batch_size=config.micro_batch_size)


def _write_recovery_event(job_dir: Path, event: dict[str, Any]) -> None:
    _jsonl_append(job_dir / "recovery.jsonl", {**event, "unix": time.time()})


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None], device: torch.device) -> None:
    torch.set_rng_state(state[0].cpu())
    if device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in text or "out of memory" in text


def _probe_micro_batch_size(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    *,
    job_id: str,
    seed: int,
    config: RunConfig,
    device: torch.device,
    precision: str,
) -> int:
    if config.micro_batch_size > 0:
        return min(config.micro_batch_size, config.effective_batch_size)
    if device.type != "cuda":
        return min(config.effective_batch_size, max(config.micro_batch_candidates))
    candidates = sorted(
        {
            min(int(value), config.effective_batch_size)
            for value in config.micro_batch_candidates
            if int(value) >= config.recovery.minimum_micro_batch_size
        },
        reverse=True,
    )
    if not candidates:
        raise ValueError("no valid micro-batch candidates")
    rng = _capture_rng(device)
    probe_operator = "aggregation.sum" if job_id != "base.common" else None
    for candidate in candidates:
        try:
            model.zero_grad(set_to_none=True)
            input_ids, labels = factory.batch(
                job_id,
                seed=seed + 700_000,
                split="train",
                step=-1,
                batch_size=candidate,
                device=device,
                response_only=config.response_only_loss,
                forced_operator=probe_operator,
            )
            with _autocast(device, precision):
                probe_loss = _loss(model(input_ids), labels)
            probe_loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            _restore_rng(rng, device)
            return candidate
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if not _is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            _restore_rng(rng, device)
    raise RuntimeError(
        f"no micro-batch candidate fit the GPU; minimum candidate was {min(candidates)}. "
        "Reduce context/model size or the configured minimum."
    )


def _targets_for_optimizer_step(job_id: str, config: RunConfig) -> tuple[str | None, ...]:
    if config.is_exposure_matched_joint(job_id):
        return tuple(EXPERIMENT_OPERATORS)
    return (None,)


def _train_optimizer_step(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    factory: SyntheticTraceFactory,
    *,
    job_id: str,
    seed: int,
    split: str,
    step: int,
    config: RunConfig,
    device: torch.device,
    precision: str,
    runtime: RuntimeState,
    job_dir: Path,
) -> StepResult:
    targets = _targets_for_optimizer_step(job_id, config)
    target_examples = config.effective_batch_size
    total_examples = target_examples * len(targets)

    while True:
        rng = _capture_rng(device)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss_total = 0.0
        per_operator: dict[str, int] = {operator: 0 for operator in EXPERIMENT_OPERATORS}
        try:
            for target_index, forced_operator in enumerate(targets):
                offset = 0
                while offset < target_examples:
                    chunk = min(runtime.micro_batch_size, target_examples - offset)
                    sample_offset = target_index * target_examples + offset
                    input_ids, labels = factory.batch(
                        job_id,
                        seed=seed,
                        split=split,
                        step=step,
                        batch_size=chunk,
                        device=device,
                        response_only=config.response_only_loss,
                        sample_offset=sample_offset,
                        forced_operator=forced_operator,
                    )
                    with _autocast(device, precision):
                        raw_loss = _loss(model(input_ids), labels)
                    if not torch.isfinite(raw_loss):
                        raise NonFiniteTrainingError(f"non-finite loss at step {step}: {float(raw_loss.detach().cpu())}")
                    weight = float(chunk) / float(total_examples)
                    (raw_loss * weight).backward()
                    weighted_loss_total += float(raw_loss.detach().cpu()) * weight
                    if forced_operator is not None:
                        per_operator[forced_operator] += chunk
                    elif job_id in EXPERIMENT_OPERATORS:
                        per_operator[job_id] += chunk
                    elif job_id == "base.common" or job_id.startswith("joint."):
                        namespace = "base" if job_id == "base.common" else job_id
                        for local_index in range(chunk):
                            sampled = factory.joint_operator(
                                seed=seed,
                                split=split,
                                step=step,
                                sample_index=sample_offset + local_index,
                                namespace=namespace,
                            )
                            per_operator[sampled] += 1
                    offset += chunk
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)
            grad_norm = float(grad_norm_tensor.detach().cpu() if isinstance(grad_norm_tensor, torch.Tensor) else grad_norm_tensor)
            if not math.isfinite(grad_norm):
                raise NonFiniteTrainingError(f"non-finite gradient norm at step {step}: {grad_norm}")
            optimizer.step()
            return StepResult(
                loss=weighted_loss_total,
                grad_norm=grad_norm,
                micro_batch_size=runtime.micro_batch_size,
                supervised_examples=total_examples,
                per_operator_examples={key: value for key, value in per_operator.items() if value},
            )
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            if not _is_cuda_oom(exc):
                raise
            if device.type != "cuda":
                raise
            old = runtime.micro_batch_size
            new = max(config.recovery.minimum_micro_batch_size, old // 2)
            if new >= old:
                raise RuntimeError(f"CUDA OOM at minimum micro-batch size {old}") from exc
            runtime.micro_batch_size = new
            runtime.oom_reductions += 1
            _json_dump(job_dir / "runtime_state.json", runtime.to_dict())
            _write_recovery_event(
                job_dir,
                {
                    "type": "cuda_oom",
                    "step": step,
                    "old_micro_batch_size": old,
                    "new_micro_batch_size": new,
                    "effective_batch_size": config.effective_batch_size,
                    "action": "retry_same_step_with_gradient_accumulation",
                },
            )
            torch.cuda.empty_cache()
            _restore_rng(rng, device)


def _evaluate_loss(
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
    model.eval()
    losses: dict[str, float] = {}
    evaluation_targets: tuple[str | None, ...]
    if job_id == "base.common":
        evaluation_targets = (None,)
    else:
        evaluation_targets = tuple(EXPERIMENT_OPERATORS)
    with torch.no_grad():
        for target_index, forced_operator in enumerate(evaluation_targets):
            key = forced_operator or job_id
            total = 0.0
            for batch_index in range(config.eval_batches):
                input_ids, labels = factory.batch(
                    job_id,
                    seed=seed + 100_000,
                    split=split,
                    step=target_index * 1_000_000 + batch_index,
                    batch_size=min(micro_batch_size, config.effective_batch_size),
                    device=device,
                    response_only=config.response_only_loss,
                    forced_operator=forced_operator,
                )
                with _autocast(device, precision):
                    value = _loss(model(input_ids), labels)
                total += float(value.detach().cpu())
            losses[key] = total / config.eval_batches
    model.train()
    losses["mean"] = sum(losses.values()) / max(1, len(losses))
    return losses


def _greedy_response(model: GPTModel, prompt: list[int], response_length: int, device: torch.device) -> list[int]:
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(response_length):
            condition = ids[:, -model.config.max_seq_len :]
            next_id = int(torch.argmax(model(condition)[:, -1, :], dim=-1).item())
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


def _last_numeric_id(ids: list[int], tokenizer: FixedVocabTokenizer) -> int | None:
    result = None
    for token_id in ids:
        token = tokenizer.tokens[token_id]
        if token.startswith("<N_") and token.endswith(">"):
            result = token_id
    return result


def _evaluate_generation(
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
    splits = ("validation", "value_ood", "length_ood")
    targets: tuple[str | None, ...] = (None,) if job_id == "base.common" else tuple(EXPERIMENT_OPERATORS)
    output: dict[str, Any] = {}
    stop_id = tokenizer.token_to_id["<TRACE_STOP>"]
    for split in splits:
        split_result: dict[str, Any] = {}
        for target_index, forced_operator in enumerate(targets):
            exact = 0
            final_correct = 0
            final_count = 0
            stop_correct = 0
            token_correct = 0
            token_count = 0
            for sample_index in range(config.generation_eval_examples):
                prompt, expected, final_id, actual_operator = factory.prompt_and_expected_ids(
                    job_id,
                    seed=seed + 200_000,
                    split=split,
                    step=target_index,
                    sample_index=sample_index,
                    forced_operator=forced_operator,
                )
                if len(prompt) + len(expected) > model.config.max_seq_len:
                    raise ValueError(
                        f"evaluation sequence length {len(prompt) + len(expected)} exceeds context {model.config.max_seq_len}"
                    )
                generated = _greedy_response(model, prompt, len(expected), device)
                exact += int(generated == expected)
                token_correct += sum(int(left == right) for left, right in zip(generated, expected))
                token_count += len(expected)
                expected_stop = expected.index(stop_id) if stop_id in expected else -1
                generated_stop = generated.index(stop_id) if stop_id in generated else -2
                stop_correct += int(generated_stop == expected_stop)
                if final_id is not None:
                    final_count += 1
                    final_correct += int(_last_numeric_id(generated, tokenizer) == final_id)
            key = forced_operator or job_id
            split_result[key] = {
                "response_exact_accuracy": exact / config.generation_eval_examples,
                "response_token_accuracy": token_correct / max(1, token_count),
                "final_value_accuracy": final_correct / max(1, final_count) if final_count else None,
                "stop_position_accuracy": stop_correct / config.generation_eval_examples,
            }
        output[split] = split_result
    model.train()
    return output


def _checkpoint_payload(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    job_id: str,
    seed: int,
    tokenizer: FixedVocabTokenizer,
    parent_checkpoint: Path,
    random_initial_checkpoint: Path,
    metrics: dict[str, Any],
    runtime: RuntimeState,
    cumulative_examples: int,
    per_operator_examples: dict[str, int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "job_id": job_id,
        "seed": seed,
        "model_config": model.config.to_dict(),
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "vocab_size": tokenizer.vocab_size,
        "parent_checkpoint": str(parent_checkpoint),
        "initial_checkpoint": str(parent_checkpoint),
        "random_initial_checkpoint": str(random_initial_checkpoint),
        "metrics": metrics,
        "runtime": runtime.to_dict(),
        "cumulative_examples": cumulative_examples,
        "per_operator_examples": per_operator_examples,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def create_shared_initial_checkpoint(
    *,
    repo_root: Path,
    config: RunConfig,
    tokenizer: FixedVocabTokenizer,
    seed: int,
    device: torch.device,
) -> Path:
    output_root = _resolve_repo_path(repo_root, config.output_dir)
    path = output_root / f"seed_{seed}" / "shared_initial.pt"
    metadata_path = output_root / f"seed_{seed}" / "shared_initial.json"
    if path.exists() and metadata_path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("vocab_hash") != tokenizer.vocab_hash:
            raise RuntimeError("existing shared initial checkpoint has a different vocabulary hash")
        return path
    _set_seed(seed, config.deterministic_algorithms)
    model_config = load_config(_resolve_repo_path(repo_root, config.model_config))
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size={model_config.vocab_size} does not match tokenizer vocab_size={tokenizer.vocab_size}"
        )
    model = GPTModel(model_config).to(device)
    if model.param_count > config.max_parameters:
        raise ValueError(f"model has {model.param_count} parameters; limit is {config.max_parameters}")
    payload = {
        "format_version": 2 if config.base_model_id else 1,
        "model_state_dict": _cpu_state_dict(model),
        "seed": seed,
        "step": 0,
        "model_config": model_config.to_dict(),
        "parameter_count": model.param_count,
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "vocab_size": tokenizer.vocab_size,
    }
    _atomic_torch_save(path, payload)
    _json_dump(metadata_path, {key: value for key, value in payload.items() if key != "model_state_dict"})
    return path


def _parent_checkpoint(repo_root: Path, config: RunConfig, seed: int, job_id: str, random_initial: Path) -> Path:
    if config.base_model_id is None or job_id == config.base_model_id:
        return random_initial
    base_complete = _resolve_repo_path(repo_root, config.output_dir) / f"seed_{seed}" / config.base_model_id.replace(".", "_") / "complete.json"
    if not base_complete.exists():
        raise RuntimeError(f"base model must complete before {job_id}: missing {base_complete}")
    payload = json.loads(base_complete.read_text(encoding="utf-8"))
    return Path(payload["final_checkpoint"])


def train_job(
    *,
    repo_root: Path,
    config: RunConfig,
    job_id: str,
    seed: int,
    allow_cpu: bool = False,
) -> Path:
    if job_id not in config.jobs:
        raise KeyError(f"unknown job_id {job_id!r}; expected one of {config.jobs}")
    device = _device(config, allow_cpu)
    _configure_cuda(config, device)
    precision = _resolve_precision(config, device)
    tokenizer = FixedVocabTokenizer.from_config(_resolve_repo_path(repo_root, config.tokenizer_config))
    factory = SyntheticTraceFactory(tokenizer, config.data)
    random_initial = create_shared_initial_checkpoint(
        repo_root=repo_root,
        config=config,
        tokenizer=tokenizer,
        seed=seed,
        device=device,
    )
    parent_checkpoint = _parent_checkpoint(repo_root, config, seed, job_id, random_initial)
    model_config = load_config(_resolve_repo_path(repo_root, config.model_config))
    model = GPTModel(model_config).to(device)
    if model.param_count > config.max_parameters:
        raise ValueError(f"model has {model.param_count} parameters; limit is {config.max_parameters}")
    parent_payload = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(parent_payload["model_state_dict"])
    parent_state = {name: tensor.detach().cpu().clone() for name, tensor in parent_payload["model_state_dict"].items()}
    random_payload = torch.load(random_initial, map_location="cpu", weights_only=False)
    random_state = {name: tensor.detach().cpu().clone() for name, tensor in random_payload["model_state_dict"].items()}
    _set_seed(_stable_job_seed(seed, job_id), config.deterministic_algorithms)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    output_root = _resolve_repo_path(repo_root, config.output_dir)
    job_dir = output_root / f"seed_{seed}" / job_id.replace(".", "_")
    checkpoint_dir = job_dir / "checkpoints"
    last_path = job_dir / "last.pt"
    complete_path = job_dir / "complete.json"
    metrics_path = job_dir / "metrics.jsonl"
    index_path = job_dir / "checkpoint_index.jsonl"
    runtime_path = job_dir / "runtime_state.json"
    job_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(job_dir / "vocab.json")

    runtime = _read_runtime_state(runtime_path, config)
    if runtime.micro_batch_size <= 0:
        runtime.micro_batch_size = _probe_micro_batch_size(
            model,
            factory,
            job_id=job_id,
            seed=seed,
            config=config,
            device=device,
            precision=precision,
        )
        _json_dump(runtime_path, runtime.to_dict())
        _write_recovery_event(
            job_dir,
            {
                "type": "micro_batch_probe",
                "selected_micro_batch_size": runtime.micro_batch_size,
                "effective_batch_size": config.effective_batch_size,
                "precision": precision,
            },
        )

    manifest = {
        "experiment_id": config.experiment_id,
        "job_id": job_id,
        "seed": seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        "precision_requested": config.precision,
        "precision_resolved": precision,
        "allow_tf32": config.allow_tf32,
        "parameter_count": model.param_count,
        "parameter_limit": config.max_parameters,
        "model_config": model_config.to_dict(),
        "tokenizer": tokenizer.metadata.__dict__,
        "random_initial_checkpoint": str(random_initial),
        "parent_checkpoint": str(parent_checkpoint),
        "effective_batch_size": config.effective_batch_size,
        "initial_micro_batch_size": runtime.micro_batch_size,
        "exposure_matched_joint": config.is_exposure_matched_joint(job_id),
        "python": sys.version,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "torch": torch.__version__,
        "created_unix": time.time(),
    }
    _json_dump(job_dir / "run_manifest.json", manifest)

    start_step = 0
    last_metrics: dict[str, Any] = {}
    cumulative_examples = 0
    per_operator_examples = {operator: 0 for operator in EXPERIMENT_OPERATORS}
    if last_path.exists() and not complete_path.exists():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if resume.get("job_id") != job_id or int(resume.get("seed", -1)) != seed:
            raise RuntimeError("resume checkpoint identity mismatch")
        if resume.get("vocab_hash") != tokenizer.vocab_hash:
            raise RuntimeError("resume checkpoint vocabulary mismatch")
        if Path(str(resume.get("parent_checkpoint", resume.get("initial_checkpoint")))) != parent_checkpoint:
            raise RuntimeError("resume checkpoint parent mismatch")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_step = int(resume["step"])
        last_metrics = dict(resume.get("metrics", {}))
        cumulative_examples = int(resume.get("cumulative_examples", 0))
        saved_counts = resume.get("per_operator_examples", {})
        per_operator_examples.update({key: int(value) for key, value in saved_counts.items()})
        saved_runtime = resume.get("runtime", {})
        if saved_runtime:
            runtime.micro_batch_size = min(runtime.micro_batch_size, int(saved_runtime.get("micro_batch_size", runtime.micro_batch_size)))
            runtime.lr_scale = min(runtime.lr_scale, float(saved_runtime.get("lr_scale", runtime.lr_scale)))
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in resume:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])

    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(complete.get("completed_step", -1)) != config.max_steps:
            raise RuntimeError("completed run was produced with a different max_steps; use a new output directory")
        return Path(complete["final_checkpoint"])

    checkpoint_steps = set(config.resolved_checkpoint_steps)

    def checkpoint_metrics(base_metrics: dict[str, Any]) -> dict[str, Any]:
        state = _cpu_state_dict(model)
        return {
            **base_metrics,
            "parameter_delta_from_parent": _delta_summary(parent_state, state),
            "parameter_delta_from_random_initial": _delta_summary(random_state, state),
        }

    def build_payload(step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return _checkpoint_payload(
            model,
            optimizer,
            step=step,
            job_id=job_id,
            seed=seed,
            tokenizer=tokenizer,
            parent_checkpoint=parent_checkpoint,
            random_initial_checkpoint=random_initial,
            metrics=checkpoint_metrics(metrics),
            runtime=runtime,
            cumulative_examples=cumulative_examples,
            per_operator_examples=per_operator_examples,
        )

    def save_permanent(step: int, metrics: dict[str, Any], label: str | None = None) -> Path:
        checkpoint_name = label or f"step_{step:09d}.pt"
        path = checkpoint_dir / checkpoint_name
        payload = build_payload(step, metrics)
        _atomic_torch_save(path, payload)
        _atomic_torch_save(last_path, payload)
        _jsonl_append(
            index_path,
            {
                "step": step,
                "checkpoint": str(path),
                "train_loss": metrics.get("train_loss"),
                "validation_loss": metrics.get("validation_loss"),
                "generation_metrics": metrics.get("generation_metrics"),
                "cumulative_examples": cumulative_examples,
                "per_operator_examples": per_operator_examples,
                "micro_batch_size": runtime.micro_batch_size,
                "lr_scale": runtime.lr_scale,
                "saved_unix": time.time(),
            },
        )
        return path

    def save_resume(step: int, metrics: dict[str, Any]) -> None:
        _atomic_torch_save(last_path, build_payload(step, metrics))

    if start_step == 0 and 0 in checkpoint_steps and not (checkpoint_dir / "step_000000000.pt").exists():
        initial_validation = _evaluate_loss(
            model,
            factory,
            job_id=job_id,
            seed=seed,
            config=config,
            device=device,
            precision=precision,
            micro_batch_size=runtime.micro_batch_size,
        )
        last_metrics = {"train_loss": None, "validation_loss": initial_validation, "learning_rate": 0.0}
        save_permanent(0, last_metrics)

    model.train()
    running_loss = 0.0
    running_count = 0
    started = time.time()
    for step_index in range(start_step, config.max_steps):
        lr = _learning_rate(step_index, config, runtime.lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = lr
        result = _train_optimizer_step(
            model,
            optimizer,
            factory,
            job_id=job_id,
            seed=seed,
            split="train",
            step=step_index,
            config=config,
            device=device,
            precision=precision,
            runtime=runtime,
            job_dir=job_dir,
        )
        completed_step = step_index + 1
        cumulative_examples += result.supervised_examples
        for operator, count in result.per_operator_examples.items():
            per_operator_examples[operator] += count
        running_loss += result.loss
        running_count += 1

        should_log = completed_step % config.log_every == 0
        should_eval = completed_step % config.eval_every == 0 or completed_step == config.max_steps
        should_generation_eval = (
            completed_step % config.generation_eval_every == 0 or completed_step == config.max_steps
        )
        validation: dict[str, float] | None = None
        generation_metrics: dict[str, Any] | None = None
        if should_eval:
            validation = _evaluate_loss(
                model,
                factory,
                job_id=job_id,
                seed=seed,
                config=config,
                device=device,
                precision=precision,
                micro_batch_size=runtime.micro_batch_size,
            )
        if should_generation_eval:
            generation_metrics = _evaluate_generation(
                model,
                factory,
                tokenizer,
                job_id=job_id,
                seed=seed,
                config=config,
                device=device,
            )
        if should_log or should_eval or should_generation_eval:
            record = {
                "step": completed_step,
                "job_id": job_id,
                "seed": seed,
                "train_loss": running_loss / max(1, running_count),
                "learning_rate": lr,
                "lr_scale": runtime.lr_scale,
                "grad_norm": result.grad_norm,
                "micro_batch_size": runtime.micro_batch_size,
                "effective_batch_size": config.effective_batch_size,
                "validation_loss": validation,
                "generation_metrics": generation_metrics,
                "cumulative_examples": cumulative_examples,
                "per_operator_examples": dict(per_operator_examples),
                "elapsed_seconds": time.time() - started,
            }
            _jsonl_append(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            running_loss = 0.0
            running_count = 0
            last_metrics = record

        should_checkpoint = completed_step != config.max_steps and (
            completed_step in checkpoint_steps
            or completed_step % config.checkpoint_every == 0
        )
        if should_checkpoint:
            if validation is None:
                validation = _evaluate_loss(
                    model,
                    factory,
                    job_id=job_id,
                    seed=seed,
                    config=config,
                    device=device,
                    precision=precision,
                    micro_batch_size=runtime.micro_batch_size,
                )
            checkpoint_record = {
                "train_loss": result.loss,
                "validation_loss": validation,
                "generation_metrics": generation_metrics,
                "learning_rate": lr,
                "grad_norm": result.grad_norm,
                "elapsed_seconds": time.time() - started,
            }
            save_permanent(completed_step, checkpoint_record)
        elif completed_step % config.resume_every == 0:
            save_resume(completed_step, last_metrics or {"train_loss": result.loss, "learning_rate": lr})

    final_generation = _evaluate_generation(
        model,
        factory,
        tokenizer,
        job_id=job_id,
        seed=seed,
        config=config,
        device=device,
    )
    final_metrics = {
        "train_loss": last_metrics.get("train_loss"),
        "validation_loss": _evaluate_loss(
            model,
            factory,
            job_id=job_id,
            seed=seed,
            config=config,
            device=device,
            precision=precision,
            micro_batch_size=runtime.micro_batch_size,
        ),
        "generation_metrics": final_generation,
        "learning_rate": _learning_rate(config.max_steps - 1, config, runtime.lr_scale),
        "elapsed_seconds": time.time() - started,
    }
    final_path = save_permanent(config.max_steps, final_metrics, label="final.pt")
    _json_dump(
        complete_path,
        {
            "job_id": job_id,
            "seed": seed,
            "final_checkpoint": str(final_path),
            "completed_step": config.max_steps,
            "cumulative_examples": cumulative_examples,
            "per_operator_examples": per_operator_examples,
            "validation_loss": final_metrics["validation_loss"],
            "generation_metrics": final_generation,
            "completed_unix": time.time(),
        },
    )
    return final_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one resumable GPT job for the bias-fusion model factory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke-test only; production config requires CUDA")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).resolve()
    repo_root = _find_repo_root(config_path.parent)
    config = load_run_config(config_path)
    final = train_job(repo_root=repo_root, config=config, job_id=args.job, seed=args.seed, allow_cpu=args.allow_cpu)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
