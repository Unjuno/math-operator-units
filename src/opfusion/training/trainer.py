from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer
from .config import RunConfig, load_run_config
from .data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


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


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        if device.type != "cuda":
            raise RuntimeError("bf16 mode is supported only for CUDA in this experiment")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not report bfloat16 support")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def _learning_rate(step: int, config: RunConfig) -> float:
    opt = config.optimizer
    if step < opt.warmup_steps:
        return opt.learning_rate * float(step + 1) / float(max(1, opt.warmup_steps))
    progress = (step - opt.warmup_steps) / float(max(1, config.max_steps - opt.warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return opt.min_learning_rate + cosine * (opt.learning_rate - opt.min_learning_rate)


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


def _evaluate(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    *,
    seed: int,
    config: RunConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for operator_index, operator_id in enumerate(EXPERIMENT_OPERATORS):
            total = 0.0
            for batch_index in range(config.eval_batches):
                input_ids, labels = factory.batch(
                    operator_id,
                    seed=seed + 100_000,
                    split="validation",
                    step=operator_index * 1_000_000 + batch_index,
                    batch_size=config.batch_size,
                    device=device,
                )
                with _autocast(device, config.precision):
                    total += float(_loss(model(input_ids), labels).detach().cpu())
            losses[operator_id] = total / config.eval_batches
    model.train()
    losses["mean"] = sum(losses.values()) / len(EXPERIMENT_OPERATORS)
    return losses


def _checkpoint_payload(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    job_id: str,
    seed: int,
    tokenizer: FixedVocabTokenizer,
    initial_checkpoint: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "job_id": job_id,
        "seed": seed,
        "model_config": model.config.to_dict(),
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "vocab_size": tokenizer.vocab_size,
        "initial_checkpoint": str(initial_checkpoint),
        "metrics": metrics,
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
        "format_version": 1,
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
    tokenizer = FixedVocabTokenizer.from_config(_resolve_repo_path(repo_root, config.tokenizer_config))
    factory = SyntheticTraceFactory(tokenizer, config.data)
    initial_checkpoint = create_shared_initial_checkpoint(
        repo_root=repo_root,
        config=config,
        tokenizer=tokenizer,
        seed=seed,
        device=device,
    )
    model_config = load_config(_resolve_repo_path(repo_root, config.model_config))
    model = GPTModel(model_config).to(device)
    if model.param_count > config.max_parameters:
        raise ValueError(f"model has {model.param_count} parameters; limit is {config.max_parameters}")
    initial_payload = torch.load(initial_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(initial_payload["model_state_dict"])
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_payload["model_state_dict"].items()}
    # Reset stochastic training state after model construction/loading so every
    # job in a seed uses a controlled, comparable dropout trajectory.
    _set_seed(seed + 1_000_000, config.deterministic_algorithms)

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
    job_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(job_dir / "vocab.json")

    manifest = {
        "experiment_id": config.experiment_id,
        "job_id": job_id,
        "seed": seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": config.precision,
        "parameter_count": model.param_count,
        "parameter_limit": config.max_parameters,
        "model_config": model_config.to_dict(),
        "tokenizer": tokenizer.metadata.__dict__,
        "initial_checkpoint": str(initial_checkpoint),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "created_unix": time.time(),
    }
    _json_dump(job_dir / "run_manifest.json", manifest)

    start_step = 0
    last_metrics: dict[str, Any] = {}
    if last_path.exists() and not complete_path.exists():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        if resume.get("job_id") != job_id or int(resume.get("seed", -1)) != seed:
            raise RuntimeError("resume checkpoint identity mismatch")
        if resume.get("vocab_hash") != tokenizer.vocab_hash:
            raise RuntimeError("resume checkpoint vocabulary mismatch")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_step = int(resume["step"])
        last_metrics = dict(resume.get("metrics", {}))
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in resume:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])

    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(complete.get("completed_step", -1)) != config.max_steps:
            raise RuntimeError("completed run was produced with a different max_steps; use a new output directory")
        return Path(complete["final_checkpoint"])

    checkpoint_steps = set(config.checkpoint_steps)

    def save_at(step: int, metrics: dict[str, Any], label: str | None = None) -> Path:
        state = _cpu_state_dict(model)
        delta = _delta_summary(initial_state, state)
        checkpoint_name = label or f"step_{step:09d}.pt"
        path = checkpoint_dir / checkpoint_name
        payload = _checkpoint_payload(
            model,
            optimizer,
            step=step,
            job_id=job_id,
            seed=seed,
            tokenizer=tokenizer,
            initial_checkpoint=initial_checkpoint,
            metrics={**metrics, "parameter_delta": delta},
        )
        _atomic_torch_save(path, payload)
        _atomic_torch_save(last_path, payload)
        _jsonl_append(
            index_path,
            {
                "step": step,
                "checkpoint": str(path),
                "train_loss": metrics.get("train_loss"),
                "validation_loss": metrics.get("validation_loss"),
                "parameter_delta": delta,
                "saved_unix": time.time(),
            },
        )
        return path

    if start_step == 0 and 0 in checkpoint_steps and not (checkpoint_dir / "step_000000000.pt").exists():
        initial_validation = _evaluate(model, factory, seed=seed, config=config, device=device)
        last_metrics = {"train_loss": None, "validation_loss": initial_validation, "learning_rate": 0.0}
        save_at(0, last_metrics)

    model.train()
    running_loss = 0.0
    running_count = 0
    started = time.time()
    for step_index in range(start_step, config.max_steps):
        lr = _learning_rate(step_index, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        input_ids, labels = factory.batch(
            job_id,
            seed=seed,
            split="train",
            step=step_index,
            batch_size=config.batch_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.precision):
            loss = _loss(model(input_ids), labels)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm))
        optimizer.step()
        completed_step = step_index + 1
        loss_value = float(loss.detach().cpu())
        running_loss += loss_value
        running_count += 1

        should_log = completed_step % config.log_every == 0
        should_eval = completed_step % config.eval_every == 0 or completed_step == config.max_steps
        validation: dict[str, float] | None = None
        if should_eval:
            validation = _evaluate(model, factory, seed=seed, config=config, device=device)
        if should_log or should_eval:
            record = {
                "step": completed_step,
                "job_id": job_id,
                "seed": seed,
                "train_loss": running_loss / max(1, running_count),
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "validation_loss": validation,
                "elapsed_seconds": time.time() - started,
            }
            _jsonl_append(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            running_loss = 0.0
            running_count = 0
            last_metrics = record

        should_checkpoint = (
            completed_step in checkpoint_steps
            or completed_step % config.checkpoint_every == 0
            or completed_step == config.max_steps
        )
        if should_checkpoint:
            if validation is None:
                validation = _evaluate(model, factory, seed=seed, config=config, device=device)
            checkpoint_metrics = {
                "train_loss": loss_value,
                "validation_loss": validation,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "elapsed_seconds": time.time() - started,
            }
            save_at(completed_step, checkpoint_metrics)

    final_metrics = {
        "train_loss": last_metrics.get("train_loss"),
        "validation_loss": _evaluate(model, factory, seed=seed, config=config, device=device),
        "learning_rate": _learning_rate(config.max_steps - 1, config),
        "elapsed_seconds": time.time() - started,
    }
    final_path = save_at(config.max_steps, final_metrics, label="final.pt")
    _json_dump(
        complete_path,
        {
            "job_id": job_id,
            "seed": seed,
            "final_checkpoint": str(final_path),
            "completed_step": config.max_steps,
            "validation_loss": final_metrics["validation_loss"],
            "completed_unix": time.time(),
        },
    )
    return final_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one GPT operator model with resumable CUDA checkpoints")
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
