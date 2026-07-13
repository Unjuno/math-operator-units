from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from opfusion.model import GPTModel, load_config

from . import trainer as core
from . import trainer_surface as surface
from .config import RunConfig
from .data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from .design_config import load_design_run_config, model_design
from .experiment_contract import ensure_experiment_contract


_ORIGINAL_TRAIN_STEP = core._train_optimizer_step
_ORIGINAL_PARENT_CHECKPOINT = core._parent_checkpoint
_ORIGINAL_CHECKPOINT_PAYLOAD = core._checkpoint_payload
_ORIGINAL_SHARED_INITIAL = core.create_shared_initial_checkpoint
_ACTIVE_FINGERPRINTS: dict[tuple[str, int], str] = {}
_ACTIVE_SEED_FINGERPRINTS: dict[int, str] = {}


@dataclass
class _RegularizationContext:
    reference_model: GPTModel
    fingerprint: str


_CONTEXTS: dict[str, _RegularizationContext] = {}


def _parent_checkpoint_selected(
    repo_root: Path,
    config: RunConfig,
    seed: int,
    job_id: str,
    random_initial: Path,
) -> Path:
    if config.base_model_id is None or job_id == config.base_model_id:
        return random_initial
    complete_path = (
        core._resolve_repo_path(repo_root, config.output_dir)
        / f"seed_{seed}"
        / config.base_model_id.replace(".", "_")
        / "complete.json"
    )
    if not complete_path.exists():
        raise RuntimeError(f"base model must complete before {job_id}: missing {complete_path}")
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    selected = Path(str(payload.get("selected_checkpoint", payload.get("final_checkpoint"))))
    if not selected.is_absolute():
        selected = repo_root / selected
    if not selected.exists():
        raise RuntimeError(f"selected base checkpoint does not exist: {selected}")
    return selected


def _checkpoint_payload_with_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(*args, **kwargs)
    key = (str(kwargs.get("job_id")), int(kwargs.get("seed", -1)))
    fingerprint = _ACTIVE_FINGERPRINTS.get(key)
    if fingerprint:
        payload["experiment_fingerprint"] = fingerprint
    return payload


def _shared_initial_with_contract(**kwargs: Any) -> Path:
    path = _ORIGINAL_SHARED_INITIAL(**kwargs)
    seed = int(kwargs["seed"])
    fingerprint = _ACTIVE_SEED_FINGERPRINTS.get(seed)
    if not fingerprint:
        return path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    existing = payload.get("experiment_fingerprint")
    if existing not in {None, fingerprint}:
        raise RuntimeError(
            f"shared initial checkpoint fingerprint mismatch: existing={existing} current={fingerprint}"
        )
    if existing is None:
        payload["experiment_fingerprint"] = fingerprint
        core._atomic_torch_save(path, payload)
    return path


def _masked_teacher_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = labels != -100
    if not bool(mask.any()):
        return student_logits.sum() * 0.0
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return token_kl[mask].mean()


def _parameter_anchor(model: GPTModel, reference: GPTModel) -> torch.Tensor:
    total: torch.Tensor | None = None
    count = 0
    for current, parent in zip(model.parameters(), reference.parameters()):
        value = (current.float() - parent.detach().float()).pow(2).sum()
        total = value if total is None else total + value
        count += current.numel()
    if total is None:
        raise RuntimeError("model has no parameters")
    return total / max(1, count)


def _train_optimizer_step_regularized(
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
    runtime: core.RuntimeState,
    job_dir: Path,
) -> core.StepResult:
    context = _CONTEXTS.get(str(job_dir.resolve()))
    design = model_design(config)
    if context is None or job_id not in EXPERIMENT_OPERATORS:
        return _ORIGINAL_TRAIN_STEP(
            model,
            optimizer,
            factory,
            job_id=job_id,
            seed=seed,
            split=split,
            step=step,
            config=config,
            device=device,
            precision=precision,
            runtime=runtime,
            job_dir=job_dir,
        )

    targets = core._targets_for_optimizer_step(job_id, config)
    target_examples = config.effective_batch_size
    total_examples = target_examples * len(targets)
    inactive = tuple(operator for operator in EXPERIMENT_OPERATORS if operator != job_id)
    retention_per_operator = design.specialist_retention_examples_per_operator

    while True:
        rng = core._capture_rng(device)
        optimizer.zero_grad(set_to_none=True)
        task_loss_total = 0.0
        retention_loss_total = 0.0
        anchor_loss_value = 0.0
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
                    with core._autocast(device, precision):
                        raw_loss = core._loss(model(input_ids), labels)
                    if not torch.isfinite(raw_loss):
                        raise core.NonFiniteTrainingError(
                            f"non-finite task loss at step {step}: {float(raw_loss.detach().cpu())}"
                        )
                    weight = float(chunk) / float(total_examples)
                    (raw_loss * weight).backward()
                    task_loss_total += float(raw_loss.detach().cpu()) * weight
                    per_operator[job_id] += chunk
                    offset += chunk

            if design.specialist_retention_kl_weight > 0.0:
                retention_total = retention_per_operator * len(inactive)
                for inactive_index, inactive_operator in enumerate(inactive):
                    offset = 0
                    while offset < retention_per_operator:
                        chunk = min(runtime.micro_batch_size, retention_per_operator - offset)
                        input_ids, labels = factory.batch(
                            "base.common",
                            seed=seed + 900_000,
                            split=split,
                            step=step,
                            batch_size=chunk,
                            device=device,
                            response_only=config.response_only_loss,
                            sample_offset=inactive_index * retention_per_operator + offset,
                            forced_operator=inactive_operator,
                        )
                        with core._autocast(device, precision):
                            student_logits = model(input_ids)
                            with torch.no_grad():
                                teacher_logits = context.reference_model(input_ids)
                            kl = _masked_teacher_kl(student_logits, teacher_logits, labels)
                        if not torch.isfinite(kl):
                            raise core.NonFiniteTrainingError(
                                f"non-finite retention KL at step {step}: {float(kl.detach().cpu())}"
                            )
                        weight = float(chunk) / float(max(1, retention_total))
                        scaled = kl * design.specialist_retention_kl_weight * weight
                        scaled.backward()
                        retention_loss_total += float(kl.detach().cpu()) * weight
                        offset += chunk

            if design.specialist_parameter_anchor_weight > 0.0:
                anchor = _parameter_anchor(model, context.reference_model)
                if not torch.isfinite(anchor):
                    raise core.NonFiniteTrainingError(
                        f"non-finite parameter anchor at step {step}: {float(anchor.detach().cpu())}"
                    )
                (anchor * design.specialist_parameter_anchor_weight).backward()
                anchor_loss_value = float(anchor.detach().cpu())

            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.grad_clip_norm
            )
            grad_norm = float(
                grad_norm_tensor.detach().cpu()
                if isinstance(grad_norm_tensor, torch.Tensor)
                else grad_norm_tensor
            )
            if not math.isfinite(grad_norm):
                raise core.NonFiniteTrainingError(
                    f"non-finite gradient norm at step {step}: {grad_norm}"
                )
            optimizer.step()
            combined = (
                task_loss_total
                + design.specialist_retention_kl_weight * retention_loss_total
                + design.specialist_parameter_anchor_weight * anchor_loss_value
            )
            if (step + 1) % config.log_every == 0:
                core._jsonl_append(
                    job_dir / "regularization.jsonl",
                    {
                        "step": step + 1,
                        "task_loss": task_loss_total,
                        "retention_kl": retention_loss_total,
                        "retention_kl_weight": design.specialist_retention_kl_weight,
                        "parameter_anchor_mse": anchor_loss_value,
                        "parameter_anchor_weight": design.specialist_parameter_anchor_weight,
                        "inactive_operators": list(inactive),
                        "retention_examples_per_inactive_operator": retention_per_operator,
                    },
                )
            return core.StepResult(
                loss=combined,
                grad_norm=grad_norm,
                micro_batch_size=runtime.micro_batch_size,
                supervised_examples=total_examples,
                per_operator_examples={key: value for key, value in per_operator.items() if value},
            )
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            if not core._is_cuda_oom(exc):
                raise
            if device.type != "cuda":
                raise
            old = runtime.micro_batch_size
            new = max(config.recovery.minimum_micro_batch_size, old // 2)
            if new >= old:
                raise RuntimeError(f"CUDA OOM at minimum micro-batch size {old}") from exc
            runtime.micro_batch_size = new
            runtime.oom_reductions += 1
            core._json_dump(job_dir / "runtime_state.json", runtime.to_dict())
            core._write_recovery_event(
                job_dir,
                {
                    "type": "cuda_oom",
                    "step": step,
                    "old_micro_batch_size": old,
                    "new_micro_batch_size": new,
                    "effective_batch_size": config.effective_batch_size,
                    "retention_active": True,
                    "action": "retry_same_step_with_gradient_accumulation",
                },
            )
            torch.cuda.empty_cache()
            core._restore_rng(rng, device)


def _validation_score(row: dict[str, Any], job_id: str) -> float | None:
    validation = row.get("validation_loss")
    if not isinstance(validation, dict):
        return None
    if job_id in EXPERIMENT_OPERATORS:
        value = validation.get(job_id)
    else:
        value = validation.get("mean", validation.get(job_id))
    if value is None:
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _select_validation_checkpoint(
    *,
    repo_root: Path,
    config: RunConfig,
    job_id: str,
    seed: int,
    fingerprint: str,
) -> Path:
    job_dir = (
        core._resolve_repo_path(repo_root, config.output_dir)
        / f"seed_{seed}"
        / job_id.replace(".", "_")
    )
    index_path = job_dir / "checkpoint_index.jsonl"
    if not index_path.exists():
        raise RuntimeError(f"checkpoint index is missing after training: {index_path}")
    candidates: list[tuple[float, int, Path]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row.get("step", 0))
        score = _validation_score(row, job_id)
        if step <= 0 or score is None:
            continue
        path = Path(str(row["checkpoint"]))
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            candidates.append((score, step, path))
    if not candidates:
        raise RuntimeError(f"no positive-step validation checkpoint is selectable for {job_id}")
    score, step, source = min(candidates, key=lambda item: (item[0], item[1]))
    selected_path = job_dir / "selected.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["experiment_fingerprint"] = fingerprint
    payload["selection"] = {
        "metric": "validation_nll",
        "score": score,
        "step": step,
        "source_checkpoint": str(source),
    }
    core._atomic_torch_save(selected_path, payload)

    complete_path = job_dir / "complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.update(
        {
            "experiment_fingerprint": fingerprint,
            "selected_checkpoint": str(selected_path),
            "selected_step": step,
            "selection_metric": "validation_nll",
            "selection_score": score,
            "final_checkpoint_is_selected": Path(str(complete["final_checkpoint"])) == source,
        }
    )
    core._json_dump(complete_path, complete)
    manifest_path = job_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiment_fingerprint"] = fingerprint
        manifest["model_design"] = model_design(config).to_dict()
        manifest["selected_checkpoint"] = str(selected_path)
        manifest["selected_step"] = step
        manifest["selection_score"] = score
        core._json_dump(manifest_path, manifest)
    return selected_path


def _install_design_runtime() -> None:
    core._parent_checkpoint = _parent_checkpoint_selected
    core._checkpoint_payload = _checkpoint_payload_with_contract
    core.create_shared_initial_checkpoint = _shared_initial_with_contract
    core._train_optimizer_step = _train_optimizer_step_regularized


def train_job(
    *,
    repo_root: Path,
    config: RunConfig,
    job_id: str,
    seed: int,
    allow_cpu: bool = False,
) -> Path:
    _install_design_runtime()
    contract = ensure_experiment_contract(repo_root, config)
    fingerprint = str(contract["fingerprint"])
    _ACTIVE_FINGERPRINTS[(job_id, seed)] = fingerprint
    _ACTIVE_SEED_FINGERPRINTS[seed] = fingerprint
    job_dir = (
        core._resolve_repo_path(repo_root, config.output_dir)
        / f"seed_{seed}"
        / job_id.replace(".", "_")
    )
    design = model_design(config)
    try:
        if job_id in EXPERIMENT_OPERATORS and (
            design.specialist_retention_kl_weight > 0.0
            or design.specialist_parameter_anchor_weight > 0.0
        ):
            random_initial = core._resolve_repo_path(repo_root, config.output_dir) / f"seed_{seed}" / "shared_initial.pt"
            parent = _parent_checkpoint_selected(repo_root, config, seed, job_id, random_initial)
            device = core._device(config, allow_cpu)
            model_config = load_config(core._resolve_repo_path(repo_root, config.model_config))
            reference = GPTModel(model_config).to(device)
            parent_payload = torch.load(parent, map_location=device, weights_only=False)
            if parent_payload.get("experiment_fingerprint") not in {None, fingerprint}:
                raise RuntimeError("parent checkpoint experiment fingerprint mismatch")
            reference.load_state_dict(parent_payload["model_state_dict"])
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            _CONTEXTS[str(job_dir.resolve())] = _RegularizationContext(reference, fingerprint)
        surface.train_job(
            repo_root=repo_root,
            config=config,
            job_id=job_id,
            seed=seed,
            allow_cpu=allow_cpu,
        )
        return _select_validation_checkpoint(
            repo_root=repo_root,
            config=config,
            job_id=job_id,
            seed=seed,
            fingerprint=fingerprint,
        )
    finally:
        _CONTEXTS.pop(str(job_dir.resolve()), None)
        _ACTIVE_FINGERPRINTS.pop((job_id, seed), None)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one model with strict fingerprints, optional retention, and validation selection"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--allow-cpu", action="store_true", help="smoke/pilot only")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(args.config).resolve()
    repo_root = core._find_repo_root(config_path.parent)
    config = load_design_run_config(config_path)
    selected = train_job(
        repo_root=repo_root,
        config=config,
        job_id=args.job,
        seed=args.seed,
        allow_cpu=args.allow_cpu,
    )
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
