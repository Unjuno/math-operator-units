from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opfusion.io import load_yaml
from .data import EXPERIMENT_OPERATORS, SyntheticDataConfig


DEFAULT_CHECKPOINT_STEPS = (0, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 200_000)
DEFAULT_CHECKPOINT_FRACTIONS = (
    0.0,
    0.001,
    0.003,
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.0,
)


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class RecoveryConfig:
    max_retries_per_job: int = 5
    minimum_micro_batch_size: int = 4
    non_finite_lr_factor: float = 0.5
    max_lr_reductions: int = 2
    restart_delay_seconds: int = 30

    def validate(self) -> None:
        if self.max_retries_per_job < 0:
            raise ValueError("max_retries_per_job must be nonnegative")
        if self.minimum_micro_batch_size <= 0:
            raise ValueError("minimum_micro_batch_size must be positive")
        if not 0.0 < self.non_finite_lr_factor < 1.0:
            raise ValueError("non_finite_lr_factor must be in (0, 1)")
        if self.max_lr_reductions < 0:
            raise ValueError("max_lr_reductions must be nonnegative")


@dataclass(frozen=True)
class RunConfig:
    experiment_id: str
    output_dir: str
    model_config: str
    tokenizer_config: str
    operators: tuple[str, ...] = EXPERIMENT_OPERATORS
    # V1 compatibility: when base_model_id is None, jobs branch directly from
    # shared_initial.pt. Active production configs train base.common first.
    base_model_id: str | None = None
    joint_model_ids: tuple[str, ...] = ("joint.all_five",)
    seeds: tuple[int, ...] = (0, 1, 2)
    require_cuda: bool = True
    precision: str = "fp32"
    max_parameters: int = 1_000_000
    deterministic_algorithms: bool = False
    allow_tf32: bool = False
    continue_on_error: bool = False
    response_only_loss: bool = False
    effective_batch_size: int = 128
    # 0 means probe candidates on the actual CUDA device.
    micro_batch_size: int = 0
    micro_batch_candidates: tuple[int, ...] = (128, 64, 32, 16, 8, 4)
    max_steps: int = 200_000
    eval_every: int = 1_000
    # Legacy setting retained for old configs. Active configs use eval_examples
    # so metric sample count does not depend on the selected micro-batch.
    eval_batches: int = 8
    eval_examples: int = 256
    generation_eval_every: int = 5_000
    generation_eval_examples: int = 8
    generation_max_new_tokens: int = 256
    checkpoint_every: int = 10_000
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    checkpoint_fractions: tuple[float, ...] = ()
    resume_every: int = 500
    log_every: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)

    @property
    def jobs(self) -> tuple[str, ...]:
        items: list[str] = []
        if self.base_model_id is not None:
            items.append(self.base_model_id)
        items.extend(self.operators)
        items.extend(self.joint_model_ids)
        return tuple(items)

    @property
    def primary_joint_model_id(self) -> str:
        return self.joint_model_ids[0]

    @property
    def joint_model_id(self) -> str:
        """Compatibility alias for V1 callers that support one joint model."""
        return self.primary_joint_model_id

    @property
    def batch_size(self) -> int:
        """Compatibility alias for V1 callers before gradient accumulation."""
        return self.effective_batch_size

    @property
    def resolved_checkpoint_steps(self) -> tuple[int, ...]:
        values = set(int(step) for step in self.checkpoint_steps if 0 <= int(step) <= self.max_steps)
        for fraction in self.checkpoint_fractions:
            values.add(int(round(self.max_steps * float(fraction))))
        values.add(0)
        values.add(self.max_steps)
        return tuple(sorted(values))

    def is_joint(self, job_id: str) -> bool:
        return job_id in self.joint_model_ids

    def is_exposure_matched_joint(self, job_id: str) -> bool:
        return self.is_joint(job_id) and "exposure_matched" in job_id

    def validate(self) -> None:
        if tuple(self.operators) != EXPERIMENT_OPERATORS:
            raise ValueError(f"operator factory requires exactly {EXPERIMENT_OPERATORS}")
        if not self.joint_model_ids:
            raise ValueError("at least one joint model id is required")
        if len(set(self.jobs)) != len(self.jobs):
            raise ValueError("job ids must be unique")
        if self.precision not in {"auto", "fp32", "bf16"}:
            raise ValueError("precision must be auto, fp32, or bf16")
        if self.max_parameters <= 0 or self.max_parameters > 1_000_000:
            raise ValueError("max_parameters must be in (0, 1_000_000]")
        if self.effective_batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("effective_batch_size and max_steps must be positive")
        if self.micro_batch_size < 0:
            raise ValueError("micro_batch_size must be zero (auto) or positive")
        if not self.micro_batch_candidates or any(value <= 0 for value in self.micro_batch_candidates):
            raise ValueError("micro_batch_candidates must contain positive integers")
        if self.eval_every <= 0 or self.checkpoint_every <= 0 or self.log_every <= 0 or self.resume_every <= 0:
            raise ValueError("eval/checkpoint/log/resume intervals must be positive")
        if self.eval_batches <= 0 or self.eval_examples <= 0:
            raise ValueError("evaluation settings must be positive")
        if self.generation_eval_every <= 0 or self.generation_eval_examples < 0 or self.generation_max_new_tokens <= 0:
            raise ValueError("generation evaluation settings are invalid")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if any(not 0.0 <= fraction <= 1.0 for fraction in self.checkpoint_fractions):
            raise ValueError("checkpoint fractions must be in [0, 1]")
        self.recovery.validate()
        self.data.validate()


def _tuple(value: Any, fallback: tuple[Any, ...]) -> tuple[Any, ...]:
    if value is None:
        return fallback
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected list/tuple, got {type(value).__name__}")
    return tuple(value)


def load_run_config(path: str | Path) -> RunConfig:
    raw = load_yaml(path)
    experiment = raw.get("experiment", raw)
    train = experiment.get("train", {})
    optimizer_raw = train.get("optimizer", {})
    recovery_raw = experiment.get("recovery", {})
    data_raw = experiment.get("data", {})
    optimizer = OptimizerConfig(
        learning_rate=float(optimizer_raw.get("learning_rate", 3e-4)),
        min_learning_rate=float(optimizer_raw.get("min_learning_rate", 3e-5)),
        warmup_steps=int(optimizer_raw.get("warmup_steps", 2_000)),
        weight_decay=float(optimizer_raw.get("weight_decay", 0.1)),
        beta1=float(optimizer_raw.get("beta1", 0.9)),
        beta2=float(optimizer_raw.get("beta2", 0.95)),
        grad_clip_norm=float(optimizer_raw.get("grad_clip_norm", 1.0)),
    )
    recovery = RecoveryConfig(
        max_retries_per_job=int(recovery_raw.get("max_retries_per_job", 5)),
        minimum_micro_batch_size=int(recovery_raw.get("minimum_micro_batch_size", 4)),
        non_finite_lr_factor=float(recovery_raw.get("non_finite_lr_factor", 0.5)),
        max_lr_reductions=int(recovery_raw.get("max_lr_reductions", 2)),
        restart_delay_seconds=int(recovery_raw.get("restart_delay_seconds", 30)),
    )
    data = SyntheticDataConfig(
        operand_min=int(data_raw.get("operand_min", -64)),
        operand_max=int(data_raw.get("operand_max", 64)),
        min_terms=int(data_raw.get("min_terms", 3)),
        max_terms=int(data_raw.get("max_terms", 8)),
        numeric_token_min=int(data_raw.get("numeric_token_min", -1024)),
        numeric_token_max=int(data_raw.get("numeric_token_max", 1024)),
        value_ood_abs_min=int(data_raw.get("value_ood_abs_min", data_raw.get("operand_ood_abs_min", 65))),
        value_ood_abs_max=int(data_raw.get("value_ood_abs_max", data_raw.get("operand_ood_abs_max", 80))),
        length_ood_min_terms=int(data_raw.get("length_ood_min_terms", 9)),
        length_ood_max_terms=int(data_raw.get("length_ood_max_terms", 10)),
        partition_modulus=int(data_raw.get("partition_modulus", 100)),
        train_bucket_end=int(data_raw.get("train_bucket_end", 70)),
        validation_bucket_end=int(data_raw.get("validation_bucket_end", 85)),
        full_trace_weight=int(data_raw.get("full_trace_weight", 60)),
        continuation_weight=int(data_raw.get("continuation_weight", 25)),
        terminal_weight=int(data_raw.get("terminal_weight", 15)),
        max_partition_attempts=int(data_raw.get("max_partition_attempts", 20_000)),
        randomized_train_reduction=bool(data_raw.get("randomized_train_reduction", True)),
    )
    legacy_joint = str(experiment.get("joint_model_id", "joint.all_five"))
    joint_ids = tuple(str(value) for value in _tuple(experiment.get("joint_model_ids"), (legacy_joint,)))
    checkpoint_fractions = tuple(float(value) for value in _tuple(train.get("checkpoint_fractions"), ()))
    config = RunConfig(
        experiment_id=str(experiment["id"]),
        output_dir=str(experiment["output_dir"]),
        model_config=str(experiment["model_config"]),
        tokenizer_config=str(experiment["tokenizer_config"]),
        operators=tuple(str(value) for value in _tuple(experiment.get("operators"), EXPERIMENT_OPERATORS)),
        base_model_id=(str(experiment["base_model_id"]) if experiment.get("base_model_id") else None),
        joint_model_ids=joint_ids,
        seeds=tuple(int(seed) for seed in _tuple(experiment.get("seeds"), (0, 1, 2))),
        require_cuda=bool(experiment.get("require_cuda", True)),
        precision=str(experiment.get("precision", "fp32")),
        max_parameters=int(experiment.get("max_parameters", 1_000_000)),
        deterministic_algorithms=bool(experiment.get("deterministic_algorithms", False)),
        allow_tf32=bool(experiment.get("allow_tf32", False)),
        continue_on_error=bool(experiment.get("continue_on_error", False)),
        response_only_loss=bool(train.get("response_only_loss", False)),
        effective_batch_size=int(train.get("effective_batch_size", train.get("batch_size", 128))),
        micro_batch_size=int(train.get("micro_batch_size", train.get("batch_size", 128))),
        micro_batch_candidates=tuple(int(value) for value in _tuple(train.get("micro_batch_candidates"), (128, 64, 32, 16, 8, 4))),
        max_steps=int(train.get("max_steps", 200_000)),
        eval_every=int(train.get("eval_every", 1_000)),
        eval_batches=int(train.get("eval_batches", 8)),
        eval_examples=int(train.get("eval_examples", train.get("eval_batches", 8) * train.get("batch_size", 128))),
        generation_eval_every=int(train.get("generation_eval_every", 5_000)),
        generation_eval_examples=int(train.get("generation_eval_examples", 8)),
        generation_max_new_tokens=int(train.get("generation_max_new_tokens", 256)),
        checkpoint_every=int(train.get("checkpoint_every", 10_000)),
        checkpoint_steps=tuple(int(step) for step in _tuple(train.get("checkpoint_steps"), DEFAULT_CHECKPOINT_STEPS)),
        checkpoint_fractions=checkpoint_fractions,
        resume_every=int(train.get("resume_every", 500)),
        log_every=int(train.get("log_every", 100)),
        optimizer=optimizer,
        recovery=recovery,
        data=data,
    )
    config.validate()
    return config
