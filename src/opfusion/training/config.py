from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opfusion.io import load_yaml
from .data import EXPERIMENT_OPERATORS, SyntheticDataConfig


DEFAULT_CHECKPOINT_STEPS = (0, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 200_000)


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
class RunConfig:
    experiment_id: str
    output_dir: str
    model_config: str
    tokenizer_config: str
    operators: tuple[str, ...] = EXPERIMENT_OPERATORS
    joint_model_id: str = "joint.all_five"
    seeds: tuple[int, ...] = (0, 1, 2)
    require_cuda: bool = True
    precision: str = "fp32"
    max_parameters: int = 1_000_000
    deterministic_algorithms: bool = False
    continue_on_error: bool = False
    batch_size: int = 128
    max_steps: int = 200_000
    eval_every: int = 1_000
    eval_batches: int = 8
    checkpoint_every: int = 10_000
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    log_every: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)

    @property
    def jobs(self) -> tuple[str, ...]:
        return (*self.operators, self.joint_model_id)

    def validate(self) -> None:
        if tuple(self.operators) != EXPERIMENT_OPERATORS:
            raise ValueError(f"operator factory v1 requires exactly {EXPERIMENT_OPERATORS}")
        if self.joint_model_id != "joint.all_five":
            raise ValueError("joint_model_id must be joint.all_five")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.max_parameters <= 0 or self.max_parameters > 1_000_000:
            raise ValueError("max_parameters must be in (0, 1_000_000]")
        if self.batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if self.eval_every <= 0 or self.checkpoint_every <= 0 or self.log_every <= 0:
            raise ValueError("eval/checkpoint/log intervals must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
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
    data = SyntheticDataConfig(
        operand_min=int(data_raw.get("operand_min", -64)),
        operand_max=int(data_raw.get("operand_max", 64)),
        min_terms=int(data_raw.get("min_terms", 3)),
        max_terms=int(data_raw.get("max_terms", 8)),
        numeric_token_min=int(data_raw.get("numeric_token_min", -1024)),
        numeric_token_max=int(data_raw.get("numeric_token_max", 1024)),
    )
    config = RunConfig(
        experiment_id=str(experiment["id"]),
        output_dir=str(experiment["output_dir"]),
        model_config=str(experiment["model_config"]),
        tokenizer_config=str(experiment["tokenizer_config"]),
        operators=_tuple(experiment.get("operators"), EXPERIMENT_OPERATORS),
        joint_model_id=str(experiment.get("joint_model_id", "joint.all_five")),
        seeds=tuple(int(seed) for seed in _tuple(experiment.get("seeds"), (0, 1, 2))),
        require_cuda=bool(experiment.get("require_cuda", True)),
        precision=str(experiment.get("precision", "fp32")),
        max_parameters=int(experiment.get("max_parameters", 1_000_000)),
        deterministic_algorithms=bool(experiment.get("deterministic_algorithms", False)),
        continue_on_error=bool(experiment.get("continue_on_error", False)),
        batch_size=int(train.get("batch_size", 128)),
        max_steps=int(train.get("max_steps", 200_000)),
        eval_every=int(train.get("eval_every", 1_000)),
        eval_batches=int(train.get("eval_batches", 8)),
        checkpoint_every=int(train.get("checkpoint_every", 10_000)),
        checkpoint_steps=tuple(int(step) for step in _tuple(train.get("checkpoint_steps"), DEFAULT_CHECKPOINT_STEPS)),
        log_every=int(train.get("log_every", 100)),
        optimizer=optimizer,
        data=data,
    )
    config.validate()
    return config
