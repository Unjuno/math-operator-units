from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opfusion.io import load_yaml

from .config import RunConfig, load_run_config as load_base_run_config


@dataclass(frozen=True)
class ModelDesignConfig:
    """Model-construction controls that must remain explicit in experiment YAML.

    These controls are kept outside the legacy ``RunConfig`` schema so older
    typed-token configurations remain loadable without silently acquiring new
    behavior. ``load_design_run_config`` attaches one validated instance to the
    returned run and data configs.
    """

    base_target_mode: str = "identity"
    base_weak_operand_abs_max: int = 8
    base_weak_max_terms: int = 4
    specialist_retention_kl_weight: float = 0.0
    specialist_retention_examples_per_operator: int = 0
    specialist_parameter_anchor_weight: float = 0.0
    selection_metric: str = "validation_nll"
    strict_experiment_fingerprint: bool = True

    def validate(self) -> None:
        if self.base_target_mode not in {"identity", "weak_multitask"}:
            raise ValueError("base_target_mode must be identity or weak_multitask")
        if self.base_weak_operand_abs_max <= 0:
            raise ValueError("base_weak_operand_abs_max must be positive")
        if self.base_weak_max_terms < 2:
            raise ValueError("base_weak_max_terms must be at least 2")
        if self.specialist_retention_kl_weight < 0.0:
            raise ValueError("specialist_retention_kl_weight must be nonnegative")
        if self.specialist_retention_examples_per_operator < 0:
            raise ValueError("specialist_retention_examples_per_operator must be nonnegative")
        if self.specialist_parameter_anchor_weight < 0.0:
            raise ValueError("specialist_parameter_anchor_weight must be nonnegative")
        if self.specialist_retention_kl_weight > 0.0 and self.specialist_retention_examples_per_operator <= 0:
            raise ValueError("positive retention KL requires retention examples per inactive operator")
        if self.selection_metric != "validation_nll":
            raise ValueError("only validation_nll checkpoint selection is currently supported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_target_mode": self.base_target_mode,
            "base_weak_operand_abs_max": self.base_weak_operand_abs_max,
            "base_weak_max_terms": self.base_weak_max_terms,
            "specialist_retention_kl_weight": self.specialist_retention_kl_weight,
            "specialist_retention_examples_per_operator": self.specialist_retention_examples_per_operator,
            "specialist_parameter_anchor_weight": self.specialist_parameter_anchor_weight,
            "selection_metric": self.selection_metric,
            "strict_experiment_fingerprint": self.strict_experiment_fingerprint,
        }


def load_model_design(path: str | Path) -> ModelDesignConfig:
    raw = load_yaml(path)
    experiment = raw.get("experiment", raw)
    section = experiment.get("model_design", {})
    regularization = section.get("specialist_regularization", {})
    design = ModelDesignConfig(
        base_target_mode=str(section.get("base_target_mode", "identity")),
        base_weak_operand_abs_max=int(section.get("base_weak_operand_abs_max", 8)),
        base_weak_max_terms=int(section.get("base_weak_max_terms", 4)),
        specialist_retention_kl_weight=float(regularization.get("retention_kl_weight", 0.0)),
        specialist_retention_examples_per_operator=int(
            regularization.get("retention_examples_per_inactive_operator", 0)
        ),
        specialist_parameter_anchor_weight=float(regularization.get("parameter_anchor_weight", 0.0)),
        selection_metric=str(section.get("selection_metric", "validation_nll")),
        strict_experiment_fingerprint=bool(section.get("strict_experiment_fingerprint", True)),
    )
    design.validate()
    return design


def attach_model_design(config: RunConfig, design: ModelDesignConfig) -> RunConfig:
    # RunConfig and SyntheticDataConfig are frozen dataclasses but intentionally
    # do not use slots. Attaching immutable metadata preserves compatibility with
    # legacy code while making the active model design available to factories.
    object.__setattr__(config, "_model_design", design)
    object.__setattr__(config.data, "_model_design", design)
    return config


def model_design(config: RunConfig | object) -> ModelDesignConfig:
    value = getattr(config, "_model_design", None)
    return value if isinstance(value, ModelDesignConfig) else ModelDesignConfig()


def load_design_run_config(path: str | Path) -> RunConfig:
    config = load_base_run_config(path)
    return attach_model_design(config, load_model_design(path))
