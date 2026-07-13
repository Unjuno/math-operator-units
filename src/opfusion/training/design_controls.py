from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .data import SyntheticTraceFactory, TrainingExample
from .design_config import model_design


_BASE_TRAINING_EXAMPLE: Callable[..., TrainingExample] | None = None


def _delegate(
    factory: SyntheticTraceFactory,
    job_id: str,
    *,
    seed: int,
    split: str,
    step: int,
    sample_index: int,
    forced_operator: str | None,
) -> TrainingExample:
    if _BASE_TRAINING_EXAMPLE is None:
        raise RuntimeError("model-design controls were not installed")
    return _BASE_TRAINING_EXAMPLE(
        factory,
        job_id,
        seed=seed,
        split=split,
        step=step,
        sample_index=sample_index,
        forced_operator=forced_operator,
    )


def _restricted_factory(
    factory: SyntheticTraceFactory,
    *,
    operand_abs_max: int,
    max_terms: int,
) -> SyntheticTraceFactory:
    key = (operand_abs_max, max_terms)
    cache = getattr(factory, "_model_design_factory_cache", None)
    if cache is None:
        cache = {}
        setattr(factory, "_model_design_factory_cache", cache)
    if key in cache:
        return cache[key]
    lower = max(factory.config.operand_min, -operand_abs_max)
    upper = min(factory.config.operand_max, operand_abs_max)
    restricted_max_terms = min(factory.config.max_terms, max_terms)
    restricted_min_terms = min(factory.config.min_terms, restricted_max_terms)
    restricted_config = replace(
        factory.config,
        operand_min=lower,
        operand_max=upper,
        min_terms=restricted_min_terms,
        max_terms=restricted_max_terms,
    )
    restricted = SyntheticTraceFactory(factory.tokenizer, restricted_config)
    cache[key] = restricted
    return restricted


def training_example_with_model_design(
    self: SyntheticTraceFactory,
    job_id: str,
    *,
    seed: int,
    split: str,
    step: int,
    sample_index: int,
    forced_operator: str | None = None,
) -> TrainingExample:
    design = model_design(self.config)
    if job_id != "base.common" or design.base_target_mode == "identity":
        return _delegate(
            self,
            job_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
            forced_operator=forced_operator,
        )

    operator_id = forced_operator or self.joint_operator(
        seed=seed,
        split=split,
        step=step,
        sample_index=sample_index,
        namespace="base",
    )
    weak_factory = _restricted_factory(
        self,
        operand_abs_max=design.base_weak_operand_abs_max,
        max_terms=design.base_weak_max_terms,
    )
    candidate = _delegate(
        weak_factory,
        operator_id,
        seed=seed,
        split=split,
        step=step,
        sample_index=sample_index,
        forced_operator=operator_id,
    )
    # Keep the normal verifier task label; only the job identity changes.
    return replace(candidate, job_id="base.common")


def install_model_design_controls() -> None:
    global _BASE_TRAINING_EXAMPLE
    # This function is called after install_strict_verifier(), so capture the
    # shared-prefix/strict behavior rather than the raw legacy generator.
    _BASE_TRAINING_EXAMPLE = SyntheticTraceFactory.training_example
    SyntheticTraceFactory.training_example = training_example_with_model_design  # type: ignore[method-assign]
