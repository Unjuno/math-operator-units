from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import torch

from opfusion import fusion_stateful_dual_timescale as dual_timescale
from opfusion import fusion_stateful_dual_timescale_confirmatory as confirmatory
from opfusion import fusion_stateful_dual_timescale_confirmatory_deterministic as deterministic
from opfusion import fusion_stateful_dual_timescale_init_ensemble as ensemble
from opfusion import fusion_stateful_mixture as implementation
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort
from opfusion.fusion_validity_mixture import SourceValidityMixer
from opfusion.fusion_verify import (
    _dataset,
    _empty_gold_counter,
    _finalize_gold_counter,
    _update_gold_counter,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


ENV_ORACLE_STRENGTH = "OPFUSION_ORACLE_OPERATOR_STRENGTH"


def oracle_operator_prior(
    operator: str,
    *,
    source_count: int,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive log-weight prior for the matching specialist.

    Source index zero is Base. Specialist indices follow EXPERIMENT_OPERATORS.
    The prior remains finite, so every source keeps strictly positive weight.
    """
    if source_count != len(EXPERIMENT_OPERATORS) + 1:
        raise ValueError(
            f"expected {len(EXPERIMENT_OPERATORS) + 1} sources, got {source_count}"
        )
    try:
        specialist_index = 1 + EXPERIMENT_OPERATORS.index(operator)
    except ValueError as exc:
        raise ValueError(f"unknown operator: {operator}") from exc
    prior = torch.zeros(source_count, device=device, dtype=dtype)
    prior[specialist_index] = float(strength)
    return prior


def combine_oracle_operator_states(
    fast_state: torch.Tensor,
    slow_state: torch.Tensor,
    *,
    candidate: dual_timescale.DualTimescaleCandidate,
    operator: str,
    strength: float,
) -> torch.Tensor:
    """Combine dual-timescale reliability with an oracle operator-state prior."""
    if fast_state.shape != slow_state.shape:
        raise ValueError("fast_state and slow_state must have the same shape")
    combined = (
        (1.0 - float(candidate.slow_mix)) * fast_state
        + float(candidate.slow_mix) * slow_state
    ) / float(candidate.temperature)
    prior = oracle_operator_prior(
        operator,
        source_count=int(combined.shape[-1]),
        strength=strength,
        device=combined.device,
        dtype=combined.dtype,
    )
    return torch.softmax(combined + prior, dim=-1)


def _generate_oracle_operator(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: SourceValidityMixer,
    candidate: dual_timescale.DualTimescaleCandidate,
    operator: str,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    strength = float(os.environ.get(ENV_ORACLE_STRENGTH, "0.0"))
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    fast_state: torch.Tensor | None = None
    slow_state: torch.Tensor | None = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    timescale_gap_sum = 0.0
    oracle_weight_sum = 0.0
    positions = 0
    oracle_index = 1 + EXPERIMENT_OPERATORS.index(operator)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = implementation._source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()

            if fast_state is None or slow_state is None:
                fast_state = instant_state
                slow_state = instant_state
                state_change = instant_state.new_tensor(0.0)
            else:
                previous_combined = (
                    (1.0 - float(candidate.slow_mix)) * fast_state
                    + float(candidate.slow_mix) * slow_state
                )
                fast_state = (
                    float(candidate.fast_memory) * fast_state
                    + (1.0 - float(candidate.fast_memory)) * instant_state
                )
                slow_state = (
                    float(candidate.slow_memory) * slow_state
                    + (1.0 - float(candidate.slow_memory)) * instant_state
                )
                current_combined = (
                    (1.0 - float(candidate.slow_mix)) * fast_state
                    + float(candidate.slow_mix) * slow_state
                )
                state_change = (current_combined - previous_combined).abs().mean()

            weights = combine_oracle_operator_states(
                fast_state,
                slow_state,
                candidate=candidate,
                operator=operator,
                strength=strength,
            )
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0.0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                token_support = token_support - token_support.mean()
                fast_state = fast_state + float(candidate.feedback) * token_support
                slow_state = (
                    slow_state + 0.25 * float(candidate.feedback) * token_support
                )

            entropy_sum += float(
                (-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu()
            )
            max_weight_sum += float(weights.max().detach().cpu())
            state_change_sum += float(state_change.detach().cpu())
            timescale_gap_sum += float(
                (fast_state - slow_state).abs().mean().detach().cpu()
            )
            oracle_weight_sum += float(weights[oracle_index].detach().cpu())
            positions += 1

            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == eos_id:
                break

    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
        "mean_timescale_gap": timescale_gap_sum / max(1, positions),
        "mean_oracle_source_weight": oracle_weight_sum / max(1, positions),
    }


def evaluate_candidates_oracle_operator(
    cohort: Cohort,
    *,
    root,
    mixer: SourceValidityMixer,
    candidates,
    examples_per_operator: int,
    evaluation_seed: int,
    max_new_tokens: int,
    device: torch.device,
    include_bias_mean: bool,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    dataset = _dataset(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=examples_per_operator,
        evaluation_seed=evaluation_seed,
    )
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    methods = [candidate.candidate_id for candidate in candidates]
    if include_bias_mean:
        methods = ["bias_mean", *methods]
    metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    diagnostics: dict[str, dict[str, Any]] = {
        candidate.candidate_id: {} for candidate in candidates
    }

    diagnostic_keys = (
        "mean_weight_entropy",
        "mean_max_weight",
        "mean_state_change",
        "mean_timescale_gap",
        "mean_oracle_source_weight",
    )
    for operator in EXPERIMENT_OPERATORS:
        counters = {method: _empty_gold_counter() for method in methods}
        accumulators = {
            candidate.candidate_id: {key: 0.0 for key in diagnostic_keys}
            for candidate in candidates
        }
        counts = {candidate.candidate_id: 0 for candidate in candidates}
        for example, prompt, expected in dataset[operator]:
            if include_bias_mean:
                generated = implementation._generate_bias_mean(
                    base=base,
                    units=units,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    counters["bias_mean"],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
            for candidate in candidates:
                generated, row = _generate_oracle_operator(
                    base=base,
                    units=units,
                    mixer=mixer,
                    candidate=candidate,
                    operator=operator,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    counters[candidate.candidate_id],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
                for key in diagnostic_keys:
                    accumulators[candidate.candidate_id][key] += row[key]
                counts[candidate.candidate_id] += 1

        for method in methods:
            metrics[method][operator] = _finalize_gold_counter(counters[method])
        for candidate in candidates:
            name = candidate.candidate_id
            diagnostics[name][operator] = {
                key: accumulators[name][key] / max(1, counts[name])
                for key in diagnostic_keys
            }

    del base, units
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "composition_metrics": metrics,
        "state_diagnostics": diagnostics,
    }


_original_search = implementation.search_stateful_mixtures


def _search_oracle_operator(**kwargs: Any) -> dict[str, Any]:
    report = _original_search(**kwargs)
    strength = float(os.environ.get(ENV_ORACLE_STRENGTH, "0.0"))
    report["evaluation_role"] = (
        "validation_only_oracle_operator_conditioned_dual_timescale_ensemble"
    )
    report["algebra"] = (
        "five-mixer arithmetic ensemble with dual-timescale reliability and a "
        "finite additive log-weight prior for the externally supplied current operator"
    )
    report["claim_boundary"] = (
        "the current operator label is supplied as oracle state; Base and all five "
        "specialists remain active with strictly positive weights; no final answer, "
        "intermediate value, valid-token mask, or final IID/OOD split is exposed"
    )
    report["oracle_operator_strength"] = strength
    return report


implementation.evaluate_candidates = evaluate_candidates_oracle_operator
implementation.search_stateful_mixtures = _search_oracle_operator


def main() -> int:
    return deterministic.main()


if __name__ == "__main__":
    raise SystemExit(main())
