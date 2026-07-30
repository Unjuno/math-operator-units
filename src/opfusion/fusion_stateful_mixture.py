from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from opfusion.fusion_compose import _aggregate_reports, _ranking_key, fixed_compose
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_sparse_valid import SparseValidBatch, _collect_valid_batch
from opfusion.fusion_validity_mixture import (
    SourceValidityMixer,
    export_parameters,
    fit_mixer,
    mixer_metrics,
)
from opfusion.fusion_verify import (
    _dataset,
    _empty_gold_counter,
    _finalize_gold_counter,
    _next_logits,
    _update_gold_counter,
)
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


DEFAULT_CALIBRATION_SEED = 721_000
DEFAULT_STATE_SEARCH_SEED = 721_500
DEFAULT_VERIFICATION_SEED = 722_000


@dataclass(frozen=True)
class StateCandidate:
    candidate_id: str
    memory: float
    feedback: float
    temperature: float


def candidate_grid() -> tuple[StateCandidate, ...]:
    rows: list[StateCandidate] = []
    for memory in (0.0, 0.5, 0.8, 0.95):
        for feedback in (0.0, 0.15, 0.35):
            for temperature in (0.75, 1.0, 1.25):
                rows.append(
                    StateCandidate(
                        candidate_id=f"state_m{memory}_f{feedback}_t{temperature}",
                        memory=memory,
                        feedback=feedback,
                        temperature=temperature,
                    )
                )
    return tuple(rows)


def _source_logits(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    ids: torch.Tensor,
) -> torch.Tensor:
    base_logits = _next_logits(base, ids)
    unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
    return torch.cat([base_logits.unsqueeze(0), unit_logits], dim=0)


def _generate_stateful(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: SourceValidityMixer,
    candidate: StateCandidate,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    state: torch.Tensor | None = None
    entropy_sum = 0.0
    max_weight_sum = 0.0
    state_change_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            sources = _source_logits(base=base, units=units, ids=ids)
            _, instant_weights = mixer.compose(sources)
            instant_state = instant_weights.clamp_min(1e-9).log()
            if state is None:
                state = instant_state
                state_change = instant_state.new_tensor(0.0)
            else:
                previous = state
                state = float(candidate.memory) * state + (1.0 - float(candidate.memory)) * instant_state
                state_change = (state - previous).abs().mean()
            weights = torch.softmax(state / float(candidate.temperature), dim=-1)
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
            next_id = int(torch.argmax(mixture, dim=-1).item())
            output.append(next_id)

            if candidate.feedback > 0:
                token_support = torch.log_softmax(sources.float(), dim=-1)[:, next_id]
                token_support = token_support - token_support.mean()
                state = state + float(candidate.feedback) * token_support

            entropy_sum += float((-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu())
            max_weight_sum += float(weights.max().detach().cpu())
            state_change_sum += float(state_change.detach().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_state_change": state_change_sum / max(1, positions),
    }


def _generate_bias_mean(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            logits = fixed_compose(base_logits, unit_logits, mode="bias_mean")
            next_id = int(torch.argmax(logits, dim=-1).item())
            output.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output


def evaluate_candidates(
    cohort: Cohort,
    *,
    root: Path,
    mixer: SourceValidityMixer,
    candidates: Sequence[StateCandidate],
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
    units = {operator: _load_model(path, device=device, tokenizer=tokenizer) for operator, path in cohort.unit_checkpoints.items()}
    methods = [candidate.candidate_id for candidate in candidates]
    if include_bias_mean:
        methods = ["bias_mean", *methods]
    metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    diagnostics: dict[str, dict[str, Any]] = {candidate.candidate_id: {} for candidate in candidates}

    for operator in EXPERIMENT_OPERATORS:
        counters = {method: _empty_gold_counter() for method in methods}
        entropy_acc = {candidate.candidate_id: 0.0 for candidate in candidates}
        maximum_acc = {candidate.candidate_id: 0.0 for candidate in candidates}
        change_acc = {candidate.candidate_id: 0.0 for candidate in candidates}
        counts = {candidate.candidate_id: 0 for candidate in candidates}
        for example, prompt, expected in dataset[operator]:
            if include_bias_mean:
                generated = _generate_bias_mean(
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
                generated, row = _generate_stateful(
                    base=base,
                    units=units,
                    mixer=mixer,
                    candidate=candidate,
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
                entropy_acc[candidate.candidate_id] += row["mean_weight_entropy"]
                maximum_acc[candidate.candidate_id] += row["mean_max_weight"]
                change_acc[candidate.candidate_id] += row["mean_state_change"]
                counts[candidate.candidate_id] += 1
        for method in methods:
            metrics[method][operator] = _finalize_gold_counter(counters[method])
        for candidate in candidates:
            name = candidate.candidate_id
            diagnostics[name][operator] = {
                "mean_weight_entropy": entropy_acc[name] / max(1, counts[name]),
                "mean_max_weight": maximum_acc[name] / max(1, counts[name]),
                "mean_state_change": change_acc[name] / max(1, counts[name]),
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


def search_stateful_mixtures(
    *,
    root: Path,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    hidden_size: int,
    sketch_size: int,
    state_search_examples_per_operator: int,
    verification_examples_per_operator: int,
    maximum_verification_candidates: int,
    calibration_seed: int,
    state_search_seed: int,
    verification_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    cohorts = sorted(discover_cohorts(root, "fusion-factory"), key=lambda item: int(item.metadata.get("seed", 0)))
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    calibration_batches = [
        _collect_valid_batch(
            cohort,
            root=root,
            examples_per_operator=calibration_examples_per_operator,
            data_seed=calibration_seed,
            max_prefixes_per_example=max_prefixes_per_example,
            max_positions=max_positions_per_cohort,
            device=device,
        )[0]
        for cohort in cohorts[:2]
    ]
    calibration = SparseValidBatch(
        base_logits=torch.cat([batch.base_logits for batch in calibration_batches], dim=0),
        unit_logits=torch.cat([batch.unit_logits for batch in calibration_batches], dim=0),
        valid_mask=torch.cat([batch.valid_mask for batch in calibration_batches], dim=0),
    )
    holdout, holdout_collection = _collect_valid_batch(
        cohorts[2],
        root=root,
        examples_per_operator=calibration_examples_per_operator,
        data_seed=calibration_seed + 500,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions=max_positions_per_cohort,
        device=device,
    )
    vocabulary_size = int(calibration.base_logits.shape[-1])
    mixer, fit_report = fit_mixer(
        batch=calibration,
        vocabulary_size=vocabulary_size,
        hidden_size=hidden_size,
        sketch_size=sketch_size,
        temperature=0.75,
        target_temperature=0.35,
        source_supervision_weight=1.0,
        entropy_target=math.log(2.0),
        entropy_penalty=0.2,
        l2_weight=0.001,
        learning_rate=learning_rate,
        steps=fit_steps,
        batch_positions=fit_batch_positions,
        seed=calibration_seed,
    )
    fit_report["holdout_metrics"] = mixer_metrics(mixer, holdout, target_temperature=0.35)
    mixer.eval()

    candidates = candidate_grid()
    search_reports = [
        evaluate_candidates(
            cohort,
            root=root,
            mixer=mixer,
            candidates=candidates,
            examples_per_operator=state_search_examples_per_operator,
            evaluation_seed=state_search_seed,
            max_new_tokens=max_new_tokens,
            device=device,
            include_bias_mean=False,
        )
        for cohort in cohorts[:2]
    ]
    search_methods = [candidate.candidate_id for candidate in candidates]
    search_aggregate = _aggregate_reports(search_reports, search_methods)
    search_ranked = sorted(search_aggregate, key=_ranking_key)
    selected_ids = [row["method"] for row in search_ranked[:maximum_verification_candidates]]
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = [by_id[name] for name in selected_ids]

    verification_reports = [
        evaluate_candidates(
            cohort,
            root=root,
            mixer=mixer,
            candidates=selected,
            examples_per_operator=verification_examples_per_operator,
            evaluation_seed=verification_seed,
            max_new_tokens=max_new_tokens,
            device=device,
            include_bias_mean=True,
        )
        for cohort in cohorts
    ]
    verification_methods = ["bias_mean", *selected_ids]
    aggregate = _aggregate_reports(verification_reports, verification_methods)
    ranked = sorted(aggregate, key=_ranking_key)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_continuous_stateful_source_mixture",
        "claim_boundary": "all six sources remain positive and active; no operator labels or discrete switching; final splits unopened",
        "algebra": "continuous source-validity mixture with exponentially persistent reliability and optional self-support feedback",
        "calibration_seed": calibration_seed,
        "state_search_seed": state_search_seed,
        "verification_seed": verification_seed,
        "holdout_collection": holdout_collection,
        "mixer_fit": fit_report,
        "mixer_parameters": export_parameters(mixer),
        "candidate_count": len(candidates),
        "state_search_reports": search_reports,
        "state_search_ranked": search_ranked,
        "selected_candidates": [candidate.__dict__ for candidate in selected],
        "verification_reports": verification_reports,
        "aggregate_methods": aggregate,
        "ranked_methods": ranked,
        "recommended_validation_composition": ranked[0] if ranked else None,
        "passing_methods": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search continuous stateful all-source fusion laws")
    parser.add_argument("--calibration-examples-per-operator", type=int, default=6)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=1024)
    parser.add_argument("--fit-steps", type=int, default=400)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sketch-size", type=int, default=8)
    parser.add_argument("--state-search-examples-per-operator", type=int, default=2)
    parser.add_argument("--verification-examples-per-operator", type=int, default=8)
    parser.add_argument("--maximum-verification-candidates", type=int, default=3)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--state-search-seed", type=int, default=DEFAULT_STATE_SEARCH_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_stateful_mixture/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(__file__).resolve().parents[2]
    report = search_stateful_mixtures(
        root=root,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        max_prefixes_per_example=args.max_prefixes_per_example,
        max_positions_per_cohort=args.max_positions_per_cohort,
        fit_steps=args.fit_steps,
        fit_batch_positions=args.fit_batch_positions,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        sketch_size=args.sketch_size,
        state_search_examples_per_operator=args.state_search_examples_per_operator,
        verification_examples_per_operator=args.verification_examples_per_operator,
        maximum_verification_candidates=args.maximum_verification_candidates,
        calibration_seed=args.calibration_seed,
        state_search_seed=args.state_search_seed,
        verification_seed=args.verification_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report.get("selected_candidates"), ensure_ascii=False, sort_keys=True))
    print(json.dumps(report.get("recommended_validation_composition"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
