from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from opfusion.fusion_compose import _aggregate_reports, _ranking_key, fixed_compose
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_sparse_valid import SparseValidBatch, _collect_valid_batch, valid_set_loss
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


DEFAULT_CALIBRATION_SEED = 715_000
DEFAULT_HOLDOUT_SEED = 715_500
DEFAULT_VERIFICATION_SEED = 716_000
SOURCE_NAMES = ("base", *EXPERIMENT_OPERATORS)


class SourceValidityMixer(nn.Module):
    """Continuous probability mixture predicted from simultaneous source evidence.

    Base and all five specialists are evaluated at every position. A single shared
    scorer is applied to every source; it receives no source identity, operator id,
    task label, or subset mask. The output is a soft distribution over all six
    source distributions, and the next-token distribution is their probability
    mixture rather than a discrete source selection.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        hidden_size: int = 20,
        sketch_size: int = 16,
        temperature: float = 1.0,
        projection_seed: int = 99173,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 1 or hidden_size <= 0 or sketch_size <= 0 or temperature <= 0:
            raise ValueError("invalid mixer dimensions or temperature")
        self.temperature = float(temperature)
        generator = torch.Generator(device="cpu").manual_seed(projection_seed)
        projection = torch.randn(vocabulary_size, sketch_size, generator=generator) / math.sqrt(vocabulary_size)
        self.register_buffer("projection", projection, persistent=True)
        feature_size = 8 + sketch_size
        self.score_network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top = logits.float().topk(k=2, dim=-1).values
        return top[..., 0] - top[..., 1]

    def features(self, source_logits: torch.Tensor) -> torch.Tensor:
        if source_logits.shape[-2] != len(SOURCE_NAMES):
            raise ValueError("expected Base plus all five specialists")
        probabilities = torch.softmax(source_logits.float(), dim=-1)
        log_probabilities = torch.log_softmax(source_logits.float(), dim=-1)
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        margin = self._margin(source_logits)
        maximum_probability = probabilities.amax(dim=-1)

        base_logits = source_logits[..., :1, :]
        base_log_probabilities = log_probabilities[..., :1, :]
        bias = source_logits - base_logits
        centered = bias - bias.mean(dim=-1, keepdim=True)
        rms = centered.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        normalized = centered / rms.unsqueeze(-1).to(centered.dtype)
        positive_peak = normalized.amax(dim=-1)
        negative_peak = -normalized.amin(dim=-1)
        kl_to_base = (probabilities * (log_probabilities - base_log_probabilities)).sum(dim=-1)

        consensus = normalized.mean(dim=-2, keepdim=True)
        consensus_rms = consensus.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        cosine = (normalized * consensus).float().mean(dim=-1) / consensus_rms
        sketch = torch.matmul(centered.float(), self.projection.float())

        return torch.cat(
            [
                torch.stack(
                    [
                        entropy,
                        margin,
                        maximum_probability,
                        kl_to_base,
                        rms.log(),
                        positive_peak,
                        negative_peak,
                        cosine,
                    ],
                    dim=-1,
                ),
                sketch,
            ],
            dim=-1,
        )

    def compose(self, source_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(source_logits)
        scores = self.score_network(features).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        probabilities = torch.softmax(source_logits.float(), dim=-1)
        mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=-2).clamp_min(1e-12)
        return mixture.log(), weights

    def forward(self, source_logits: torch.Tensor) -> torch.Tensor:
        return self.compose(source_logits)[0]


def source_stack(batch: SparseValidBatch) -> torch.Tensor:
    return torch.cat([batch.base_logits.unsqueeze(-2), batch.unit_logits], dim=-2)


def validity_targets(source_logits: torch.Tensor, valid_mask: torch.Tensor, *, target_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(source_logits.float(), dim=-1)
    valid_mass = (probabilities * valid_mask.unsqueeze(-2).float()).sum(dim=-1).clamp_min(1e-9)
    target = torch.softmax(valid_mass.log() / float(target_temperature), dim=-1)
    oracle = valid_mass.argmax(dim=-1)
    return target, oracle, valid_mass


def export_parameters(model: SourceValidityMixer) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def load_parameters(model: SourceValidityMixer, values: Mapping[str, Any]) -> None:
    state = model.state_dict()
    for name, raw in values.items():
        if name not in state:
            raise KeyError(name)
        state[name] = torch.tensor(raw, dtype=state[name].dtype, device=state[name].device)
    model.load_state_dict(state)


def mixer_metrics(
    model: SourceValidityMixer,
    batch: SparseValidBatch,
    *,
    target_temperature: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        sources = source_stack(batch)
        logits, weights = model.compose(sources)
        targets, oracle, valid_mass = validity_targets(
            sources, batch.valid_mask, target_temperature=target_temperature
        )
        prediction = logits.argmax(dim=-1)
        valid_top1 = batch.valid_mask.gather(1, prediction.unsqueeze(-1)).squeeze(-1)
        selected = weights.argmax(dim=-1)
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1)
        probability_mass = (torch.softmax(logits.float(), dim=-1) * batch.valid_mask.float()).sum(dim=-1)
        source_kl = (targets * (targets.clamp_min(1e-9).log() - weights.clamp_min(1e-9).log())).sum(dim=-1)
        selected_valid_mass = valid_mass.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    return {
        "valid_set_nll": float(valid_set_loss(logits, batch.valid_mask).detach().cpu()),
        "valid_top1_accuracy": float(valid_top1.float().mean().detach().cpu()),
        "valid_probability_mass": float(probability_mass.mean().detach().cpu()),
        "oracle_source_match": float((selected == oracle).float().mean().detach().cpu()),
        "selected_source_valid_mass": float(selected_valid_mass.mean().detach().cpu()),
        "source_target_kl": float(source_kl.mean().detach().cpu()),
        "mean_weight_entropy": float(entropy.mean().detach().cpu()),
        "effective_source_count": float(entropy.exp().mean().detach().cpu()),
        "mean_max_weight": float(weights.amax(dim=-1).mean().detach().cpu()),
    }


def fit_mixer(
    *,
    batch: SparseValidBatch,
    vocabulary_size: int,
    hidden_size: int,
    sketch_size: int,
    temperature: float,
    target_temperature: float,
    source_supervision_weight: float,
    entropy_target: float,
    entropy_penalty: float,
    l2_weight: float,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    seed: int,
) -> tuple[SourceValidityMixer, dict[str, Any]]:
    device = batch.base_logits.device
    model = SourceValidityMixer(
        vocabulary_size=vocabulary_size,
        hidden_size=hidden_size,
        sketch_size=sketch_size,
        temperature=temperature,
    ).to(device)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(batch.positions, generator=generator)
    cursor = 0
    losses: list[float] = []

    model.train()
    for _ in range(steps):
        if cursor + batch_positions > batch.positions:
            permutation = torch.randperm(batch.positions, generator=generator)
            cursor = 0
        cpu_indices = permutation[cursor : cursor + min(batch_positions, batch.positions)]
        cursor += int(cpu_indices.numel())
        indices = cpu_indices.to(device)
        sources = torch.cat(
            [
                batch.base_logits.index_select(0, indices).unsqueeze(-2),
                batch.unit_logits.index_select(0, indices),
            ],
            dim=-2,
        )
        valid_mask = batch.valid_mask.index_select(0, indices)
        logits, weights = model.compose(sources)
        targets, _, _ = validity_targets(
            sources, valid_mask, target_temperature=target_temperature
        )
        valid_loss = valid_set_loss(logits, valid_mask)
        source_kl = (targets * (targets.clamp_min(1e-9).log() - weights.clamp_min(1e-9).log())).sum(dim=-1).mean()
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
        entropy_shortfall = F.relu(logits.new_tensor(float(entropy_target)) - entropy)
        regularizer = sum(
            (parameter - initial[name]).float().pow(2).mean()
            for name, parameter in model.named_parameters()
        )
        loss = (
            valid_loss
            + float(source_supervision_weight) * source_kl
            + float(entropy_penalty) * entropy_shortfall
            + float(l2_weight) * regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, {
        "steps": steps,
        "temperature": temperature,
        "target_temperature": target_temperature,
        "source_supervision_weight": source_supervision_weight,
        "entropy_target": entropy_target,
        "entropy_penalty": entropy_penalty,
        "optimization_first": losses[0] if losses else None,
        "optimization_last": losses[-1] if losses else None,
        "train_metrics": mixer_metrics(model, batch, target_temperature=target_temperature),
        "parameters": export_parameters(model),
    }


def _generate_mixture(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    model: SourceValidityMixer,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    entropy_sum = 0.0
    max_weight_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            sources = torch.cat([base_logits.unsqueeze(0), unit_logits], dim=0)
            logits, weights = model.compose(sources)
            next_id = int(torch.argmax(logits, dim=-1).item())
            output.append(next_id)
            entropy_sum += float((-(weights * weights.clamp_min(1e-9).log()).sum()).detach().cpu())
            max_weight_sum += float(weights.max().detach().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
    }


def _generate_bias_mean(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
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


def evaluate_cohort(
    cohort: Cohort,
    *,
    root: Path,
    model_specs: Mapping[str, Mapping[str, Any]],
    vocabulary_size: int,
    hidden_size: int,
    sketch_size: int,
    examples_per_operator: int,
    verification_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    dataset = _dataset(
        factory=factory,
        tokenizer=tokenizer,
        examples_per_operator=examples_per_operator,
        evaluation_seed=verification_seed,
    )
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {
        operator: _load_model(path, device=device, tokenizer=tokenizer)
        for operator, path in cohort.unit_checkpoints.items()
    }
    models: dict[str, SourceValidityMixer] = {}
    for name, spec in model_specs.items():
        model = SourceValidityMixer(
            vocabulary_size=vocabulary_size,
            hidden_size=hidden_size,
            sketch_size=sketch_size,
            temperature=float(spec["temperature"]),
        ).to(device)
        load_parameters(model, spec["parameters"])
        model.eval()
        models[name] = model

    methods = ["bias_mean", *models]
    metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    diagnostics: dict[str, dict[str, Any]] = {method: {} for method in models}
    for operator in EXPERIMENT_OPERATORS:
        counters = {method: _empty_gold_counter() for method in methods}
        entropy_sum = {method: 0.0 for method in models}
        max_weight_sum = {method: 0.0 for method in models}
        counts = {method: 0 for method in models}
        for example, prompt, expected in dataset[operator]:
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
            for method, model in models.items():
                generated, row = _generate_mixture(
                    base=base,
                    units=units,
                    model=model,
                    prompt=prompt,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                _update_gold_counter(
                    counters[method],
                    factory=factory,
                    example=example,
                    generated=generated,
                    expected=expected,
                )
                entropy_sum[method] += row["mean_weight_entropy"]
                max_weight_sum[method] += row["mean_max_weight"]
                counts[method] += 1
        for method in methods:
            metrics[method][operator] = _finalize_gold_counter(counters[method])
        for method in models:
            diagnostics[method][operator] = {
                "mean_weight_entropy": entropy_sum[method] / max(1, counts[method]),
                "effective_source_count": math.exp(entropy_sum[method] / max(1, counts[method])),
                "mean_max_weight": max_weight_sum[method] / max(1, counts[method]),
            }
    del base, units, models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "composition_metrics": metrics,
        "weight_diagnostics": diagnostics,
    }


def search_validity_mixtures(
    *,
    root: Path,
    calibration_examples_per_operator: int,
    max_prefixes_per_example: int,
    max_positions_per_cohort: int,
    verification_examples_per_operator: int,
    fit_steps: int,
    fit_batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    hidden_size: int,
    sketch_size: int,
    calibration_seed: int,
    holdout_seed: int,
    verification_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
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
        data_seed=holdout_seed,
        max_prefixes_per_example=max_prefixes_per_example,
        max_positions=max_positions_per_cohort,
        device=device,
    )
    vocabulary_size = int(calibration.base_logits.shape[-1])

    settings = {
        "validity_sharp": {
            "temperature": 0.5,
            "target_temperature": 0.25,
            "source_supervision_weight": 1.0,
            "entropy_target": 0.0,
            "entropy_penalty": 0.0,
        },
        "validity_two_source": {
            "temperature": 0.75,
            "target_temperature": 0.35,
            "source_supervision_weight": 1.0,
            "entropy_target": math.log(2.0),
            "entropy_penalty": 0.2,
        },
        "validity_three_source": {
            "temperature": 1.0,
            "target_temperature": 0.5,
            "source_supervision_weight": 0.75,
            "entropy_target": math.log(3.0),
            "entropy_penalty": 0.2,
        },
        "validity_joint_objective": {
            "temperature": 0.75,
            "target_temperature": 0.35,
            "source_supervision_weight": 0.25,
            "entropy_target": math.log(2.0),
            "entropy_penalty": 0.1,
        },
    }
    fit_reports: dict[str, Any] = {}
    for index, (name, setting) in enumerate(settings.items()):
        model, report = fit_mixer(
            batch=calibration,
            vocabulary_size=vocabulary_size,
            hidden_size=hidden_size,
            sketch_size=sketch_size,
            temperature=float(setting["temperature"]),
            target_temperature=float(setting["target_temperature"]),
            source_supervision_weight=float(setting["source_supervision_weight"]),
            entropy_target=float(setting["entropy_target"]),
            entropy_penalty=float(setting["entropy_penalty"]),
            l2_weight=l2_weight,
            learning_rate=learning_rate,
            steps=fit_steps,
            batch_positions=fit_batch_positions,
            seed=calibration_seed + index,
        )
        report["holdout_metrics"] = mixer_metrics(
            model,
            holdout,
            target_temperature=float(setting["target_temperature"]),
        )
        fit_reports[name] = report
        del model

    holdout_ranking = sorted(
        fit_reports,
        key=lambda name: (
            -float(fit_reports[name]["holdout_metrics"]["valid_top1_accuracy"]),
            float(fit_reports[name]["holdout_metrics"]["valid_set_nll"]),
            -float(fit_reports[name]["holdout_metrics"]["effective_source_count"]),
        ),
    )
    selected_names = holdout_ranking[:3]
    model_specs = {
        name: {
            "temperature": settings[name]["temperature"],
            "parameters": fit_reports[name]["parameters"],
        }
        for name in selected_names
    }
    cohort_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            model_specs=model_specs,
            vocabulary_size=vocabulary_size,
            hidden_size=hidden_size,
            sketch_size=sketch_size,
            examples_per_operator=verification_examples_per_operator,
            verification_seed=verification_seed,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for cohort in cohorts
    ]
    methods = ["bias_mean", *selected_names]
    aggregate = _aggregate_reports(cohort_reports, methods)
    ranked = sorted(aggregate, key=_ranking_key)
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "validation_only_soft_source_validity_mixture",
        "claim_boundary": "all six sources contribute through a continuous probability mixture; no operator labels or discrete switching; final splits unopened",
        "source_names": list(SOURCE_NAMES),
        "calibration_seed": calibration_seed,
        "holdout_seed": holdout_seed,
        "verification_seed": verification_seed,
        "calibration_examples_per_operator": calibration_examples_per_operator,
        "verification_examples_per_operator": verification_examples_per_operator,
        "holdout_collection": holdout_collection,
        "settings": settings,
        "fit_reports": fit_reports,
        "holdout_ranking": holdout_ranking,
        "autoregressive_selected_candidates": selected_names,
        "cohort_reports": cohort_reports,
        "aggregate_methods": aggregate,
        "ranked_methods": ranked,
        "recommended_validation_composition": ranked[0] if ranked else None,
        "passing_methods": [row for row in ranked if row["passes_validation_gate"]],
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Learn soft source-validity probability mixtures")
    parser.add_argument("--calibration-examples-per-operator", type=int, default=4)
    parser.add_argument("--max-prefixes-per-example", type=int, default=32)
    parser.add_argument("--max-positions-per-cohort", type=int, default=768)
    parser.add_argument("--verification-examples-per-operator", type=int, default=8)
    parser.add_argument("--fit-steps", type=int, default=240)
    parser.add_argument("--fit-batch-positions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=20)
    parser.add_argument("--sketch-size", type=int, default=16)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_validity_mixture/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    report = search_validity_mixtures(
        root=root,
        calibration_examples_per_operator=args.calibration_examples_per_operator,
        max_prefixes_per_example=args.max_prefixes_per_example,
        max_positions_per_cohort=args.max_positions_per_cohort,
        verification_examples_per_operator=args.verification_examples_per_operator,
        fit_steps=args.fit_steps,
        fit_batch_positions=args.fit_batch_positions,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
        hidden_size=args.hidden_size,
        sketch_size=args.sketch_size,
        calibration_seed=args.calibration_seed,
        holdout_seed=args.holdout_seed,
        verification_seed=args.verification_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report.get("holdout_ranking"), ensure_ascii=False))
    print(json.dumps(report.get("recommended_validation_composition"), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
