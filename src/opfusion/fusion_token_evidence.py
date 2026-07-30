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
from opfusion.fusion_coordinate_gate import build_token_features
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


DEFAULT_CALIBRATION_SEED = 719_000
DEFAULT_HOLDOUT_SEED = 719_500
DEFAULT_VERIFICATION_SEED = 720_000
SOURCE_COUNT = 1 + len(EXPERIMENT_OPERATORS)


def source_stack(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
    if unit_logits.shape[-2] != len(EXPERIMENT_OPERATORS):
        raise ValueError("all five specialist fields are required")
    if unit_logits.shape[:-2] != base_logits.shape[:-1] or unit_logits.shape[-1] != base_logits.shape[-1]:
        raise ValueError("base and unit logit shapes are incompatible")
    return torch.cat([base_logits.unsqueeze(-2), unit_logits], dim=-2)


class TokenEvidenceCompositor(nn.Module):
    """Permutation-equivariant token scorer over Base and all specialist fields.

    The shared coordinate scorer has no source identity, operator id, task label, or
    subset mask. It receives source-local rank/confidence evidence plus permutation-
    invariant cross-source statistics for the candidate token. All six source
    distributions contribute continuously to a coordinate-wise probability mixture.
    """

    def __init__(
        self,
        *,
        token_features: torch.Tensor,
        hidden_size: int = 32,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if token_features.ndim != 2 or token_features.shape[0] <= 1:
            raise ValueError("token_features must be [vocabulary, features]")
        if hidden_size <= 0 or gate_temperature <= 0:
            raise ValueError("invalid hidden size or gate temperature")
        self.gate_temperature = float(gate_temperature)
        self.register_buffer("token_features", token_features.float(), persistent=True)
        # 8 source-local + 17 cross-source + token metadata.
        feature_size = 25 + int(token_features.shape[-1])
        self.gate_network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.raw_base_mix = nn.Parameter(torch.tensor(0.0))
        self.raw_evidence_scale = nn.Parameter(torch.tensor(-0.4327521))  # softplus ~= 0.5

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top = logits.float().topk(k=2, dim=-1).values
        return top[..., 0] - top[..., 1]

    def features(self, source_logits: torch.Tensor) -> torch.Tensor:
        if source_logits.shape[-2] != SOURCE_COUNT:
            raise ValueError("expected Base plus all five specialists")
        logits = source_logits.float()
        log_prob = torch.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        relative = logits - logits.amax(dim=-1, keepdim=True)
        source_mean = logits.mean(dim=-1, keepdim=True)
        source_std = logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        standardized = (logits - source_mean) / source_std
        entropy = -(prob * log_prob).sum(dim=-1)
        margin = self._margin(logits)
        max_probability = prob.amax(dim=-1)

        base_logits = logits[..., :1, :]
        base_relative = relative[..., :1, :]
        base_prob = prob[..., :1, :]
        gain = logits - base_logits
        gain_scale = gain.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        normalized_gain = gain / gain_scale.unsqueeze(-1)

        ordered_relative = relative.topk(k=2, dim=-2).values
        ordered_prob = prob.topk(k=2, dim=-2).values
        max_relative = ordered_relative[..., 0, :]
        second_relative = ordered_relative[..., 1, :]
        mean_relative = relative.mean(dim=-2)
        std_relative = relative.std(dim=-2, unbiased=False)
        max_prob = ordered_prob[..., 0, :]
        second_prob = ordered_prob[..., 1, :]
        mean_prob = prob.mean(dim=-2)
        std_prob = prob.std(dim=-2, unbiased=False)
        max_gain = normalized_gain.amax(dim=-2)
        min_gain = normalized_gain.amin(dim=-2)
        mean_gain = normalized_gain.mean(dim=-2)
        std_gain = normalized_gain.std(dim=-2, unbiased=False)
        positive_gain_fraction = (normalized_gain > 0).float().mean(dim=-2)

        top1_ids = logits.argmax(dim=-1)
        top1_support = F.one_hot(top1_ids, num_classes=logits.shape[-1]).float().mean(dim=-2)
        top3_ids = logits.topk(k=min(3, logits.shape[-1]), dim=-1).indices
        top3_support = torch.zeros_like(prob).scatter_(-1, top3_ids, 1.0).mean(dim=-2)
        within_one = (relative >= -1.0).float().mean(dim=-2)
        within_two = (relative >= -2.0).float().mean(dim=-2)

        cross = torch.stack(
            [
                max_relative,
                second_relative,
                mean_relative,
                std_relative,
                max_prob,
                second_prob,
                mean_prob,
                std_prob,
                max_gain,
                min_gain,
                mean_gain,
                std_gain,
                positive_gain_fraction,
                top1_support,
                top3_support,
                within_one,
                within_two,
            ],
            dim=-1,
        )
        cross = cross.unsqueeze(-3).expand(*logits.shape, cross.shape[-1])

        source_local = torch.stack(
            [
                relative,
                standardized,
                prob,
                log_prob.clamp(-30.0, 0.0).div(30.0),
                normalized_gain,
                entropy.unsqueeze(-1).expand_as(logits).div(math.log(logits.shape[-1])),
                margin.unsqueeze(-1).expand_as(logits).div(20.0).clamp(0.0, 1.0),
                max_probability.unsqueeze(-1).expand_as(logits),
            ],
            dim=-1,
        )

        metadata = self.token_features.to(logits.device)
        metadata = metadata.view(*([1] * (logits.ndim - 2)), 1, metadata.shape[0], metadata.shape[1])
        metadata = metadata.expand(*logits.shape, metadata.shape[-1])
        return torch.cat([source_local, cross, metadata], dim=-1)

    def compose(self, source_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(source_logits)
        gate_logits = self.gate_network(features).squeeze(-1) / self.gate_temperature
        gates = torch.sigmoid(gate_logits).clamp_min(1e-6)
        source_log_prob = torch.log_softmax(source_logits.float(), dim=-1)

        # Coordinate-wise geometric evidence pool. Every source retains positive,
        # continuous influence; the learned gates only attenuate its token evidence.
        log_weighted = source_log_prob + gates.log()
        pooled = torch.logsumexp(log_weighted, dim=-2) - torch.log(gates.sum(dim=-2).clamp_min(1e-6))

        base_log_prob = source_log_prob[..., 0, :]
        base_mix = torch.sigmoid(self.raw_base_mix)
        evidence_scale = F.softplus(self.raw_evidence_scale) + 1e-5
        fused = base_mix * base_log_prob + (1.0 - base_mix) * pooled
        fused = base_log_prob + evidence_scale * (fused - base_log_prob)
        fused = torch.log_softmax(fused, dim=-1)
        return fused, gates, gates.sum(dim=-2)

    def forward(self, source_logits: torch.Tensor) -> torch.Tensor:
        return self.compose(source_logits)[0]


def export_parameters(model: TokenEvidenceCompositor) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def load_parameters(model: TokenEvidenceCompositor, values: Mapping[str, Any]) -> None:
    state = model.state_dict()
    for name, raw in values.items():
        if name not in state:
            raise KeyError(name)
        state[name] = torch.tensor(raw, dtype=state[name].dtype, device=state[name].device)
    model.load_state_dict(state)


def batch_metrics(model: TokenEvidenceCompositor, batch: SparseValidBatch, *, chunk_size: int = 32) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    mass_sum = 0.0
    gate_sum = 0.0
    effective_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, batch.positions, max(1, chunk_size)):
            end = min(batch.positions, start + max(1, chunk_size))
            sources = source_stack(batch.base_logits[start:end], batch.unit_logits[start:end])
            mask = batch.valid_mask[start:end]
            logits, gates, gate_total = model.compose(sources)
            rows = end - start
            loss_sum += float(valid_set_loss(logits, mask).detach().cpu()) * rows
            prediction = logits.argmax(dim=-1)
            correct_sum += float(mask.gather(1, prediction.unsqueeze(-1)).squeeze(-1).float().sum().detach().cpu())
            mass_sum += float((logits.exp() * mask.float()).sum(dim=-1).sum().detach().cpu())
            gate_sum += float(gates.mean(dim=(-2, -1)).sum().detach().cpu())
            normalized_gate = gates / gate_total.unsqueeze(-2).clamp_min(1e-6)
            entropy = -(normalized_gate * normalized_gate.clamp_min(1e-9).log()).sum(dim=-2)
            effective_sum += float(entropy.exp().mean(dim=-1).sum().detach().cpu())
            count += rows
    return {
        "valid_set_nll": loss_sum / max(1, count),
        "valid_top1_accuracy": correct_sum / max(1, count),
        "valid_probability_mass": mass_sum / max(1, count),
        "mean_coordinate_gate": gate_sum / max(1, count),
        "mean_effective_sources_per_token": effective_sum / max(1, count),
        "base_mix": float(torch.sigmoid(model.raw_base_mix).detach().cpu()),
        "evidence_scale": float((F.softplus(model.raw_evidence_scale) + 1e-5).detach().cpu()),
    }


def fit_token_evidence(
    *,
    batch: SparseValidBatch,
    token_features: torch.Tensor,
    hidden_size: int,
    gate_temperature: float,
    steps: int,
    batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    gate_penalty: float,
    seed: int,
) -> tuple[TokenEvidenceCompositor, dict[str, Any]]:
    model = TokenEvidenceCompositor(
        token_features=token_features,
        hidden_size=hidden_size,
        gate_temperature=gate_temperature,
    ).to(batch.base_logits.device)
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
        indices = cpu_indices.to(batch.base_logits.device)
        sources = source_stack(
            batch.base_logits.index_select(0, indices),
            batch.unit_logits.index_select(0, indices),
        )
        mask = batch.valid_mask.index_select(0, indices)
        logits, gates, _ = model.compose(sources)
        objective = valid_set_loss(logits, mask)
        regularizer = sum(
            (parameter - initial[name]).float().pow(2).mean()
            for name, parameter in model.named_parameters()
        )
        loss = objective + float(gate_penalty) * gates.mean() + float(l2_weight) * regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, {
        "steps": steps,
        "batch_positions": batch_positions,
        "learning_rate": learning_rate,
        "l2_weight": l2_weight,
        "gate_penalty": gate_penalty,
        "gate_temperature": gate_temperature,
        "optimization_first": losses[0] if losses else None,
        "optimization_last": losses[-1] if losses else None,
        "train_metrics": batch_metrics(model, batch),
        "parameters": export_parameters(model),
    }


def _generate_model(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    model: TokenEvidenceCompositor,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    gate_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            logits, gates, _ = model.compose(source_stack(base_logits, unit_logits))
            next_id = int(torch.argmax(logits, dim=-1).item())
            output.append(next_id)
            gate_sum += float(gates.mean().detach().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {"mean_coordinate_gate": gate_sum / max(1, positions)}


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
    token_features: torch.Tensor,
    hidden_size: int,
    examples_per_operator: int,
    verification_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    run = load_run_config(cohort.config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    dataset = _dataset(factory=factory, tokenizer=tokenizer, examples_per_operator=examples_per_operator, evaluation_seed=verification_seed)
    base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
    units = {operator: _load_model(path, device=device, tokenizer=tokenizer) for operator, path in cohort.unit_checkpoints.items()}
    models: dict[str, TokenEvidenceCompositor] = {}
    for name, spec in model_specs.items():
        model = TokenEvidenceCompositor(
            token_features=token_features,
            hidden_size=hidden_size,
            gate_temperature=float(spec["gate_temperature"]),
        ).to(device)
        load_parameters(model, spec["parameters"])
        model.eval()
        models[name] = model

    methods = ["bias_mean", *models]
    metrics: dict[str, dict[str, Any]] = {method: {} for method in methods}
    diagnostics: dict[str, dict[str, Any]] = {method: {} for method in models}
    for operator in EXPERIMENT_OPERATORS:
        counters = {method: _empty_gold_counter() for method in methods}
        gate_acc = {method: 0.0 for method in models}
        counts = {method: 0 for method in models}
        for example, prompt, expected in dataset[operator]:
            generated = _generate_bias_mean(base=base, units=units, prompt=prompt, eos_id=tokenizer.eos_id, max_new_tokens=max_new_tokens, device=device)
            _update_gold_counter(counters["bias_mean"], factory=factory, example=example, generated=generated, expected=expected)
            for method, model in models.items():
                generated, row = _generate_model(base=base, units=units, model=model, prompt=prompt, eos_id=tokenizer.eos_id, max_new_tokens=max_new_tokens, device=device)
                _update_gold_counter(counters[method], factory=factory, example=example, generated=generated, expected=expected)
                gate_acc[method] += row["mean_coordinate_gate"]
                counts[method] += 1
        for method in methods:
            metrics[method][operator] = _finalize_gold_counter(counters[method])
        for method in models:
            diagnostics[method][operator] = {"mean_coordinate_gate": gate_acc[method] / max(1, counts[method])}
    del base, units, models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "composition_metrics": metrics,
        "gate_diagnostics": diagnostics,
    }


def search_token_evidence(
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
    calibration_seed: int,
    holdout_seed: int,
    verification_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    cohorts = sorted(discover_cohorts(root, "fusion-factory"), key=lambda item: int(item.metadata.get("seed", 0)))
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")
    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    token_features = build_token_features(tokenizer).to(device)

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

    settings = {
        "token_evidence_soft": {"gate_temperature": 1.0, "gate_penalty": 0.0},
        "token_evidence_sparse": {"gate_temperature": 1.0, "gate_penalty": 0.002},
        "token_evidence_sharp": {"gate_temperature": 0.5, "gate_penalty": 0.001},
    }
    fit_reports: dict[str, Any] = {}
    for index, (name, setting) in enumerate(settings.items()):
        model, report = fit_token_evidence(
            batch=calibration,
            token_features=token_features,
            hidden_size=hidden_size,
            gate_temperature=float(setting["gate_temperature"]),
            steps=fit_steps,
            batch_positions=fit_batch_positions,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            gate_penalty=float(setting["gate_penalty"]),
            seed=calibration_seed + index,
        )
        report["holdout_metrics"] = batch_metrics(model, holdout)
        fit_reports[name] = report
        del model
    holdout_ranking = sorted(
        fit_reports,
        key=lambda name: (
            -float(fit_reports[name]["holdout_metrics"]["valid_top1_accuracy"]),
            float(fit_reports[name]["holdout_metrics"]["valid_set_nll"]),
        ),
    )
    selected_names = holdout_ranking[:2]
    model_specs = {
        name: {
            "gate_temperature": settings[name]["gate_temperature"],
            "parameters": fit_reports[name]["parameters"],
        }
        for name in selected_names
    }
    cohort_reports = [
        evaluate_cohort(
            cohort,
            root=root,
            model_specs=model_specs,
            token_features=token_features,
            hidden_size=hidden_size,
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
        "evaluation_role": "validation_only_permutation_invariant_token_evidence",
        "claim_boundary": "Base and all five specialists contribute continuously at every token; no source/operator identity or discrete switching; final splits unopened",
        "algebra": "shared candidate-token scorer and coordinate-wise geometric probability mixture",
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
    parser = argparse.ArgumentParser(description="Learn permutation-invariant candidate-token evidence composition")
    parser.add_argument("--calibration-examples-per-operator", type=int, default=4)
    parser.add_argument("--max-prefixes-per-example", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=512)
    parser.add_argument("--verification-examples-per-operator", type=int, default=8)
    parser.add_argument("--fit-steps", type=int, default=240)
    parser.add_argument("--fit-batch-positions", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_token_evidence/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(__file__).resolve().parents[2]
    report = search_token_evidence(
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
