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


DEFAULT_CALIBRATION_SEED = 717_000
DEFAULT_HOLDOUT_SEED = 717_500
DEFAULT_VERIFICATION_SEED = 718_000


def build_token_features(tokenizer: FixedVocabTokenizer) -> torch.Tensor:
    """Vocabulary metadata shared across model seeds, without operator labels."""
    rows: list[list[float]] = []
    structural = {"=", "<PLUS>", "<COMMA>", "<LBRACK>", "<RBRACK>"}
    special = {"<PAD>", "<BOS>", "<EOS>", "<UNK>", "<RESPONSE>"}
    for token in tokenizer.tokens:
        numeric = token.startswith("<N_") and token.endswith(">")
        value = 0.0
        if numeric:
            try:
                value = max(-1.0, min(1.0, int(token[3:-1]) / 1024.0))
            except ValueError:
                numeric = False
        rows.append(
            [
                float(numeric),
                float(token in structural),
                float(token.startswith("<OP_")),
                float(token in special),
                value,
                abs(value),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


class CoordinateGateCompositor(nn.Module):
    """Permutation-equivariant, vocabulary-coordinate bias suppression.

    Every specialist is evaluated at every position. One shared network scores each
    specialist/token coordinate from simultaneous logit evidence. It receives no
    specialist identity, operator id, task label, or subset mask. Raw Base-relative
    fields are never RMS-equalized; RMS is used only as a feature and threshold scale.
    """

    def __init__(
        self,
        *,
        token_features: torch.Tensor,
        hidden_size: int = 24,
        use_global_gate: bool = True,
        allow_threshold: bool = True,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if token_features.ndim != 2 or token_features.shape[0] <= 1:
            raise ValueError("token_features must be [vocabulary, features]")
        if hidden_size <= 0 or gate_temperature <= 0:
            raise ValueError("invalid hidden size or temperature")
        self.use_global_gate = bool(use_global_gate)
        self.allow_threshold = bool(allow_threshold)
        self.gate_temperature = float(gate_temperature)
        self.register_buffer("token_features", token_features.float(), persistent=True)
        coordinate_feature_size = 12 + int(token_features.shape[-1])
        self.coordinate_network = nn.Sequential(
            nn.Linear(coordinate_feature_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        if self.use_global_gate:
            self.global_network = nn.Sequential(
                nn.Linear(6, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.global_network = None
        self.raw_alpha = nn.Parameter(torch.tensor(-1.0502256))
        if self.allow_threshold:
            self.raw_threshold = nn.Parameter(torch.tensor(-2.9706281))
        else:
            self.register_buffer("raw_threshold", torch.tensor(float("-inf")), persistent=True)

    @staticmethod
    def centered_biases(base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        if unit_logits.shape[-2] != len(EXPERIMENT_OPERATORS):
            raise ValueError("all five specialist fields are required")
        if unit_logits.shape[:-2] != base_logits.shape[:-1] or unit_logits.shape[-1] != base_logits.shape[-1]:
            raise ValueError("base and unit logit shapes are incompatible")
        bias = unit_logits - base_logits.unsqueeze(-2)
        return bias - bias.mean(dim=-1, keepdim=True)

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top = logits.float().topk(k=2, dim=-1).values
        return top[..., 0] - top[..., 1]

    def _features(
        self,
        base_logits: torch.Tensor,
        unit_logits: torch.Tensor,
        centered: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rms = centered.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        normalized = centered.float() / rms.unsqueeze(-1)
        consensus = normalized.mean(dim=-2, keepdim=True)
        consensus_rms = consensus.pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
        cosine = (normalized * consensus).mean(dim=-1) / consensus_rms

        base_log_prob = torch.log_softmax(base_logits.float(), dim=-1)
        unit_log_prob = torch.log_softmax(unit_logits.float(), dim=-1)
        base_prob = base_log_prob.exp()
        unit_prob = unit_log_prob.exp()
        base_relative_top = base_logits.float() - base_logits.float().amax(dim=-1, keepdim=True)
        unit_relative_top = unit_logits.float() - unit_logits.float().amax(dim=-1, keepdim=True)

        positive_fraction = (normalized > 0).float().mean(dim=-2, keepdim=True)
        same_sign_fraction = torch.where(normalized >= 0, positive_fraction, 1.0 - positive_fraction)
        maximum_abs = normalized.abs().amax(dim=-2, keepdim=True).clamp_min(1e-6)
        relative_magnitude = normalized.abs() / maximum_abs

        token_features = self.token_features.to(base_logits.device)
        token_features = token_features.view(*([1] * (centered.ndim - 2)), 1, token_features.shape[0], token_features.shape[1])
        token_features = token_features.expand(*centered.shape, token_features.shape[-1])

        coordinate_features = torch.cat(
            [
                normalized.unsqueeze(-1),
                normalized.abs().unsqueeze(-1),
                torch.tanh(centered.float() / rms.unsqueeze(-1)).unsqueeze(-1),
                (unit_log_prob - base_log_prob.unsqueeze(-2)).clamp(-20.0, 20.0).div(20.0).unsqueeze(-1),
                unit_relative_top.clamp(-20.0, 0.0).div(20.0).unsqueeze(-1),
                base_relative_top.unsqueeze(-2).expand_as(centered).clamp(-20.0, 0.0).div(20.0).unsqueeze(-1),
                consensus.expand_as(normalized).unsqueeze(-1),
                (normalized - consensus).unsqueeze(-1),
                same_sign_fraction.expand_as(normalized).unsqueeze(-1),
                relative_magnitude.unsqueeze(-1),
                unit_prob.unsqueeze(-1),
                base_prob.unsqueeze(-2).expand_as(unit_prob).unsqueeze(-1),
                token_features,
            ],
            dim=-1,
        )

        base_entropy = -(base_prob * base_log_prob).sum(dim=-1)
        unit_entropy = -(unit_prob * unit_log_prob).sum(dim=-1)
        entropy_gain = base_entropy.unsqueeze(-1) - unit_entropy
        margin_gain = self._margin(unit_logits) - self._margin(base_logits).unsqueeze(-1)
        top_ids = unit_logits.argmax(dim=-1)
        top_agreement = (top_ids.unsqueeze(-1) == top_ids.unsqueeze(-2)).float().mean(dim=-1)
        global_features = torch.stack(
            [
                rms.log(),
                entropy_gain,
                margin_gain,
                cosine,
                top_agreement,
                normalized.abs().amax(dim=-1),
            ],
            dim=-1,
        )
        return coordinate_features, global_features, rms

    def compose(
        self,
        base_logits: torch.Tensor,
        unit_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = self.centered_biases(base_logits, unit_logits)
        coordinate_features, global_features, rms = self._features(base_logits, unit_logits, centered)
        local_gate = torch.sigmoid(self.coordinate_network(coordinate_features).squeeze(-1) / self.gate_temperature)
        if self.global_network is None:
            global_gate = torch.ones_like(rms)
        else:
            global_gate = torch.sigmoid(self.global_network(global_features).squeeze(-1) / self.gate_temperature)
        gate = local_gate * global_gate.unsqueeze(-1)

        threshold_ratio = F.softplus(self.raw_threshold) if self.allow_threshold else centered.new_tensor(0.0)
        threshold = threshold_ratio * rms
        shrunk = centered.sign() * F.relu(centered.abs() - threshold.unsqueeze(-1).to(centered.dtype))
        residual = (gate.to(shrunk.dtype) * shrunk).sum(dim=-2)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        alpha = F.softplus(self.raw_alpha) + 1e-5
        cap = (4.0 * rms.median(dim=-1).values).clamp_min(1e-4).unsqueeze(-1).to(residual.dtype)
        bounded = cap * torch.tanh(alpha * residual / cap)
        return base_logits + bounded, gate, global_gate, threshold_ratio

    def forward(self, base_logits: torch.Tensor, unit_logits: torch.Tensor) -> torch.Tensor:
        return self.compose(base_logits, unit_logits)[0]


def export_parameters(model: CoordinateGateCompositor) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def load_parameters(model: CoordinateGateCompositor, values: Mapping[str, Any]) -> None:
    state = model.state_dict()
    for name, raw in values.items():
        if name not in state:
            raise KeyError(name)
        state[name] = torch.tensor(raw, dtype=state[name].dtype, device=state[name].device)
    model.load_state_dict(state)


def batch_metrics(model: CoordinateGateCompositor, batch: SparseValidBatch, *, chunk_size: int = 32) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    mass_sum = 0.0
    gate_sum = 0.0
    global_sum = 0.0
    active_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, batch.positions, max(1, chunk_size)):
            end = min(batch.positions, start + max(1, chunk_size))
            base = batch.base_logits[start:end]
            units = batch.unit_logits[start:end]
            mask = batch.valid_mask[start:end]
            logits, gate, global_gate, threshold_ratio = model.compose(base, units)
            rows = end - start
            loss_sum += float(valid_set_loss(logits, mask).detach().cpu()) * rows
            prediction = logits.argmax(dim=-1)
            correct_sum += float(mask.gather(1, prediction.unsqueeze(-1)).squeeze(-1).float().sum().detach().cpu())
            mass = (torch.softmax(logits.float(), dim=-1) * mask.float()).sum(dim=-1)
            mass_sum += float(mass.sum().detach().cpu())
            gate_sum += float(gate.mean(dim=(-2, -1)).sum().detach().cpu())
            global_sum += float(global_gate.mean(dim=-1).sum().detach().cpu())
            active_sum += float((gate > 0.5).float().mean(dim=(-2, -1)).sum().detach().cpu())
            count += rows
    return {
        "valid_set_nll": loss_sum / max(1, count),
        "valid_top1_accuracy": correct_sum / max(1, count),
        "valid_probability_mass": mass_sum / max(1, count),
        "mean_coordinate_gate": gate_sum / max(1, count),
        "mean_global_gate": global_sum / max(1, count),
        "coordinate_gate_above_half": active_sum / max(1, count),
        "threshold_ratio": float((F.softplus(model.raw_threshold) if model.allow_threshold else torch.tensor(0.0)).detach().cpu()),
        "alpha": float((F.softplus(model.raw_alpha) + 1e-5).detach().cpu()),
    }


def fit_coordinate_gate(
    *,
    batch: SparseValidBatch,
    token_features: torch.Tensor,
    hidden_size: int,
    use_global_gate: bool,
    allow_threshold: bool,
    gate_temperature: float,
    steps: int,
    batch_positions: int,
    learning_rate: float,
    l2_weight: float,
    gate_penalty: float,
    seed: int,
) -> tuple[CoordinateGateCompositor, dict[str, Any]]:
    model = CoordinateGateCompositor(
        token_features=token_features,
        hidden_size=hidden_size,
        use_global_gate=use_global_gate,
        allow_threshold=allow_threshold,
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
        logits, gate, _, _ = model.compose(
            batch.base_logits.index_select(0, indices),
            batch.unit_logits.index_select(0, indices),
        )
        objective = valid_set_loss(logits, batch.valid_mask.index_select(0, indices))
        regularizer = sum(
            (parameter - initial[name]).float().pow(2).mean()
            for name, parameter in model.named_parameters()
        )
        loss = objective + float(gate_penalty) * gate.mean() + float(l2_weight) * regularizer
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
        "use_global_gate": use_global_gate,
        "allow_threshold": allow_threshold,
        "gate_temperature": gate_temperature,
        "optimization_first": losses[0] if losses else None,
        "optimization_last": losses[-1] if losses else None,
        "train_metrics": batch_metrics(model, batch),
        "parameters": export_parameters(model),
    }


def _generate_gate(
    *,
    base: nn.Module,
    units: Mapping[str, nn.Module],
    model: CoordinateGateCompositor,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, float]]:
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    gate_sum = 0.0
    global_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_logits = _next_logits(base, ids)
            unit_logits = torch.stack([_next_logits(units[name], ids) for name in EXPERIMENT_OPERATORS], dim=0)
            logits, gate, global_gate, _ = model.compose(base_logits, unit_logits)
            next_id = int(torch.argmax(logits, dim=-1).item())
            output.append(next_id)
            gate_sum += float(gate.mean().detach().cpu())
            global_sum += float(global_gate.mean().detach().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break
    return output, {
        "mean_coordinate_gate": gate_sum / max(1, positions),
        "mean_global_gate": global_sum / max(1, positions),
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
    models: dict[str, CoordinateGateCompositor] = {}
    for name, spec in model_specs.items():
        model = CoordinateGateCompositor(
            token_features=token_features,
            hidden_size=hidden_size,
            use_global_gate=bool(spec["use_global_gate"]),
            allow_threshold=bool(spec["allow_threshold"]),
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
        global_acc = {method: 0.0 for method in models}
        count = {method: 0 for method in models}
        for example, prompt, expected in dataset[operator]:
            generated = _generate_bias_mean(base=base, units=units, prompt=prompt, eos_id=tokenizer.eos_id, max_new_tokens=max_new_tokens, device=device)
            _update_gold_counter(counters["bias_mean"], factory=factory, example=example, generated=generated, expected=expected)
            for method, model in models.items():
                generated, row = _generate_gate(base=base, units=units, model=model, prompt=prompt, eos_id=tokenizer.eos_id, max_new_tokens=max_new_tokens, device=device)
                _update_gold_counter(counters[method], factory=factory, example=example, generated=generated, expected=expected)
                gate_acc[method] += row["mean_coordinate_gate"]
                global_acc[method] += row["mean_global_gate"]
                count[method] += 1
        for method in methods:
            metrics[method][operator] = _finalize_gold_counter(counters[method])
        for method in models:
            diagnostics[method][operator] = {
                "mean_coordinate_gate": gate_acc[method] / max(1, count[method]),
                "mean_global_gate": global_acc[method] / max(1, count[method]),
            }
    del base, units, models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "cohort_id": cohort.cohort_id,
        "model_seed": cohort.metadata.get("seed"),
        "composition_metrics": metrics,
        "gate_diagnostics": diagnostics,
    }


def search_coordinate_gates(
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
        "coordinate_local_sparse": {"use_global_gate": False, "allow_threshold": True, "gate_temperature": 1.0, "gate_penalty": 0.005},
        "coordinate_global_sparse": {"use_global_gate": True, "allow_threshold": True, "gate_temperature": 1.0, "gate_penalty": 0.005},
        "coordinate_global_sharp": {"use_global_gate": True, "allow_threshold": True, "gate_temperature": 0.5, "gate_penalty": 0.01},
        "coordinate_global_dense": {"use_global_gate": True, "allow_threshold": False, "gate_temperature": 1.0, "gate_penalty": 0.0},
    }
    fit_reports: dict[str, Any] = {}
    for index, (name, setting) in enumerate(settings.items()):
        model, report = fit_coordinate_gate(
            batch=calibration,
            token_features=token_features,
            hidden_size=hidden_size,
            use_global_gate=bool(setting["use_global_gate"]),
            allow_threshold=bool(setting["allow_threshold"]),
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
            "use_global_gate": settings[name]["use_global_gate"],
            "allow_threshold": settings[name]["allow_threshold"],
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
        "evaluation_role": "validation_only_permutation_equivariant_coordinate_gate",
        "claim_boundary": "all five units evaluated at every token; shared coordinate gate has no operator or specialist identity; final splits unopened",
        "algebra": "raw Base-relative fields with learned per-coordinate continuous suppression",
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
    parser = argparse.ArgumentParser(description="Learn permutation-equivariant coordinate-wise bias suppression")
    parser.add_argument("--calibration-examples-per-operator", type=int, default=4)
    parser.add_argument("--max-prefixes-per-example", type=int, default=32)
    parser.add_argument("--max-positions-per-cohort", type=int, default=768)
    parser.add_argument("--verification-examples-per-operator", type=int, default=8)
    parser.add_argument("--fit-steps", type=int, default=260)
    parser.add_argument("--fit-batch-positions", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--verification-seed", type=int, default=DEFAULT_VERIFICATION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluations/fusion_coordinate_gate/summary.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(__file__).resolve().parents[2]
    report = search_coordinate_gates(
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
