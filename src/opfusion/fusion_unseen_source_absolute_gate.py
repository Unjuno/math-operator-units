from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion import fusion_unseen_source_generalization as baseline
from opfusion import fusion_unseen_source_invariant as invariant


DEFAULT_GATE_PENALTY = 0.01
_ACTIVE_SCORER: "AbsoluteGateScorer | None" = None
_ACTIVE_HELDOUT: str | None = None
_ACTIVE_INCLUDE_HELDOUT = True


class AbsoluteGateScorer(nn.Module):
    """Assign an absolute nonnegative gate to each specialist independently."""

    def __init__(self, *, hidden_size: int = 16) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.net = nn.Sequential(
            nn.Linear(len(invariant.FEATURE_NAMES), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(len(invariant.FEATURE_NAMES)))
        self.register_buffer("feature_std", torch.ones(len(invariant.FEATURE_NAMES)))

    def set_normalization(self, specialist_features: torch.Tensor) -> None:
        if specialist_features.ndim != 3 or specialist_features.shape[-1] != len(invariant.FEATURE_NAMES):
            raise ValueError("specialist_features must have shape [positions, specialists, features]")
        flat = specialist_features.reshape(-1, specialist_features.shape[-1]).float()
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-4))

    def score_features(self, specialist_features: torch.Tensor) -> torch.Tensor:
        normalized = (specialist_features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)

    def gates_from_features(self, specialist_features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.score_features(specialist_features))

    def gates_from_logits(self, source_logits: torch.Tensor) -> torch.Tensor:
        return self.gates_from_features(invariant._source_features(source_logits)[1:])


def _collect_invariant_batch(*args: Any, **kwargs: Any) -> baseline.TrainingBatch:
    old_names = baseline.FEATURE_NAMES
    old_features = baseline._source_features
    baseline.FEATURE_NAMES = invariant.FEATURE_NAMES
    baseline._source_features = invariant._source_features
    try:
        return baseline.collect_training_batch(*args, **kwargs)
    finally:
        baseline.FEATURE_NAMES = old_names
        baseline._source_features = old_features


def fit_absolute_gate_scorer(
    batch: baseline.TrainingBatch,
    *,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    gate_penalty: float,
    seed: int,
    device: torch.device,
) -> tuple[AbsoluteGateScorer, dict[str, Any]]:
    if batch.features.shape[1] < 2:
        raise ValueError("batch must contain Base plus at least one specialist")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    scorer = AbsoluteGateScorer(hidden_size=hidden_size).to(device)
    scorer.set_normalization(batch.features[:, 1:, :])
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    nlls: list[float] = []
    gate_means: list[float] = []
    gate_maxes: list[float] = []

    for _ in range(steps):
        if batch_positions >= batch.positions:
            indices = torch.arange(batch.positions, device=device)
        else:
            indices = torch.randint(0, batch.positions, (batch_positions,), device=device)
        features = batch.features.index_select(0, indices)[:, 1:, :]
        target_probabilities = batch.target_source_probabilities.index_select(0, indices)
        gates = scorer.gates_from_features(features)
        numerator = target_probabilities[:, 0] + (gates * target_probabilities[:, 1:]).sum(dim=-1)
        denominator = 1.0 + gates.sum(dim=-1)
        fused_target_probability = (numerator / denominator).clamp_min(1e-12)
        nll = -fused_target_probability.log().mean()
        loss = nll + float(gate_penalty) * gates.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        nlls.append(float(nll.detach().cpu()))
        gate_means.append(float(gates.mean().detach().cpu()))
        gate_maxes.append(float(gates.max().detach().cpu()))

    scorer.eval()
    return scorer, {
        "positions": batch.positions,
        "source_count_train": int(batch.features.shape[1]),
        "specialist_count_train": int(batch.features.shape[1] - 1),
        "feature_names": list(invariant.FEATURE_NAMES),
        "optimization_first": losses[0],
        "optimization_last": losses[-1],
        "nll_first": nlls[0],
        "nll_last": nlls[-1],
        "gate_mean_first": gate_means[0],
        "gate_mean_last": gate_means[-1],
        "gate_max_first": gate_maxes[0],
        "gate_max_last": gate_maxes[-1],
        "feature_mean": scorer.feature_mean.detach().cpu().tolist(),
        "feature_std": scorer.feature_std.detach().cpu().tolist(),
        "gate_penalty": gate_penalty,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
    }


def _generate_absolute_gate(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    mixer: torch.nn.Module,
    candidate: Any,
    operator: str,
    prompt: Sequence[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
):
    del mixer, candidate
    if _ACTIVE_SCORER is None or _ACTIVE_HELDOUT is None:
        raise RuntimeError("absolute gate scorer is not active")
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    entropy_sum = 0.0
    max_weight_sum = 0.0
    matching_weight_sum = 0.0
    gate_sum = 0.0
    gate_max_sum = 0.0
    positions = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            names, sources = baseline._selected_source_logits(
                base=base,
                units=units,
                ids=ids,
                heldout=_ACTIVE_HELDOUT,
                include_heldout=_ACTIVE_INCLUDE_HELDOUT,
            )
            gates = _ACTIVE_SCORER.gates_from_logits(sources)
            raw_weights = torch.cat([gates.new_ones(1), gates], dim=0)
            weights = raw_weights / raw_weights.sum().clamp_min(1e-12)
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=0).clamp_min(1e-12)
            next_id = int(mixture.argmax().item())
            output.append(next_id)

            entropy_sum += float((-(weights * weights.clamp_min(1e-12).log()).sum()).detach().cpu())
            max_weight_sum += float(weights.max().detach().cpu())
            gate_sum += float(gates.mean().detach().cpu()) if gates.numel() else 0.0
            gate_max_sum += float(gates.max().detach().cpu()) if gates.numel() else 0.0
            if operator in names:
                matching_weight_sum += float(weights[names.index(operator)].detach().cpu())
            positions += 1
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == eos_id:
                break

    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_oracle_source_weight": matching_weight_sum / max(1, positions),
        "mean_specialist_gate": gate_sum / max(1, positions),
        "mean_max_specialist_gate": gate_max_sum / max(1, positions),
    }


def evaluate_mode(
    scorer: AbsoluteGateScorer,
    cohorts: Sequence[baseline.Cohort],
    *,
    heldout: str,
    include_heldout: bool,
    root: Path,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    global _ACTIVE_SCORER, _ACTIVE_HELDOUT, _ACTIVE_INCLUDE_HELDOUT
    previous_generator = oracle._generate_oracle_operator
    previous_scorer = _ACTIVE_SCORER
    previous_heldout = _ACTIVE_HELDOUT
    previous_include = _ACTIVE_INCLUDE_HELDOUT
    _ACTIVE_SCORER = scorer
    _ACTIVE_HELDOUT = heldout
    _ACTIVE_INCLUDE_HELDOUT = include_heldout
    oracle._generate_oracle_operator = _generate_absolute_gate
    try:
        dummy_mixer = nn.Identity()
        reports = [
            sequential.evaluate_cohort(
                cohort,
                root=root,
                mixer=dummy_mixer,
                examples_per_pair=examples_per_pair,
                data_seed=data_seed,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            for cohort in cohorts[:3]
        ]
    finally:
        oracle._generate_oracle_operator = previous_generator
        _ACTIVE_SCORER = previous_scorer
        _ACTIVE_HELDOUT = previous_heldout
        _ACTIVE_INCLUDE_HELDOUT = previous_include
    aggregate, pairs = baseline._aggregate_reports(reports)
    return {
        "include_heldout_source": include_heldout,
        "source_count_eval": 1 + len(baseline.FUNCTIONAL_OPERATORS) - (0 if include_heldout else 1),
        "aggregate": aggregate,
        "pairs": pairs,
        "subsets": baseline._subset_summary(pairs, heldout=heldout),
        "cohort_reports": reports,
    }


def run_experiment(
    *,
    root: Path,
    heldout: str,
    train_examples_per_operator: int,
    max_positions_per_cohort: int,
    train_seed: int,
    scorer_hidden_size: int,
    scorer_learning_rate: float,
    scorer_steps: int,
    scorer_batch_positions: int,
    scorer_seed: int,
    gate_penalty: float,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    )
    cohorts = sorted(
        baseline.discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    batch = _collect_invariant_batch(
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=train_examples_per_operator,
        data_seed=train_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    scorer, fit_report = fit_absolute_gate_scorer(
        batch,
        hidden_size=scorer_hidden_size,
        learning_rate=scorer_learning_rate,
        steps=scorer_steps,
        batch_positions=scorer_batch_positions,
        gate_penalty=gate_penalty,
        seed=scorer_seed,
        device=device,
    )
    without = evaluate_mode(
        scorer,
        cohorts,
        heldout=heldout,
        include_heldout=False,
        root=root,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    with_source = evaluate_mode(
        scorer,
        cohorts,
        heldout=heldout,
        include_heldout=True,
        root=root,
        examples_per_pair=examples_per_pair,
        data_seed=data_seed,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    def delta(subset: str, metric: str) -> float:
        return float(with_source["subsets"][subset][metric]) - float(without["subsets"][subset][metric])

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "leave_one_unit_out_absolute_gate_source_fusion_pilot",
        "claim_boundary": (
            "held-out specialist is absent from training examples and source pool; scorer receives only source-local/base-relative "
            "logit-field statistics and no operator identity or source slot; specialist gates are absolute softplus outputs rather "
            "than a softmax across sources; Base has fixed raw weight 1; stage boundaries remain external; two examples per pair "
            "and one scorer initialization make this a pilot"
        ),
        "heldout_operator": heldout,
        "train_seed": train_seed,
        "scorer_seed": scorer_seed,
        "data_seed": data_seed,
        "gate_penalty": gate_penalty,
        "scorer_fit": fit_report,
        "without_heldout_source": without,
        "with_heldout_source": with_source,
        "causal_deltas": {
            "heldout_as_inner_inner_accuracy": delta("heldout_as_inner", "inner_accuracy"),
            "heldout_as_outer_oracle_outer_accuracy": delta("heldout_as_outer", "oracle_intermediate_outer_accuracy"),
            "heldout_involved_end_to_end_accuracy": delta("heldout_involved", "end_to_end_accuracy"),
            "heldout_neither_end_to_end_accuracy": delta("heldout_neither", "end_to_end_accuracy"),
            "all_end_to_end_accuracy": float(with_source["aggregate"]["end_to_end_accuracy"]) - float(without["aggregate"]["end_to_end_accuracy"]),
        },
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test absolute source gates on an unseen specialist")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=baseline.FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--train-examples-per-operator", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=baseline.DEFAULT_TRAIN_SEED)
    parser.add_argument("--scorer-hidden-size", type=int, default=16)
    parser.add_argument("--scorer-learning-rate", type=float, default=0.02)
    parser.add_argument("--scorer-steps", type=int, default=400)
    parser.add_argument("--scorer-batch-positions", type=int, default=256)
    parser.add_argument("--scorer-seed", type=int, default=baseline.DEFAULT_TRAIN_SEED)
    parser.add_argument("--gate-penalty", type=float, default=DEFAULT_GATE_PENALTY)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=baseline.DEFAULT_EVAL_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_experiment(
        root=args.root,
        heldout=args.heldout,
        train_examples_per_operator=args.train_examples_per_operator,
        max_positions_per_cohort=args.max_positions_per_cohort,
        train_seed=args.train_seed,
        scorer_hidden_size=args.scorer_hidden_size,
        scorer_learning_rate=args.scorer_learning_rate,
        scorer_steps=args.scorer_steps,
        scorer_batch_positions=args.scorer_batch_positions,
        scorer_seed=args.scorer_seed,
        gate_penalty=args.gate_penalty,
        examples_per_pair=args.examples_per_pair,
        data_seed=args.data_seed,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
