from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from opfusion import fusion_oracle_sequential_composition as sequential
from opfusion import fusion_stateful_oracle_operator as oracle
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_verify import _next_logits
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
FEATURE_NAMES = (
    "entropy",
    "max_probability",
    "top_margin",
    "log1p_kl_to_base",
    "log1p_kl_to_mean",
    "top_agrees_base",
    "top_agrees_mean",
    "log1p_delta_rms",
    "consensus_alignment",
)
DEFAULT_TRAIN_SEED = 741_000
DEFAULT_EVAL_SEED = 741_500

_ACTIVE_SCORER: "SharedSourceScorer | None" = None
_ACTIVE_HELDOUT: str | None = None
_ACTIVE_INCLUDE_HELDOUT = True


def _source_features(source_logits: torch.Tensor) -> torch.Tensor:
    """Identity-free per-source features with source 0 reserved as the base anchor."""
    if source_logits.ndim != 2 or source_logits.shape[0] < 2:
        raise ValueError("source_logits must have shape [sources>=2, vocabulary]")
    logits = source_logits.float()
    log_prob = torch.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    vocabulary = int(logits.shape[-1])

    entropy = -(prob * log_prob).sum(dim=-1) / max(math.log(max(2, vocabulary)), 1e-8)
    top2 = prob.topk(k=min(2, vocabulary), dim=-1).values
    max_probability = top2[:, 0]
    top_margin = top2[:, 0] - (top2[:, 1] if vocabulary >= 2 else 0.0)

    base_log_prob = log_prob[0].unsqueeze(0)
    kl_to_base = (prob * (log_prob - base_log_prob)).sum(dim=-1).clamp_min(0.0)
    mean_prob = prob.mean(dim=0).clamp_min(1e-12)
    mean_log_prob = mean_prob.log().unsqueeze(0)
    kl_to_mean = (prob * (log_prob - mean_log_prob)).sum(dim=-1).clamp_min(0.0)

    top_index = prob.argmax(dim=-1)
    top_agrees_base = (top_index == top_index[0]).to(prob.dtype)
    top_agrees_mean = (top_index == mean_prob.argmax()).to(prob.dtype)

    delta = logits - logits[0].unsqueeze(0)
    delta = delta - delta.mean(dim=-1, keepdim=True)
    delta_rms = delta.pow(2).mean(dim=-1).sqrt()
    consensus = delta.mean(dim=0)
    consensus_norm = consensus.norm().clamp_min(1e-8)
    raw_delta_norm = delta.norm(dim=-1)
    delta_norm = raw_delta_norm.clamp_min(1e-8)
    alignment = (delta * consensus.unsqueeze(0)).sum(dim=-1) / (delta_norm * consensus_norm)
    alignment = torch.where(raw_delta_norm > 1e-7, alignment, torch.zeros_like(alignment))

    return torch.stack(
        [
            entropy,
            max_probability,
            top_margin,
            torch.log1p(kl_to_base),
            torch.log1p(kl_to_mean),
            top_agrees_base,
            top_agrees_mean,
            torch.log1p(delta_rms),
            alignment,
        ],
        dim=-1,
    )


class SharedSourceScorer(nn.Module):
    """Apply one shared scoring rule to every source, independent of source identity/count."""

    def __init__(self, *, hidden_size: int = 16) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer("feature_std", torch.ones(len(FEATURE_NAMES)))

    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("features must have shape [positions, sources, features]")
        flat = features.reshape(-1, features.shape[-1]).float()
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-4))

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)

    def weights_from_logits(self, source_logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score_features(_source_features(source_logits)), dim=-1)


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    target_source_probabilities: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.features.shape[0])


def _selected_source_logits(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    ids: torch.Tensor,
    heldout: str,
    include_heldout: bool,
) -> tuple[list[str], torch.Tensor]:
    operators = [
        operator
        for operator in FUNCTIONAL_OPERATORS
        if include_heldout or operator != heldout
    ]
    names = ["Base", *operators]
    logits = [_next_logits(base, ids)]
    logits.extend(_next_logits(units[operator], ids) for operator in operators)
    return names, torch.stack(logits, dim=0)


def collect_training_batch(
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    heldout: str,
    examples_per_operator: int,
    data_seed: int,
    max_positions_per_cohort: int,
    device: torch.device,
) -> TrainingBatch:
    if heldout not in FUNCTIONAL_OPERATORS:
        raise ValueError(f"heldout must be one of {FUNCTIONAL_OPERATORS}")
    all_features: list[torch.Tensor] = []
    all_target_probabilities: list[torch.Tensor] = []

    for cohort in cohorts[:2]:
        run = load_run_config(cohort.config_path)
        tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
        factory = SyntheticTraceFactory(tokenizer, run.data)
        base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
        units = {
            operator: _load_model(path, device=device, tokenizer=tokenizer)
            for operator, path in cohort.unit_checkpoints.items()
            if operator in FUNCTIONAL_OPERATORS and operator != heldout
        }
        positions = 0
        with torch.no_grad():
            for operator_index, operator in enumerate(FUNCTIONAL_OPERATORS):
                if operator == heldout:
                    continue
                for sample_index in range(examples_per_operator):
                    prompt, expected, _, _ = factory.prompt_and_expected_ids(
                        operator,
                        seed=data_seed,
                        split="train",
                        step=operator_index,
                        sample_index=sample_index,
                    )
                    ids = torch.tensor([prompt], dtype=torch.long, device=device)
                    for target_id in expected:
                        _, sources = _selected_source_logits(
                            base=base,
                            units=units,
                            ids=ids,
                            heldout=heldout,
                            include_heldout=False,
                        )
                        all_features.append(_source_features(sources).detach().cpu())
                        target_probabilities = torch.softmax(sources.float(), dim=-1)[:, int(target_id)]
                        all_target_probabilities.append(target_probabilities.detach().cpu())
                        ids = torch.cat(
                            [ids, torch.tensor([[int(target_id)]], dtype=torch.long, device=device)],
                            dim=1,
                        )
                        positions += 1
                        if positions >= max_positions_per_cohort:
                            break
                    if positions >= max_positions_per_cohort:
                        break
                if positions >= max_positions_per_cohort:
                    break
        del base, units
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_features:
        raise RuntimeError("no training positions collected")
    return TrainingBatch(
        features=torch.stack(all_features, dim=0).to(device),
        target_source_probabilities=torch.stack(all_target_probabilities, dim=0).to(device),
    )


def fit_shared_scorer(
    batch: TrainingBatch,
    *,
    hidden_size: int,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    seed: int,
    device: torch.device,
) -> tuple[SharedSourceScorer, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    scorer = SharedSourceScorer(hidden_size=hidden_size).to(device)
    scorer.set_normalization(batch.features)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    entropies: list[float] = []
    for _ in range(steps):
        if batch_positions >= batch.positions:
            indices = torch.arange(batch.positions, device=device)
        else:
            indices = torch.randint(0, batch.positions, (batch_positions,), device=device)
        features = batch.features.index_select(0, indices)
        target_probabilities = batch.target_source_probabilities.index_select(0, indices)
        scores = scorer.score_features(features)
        weights = torch.softmax(scores, dim=-1)
        target_probability = (weights * target_probabilities).sum(dim=-1).clamp_min(1e-12)
        loss = -target_probability.log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        entropies.append(
            float((-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach().cpu())
        )
    scorer.eval()
    return scorer, {
        "positions": batch.positions,
        "source_count_train": int(batch.features.shape[1]),
        "feature_names": list(FEATURE_NAMES),
        "optimization_first": losses[0],
        "optimization_last": losses[-1],
        "weight_entropy_first": entropies[0],
        "weight_entropy_last": entropies[-1],
        "feature_mean": scorer.feature_mean.detach().cpu().tolist(),
        "feature_std": scorer.feature_std.detach().cpu().tolist(),
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
    }


def _generate_shared_source(
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
        raise RuntimeError("shared source scorer is not active")
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    entropy_sum = 0.0
    max_weight_sum = 0.0
    matching_weight_sum = 0.0
    positions = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            names, sources = _selected_source_logits(
                base=base,
                units=units,
                ids=ids,
                heldout=_ACTIVE_HELDOUT,
                include_heldout=_ACTIVE_INCLUDE_HELDOUT,
            )
            weights = _ACTIVE_SCORER.weights_from_logits(sources)
            probabilities = torch.softmax(sources.float(), dim=-1)
            mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=0).clamp_min(1e-12)
            next_id = int(mixture.argmax().item())
            output.append(next_id)
            entropy_sum += float((-(weights * weights.clamp_min(1e-12).log()).sum()).detach().cpu())
            max_weight_sum += float(weights.max().detach().cpu())
            if operator in names:
                matching_weight_sum += float(weights[names.index(operator)].detach().cpu())
            positions += 1
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == eos_id:
                break

    return output, {
        "mean_weight_entropy": entropy_sum / max(1, positions),
        "mean_max_weight": max_weight_sum / max(1, positions),
        "mean_oracle_source_weight": matching_weight_sum / max(1, positions),
    }


def _aggregate_reports(reports: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = sequential._empty_counter()
    pairs = {
        f"{inner}->{outer}": sequential._empty_counter()
        for inner in FUNCTIONAL_OPERATORS
        for outer in FUNCTIONAL_OPERATORS
    }
    for report in reports:
        sequential._merge_counter(
            aggregate,
            {key: int(report["aggregate"][key]) for key in sequential._empty_counter()},
        )
        for pair_id, row in report["pairs"].items():
            sequential._merge_counter(
                pairs[pair_id],
                {key: int(row[key]) for key in sequential._empty_counter()},
            )
    return sequential._finalize_counter(aggregate), {
        pair_id: sequential._finalize_counter(counter) for pair_id, counter in pairs.items()
    }


def _subset_summary(pairs: Mapping[str, Mapping[str, Any]], *, heldout: str) -> dict[str, Any]:
    subsets = {
        "heldout_as_inner": lambda inner, outer: inner == heldout,
        "heldout_as_outer": lambda inner, outer: outer == heldout,
        "heldout_involved": lambda inner, outer: inner == heldout or outer == heldout,
        "heldout_both": lambda inner, outer: inner == heldout and outer == heldout,
        "heldout_neither": lambda inner, outer: inner != heldout and outer != heldout,
    }
    result: dict[str, Any] = {}
    for name, predicate in subsets.items():
        counter = sequential._empty_counter()
        for pair_id, row in pairs.items():
            inner, outer = pair_id.split("->", 1)
            if predicate(inner, outer):
                sequential._merge_counter(
                    counter,
                    {key: int(row[key]) for key in sequential._empty_counter()},
                )
        result[name] = sequential._finalize_counter(counter)
    return result


def evaluate_mode(
    scorer: SharedSourceScorer,
    cohorts: Sequence[Cohort],
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
    oracle._generate_oracle_operator = _generate_shared_source
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
    aggregate, pairs = _aggregate_reports(reports)
    return {
        "include_heldout_source": include_heldout,
        "source_count_eval": 1 + len(FUNCTIONAL_OPERATORS) - (0 if include_heldout else 1),
        "aggregate": aggregate,
        "pairs": pairs,
        "subsets": _subset_summary(pairs, heldout=heldout),
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
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    cohorts = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda item: int(item.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")

    batch = collect_training_batch(
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=train_examples_per_operator,
        data_seed=train_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    scorer, fit_report = fit_shared_scorer(
        batch,
        hidden_size=scorer_hidden_size,
        learning_rate=scorer_learning_rate,
        steps=scorer_steps,
        batch_positions=scorer_batch_positions,
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

    inner_without = without["subsets"]["heldout_as_inner"]["inner_accuracy"]
    inner_with = with_source["subsets"]["heldout_as_inner"]["inner_accuracy"]
    outer_without = without["subsets"]["heldout_as_outer"]["oracle_intermediate_outer_accuracy"]
    outer_with = with_source["subsets"]["heldout_as_outer"]["oracle_intermediate_outer_accuracy"]

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "leave_one_unit_out_identity_free_source_fusion_pilot",
        "claim_boundary": (
            "one functional specialist is excluded from both controller training examples and the training source pool; "
            "the fusion scorer receives no operator token, prompt embedding, source identity, or source slot embedding; "
            "the same scorer is shared across sources and is evaluated when the held-out specialist is first added; "
            "base remains a distinguished anchor for relative field features; stage boundaries remain external; "
            "two examples per ordered pair and one scorer initialization make this a pilot rather than a final estimate"
        ),
        "heldout_operator": heldout,
        "train_seed": train_seed,
        "scorer_seed": scorer_seed,
        "data_seed": data_seed,
        "scorer_fit": fit_report,
        "without_heldout_source": without,
        "with_heldout_source": with_source,
        "causal_deltas": {
            "heldout_as_inner_inner_accuracy": float(inner_with) - float(inner_without),
            "heldout_as_outer_oracle_outer_accuracy": float(outer_with) - float(outer_without),
            "heldout_involved_end_to_end_accuracy": float(
                with_source["subsets"]["heldout_involved"]["end_to_end_accuracy"]
            )
            - float(without["subsets"]["heldout_involved"]["end_to_end_accuracy"]),
        },
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test identity-free shared source scoring on an unseen specialist"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--train-examples-per-operator", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--scorer-hidden-size", type=int, default=16)
    parser.add_argument("--scorer-learning-rate", type=float, default=0.02)
    parser.add_argument("--scorer-steps", type=int, default=400)
    parser.add_argument("--scorer-batch-positions", type=int, default=256)
    parser.add_argument("--scorer-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_EVAL_SEED)
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
