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
from opfusion import fusion_unseen_source_generalization as summary_baseline
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
DEFAULT_TRAIN_SEED = 741_000
DEFAULT_EVAL_SEED = 741_500
DEFAULT_SKETCH_SEED = 743_000


class FieldSketchSourceScorer(nn.Module):
    """Shared permutation-equivariant scorer over compressed full logit fields.

    The projection is fixed and token-coordinate aligned, so the scorer can learn
    geometry in vocabulary space without receiving source IDs or source slots.
    Base remains a distinguished reference field.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        sketch_size: int = 32,
        hidden_size: int = 64,
        sketch_seed: int = DEFAULT_SKETCH_SEED,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 1 or sketch_size <= 0 or hidden_size <= 0:
            raise ValueError("invalid field-sketch dimensions")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sketch_seed))
        projection = torch.randn(vocabulary_size, sketch_size, generator=generator)
        projection = projection / math.sqrt(float(sketch_size))
        self.register_buffer("projection", projection)
        self.sketch_size = int(sketch_size)
        self.summary_size = len(summary_baseline.FEATURE_NAMES)
        self.feature_size = self.summary_size + 4 * self.sketch_size
        self.net = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(self.feature_size))
        self.register_buffer("feature_std", torch.ones(self.feature_size))

    def field_features(self, source_logits: torch.Tensor) -> torch.Tensor:
        if source_logits.ndim != 2 or source_logits.shape[1] != self.projection.shape[0]:
            raise ValueError("source_logits must match [sources, vocabulary_size]")
        logits = source_logits.float()
        log_prob = torch.log_softmax(logits, dim=-1)
        centered_log_prob = log_prob - log_prob.mean(dim=-1, keepdim=True)
        base_field = centered_log_prob[0].unsqueeze(0)
        delta = centered_log_prob - base_field
        scale = math.sqrt(float(logits.shape[-1]))
        source_sketch = centered_log_prob @ self.projection / scale
        delta_sketch = delta @ self.projection / scale

        source_count = int(source_logits.shape[0])
        if source_count <= 1:
            raise ValueError("at least two sources are required")
        peer_mean = (delta_sketch.sum(dim=0, keepdim=True) - delta_sketch) / float(source_count - 1)
        peer_residual = delta_sketch - peer_mean
        base_sketch = source_sketch[0].unsqueeze(0).expand(source_count, -1)
        summaries = summary_baseline._source_features(source_logits)
        return torch.cat(
            [summaries, base_sketch, delta_sketch, peer_mean, peer_residual],
            dim=-1,
        )

    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.feature_size:
            raise ValueError("features must have shape [positions, sources, feature_size]")
        flat = features.reshape(-1, self.feature_size).float()
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-4))

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)

    def weights_from_logits(self, source_logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score_features(self.field_features(source_logits)), dim=-1)


@dataclass(frozen=True)
class SketchTrainingBatch:
    features: torch.Tensor
    target_source_probabilities: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.features.shape[0])


def collect_sketch_training_batch(
    scorer: FieldSketchSourceScorer,
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    heldout: str,
    examples_per_operator: int,
    data_seed: int,
    max_positions_per_cohort: int,
    device: torch.device,
) -> SketchTrainingBatch:
    if heldout not in FUNCTIONAL_OPERATORS:
        raise ValueError(f"heldout must be one of {FUNCTIONAL_OPERATORS}")
    all_features: list[torch.Tensor] = []
    all_target_probabilities: list[torch.Tensor] = []

    for cohort in cohorts[:2]:
        run = load_run_config(cohort.config_path)
        tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
        factory = SyntheticTraceFactory(tokenizer, run.data)
        if tokenizer.vocab_size != int(scorer.projection.shape[0]):
            raise RuntimeError("all cohorts must share the field-sketch vocabulary")
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
                        _, sources = summary_baseline._selected_source_logits(
                            base=base,
                            units=units,
                            ids=ids,
                            heldout=heldout,
                            include_heldout=False,
                        )
                        all_features.append(scorer.field_features(sources).detach().cpu())
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
        raise RuntimeError("no field-sketch training positions collected")
    return SketchTrainingBatch(
        features=torch.stack(all_features, dim=0).to(device),
        target_source_probabilities=torch.stack(all_target_probabilities, dim=0).to(device),
    )


def fit_field_scorer(
    scorer: FieldSketchSourceScorer,
    batch: SketchTrainingBatch,
    *,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    scorer.set_normalization(batch.features)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    entropies: list[float] = []
    scorer.train()
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
    return {
        "positions": batch.positions,
        "source_count_train": int(batch.features.shape[1]),
        "feature_size": scorer.feature_size,
        "sketch_size": scorer.sketch_size,
        "optimization_first": losses[0],
        "optimization_last": losses[-1],
        "weight_entropy_first": entropies[0],
        "weight_entropy_last": entropies[-1],
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
    }


def run_experiment(
    *,
    root: Path,
    heldout: str,
    train_examples_per_operator: int,
    max_positions_per_cohort: int,
    train_seed: int,
    sketch_size: int,
    sketch_seed: int,
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
    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)

    torch.manual_seed(scorer_seed)
    scorer = FieldSketchSourceScorer(
        vocabulary_size=tokenizer.vocab_size,
        sketch_size=sketch_size,
        hidden_size=scorer_hidden_size,
        sketch_seed=sketch_seed,
    ).to(device)
    batch = collect_sketch_training_batch(
        scorer,
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=train_examples_per_operator,
        data_seed=train_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    fit_report = fit_field_scorer(
        scorer,
        batch,
        learning_rate=scorer_learning_rate,
        steps=scorer_steps,
        batch_positions=scorer_batch_positions,
        seed=scorer_seed,
        device=device,
    )
    without = summary_baseline.evaluate_mode(
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
    with_source = summary_baseline.evaluate_mode(
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

    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "leave_one_unit_out_full_field_sketch_pilot",
        "claim_boundary": (
            "held-out specialist is absent from training examples and source pool; scorer receives no operator, prompt, "
            "source identity, or slot embedding; a fixed vocabulary-coordinate projection preserves richer field geometry; "
            "Base remains a distinguished reference; external stage boundaries remain; two examples per pair and one "
            "scorer initialization make this a pilot"
        ),
        "heldout_operator": heldout,
        "train_seed": train_seed,
        "scorer_seed": scorer_seed,
        "sketch_seed": sketch_seed,
        "data_seed": data_seed,
        "scorer_fit": fit_report,
        "without_heldout_source": without,
        "with_heldout_source": with_source,
        "causal_deltas": {
            "heldout_as_inner_inner_accuracy": float(
                with_source["subsets"]["heldout_as_inner"]["inner_accuracy"]
            )
            - float(without["subsets"]["heldout_as_inner"]["inner_accuracy"]),
            "heldout_as_outer_oracle_outer_accuracy": float(
                with_source["subsets"]["heldout_as_outer"]["oracle_intermediate_outer_accuracy"]
            )
            - float(without["subsets"]["heldout_as_outer"]["oracle_intermediate_outer_accuracy"]),
            "heldout_involved_end_to_end_accuracy": float(
                with_source["subsets"]["heldout_involved"]["end_to_end_accuracy"]
            )
            - float(without["subsets"]["heldout_involved"]["end_to_end_accuracy"]),
            "heldout_neither_end_to_end_accuracy": float(
                with_source["subsets"]["heldout_neither"]["end_to_end_accuracy"]
            )
            - float(without["subsets"]["heldout_neither"]["end_to_end_accuracy"]),
        },
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test richer full-field sketches for identity-free unseen-source fusion"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--train-examples-per-operator", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--sketch-size", type=int, default=32)
    parser.add_argument("--sketch-seed", type=int, default=DEFAULT_SKETCH_SEED)
    parser.add_argument("--scorer-hidden-size", type=int, default=64)
    parser.add_argument("--scorer-learning-rate", type=float, default=0.01)
    parser.add_argument("--scorer-steps", type=int, default=600)
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
        sketch_size=args.sketch_size,
        sketch_seed=args.sketch_seed,
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
