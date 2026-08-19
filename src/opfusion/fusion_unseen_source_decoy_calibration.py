from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from opfusion import fusion_unseen_source_residual_gating as residual
from opfusion import fusion_unseen_source_residual_gating_runner as residual_runner
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config


FUNCTIONAL_OPERATORS = residual.FUNCTIONAL_OPERATORS
DEFAULT_TRAIN_SEED = 741_000
DEFAULT_EVAL_SEED = 741_500
DEFAULT_SKETCH_SEED = 743_000


@dataclass(frozen=True)
class DecoyTrainingBatch:
    features: torch.Tensor
    source_logits: torch.Tensor
    target_ids: torch.Tensor
    decoy_source_logits: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.features.shape[0])


def build_cross_prefix_decoys(source_logits: torch.Tensor) -> torch.Tensor:
    """Build plausible-but-context-mismatched sources from other training prefixes.

    For each position p, choose a specialist delta from a distant position q and
    transplant that centered delta onto p's Base field. No source/operator label
    is used. The donor specialist cycles across available seen specialists.
    """
    if source_logits.ndim != 3 or source_logits.shape[1] < 2:
        raise ValueError("source_logits must be [positions, base+specialists, vocabulary]")
    positions, source_count, _ = source_logits.shape
    if positions < 2:
        raise ValueError("at least two positions are required for cross-prefix decoys")
    specialist_count = source_count - 1
    device = source_logits.device
    position_index = torch.arange(positions, device=device)
    offset = max(1, positions // 2)
    donor_position = (position_index + offset) % positions
    donor_specialist = 1 + (position_index % specialist_count)
    donor = source_logits[donor_position, donor_specialist].float()
    donor_base = source_logits[donor_position, 0].float()
    delta = donor - donor_base
    delta = delta - delta.mean(dim=-1, keepdim=True)
    current_base = source_logits[:, 0].float()
    return current_base + delta


def batched_base_relative_features(
    source_logits: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Vectorized counterpart of residual._base_relative_features."""
    if source_logits.ndim != 3 or source_logits.shape[1] < 2:
        raise ValueError("source_logits must be [batch, base+specialists, vocabulary]")
    if source_logits.shape[-1] != projection.shape[0]:
        raise ValueError("projection vocabulary mismatch")

    logits = source_logits.float()
    log_prob = torch.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    vocabulary = int(logits.shape[-1])

    base_log_prob = log_prob[:, 0, :]
    base_prob = prob[:, 0, :]
    specialist_log_prob = log_prob[:, 1:, :]
    specialist_prob = prob[:, 1:, :]

    entropy = -(specialist_prob * specialist_log_prob).sum(dim=-1) / max(
        math.log(max(2, vocabulary)), 1e-8
    )
    top2 = specialist_prob.topk(k=min(2, vocabulary), dim=-1).values
    max_probability = top2[..., 0]
    top_margin = top2[..., 0] - (top2[..., 1] if vocabulary >= 2 else 0.0)
    kl_to_base = (
        specialist_prob * (specialist_log_prob - base_log_prob.unsqueeze(1))
    ).sum(dim=-1).clamp_min(0.0)
    top_agrees_base = (
        specialist_prob.argmax(dim=-1) == base_prob.argmax(dim=-1).unsqueeze(1)
    ).to(specialist_prob.dtype)

    centered = log_prob - log_prob.mean(dim=-1, keepdim=True)
    base_field = centered[:, 0, :]
    specialist_field = centered[:, 1:, :]
    delta = specialist_field - base_field.unsqueeze(1)
    delta_rms = delta.pow(2).mean(dim=-1).sqrt()
    scale = math.sqrt(float(vocabulary))
    base_sketch = (base_field @ projection / scale).unsqueeze(1).expand(
        -1, delta.shape[1], -1
    )
    source_sketch = specialist_field @ projection / scale
    delta_sketch = delta @ projection / scale

    summaries = torch.stack(
        [
            entropy,
            max_probability,
            top_margin,
            torch.log1p(kl_to_base),
            top_agrees_base,
            torch.log1p(delta_rms),
        ],
        dim=-1,
    )
    return torch.cat([summaries, base_sketch, source_sketch, delta_sketch], dim=-1)


def make_decoy_batch(batch: residual.ResidualTrainingBatch) -> DecoyTrainingBatch:
    return DecoyTrainingBatch(
        features=batch.features,
        source_logits=batch.source_logits,
        target_ids=batch.target_ids,
        decoy_source_logits=build_cross_prefix_decoys(batch.source_logits),
    )


def fit_with_decoys(
    scorer: residual.IndependentResidualGate,
    batch: DecoyTrainingBatch,
    *,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    real_gate_penalty: float,
    decoy_gate_penalty: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    scorer.set_normalization(batch.features)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=1e-4)

    losses: list[float] = []
    nlls: list[float] = []
    real_gate_means: list[float] = []
    decoy_gate_means: list[float] = []
    scorer.train()
    for _ in range(steps):
        if batch_positions >= batch.positions:
            indices = torch.arange(batch.positions, device=device)
        else:
            indices = torch.randint(0, batch.positions, (batch_positions,), device=device)

        real_sources = batch.source_logits.index_select(0, indices)
        decoy_sources = batch.decoy_source_logits.index_select(0, indices).unsqueeze(1)
        augmented_sources = torch.cat([real_sources, decoy_sources], dim=1)
        features = batched_base_relative_features(augmented_sources, scorer.projection)
        gates = scorer.gates_from_features(features)
        fused = residual.fuse_residual_logits(augmented_sources, gates)
        target = batch.target_ids.index_select(0, indices)
        nll = torch.nn.functional.cross_entropy(fused, target)
        real_gates = gates[:, :-1]
        decoy_gate = gates[:, -1]
        loss = (
            nll
            + float(real_gate_penalty) * real_gates.mean()
            + float(decoy_gate_penalty) * decoy_gate.mean()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        nlls.append(float(nll.detach().cpu()))
        real_gate_means.append(float(real_gates.mean().detach().cpu()))
        decoy_gate_means.append(float(decoy_gate.mean().detach().cpu()))

    scorer.eval()
    return {
        "positions": batch.positions,
        "specialist_count_train": int(batch.source_logits.shape[1] - 1),
        "feature_size": scorer.feature_size,
        "sketch_size": scorer.sketch_size,
        "max_gate": scorer.max_gate,
        "real_gate_penalty": real_gate_penalty,
        "decoy_gate_penalty": decoy_gate_penalty,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "nll_first": nlls[0],
        "nll_last": nlls[-1],
        "real_gate_mean_first": real_gate_means[0],
        "real_gate_mean_last": real_gate_means[-1],
        "decoy_gate_mean_first": decoy_gate_means[0],
        "decoy_gate_mean_last": decoy_gate_means[-1],
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
    hidden_size: int,
    max_gate: float,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    real_gate_penalty: float,
    decoy_gate_penalty: float,
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
    cohorts: Sequence[Cohort] = sorted(
        discover_cohorts(root, "fusion-factory"),
        key=lambda cohort: int(cohort.metadata.get("seed", 0)),
    )
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")
    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)

    torch.manual_seed(scorer_seed)
    scorer = residual.IndependentResidualGate(
        vocabulary_size=tokenizer.vocab_size,
        sketch_size=sketch_size,
        hidden_size=hidden_size,
        sketch_seed=sketch_seed,
        max_gate=max_gate,
    ).to(device)
    base_batch = residual.collect_training_batch(
        scorer,
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=train_examples_per_operator,
        data_seed=train_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    batch = make_decoy_batch(base_batch)
    fit_report = fit_with_decoys(
        scorer,
        batch,
        learning_rate=learning_rate,
        steps=steps,
        batch_positions=batch_positions,
        real_gate_penalty=real_gate_penalty,
        decoy_gate_penalty=decoy_gate_penalty,
        seed=scorer_seed,
        device=device,
    )

    residual._generate_residual = residual_runner._generate_residual_compatible
    without = residual.evaluate_mode(
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
    with_source = residual.evaluate_mode(
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
        "evaluation_role": "leave_one_unit_out_cross_prefix_decoy_calibration_pilot",
        "claim_boundary": (
            "held-out specialist is absent from training data and source pool; gate receives no operator, prompt embedding, "
            "source identity, or slot; decoys are centered seen-specialist deltas transplanted from distant training prefixes; "
            "one scorer initialization, two examples per ordered pair, Base privileged, external stage boundaries remain"
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
            "heldout_involved_e2e": float(
                with_source["subsets"]["heldout_involved"]["end_to_end_accuracy"]
            )
            - float(without["subsets"]["heldout_involved"]["end_to_end_accuracy"]),
            "heldout_neither_e2e": float(
                with_source["subsets"]["heldout_neither"]["end_to_end_accuracy"]
            )
            - float(without["subsets"]["heldout_neither"]["end_to_end_accuracy"]),
            "all_e2e": float(with_source["aggregate"]["end_to_end_accuracy"])
            - float(without["aggregate"]["end_to_end_accuracy"]),
        },
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate identity-free residual gates with cross-prefix null decoys"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--heldout", choices=FUNCTIONAL_OPERATORS, required=True)
    parser.add_argument("--train-examples-per-operator", type=int, default=24)
    parser.add_argument("--max-positions-per-cohort", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--sketch-size", type=int, default=32)
    parser.add_argument("--sketch-seed", type=int, default=DEFAULT_SKETCH_SEED)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-gate", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-positions", type=int, default=256)
    parser.add_argument("--real-gate-penalty", type=float, default=0.01)
    parser.add_argument("--decoy-gate-penalty", type=float, default=0.1)
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
        hidden_size=args.hidden_size,
        max_gate=args.max_gate,
        learning_rate=args.learning_rate,
        steps=args.steps,
        batch_positions=args.batch_positions,
        real_gate_penalty=args.real_gate_penalty,
        decoy_gate_penalty=args.decoy_gate_penalty,
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
