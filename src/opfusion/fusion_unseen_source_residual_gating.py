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
from opfusion import fusion_unseen_source_generalization as unseen
from opfusion.fusion_eval import _load_model
from opfusion.fusion_search import Cohort, discover_cohorts
from opfusion.fusion_verify import _next_logits
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


FUNCTIONAL_OPERATORS = sequential.FUNCTIONAL_OPERATORS
DEFAULT_TRAIN_SEED = 741_000
DEFAULT_EVAL_SEED = 741_500
DEFAULT_SKETCH_SEED = 743_000
BASE_RELATIVE_SUMMARY_NAMES = (
    "entropy",
    "max_probability",
    "top_margin",
    "log1p_kl_to_base",
    "top_agrees_base",
    "log1p_delta_rms",
)

_ACTIVE_SCORER: "IndependentResidualGate | None" = None
_ACTIVE_HELDOUT: str | None = None
_ACTIVE_INCLUDE_HELDOUT = True


def _base_relative_features(source_logits: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """Features for specialists only; each row depends only on that specialist and Base."""
    if source_logits.ndim != 2 or source_logits.shape[0] < 2:
        raise ValueError("source_logits must have shape [base+specialists, vocabulary]")
    if source_logits.shape[1] != projection.shape[0]:
        raise ValueError("projection vocabulary mismatch")
    logits = source_logits.float()
    log_prob = torch.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    vocabulary = int(logits.shape[-1])

    base_log_prob = log_prob[0]
    base_prob = prob[0]
    specialist_log_prob = log_prob[1:]
    specialist_prob = prob[1:]
    entropy = -(specialist_prob * specialist_log_prob).sum(dim=-1) / max(
        math.log(max(2, vocabulary)), 1e-8
    )
    top2 = specialist_prob.topk(k=min(2, vocabulary), dim=-1).values
    max_probability = top2[:, 0]
    top_margin = top2[:, 0] - (top2[:, 1] if vocabulary >= 2 else 0.0)
    kl_to_base = (
        specialist_prob * (specialist_log_prob - base_log_prob.unsqueeze(0))
    ).sum(dim=-1).clamp_min(0.0)
    top_agrees_base = (
        specialist_prob.argmax(dim=-1) == base_prob.argmax()
    ).to(specialist_prob.dtype)

    centered = log_prob - log_prob.mean(dim=-1, keepdim=True)
    base_field = centered[0]
    specialist_field = centered[1:]
    delta = specialist_field - base_field.unsqueeze(0)
    delta_rms = delta.pow(2).mean(dim=-1).sqrt()
    scale = math.sqrt(float(vocabulary))
    base_sketch = (base_field @ projection / scale).unsqueeze(0).expand(delta.shape[0], -1)
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


class IndependentResidualGate(nn.Module):
    """Shared independent specialist gate with exact source-set extension invariance at a fixed prefix."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        sketch_size: int = 32,
        hidden_size: int = 64,
        sketch_seed: int = DEFAULT_SKETCH_SEED,
        max_gate: float = 4.0,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 1 or sketch_size <= 0 or hidden_size <= 0 or max_gate <= 0:
            raise ValueError("invalid residual gate configuration")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sketch_seed))
        projection = torch.randn(vocabulary_size, sketch_size, generator=generator)
        projection = projection / math.sqrt(float(sketch_size))
        self.register_buffer("projection", projection)
        self.sketch_size = int(sketch_size)
        self.max_gate = float(max_gate)
        self.feature_size = len(BASE_RELATIVE_SUMMARY_NAMES) + 3 * self.sketch_size
        self.net = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(self.feature_size))
        self.register_buffer("feature_std", torch.ones(self.feature_size))

    def features_from_logits(self, source_logits: torch.Tensor) -> torch.Tensor:
        return _base_relative_features(source_logits, self.projection)

    def set_normalization(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.feature_size:
            raise ValueError("features must have shape [positions, specialists, feature_size]")
        flat = features.reshape(-1, self.feature_size).float()
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-4))

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.net(normalized).squeeze(-1)

    def gates_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_gate * torch.sigmoid(self.score_features(features))

    def gates_from_logits(self, source_logits: torch.Tensor) -> torch.Tensor:
        return self.gates_from_features(self.features_from_logits(source_logits))


def fuse_residual_logits(source_logits: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    if source_logits.ndim == 2:
        if gates.ndim != 1 or gates.shape[0] != source_logits.shape[0] - 1:
            raise ValueError("gate count must equal specialist count")
        base = source_logits[0].float()
        delta = source_logits[1:].float() - base.unsqueeze(0)
        delta = delta - delta.mean(dim=-1, keepdim=True)
        return base + (gates.unsqueeze(-1) * delta).sum(dim=0)
    if source_logits.ndim == 3:
        if gates.ndim != 2 or gates.shape[:2] != (source_logits.shape[0], source_logits.shape[1] - 1):
            raise ValueError("batched gate shape mismatch")
        base = source_logits[:, 0, :].float()
        delta = source_logits[:, 1:, :].float() - base.unsqueeze(1)
        delta = delta - delta.mean(dim=-1, keepdim=True)
        return base + (gates.unsqueeze(-1) * delta).sum(dim=1)
    raise ValueError("source_logits must be rank 2 or 3")


@dataclass(frozen=True)
class ResidualTrainingBatch:
    features: torch.Tensor
    source_logits: torch.Tensor
    target_ids: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.features.shape[0])


def _selected_logits(
    *,
    base: torch.nn.Module,
    units: Mapping[str, torch.nn.Module],
    ids: torch.Tensor,
    heldout: str,
    include_heldout: bool,
) -> tuple[list[str], torch.Tensor]:
    operators = [op for op in FUNCTIONAL_OPERATORS if include_heldout or op != heldout]
    names = ["Base", *operators]
    logits = [_next_logits(base, ids)]
    logits.extend(_next_logits(units[op], ids) for op in operators)
    return names, torch.stack(logits, dim=0)


def collect_training_batch(
    scorer: IndependentResidualGate,
    cohorts: Sequence[Cohort],
    *,
    root: Path,
    heldout: str,
    examples_per_operator: int,
    data_seed: int,
    max_positions_per_cohort: int,
    device: torch.device,
) -> ResidualTrainingBatch:
    features: list[torch.Tensor] = []
    logits_rows: list[torch.Tensor] = []
    targets: list[int] = []
    for cohort in cohorts[:2]:
        run = load_run_config(cohort.config_path)
        tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
        if tokenizer.vocab_size != scorer.projection.shape[0]:
            raise RuntimeError("all cohorts must share a vocabulary")
        factory = SyntheticTraceFactory(tokenizer, run.data)
        base = _load_model(cohort.base_checkpoint, device=device, tokenizer=tokenizer)
        units = {
            op: _load_model(path, device=device, tokenizer=tokenizer)
            for op, path in cohort.unit_checkpoints.items()
            if op in FUNCTIONAL_OPERATORS and op != heldout
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
                        _, source_logits = _selected_logits(
                            base=base,
                            units=units,
                            ids=ids,
                            heldout=heldout,
                            include_heldout=False,
                        )
                        features.append(scorer.features_from_logits(source_logits).detach().cpu())
                        logits_rows.append(source_logits.detach().float().cpu())
                        targets.append(int(target_id))
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
    if not features:
        raise RuntimeError("no residual-gate training positions collected")
    return ResidualTrainingBatch(
        features=torch.stack(features).to(device),
        source_logits=torch.stack(logits_rows).to(device),
        target_ids=torch.tensor(targets, dtype=torch.long, device=device),
    )


def fit_residual_gate(
    scorer: IndependentResidualGate,
    batch: ResidualTrainingBatch,
    *,
    learning_rate: float,
    steps: int,
    batch_positions: int,
    gate_penalty: float,
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
    gate_means: list[float] = []
    scorer.train()
    for _ in range(steps):
        if batch_positions >= batch.positions:
            indices = torch.arange(batch.positions, device=device)
        else:
            indices = torch.randint(0, batch.positions, (batch_positions,), device=device)
        feat = batch.features.index_select(0, indices)
        sources = batch.source_logits.index_select(0, indices)
        target = batch.target_ids.index_select(0, indices)
        gates = scorer.gates_from_features(feat)
        fused = fuse_residual_logits(sources, gates)
        nll = torch.nn.functional.cross_entropy(fused, target)
        loss = nll + float(gate_penalty) * gates.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        nlls.append(float(nll.detach().cpu()))
        gate_means.append(float(gates.mean().detach().cpu()))
    scorer.eval()
    return {
        "positions": batch.positions,
        "specialist_count_train": int(batch.features.shape[1]),
        "feature_size": scorer.feature_size,
        "sketch_size": scorer.sketch_size,
        "max_gate": scorer.max_gate,
        "gate_penalty": gate_penalty,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "nll_first": nlls[0],
        "nll_last": nlls[-1],
        "mean_gate_first": gate_means[0],
        "mean_gate_last": gate_means[-1],
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
    }


def _generate_residual(
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
        raise RuntimeError("residual gate is not active")
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    output: list[int] = []
    matching_gate_sum = 0.0
    gate_sum = 0.0
    max_gate_sum = 0.0
    positions = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            names, sources = _selected_logits(
                base=base,
                units=units,
                ids=ids,
                heldout=_ACTIVE_HELDOUT,
                include_heldout=_ACTIVE_INCLUDE_HELDOUT,
            )
            gates = _ACTIVE_SCORER.gates_from_logits(sources)
            fused = fuse_residual_logits(sources, gates)
            next_id = int(fused.argmax().item())
            output.append(next_id)
            gate_sum += float(gates.mean().detach().cpu())
            max_gate_sum += float(gates.max().detach().cpu())
            if operator in names[1:]:
                matching_gate_sum += float(gates[names[1:].index(operator)].detach().cpu())
            positions += 1
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == eos_id:
                break
    return output, {
        "mean_gate": gate_sum / max(1, positions),
        "mean_max_gate": max_gate_sum / max(1, positions),
        "mean_oracle_source_gate": matching_gate_sum / max(1, positions),
    }


def evaluate_mode(
    scorer: IndependentResidualGate,
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
    old_scorer, old_heldout, old_include = _ACTIVE_SCORER, _ACTIVE_HELDOUT, _ACTIVE_INCLUDE_HELDOUT
    _ACTIVE_SCORER = scorer
    _ACTIVE_HELDOUT = heldout
    _ACTIVE_INCLUDE_HELDOUT = include_heldout
    oracle._generate_oracle_operator = _generate_residual
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
        _ACTIVE_SCORER, _ACTIVE_HELDOUT, _ACTIVE_INCLUDE_HELDOUT = old_scorer, old_heldout, old_include
    aggregate, pairs = unseen._aggregate_reports(reports)
    return {
        "include_heldout_source": include_heldout,
        "aggregate": aggregate,
        "pairs": pairs,
        "subsets": unseen._subset_summary(pairs, heldout=heldout),
        "cohort_reports": reports,
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
    gate_penalty: float,
    scorer_seed: int,
    examples_per_pair: int,
    data_seed: int,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name
    )
    cohorts = sorted(discover_cohorts(root, "fusion-factory"), key=lambda c: int(c.metadata.get("seed", 0)))
    if len(cohorts) < 3:
        raise RuntimeError(f"expected three complete cohorts, found {len(cohorts)}")
    run = load_run_config(cohorts[0].config_path)
    tokenizer = FixedVocabTokenizer.from_config(root / run.tokenizer_config)
    torch.manual_seed(scorer_seed)
    scorer = IndependentResidualGate(
        vocabulary_size=tokenizer.vocab_size,
        sketch_size=sketch_size,
        hidden_size=hidden_size,
        sketch_seed=sketch_seed,
        max_gate=max_gate,
    ).to(device)
    batch = collect_training_batch(
        scorer,
        cohorts,
        root=root,
        heldout=heldout,
        examples_per_operator=train_examples_per_operator,
        data_seed=train_seed,
        max_positions_per_cohort=max_positions_per_cohort,
        device=device,
    )
    fit_report = fit_residual_gate(
        scorer,
        batch,
        learning_rate=learning_rate,
        steps=steps,
        batch_positions=batch_positions,
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
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_role": "leave_one_unit_out_independent_residual_gate_pilot",
        "claim_boundary": (
            "held-out specialist absent from training data/source pool; no operator, prompt, source identity, or slot input; "
            "existing source gates depend only on Base and that source, so adding another source cannot alter their gates at a fixed prefix; "
            "one scorer initialization and two examples per ordered pair; external stage boundaries remain"
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
            "heldout_involved_e2e": float(with_source["subsets"]["heldout_involved"]["end_to_end_accuracy"])
            - float(without["subsets"]["heldout_involved"]["end_to_end_accuracy"]),
            "heldout_neither_e2e": float(with_source["subsets"]["heldout_neither"]["end_to_end_accuracy"])
            - float(without["subsets"]["heldout_neither"]["end_to_end_accuracy"]),
            "all_e2e": float(with_source["aggregate"]["end_to_end_accuracy"])
            - float(without["aggregate"]["end_to_end_accuracy"]),
        },
        "production_go": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test null-invariant independent residual gating on unseen specialists")
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
    parser.add_argument("--gate-penalty", type=float, default=0.0)
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
        gate_penalty=args.gate_penalty,
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
