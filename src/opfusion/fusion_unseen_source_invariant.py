from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from opfusion import fusion_unseen_source_generalization as baseline


FEATURE_NAMES = (
    "entropy",
    "max_probability",
    "top_margin",
    "log1p_kl_to_base",
    "top_agrees_base",
    "log1p_delta_rms",
)


def _source_features(source_logits: torch.Tensor) -> torch.Tensor:
    """Source-local/base-relative features invariant to insertion of other specialists."""
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
    top_index = prob.argmax(dim=-1)
    top_agrees_base = (top_index == top_index[0]).to(prob.dtype)

    delta = logits - logits[0].unsqueeze(0)
    delta = delta - delta.mean(dim=-1, keepdim=True)
    delta_rms = delta.pow(2).mean(dim=-1).sqrt()

    return torch.stack(
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


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    old_names = baseline.FEATURE_NAMES
    old_features = baseline._source_features
    baseline.FEATURE_NAMES = FEATURE_NAMES
    baseline._source_features = _source_features
    try:
        report = baseline.run_experiment(**kwargs)
    finally:
        baseline.FEATURE_NAMES = old_names
        baseline._source_features = old_features

    report["schema_version"] = 2
    report["evaluation_role"] = "leave_one_unit_out_base_relative_insertion_invariant_pilot"
    report["feature_mode"] = "base_relative_source_local"
    report["removed_set_dependent_features"] = [
        "log1p_kl_to_mean",
        "top_agrees_mean",
        "consensus_alignment",
    ]
    report["claim_boundary"] = (
        report["claim_boundary"]
        + "; fusion features are source-local/base-relative so appending a specialist does not change "
        "the raw features of pre-existing sources, although softmax normalization still couples their final weights"
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablate source-set-dependent features in unseen-specialist fusion"
    )
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
