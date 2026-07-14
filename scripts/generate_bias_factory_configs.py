#!/usr/bin/env python3
"""Generate all 24 bias-factory experiment configs.

Output:
  configs/experiments/bias_factory/
    joint_{size}.yaml         (4 files, 5 operators each)
    spec_{size}_{op}.yaml     (20 files, 1 operator each)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/experiments/bias_factory"
OUT.mkdir(parents=True, exist_ok=True)

OPERATORS = [
    ("sum", "aggregation.sum"),
    ("neg", "scalar.neg"),
    ("add", "scalar.add"),
    ("min", "scalar.min"),
    ("max", "scalar.max"),
]

SIZES = [
    ("nano",   "gpt_operator_nano_surface_v3.yaml",   200000),
    ("small",  "gpt_operator_small_surface_v3.yaml",  400000),
    ("medium", "gpt_operator_medium_surface_v3.yaml", 600000),
    ("1m",     "gpt_operator_1m_surface_v3.yaml",     1100000),
]

SNAPSHOT_STEPS = [39, 156, 635, 2559, 9800]


def _header(size: str) -> str:
    return f"""experiment:
  id: bias_factory_{size}
  output_dir: runs/bias_factory/{size}
  tokenizer_config: configs/tokenizer/operator_experiment_surface_v3.yaml
  base_model_id: base.common
  joint_model_ids: [joint.all_five.exposure_matched]
  seeds: [0]
  require_cuda: true
  precision: auto
  allow_tf32: false
  deterministic_algorithms: true
  continue_on_error: false

  model_design:
    base_target_mode: identity
    strict_experiment_fingerprint: false

  data:
    operand_min: -64
    operand_max: 64
    min_terms: 3
    max_terms: 8
    numeric_token_min: -1024
    numeric_token_max: 1024
    operand_ood_abs_min: 65
    operand_ood_abs_max: 80
    length_ood_min_terms: 9
    length_ood_max_terms: 10
    partition_modulus: 100
    train_bucket_end: 70
    validation_bucket_end: 85
    full_trace_weight: 100
    continuation_weight: 0
    terminal_weight: 0
    max_partition_attempts: 20000
    randomized_train_reduction: false

  train:
    response_only_loss: true
    effective_batch_size: 512
    micro_batch_size: 0
    micro_batch_candidates: [512, 256, 128, 64, 32, 16, 8, 4]
    max_steps: 9800
    eval_every: 1000
    eval_batches: 1
    eval_examples: 128
    generation_eval_every: 2000
    generation_eval_examples: 16
    generation_max_new_tokens: 256
    log_every: 200
    resume_every: 500
    checkpoint_every: 500
    checkpoint_steps: [{','.join(str(s) for s in SNAPSHOT_STEPS)}]
    optimizer:
      learning_rate: 0.0003
      min_learning_rate: 0.00003
      warmup_steps: 250
      weight_decay: 0.1
      beta1: 0.9
      beta2: 0.95
      grad_clip_norm: 1.0

  recovery:
    max_retries_per_job: 5
    minimum_micro_batch_size: 4
    non_finite_lr_factor: 0.5
    max_lr_reductions: 2
    restart_delay_seconds: 10
"""


def main() -> int:
    count = 0

    for size_name, model_cfg, param_limit in SIZES:
        # --- joint (all 5 operators) ---
        path = OUT / f"joint_{size_name}.yaml"
        text = _header(size_name)
        text += f"""  model_config: configs/model/{model_cfg}
  max_parameters: {param_limit}
  operators: [scalar.add, aggregation.sum, scalar.neg, scalar.min, scalar.max]
"""
        path.write_text(text)
        count += 1
        print(f"  {path.name}")

        # --- specialist (1 operator each) ---
        for op_short, op_full in OPERATORS:
            spec_name = f"{size_name}_{op_short}"
            exp_id = f"bias_factory_spec_{spec_name}"
            path = OUT / f"spec_{spec_name}.yaml"
            text = _header(size_name).replace(
                f"id: bias_factory_{size_name}",
                f"id: {exp_id}",
                1
            ).replace(
                f"output_dir: runs/bias_factory/{size_name}",
                f"output_dir: runs/bias_factory/spec_{spec_name}",
                1
            )
            text += f"""  model_config: configs/model/{model_cfg}
  max_parameters: {param_limit}
  operators: [{op_full}]
"""
            path.write_text(text)
            count += 1
            print(f"  {path.name}")

    print(f"\nGenerated {count} configs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
