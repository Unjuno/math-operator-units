#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TEMPLATE="configs/experiments/bias_factory/joint_base.yaml"
RUNS_DIR="runs/bias_factory"
mkdir -p "$RUNS_DIR" logs

# effective_batch_size=512 → per-step per-operator ≈ 102
# snapshot steps for 4K/16K/65K/262K/1M examples per operator
# step = examples_per_operator / (eff_batch / 5)
SNAPSHOT_STEPS="39,157,637,2569,9804"

# model sizes: name, model_config, param_limit, max_steps
MODELS=(
  "nano:gpt_operator_nano_surface_v3.yaml:200000:9804"
  "small:gpt_operator_small_surface_v3.yaml:400000:9804"
  "medium:gpt_operator_medium_surface_v3.yaml:600000:9804"
  "1m:gpt_operator_1m_surface_v3.yaml:1100000:9804"
)

for entry in "${MODELS[@]}"; do
  IFS=":" read -r name model_config param_limit max_steps <<< "$entry"

  output_dir="$RUNS_DIR/$name"
  complete="$output_dir/seed_0/base_common/complete.json"

  if [[ -f "$complete" ]]; then
    echo "Reusing completed: $name"
    continue
  fi

  echo "Generating config for $name..."

  cfg="$RUNS_DIR/${name}_config.yaml"
  sed \
    -e "s/{size}/$name/g" \
    -e "s|{model_config}|$model_config|g" \
    -e "s/{param_limit}/$param_limit/g" \
    -e "s/{max_steps}/$max_steps/g" \
    -e "s/{snapshot_steps}/$SNAPSHOT_STEPS/g" \
    "$TEMPLATE" > "$cfg"

  echo "Training $name..."
  .venv/bin/opfusion-train-one-design \
    --config "$cfg" --job base.common --seed 0 \
    2>&1 | tee "logs/bias_factory_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"

  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "FAILED: $name" >&2; exit 1; }
  echo "Completed: $name"
done

echo "All bias factory models completed."
echo "Snapshots at steps: $SNAPSHOT_STEPS"
echo "  (~4K, 16K, 65K, 262K, 1M examples/operator)"
