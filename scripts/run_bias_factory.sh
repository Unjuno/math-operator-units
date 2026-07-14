#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUNS_DIR="runs/bias_factory"
mkdir -p "$RUNS_DIR" logs

CONFIGS=(
    "nano:configs/experiments/bias_factory/bias_factory_nano.yaml"
    "small:configs/experiments/bias_factory/bias_factory_small.yaml"
    "medium:configs/experiments/bias_factory/bias_factory_medium.yaml"
    "1m:configs/experiments/bias_factory/bias_factory_1m.yaml"
)

for entry in "${CONFIGS[@]}"; do
    IFS=":" read -r name cfg <<< "$entry"
    complete="$RUNS_DIR/$name/seed_0/base_common/complete.json"

    if [[ -f "$complete" ]]; then
        echo "Reusing completed: $name ($complete)"
        continue
    fi

    echo "========================================"
    echo "Training: $name"
    echo "Config:  $cfg"
    echo "========================================"

    .venv/bin/opfusion-train-one-design \
        --config "$cfg" --job base.common --seed 0 \
        2>&1 | tee "logs/bias_factory_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"

    exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        echo "FAILED: $name (exit code $exit_code)" >&2
        exit 1
    fi
    echo "Completed: $name"
done

echo ""
echo "========================================"
echo "Bias Factory complete — 4 models × 5 snapshots"
echo "Checkpoints at steps 39/156/635/2559/9800"
echo "  ≈ 4K/16K/65K/262K/1M examples per operator"
echo "========================================"
