#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-foreground}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_BATCH="${TRAIN_BATCH:-$ROOT/.venv/bin/opfusion-train-batch-design}"
AUDIT_DATA="${AUDIT_DATA:-$ROOT/.venv/bin/opfusion-audit-data-design}"
EVALUATE="${EVALUATE:-$ROOT/.venv/bin/opfusion-evaluate-fusion}"
REPO_AUDIT="${REPO_AUDIT:-$ROOT/.venv/bin/opfusion-audit}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-64}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-256}"
LOCK_FILE="${LOCK_FILE:-$ROOT/runs/model_design_pilot/pilot.lock}"

if [[ "$MODE" == "detach" && "${OPFUSION_PILOT_CHILD:-0}" != "1" ]]; then
  mkdir -p logs runs/model_design_pilot
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="logs/model_design_pilot_${stamp}.log"
  command=(env OPFUSION_PILOT_CHILD=1 bash "$0" foreground)
  if command -v systemd-inhibit >/dev/null 2>&1; then
    command=(systemd-inhibit --what=sleep:shutdown --why="bias fusion model-design pilot" --mode=block "${command[@]}")
  fi
  nohup "${command[@]}" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > runs/model_design_pilot/pilot.pid
  echo "started model-design pilot PID $pid; log: $log"
  exit 0
fi

for executable in "$PYTHON" "$TRAIN_BATCH" "$AUDIT_DATA" "$EVALUATE" "$REPO_AUDIT"; do
  if [[ ! -x "$executable" ]]; then
    echo "missing executable: $executable; run bash scripts/bootstrap_arch_linux.sh" >&2
    exit 1
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 1
fi

mkdir -p runs/model_design_pilot evaluations/model_design_pilot audits/model_design_pilot logs
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another model-design pilot holds $LOCK_FILE" >&2
  exit 73
fi

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the model-design pilot")
props = torch.cuda.get_device_properties(0)
print(f"CUDA device: {props.name}; VRAM={props.total_memory / 1024**3:.1f} GiB")
PY
"$PYTHON" -m pytest -q
"$REPO_AUDIT" . --data-samples-per-operator 32

conditions=(
  identity_unanchored
  identity_retention
  weak_unanchored
  weak_retention
)

for condition in "${conditions[@]}"; do
  config="configs/experiments/model_design_pilot_${condition}.yaml"
  output="runs/model_design_pilot/${condition}"
  echo "=== MODEL DESIGN PILOT: ${condition} ==="
  "$AUDIT_DATA" \
    --config "$config" \
    --samples-per-operator "$AUDIT_SAMPLES" \
    --out "audits/model_design_pilot/${condition}.json"
  "$TRAIN_BATCH" --config "$config" --plan-only
  "$TRAIN_BATCH" --config "$config"
  manifest="$output/seed_0/fusion_subsets/subset_31.json"
  for split in validation test; do
    "$EVALUATE" \
      --config "$config" \
      --manifest "$manifest" \
      --split "$split" \
      --examples-per-operator "$EVAL_EXAMPLES" \
      --out "evaluations/model_design_pilot/${condition}_${split}.json"
  done
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path
conditions = ["identity_unanchored", "identity_retention", "weak_unanchored", "weak_retention"]
records = []
for condition in conditions:
    root = Path("runs/model_design_pilot") / condition / "seed_0"
    selected = {}
    for complete in root.glob("*/complete.json"):
        payload = json.loads(complete.read_text(encoding="utf-8"))
        selected[payload["job_id"]] = {
            "selected_step": payload.get("selected_step"),
            "selection_score": payload.get("selection_score"),
            "selected_checkpoint": payload.get("selected_checkpoint"),
            "final_checkpoint": payload.get("final_checkpoint"),
        }
    records.append({
        "condition": condition,
        "selected_endpoints": selected,
        "validation_report": f"evaluations/model_design_pilot/{condition}_validation.json",
        "test_report": f"evaluations/model_design_pilot/{condition}_test.json",
    })
path = Path("evaluations/model_design_pilot/index.json")
path.write_text(json.dumps({"conditions": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY

cat <<'EOF'
Model-design pilot completed.
Do not choose a production condition from training loss alone. Compare:
  1. relevant-specialist validation/test accuracy;
  2. raw-sum and bias-mean trace validity and EOS accuracy;
  3. divergence/agreement to the matched joint;
  4. selected checkpoint steps versus final steps;
  5. retention regularization logs for inactive-operator drift.
The guarded production candidate is weak_multitask + retention. Enable it only
after these reports support that choice.
EOF
