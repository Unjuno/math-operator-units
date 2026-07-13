#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-foreground}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_BATCH="${TRAIN_BATCH:-$ROOT/.venv/bin/opfusion-train-batch-design}"
AUDIT_DATA="${AUDIT_DATA:-$ROOT/.venv/bin/opfusion-audit-data-design}"
EVALUATE="${EVALUATE:-$ROOT/.venv/bin/opfusion-evaluate-fusion}"
DIAGNOSTICS="${DIAGNOSTICS:-$ROOT/.venv/bin/opfusion-evaluate-unit-diagnostics}"
PAIR_AUDIT="${PAIR_AUDIT:-$ROOT/.venv/bin/opfusion-audit-pilot-pairs}"
REPO_AUDIT="${REPO_AUDIT:-$ROOT/.venv/bin/opfusion-audit}"
WATCHER="${WATCHER:-$ROOT/scripts/watch_model_design_pilot.sh}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-64}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-256}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
LOCK_FILE="${LOCK_FILE:-$ROOT/runs/model_design_pilot/pilot.lock}"
STATE_FILE="${STATE_FILE:-$ROOT/runs/model_design_pilot/pilot_state.json}"
PID_FILE="${PID_FILE:-$ROOT/runs/model_design_pilot/pilot.pid}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ "$MODE" != "foreground" && "$MODE" != "detach" ]]; then
  echo "usage: $0 [foreground|detach]" >&2
  exit 64
fi

if [[ "$MODE" == "detach" && "${OPFUSION_PILOT_CHILD:-0}" != "1" ]]; then
  mkdir -p logs runs/model_design_pilot
  if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "model-design pilot is already running with PID $existing_pid" >&2
      exit 73
    fi
  fi
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="logs/model_design_pilot_${stamp}.log"
  command=(env OPFUSION_PILOT_CHILD=1 PILOT_LOG="$log" CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" bash "$WATCHER")
  if command -v systemd-inhibit >/dev/null 2>&1; then
    command=(systemd-inhibit --what=sleep:shutdown --why="bias fusion model-design pilot" --mode=block "${command[@]}")
  fi
  nohup "${command[@]}" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  echo "started unattended model-design pilot PID $pid; log: $log"
  echo "status: bash scripts/status_model_design_pilot.sh"
  exit 0
fi

for executable in "$PYTHON" "$TRAIN_BATCH" "$AUDIT_DATA" "$EVALUATE" "$DIAGNOSTICS" "$PAIR_AUDIT" "$REPO_AUDIT"; do
  if [[ ! -x "$executable" ]]; then
    echo "missing executable: $executable; run bash scripts/bootstrap_arch_linux.sh" >&2
    exit 64
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 65
fi

mkdir -p runs/model_design_pilot evaluations/model_design_pilot audits/model_design_pilot logs

# Direct foreground runs own the lock. The watchdog owns it for detached runs.
if [[ "${OPFUSION_PILOT_WATCHED:-0}" != "1" ]]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "another model-design pilot holds $LOCK_FILE" >&2
    exit 73
  fi
fi

write_state() {
  local status="$1"
  local condition="${2:-}"
  local phase="${3:-}"
  local detail="${4:-}"
  "$PYTHON" - "$STATE_FILE" "$status" "$condition" "$phase" "$detail" "$$" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
existing = {}
if path.exists():
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
existing.update(
    {
        "status": sys.argv[2],
        "condition": sys.argv[3] or None,
        "phase": sys.argv[4] or None,
        "detail": sys.argv[5] or None,
        "worker_pid": int(sys.argv[6]),
        "updated_unix": time.time(),
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

check_disk() {
  local free_kb required_kb
  free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
  required_kb="$((MIN_FREE_GB * 1024 * 1024))"
  if [[ "$free_kb" -lt "$required_kb" ]]; then
    echo "need at least ${MIN_FREE_GB} GiB free disk; currently $((free_kb / 1024 / 1024)) GiB" >&2
    exit 66
  fi
}

condition_complete() {
  local condition="$1"
  local config="$2"
  local output="runs/model_design_pilot/${condition}"
  local marker="$output/pilot_condition_complete.json"
  local manifest="$output/seed_0/fusion_subsets/subset_31.json"

  # Never reuse a completion marker from a locally modified checkout.
  if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null || true)" ]]; then
    return 1
  fi
  "$PYTHON" - "$marker" "$manifest" "$config" "$output" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

marker, manifest, config, output = map(Path, sys.argv[1:])
condition = output.name
splits = ("validation",)
reports = [
    Path("evaluations/model_design_pilot") / f"{condition}_{split}.json"
    for split in splits
]
diagnostics = [
    Path("evaluations/model_design_pilot") / f"{condition}_{split}_units.json"
    for split in splits
]
for path in (marker, manifest, config, output / "experiment_contract.json", *reports, *diagnostics):
    if not path.is_file():
        raise SystemExit(1)
try:
    completion = json.loads(marker.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
    for path in (*reports, *diagnostics):
        json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    raise SystemExit(1)
config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
fingerprint = contract.get("fingerprint")
if not (
    completion.get("status") == "completed"
    and completion.get("git_commit") == commit
    and completion.get("config_sha256") == config_sha
    and completion.get("experiment_fingerprint") == fingerprint
    and manifest_payload.get("experiment_fingerprint") == fingerprint
    and tuple(completion.get("evaluation_splits", ())) == splits
):
    raise SystemExit(1)

def resolve_checkpoint(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path.cwd() / path

checkpoint_values = [
    manifest_payload.get("base_checkpoint"),
    manifest_payload.get("joint_reference_checkpoint"),
    *manifest_payload.get("unit_checkpoints", {}).values(),
]
if any(value is None or not resolve_checkpoint(value).is_file() for value in checkpoint_values):
    raise SystemExit(1)
complete_files = sorted((output / "seed_0").glob("*/complete.json"))
if len(complete_files) != 7:
    raise SystemExit(1)
for complete_path in complete_files:
    try:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(1)
    selected = payload.get("selected_checkpoint")
    final = payload.get("final_checkpoint")
    if selected is None or final is None:
        raise SystemExit(1)
    if not resolve_checkpoint(selected).is_file() or not resolve_checkpoint(final).is_file():
        raise SystemExit(1)
raise SystemExit(0)
PY
}

mark_condition_complete() {
  local condition="$1"
  local config="$2"
  local output="runs/model_design_pilot/${condition}"
  "$PYTHON" - "$condition" "$config" "$output" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

condition = sys.argv[1]
config = Path(sys.argv[2])
output = Path(sys.argv[3])
contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
splits = ["validation"]
payload = {
    "condition": condition,
    "status": "completed",
    "expected_model_count": 7,
    "git_commit": commit,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "experiment_fingerprint": contract["fingerprint"],
    "evaluation_splits": splits,
    "reports": {
        split: f"evaluations/model_design_pilot/{condition}_{split}.json"
        for split in splits
    },
    "unit_diagnostics": {
        split: f"evaluations/model_design_pilot/{condition}_{split}_units.json"
        for split in splits
    },
    "completed_unix": time.time(),
}
path = output / "pilot_condition_complete.json"
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

write_state running "" preflight "checking GPU, disk, tests, repository contract, and deterministic CUDA settings"
check_disk
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader
"$PYTHON" - <<'PY'
import os
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the model-design pilot")
props = torch.cuda.get_device_properties(0)
print(f"CUDA device: {props.name}; VRAM={props.total_memory / 1024**3:.1f} GiB")
print(f"PyTorch: {torch.__version__}; BF16={torch.cuda.is_bf16_supported()}")
print(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
PY
"$PYTHON" -m pytest -q
"$REPO_AUDIT" . --data-samples-per-operator 32

conditions=(
  identity_unanchored
  identity_retention
  weak_unanchored
  weak_retention
)
evaluation_splits=(validation)

for condition in "${conditions[@]}"; do
  config="configs/experiments/model_design_pilot_${condition}.yaml"
  output="runs/model_design_pilot/${condition}"
  if condition_complete "$condition" "$config"; then
    echo "=== MODEL DESIGN PILOT: ${condition} already complete; skipping ==="
    write_state running "$condition" skipped "verified reports, unit diagnostics, contracts, and checkpoints"
    continue
  fi

  echo "=== MODEL DESIGN PILOT: ${condition} ==="
  check_disk
  write_state running "$condition" data_audit "auditing deterministic generated data"
  "$AUDIT_DATA" \
    --config "$config" \
    --samples-per-operator "$AUDIT_SAMPLES" \
    --out "audits/model_design_pilot/${condition}.json"

  write_state running "$condition" plan "validating dependency and checkpoint plan"
  "$TRAIN_BATCH" --config "$config" --plan-only

  write_state running "$condition" training "training/resuming seven dependency-ordered models"
  "$TRAIN_BATCH" --config "$config"

  manifest="$output/seed_0/fusion_subsets/subset_31.json"
  for split in "${evaluation_splits[@]}"; do
    write_state running "$condition" "evaluation_${split}" "evaluating all-five fusion manifest"
    "$EVALUATE" \
      --config "$config" \
      --manifest "$manifest" \
      --split "$split" \
      --examples-per-operator "$EVAL_EXAMPLES" \
      --out "evaluations/model_design_pilot/${condition}_${split}.json"

    write_state running "$condition" "unit_diagnostics_${split}" "measuring relevant and inactive unit drift"
    "$DIAGNOSTICS" \
      --config "$config" \
      --manifest "$manifest" \
      --split "$split" \
      --examples-per-operator "$EVAL_EXAMPLES" \
      --out "evaluations/model_design_pilot/${condition}_${split}_units.json"
  done
  mark_condition_complete "$condition" "$config"
  if ! condition_complete "$condition" "$config"; then
    echo "condition completion verification failed: $condition" >&2
    exit 1
  fi
  write_state running "$condition" completed "condition training and diagnostics completed"
done

write_state running "" pair_audit "verifying exact shared base/joint endpoints across paired conditions"
if ! "$PAIR_AUDIT" --repo-root . --out audits/model_design_pilot/pair_consistency.json; then
  write_state failed "" pair_audit "paired conditions did not share exact base and joint model states"
  exit 67
fi

write_state running "" summarizing "building the cross-condition endpoint index"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
conditions = ["identity_unanchored", "identity_retention", "weak_unanchored", "weak_retention"]
splits = ["validation"]
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
        "reports": {
            split: f"evaluations/model_design_pilot/{condition}_{split}.json"
            for split in splits
        },
        "unit_diagnostics": {
            split: f"evaluations/model_design_pilot/{condition}_{split}_units.json"
            for split in splits
        },
    })
path = Path("evaluations/model_design_pilot/index.json")
path.write_text(
    json.dumps(
        {
            "conditions": records,
            "pair_consistency": "audits/model_design_pilot/pair_consistency.json",
            "reserved_final_splits": ["iid_test", "operand_ood", "length_ood"],
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
print(path)
PY
write_state completed "" completed "all four validation diagnostics and pair-consistency audit completed"

cat <<'EOF'
Model-design pilot completed.
Select the production design from validation only.
IID test, operand OOD, and length OOD are intentionally reserved for final evaluation.
Compare:
  1. relevant-specialist validation accuracy;
  2. raw-sum and bias-mean trace validity and EOS accuracy;
  3. total all-five interference versus the relevant specialist;
  4. per-unit inactive JSD, KL, argmax agreement, and centered-bias RMS;
  5. divergence/agreement to the matched joint;
  6. selected checkpoint steps versus final steps;
  7. retention regularization logs and pair-consistency audit warnings.
EOF
