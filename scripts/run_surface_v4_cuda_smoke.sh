#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/experiments/gpt_bias_fusion_factory_surface_v4_smoke.yaml}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TRAIN_BATCH="${TRAIN_BATCH:-$ROOT/.venv/bin/opfusion-train-batch-design}"
AUDIT_DATA="${AUDIT_DATA:-$ROOT/.venv/bin/opfusion-audit-data-design}"
EVALUATE="${EVALUATE:-$ROOT/.venv/bin/opfusion-evaluate-fusion}"
DIAGNOSTICS="${DIAGNOSTICS:-$ROOT/.venv/bin/opfusion-evaluate-unit-diagnostics}"
REPO_AUDIT="${REPO_AUDIT:-$ROOT/.venv/bin/opfusion-audit}"
SMOKE_EVALUATION_SEED="${SMOKE_EVALUATION_SEED:-702000}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-64}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for executable in "$PYTHON" "$TRAIN_BATCH" "$AUDIT_DATA" "$EVALUATE" "$DIAGNOSTICS" "$REPO_AUDIT"; do
  if [[ ! -x "$executable" ]]; then
    echo "missing executable: $executable; run bash scripts/bootstrap_arch_linux.sh" >&2
    exit 64
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; verify the NVIDIA driver and running kernel module" >&2
  exit 65
fi
if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null || true)" ]]; then
  echo "tracked files are modified; run the smoke test from a clean checkout" >&2
  exit 68
fi

readarray -t smoke_paths < <(
  "$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from opfusion.training.design_config import load_design_run_config

config_path = Path(sys.argv[1]).resolve()
run = load_design_run_config(config_path)
if not run.experiment_id.endswith("_smoke"):
    raise SystemExit("smoke config experiment id must end with _smoke")
if run.seeds != (0,) or not 0 < run.max_steps <= 3:
    raise SystemExit("smoke config must use seed 0 and at most three optimizer steps")
if not run.require_cuda:
    raise SystemExit("smoke config must require CUDA")
if not run.deterministic_algorithms or run.allow_tf32:
    raise SystemExit("smoke config must use deterministic algorithms with TF32 disabled")
output = Path(run.output_dir)
if output.is_absolute() or len(output.parts) < 2 or output.parts[0] != "runs":
    raise SystemExit("smoke output_dir must be a relative path below runs/")
print(output)
print(f"audits/{run.experiment_id}_data.json")
print(f"evaluations/{run.experiment_id}_validation.json")
print(f"evaluations/{run.experiment_id}_validation_units.json")
PY
)
OUTPUT_DIR="${smoke_paths[0]}"
AUDIT_OUT="${smoke_paths[1]}"
EVAL_OUT="${smoke_paths[2]}"
DIAGNOSTICS_OUT="${smoke_paths[3]}"
MARKER="$OUTPUT_DIR/cuda_smoke_complete.json"
MANIFEST="$OUTPUT_DIR/seed_0/fusion_subsets/subset_31.json"

smoke_complete() {
  "$PYTHON" - "$MARKER" "$CONFIG" "$OUTPUT_DIR" "$MANIFEST" "$EVAL_OUT" "$DIAGNOSTICS_OUT" "$SMOKE_EVALUATION_SEED" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

marker, config, output, manifest, evaluation, diagnostics = map(Path, sys.argv[1:7])
evaluation_seed = int(sys.argv[7])
required = (marker, config, output / "experiment_contract.json", manifest, evaluation, diagnostics)
if any(not path.is_file() for path in required) or not torch.cuda.is_available():
    raise SystemExit(1)
try:
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
    diagnostics_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
driver = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
).splitlines()[0].strip()
props = torch.cuda.get_device_properties(0)
config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
fingerprint = contract.get("fingerprint")
gpu = marker_payload.get("gpu", {})
if not (
    marker_payload.get("status") == "passed"
    and marker_payload.get("git_commit") == commit
    and marker_payload.get("config_sha256") == config_sha
    and marker_payload.get("experiment_fingerprint") == fingerprint
    and marker_payload.get("evaluation_seed") == evaluation_seed
    and marker_payload.get("torch_version") == torch.__version__
    and marker_payload.get("torch_cuda_runtime") == torch.version.cuda
    and marker_payload.get("nvidia_driver_version") == driver
    and marker_payload.get("cublas_workspace_config") == os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    and gpu.get("name") == props.name
    and gpu.get("capability") == f"{props.major}.{props.minor}"
    and gpu.get("total_memory_bytes") == props.total_memory
    and manifest_payload.get("experiment_fingerprint") == fingerprint
    and evaluation_payload.get("experiment_fingerprint") == fingerprint
    and evaluation_payload.get("evaluation_seed") == evaluation_seed
    and diagnostics_payload.get("experiment_fingerprint") == fingerprint
    and diagnostics_payload.get("evaluation_seed") == evaluation_seed
):
    raise SystemExit(1)
complete_files = sorted((output / "seed_0").glob("*/complete.json"))
if len(complete_files) != 7:
    raise SystemExit(1)
for complete_path in complete_files:
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    for key in ("selected_checkpoint", "final_checkpoint"):
        value = payload.get(key)
        if value is None:
            raise SystemExit(1)
        path = Path(str(value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise SystemExit(1)
raise SystemExit(0)
PY
}

if [[ "${FORCE_SMOKE:-0}" != "1" ]] && smoke_complete; then
  echo "CUDA smoke already passed for this commit, configuration, GPU, driver, and PyTorch stack: $MARKER"
  exit 0
fi

rm -rf "$OUTPUT_DIR"
rm -f "$AUDIT_OUT" "$EVAL_OUT" "$DIAGNOSTICS_OUT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$AUDIT_OUT")" "$(dirname "$EVAL_OUT")"

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader
"$PYTHON" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA smoke requires torch.cuda.is_available() == true")
props = torch.cuda.get_device_properties(0)
print(f"CUDA device: {props.name}; capability={props.major}.{props.minor}; VRAM={props.total_memory / 1024**3:.1f} GiB")
print(f"PyTorch: {torch.__version__}; CUDA runtime={torch.version.cuda}; BF16={torch.cuda.is_bf16_supported()}")
print(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
PY

if [[ "${SKIP_STATIC_PREFLIGHT:-0}" != "1" ]]; then
  "$PYTHON" -m pytest -q
  "$REPO_AUDIT" . --data-samples-per-operator 32
fi
"$AUDIT_DATA" \
  --config "$CONFIG" \
  --samples-per-operator "$AUDIT_SAMPLES" \
  --out "$AUDIT_OUT"
"$TRAIN_BATCH" --config "$CONFIG" --plan-only
"$TRAIN_BATCH" --config "$CONFIG"

"$EVALUATE" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --split validation \
  --examples-per-operator 8 \
  --max-new-tokens 256 \
  --evaluation-seed "$SMOKE_EVALUATION_SEED" \
  --out "$EVAL_OUT"
"$DIAGNOSTICS" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --split validation \
  --examples-per-operator 8 \
  --evaluation-seed "$SMOKE_EVALUATION_SEED" \
  --out "$DIAGNOSTICS_OUT"

"$PYTHON" - "$MARKER" "$CONFIG" "$OUTPUT_DIR" "$MANIFEST" "$EVAL_OUT" "$DIAGNOSTICS_OUT" "$SMOKE_EVALUATION_SEED" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

marker, config, output, manifest, evaluation, diagnostics = map(Path, sys.argv[1:7])
evaluation_seed = int(sys.argv[7])
contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
diagnostics_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
fingerprint = contract["fingerprint"]
if manifest_payload.get("experiment_fingerprint") != fingerprint:
    raise SystemExit("smoke manifest fingerprint mismatch")
if evaluation_payload.get("experiment_fingerprint") != fingerprint:
    raise SystemExit("smoke evaluation fingerprint mismatch")
if diagnostics_payload.get("experiment_fingerprint") != fingerprint:
    raise SystemExit("smoke diagnostics fingerprint mismatch")
if evaluation_payload.get("evaluation_seed") != evaluation_seed:
    raise SystemExit("smoke evaluation seed mismatch")
if diagnostics_payload.get("evaluation_seed") != evaluation_seed:
    raise SystemExit("smoke diagnostics seed mismatch")
complete_files = sorted((output / "seed_0").glob("*/complete.json"))
if len(complete_files) != 7:
    raise SystemExit(f"expected seven completed smoke models, found {len(complete_files)}")
for complete_path in complete_files:
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    for key in ("selected_checkpoint", "final_checkpoint"):
        value = payload.get(key)
        if value is None:
            raise SystemExit(f"{key} missing from {complete_path}")
        path = Path(str(value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise SystemExit(f"checkpoint missing: {path}")
props = torch.cuda.get_device_properties(0)
driver = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
).splitlines()[0].strip()
payload = {
    "status": "passed",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "config": str(config),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "experiment_fingerprint": fingerprint,
    "evaluation_seed": evaluation_seed,
    "model_count": len(complete_files),
    "manifest": str(manifest),
    "evaluation": str(evaluation),
    "diagnostics": str(diagnostics),
    "gpu": {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
    },
    "nvidia_driver_version": driver,
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "completed_unix": time.time(),
}
marker.parent.mkdir(parents=True, exist_ok=True)
tmp = marker.with_suffix(marker.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, marker)
print(marker)
PY

if ! smoke_complete; then
  echo "CUDA smoke completion verification failed" >&2
  exit 69
fi

echo "Real-hardware CUDA smoke passed: $MARKER"
