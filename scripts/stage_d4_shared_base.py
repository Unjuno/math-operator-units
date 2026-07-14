from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no model_state_dict: {path}")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage one immutable D4 shared Base with provenance")
    parser.add_argument("--source", required=True, help="shared Base directory containing selected.pt and complete.json")
    parser.add_argument("--destination-output", required=True, help="condition output directory")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--condition-config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).resolve()
    output = (root / args.destination_output).resolve()
    destination = output / f"seed_{args.seed}" / "base_common"
    selected = source / "selected.pt"
    complete_path = source / "complete.json"
    base_config = (root / args.base_config).resolve()
    condition_config = (root / args.condition_config).resolve()
    for required in (selected, complete_path, base_config, condition_config):
        if not required.is_file():
            raise RuntimeError(f"required shared-base input is missing: {required}")

    complete = load_json(complete_path)
    parent_fingerprint = complete.get("experiment_fingerprint")
    checkpoint_payload = torch.load(selected, map_location="cpu", weights_only=False)
    if not isinstance(parent_fingerprint, str) or checkpoint_payload.get("experiment_fingerprint") != parent_fingerprint:
        raise RuntimeError("shared Base fingerprint is missing or inconsistent")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    staged_selected = destination / "selected.pt"

    contract = {
        "abi_version": 1,
        "role": "d4_shared_parent_base",
        "git_commit": git_commit(root),
        "seed": args.seed,
        "parent_experiment_fingerprint": parent_fingerprint,
        "parent_checkpoint": str(staged_selected.relative_to(root)),
        "parent_checkpoint_sha256": sha256_file(staged_selected),
        "parent_model_state_sha256": model_state_sha256(staged_selected),
        "base_config": str(base_config.relative_to(root)),
        "base_config_sha256": sha256_file(base_config),
        "condition_config": str(condition_config.relative_to(root)),
        "condition_config_sha256": sha256_file(condition_config),
    }
    contract_path = output / "parent_base_contract.json"
    tmp = contract_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(contract_path)
    print(contract_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
