from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from opfusion.tokenizer import FixedVocabTokenizer

from .config import RunConfig
from .design_config import model_design


CONTRACT_ABI_VERSION = 2
CONTRACT_FILENAME = "experiment_contract.json"
_CODE_PATHS = (
    "src/opfusion/model/gpt.py",
    "src/opfusion/training/data.py",
    "src/opfusion/training/strict_verifier.py",
    "src/opfusion/training/design_controls.py",
    "src/opfusion/training/trainer.py",
    "src/opfusion/training/trainer_surface.py",
    "src/opfusion/training/trainer_design.py",
    "src/opfusion/training/trainer_design_hardened.py",
    "src/opfusion/training/batch.py",
    "src/opfusion/training/batch_design.py",
    "src/opfusion/training/audit_pilot_pairs.py",
    "src/opfusion/fusion_eval.py",
    "src/opfusion/fusion_eval_seeded.py",
    "src/opfusion/final_eval_guard.py",
    "src/opfusion/final_eval_policy.py",
    "src/opfusion/fusion_diagnostics.py",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_revision(repo_root: Path) -> dict[str, Any]:
    # Generated audits/evaluations must not change the fingerprint. Source-file
    # hashes already catch local code edits, while the commit records provenance.
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"commit": commit}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}


def contract_payload(repo_root: Path, config: RunConfig) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    model_path = Path(config.model_config)
    tokenizer_path = Path(config.tokenizer_config)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = repo_root / tokenizer_path
    tokenizer = FixedVocabTokenizer.from_config(tokenizer_path)
    code_hashes = {
        relative: _file_hash(repo_root / relative)
        for relative in _CODE_PATHS
        if (repo_root / relative).is_file()
    }
    return {
        "contract_abi_version": CONTRACT_ABI_VERSION,
        "experiment_id": config.experiment_id,
        "run_config": asdict(config),
        "model_design": model_design(config).to_dict(),
        "model_config_path": str(Path(config.model_config)),
        "model_config_sha256": _file_hash(model_path),
        "tokenizer_config_path": str(Path(config.tokenizer_config)),
        "tokenizer_config_sha256": _file_hash(tokenizer_path),
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "vocab_size": tokenizer.vocab_size,
        "code_sha256": code_hashes,
        "git": _git_revision(repo_root),
    }


def fingerprint_for_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def build_contract(repo_root: Path, config: RunConfig) -> dict[str, Any]:
    payload = contract_payload(repo_root, config)
    return {"fingerprint": fingerprint_for_payload(payload), "payload": payload}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ensure_experiment_contract(repo_root: Path, config: RunConfig) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = Path(config.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    contract_path = output_root / CONTRACT_FILENAME
    current = build_contract(repo_root, config)
    strict = model_design(config).strict_experiment_fingerprint

    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != current["fingerprint"]:
            raise RuntimeError(
                "experiment fingerprint mismatch for existing output directory; "
                "use a new output_dir instead of mixing checkpoints from different "
                f"configs/code revisions. existing={existing.get('fingerprint')} "
                f"current={current['fingerprint']} path={contract_path}"
            )
        return existing

    existing_entries = []
    if output_root.exists():
        existing_entries = [entry.name for entry in output_root.iterdir() if entry.name != CONTRACT_FILENAME]
    if strict and existing_entries:
        raise RuntimeError(
            "output directory contains artifacts but has no experiment contract; "
            "refusing to adopt legacy checkpoints. Select a new output_dir or move "
            f"the old run aside. path={output_root} entries={sorted(existing_entries)[:8]}"
        )
    _atomic_json(contract_path, current)
    return current
