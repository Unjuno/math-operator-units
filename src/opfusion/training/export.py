from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import torch

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer


def export_bundle(
    *,
    checkpoint: Path,
    model_config_path: Path,
    tokenizer_config_path: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = FixedVocabTokenizer.from_config(tokenizer_config_path)
    model_config = load_config(model_config_path)
    model = GPTModel(model_config)
    model.load_state_dict(payload["model_state_dict"])
    if model.param_count > 1_000_000:
        raise ValueError(f"checkpoint model exceeds 1M parameter limit: {model.param_count}")
    checkpoint_hash = payload.get("vocab_hash")
    if checkpoint_hash is not None and checkpoint_hash != tokenizer.vocab_hash:
        raise ValueError("checkpoint and tokenizer vocabulary hashes differ")

    shutil.copy2(checkpoint, output_dir / "model.pt")
    shutil.copy2(model_config_path, output_dir / "model_config.yaml")
    shutil.copy2(tokenizer_config_path, output_dir / "tokenizer_config.yaml")
    tokenizer.save_vocab(output_dir / "vocab.json")
    metadata = {
        "format": "opfusion-gpt-operator-v1",
        "job_id": payload.get("job_id"),
        "seed": payload.get("seed"),
        "step": payload.get("step"),
        "parameter_count": model.param_count,
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "source_checkpoint": str(checkpoint),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# GPT Operator Unit\n\n"
        "This bundle was produced by `math-operator-units`. It contains a GPT-only operator checkpoint, "
        "the fixed experiment tokenizer, the model configuration, and reproducibility metadata.\n\n"
        "The model is an experimental synthetic-mathematics unit and is not a general-purpose language model.\n",
        encoding="utf-8",
    )
    return output_dir


def upload_bundle(bundle_dir: Path, repo_id: str, *, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError('install the publish extra: pip install -e ".[publish]"') from exc
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(bundle_dir))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an operator GPT checkpoint as a self-contained bundle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    bundle = export_bundle(
        checkpoint=Path(args.checkpoint),
        model_config_path=Path(args.model_config),
        tokenizer_config_path=Path(args.tokenizer_config),
        output_dir=Path(args.out),
    )
    if args.repo_id:
        upload_bundle(bundle, args.repo_id, private=args.private)
    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
