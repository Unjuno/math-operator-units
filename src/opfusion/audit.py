from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opfusion.operators import load_applicability_policy
from opfusion.tokenizer import build_output_allow_list, build_vocab, build_vocab_hash, reserved_operator_tokens


def audit_repo(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    tokenizer_path = root / "configs" / "tokenizer" / "tokenizer_core_v1.yaml"
    policy_path = root / "configs" / "operators" / "applicability_policy.yaml"

    vocab = build_vocab(tokenizer_path, repo_root=root)
    reserved = reserved_operator_tokens(vocab)
    allow_list = build_output_allow_list(vocab)
    policy = load_applicability_policy(policy_path)

    return {
        "vocab_size": len(vocab),
        "vocab_hash": build_vocab_hash(tokenizer_path, repo_root=root),
        "reserved_operator_slots": len(reserved),
        "first_reserved_operator_slot": reserved[0] if reserved else None,
        "last_reserved_operator_slot": reserved[-1] if reserved else None,
        "unassigned_reserved_allowed_count": sum(1 for token, allowed in zip(vocab, allow_list) if token in reserved and allowed),
        "dispatch_policy": "false_only",
        "applicability_core_rule": policy.core_rule,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit operator-unit-fusion repository configuration.")
    parser.add_argument("repo_root", nargs="?", default=".")
    args = parser.parse_args()
    print(json.dumps(audit_repo(args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
