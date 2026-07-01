from __future__ import annotations

from pathlib import Path

from opfusion.tokenizer import build_output_allow_list, build_vocab, build_vocab_hash, reserved_operator_tokens

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "configs" / "tokenizer" / "tokenizer_core_v1.yaml"


def test_tokenizer_core_v1_has_8192_reserved_slots() -> None:
    vocab = build_vocab(TOKENIZER, repo_root=ROOT)
    reserved = reserved_operator_tokens(vocab)
    assert len(reserved) == 8192
    assert reserved[0] == "<OP_RESERVED_0000>"
    assert reserved[-1] == "<OP_RESERVED_8191>"


def test_tokenizer_core_v1_has_no_duplicate_tokens() -> None:
    vocab = build_vocab(TOKENIZER, repo_root=ROOT)
    assert len(vocab) == len(set(vocab))


def test_planned_direct_tokens_are_included_but_speculative_tokens_are_not() -> None:
    vocab = set(build_vocab(TOKENIZER, repo_root=ROOT))
    assert "<OP_SCALAR_MUL>" in vocab
    assert "<OP_VERIFY_EXACT>" in vocab
    assert "<OP_SEM_PARSE>" not in vocab
    assert "<OP_TOOL_FETCH>" not in vocab


def test_program_only_tokens_are_not_required_in_core_vocab() -> None:
    vocab = set(build_vocab(TOKENIZER, repo_root=ROOT))
    assert "<OP_SCALAR_SUB>" not in vocab


def test_vocab_hash_is_sha256_hex() -> None:
    digest = build_vocab_hash(TOKENIZER, repo_root=ROOT)
    assert len(digest) == 64
    int(digest, 16)


def test_unassigned_reserved_tokens_are_not_output_allowed() -> None:
    vocab = build_vocab(TOKENIZER, repo_root=ROOT)
    allow_list = build_output_allow_list(vocab)
    by_token = dict(zip(vocab, allow_list))
    assert by_token["<OP_RESERVED_0000>"] is False
    assert by_token["<OP_RESERVED_8191>"] is False
    assert by_token["<OP_SCALAR_ADD>"] is True
