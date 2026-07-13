from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from opfusion.io import load_yaml

VOCAB_INCLUDED_BACKLOG_STATUSES = {"active", "planned_direct"}
_TOKEN_RANGE_RE = re.compile(r"^<([A-Za-z]+)(-?\d+)>\.\.<\1(-?\d+)>$")


def _append_token(tokens: list[str], seen: set[str], token: str, source: str) -> None:
    if not isinstance(token, str) or not token:
        raise ValueError(f"invalid token from {source}: {token!r}")
    if token in seen:
        raise ValueError(f"duplicate token {token!r} from {source}")
    tokens.append(token)
    seen.add(token)


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_strings(item)


def _expand_angle_range(spec: str) -> list[str]:
    match = _TOKEN_RANGE_RE.match(spec)
    if not match:
        raise ValueError(f"unsupported token range syntax: {spec!r}")
    prefix, start_text, end_text = match.groups()
    start = int(start_text)
    end = int(end_text)
    step = 1 if end >= start else -1
    return [f"<{prefix}{i}>" for i in range(start, end + step, step)]


def _format_reserved_token(fmt: str, index: int) -> str:
    try:
        return fmt.format(index=index)
    except Exception as exc:  # pragma: no cover - defensive error path
        raise ValueError(f"invalid reserved token format {fmt!r}") from exc


def _planned_operator_tokens(repo_root: Path, config: dict[str, Any]) -> Iterable[str]:
    for name, rel_path in sorted(config.get("planned_operator_lists", {}).items()):
        data = load_yaml(repo_root / rel_path)
        for idx, operator in enumerate(data.get("operators", [])):
            if not isinstance(operator, dict):
                raise ValueError(f"operator entry {idx} in {rel_path} is not a mapping")
            status = operator.get("status")
            token = operator.get("canonical_token")
            if status in VOCAB_INCLUDED_BACKLOG_STATUSES and isinstance(token, str):
                yield token


def build_vocab(config_path: str | Path, repo_root: str | Path | None = None) -> list[str]:
    config_path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_path.parents[2]
    config = load_yaml(config_path)

    tokens: list[str] = []
    seen: set[str] = set()

    for section in ("special_tokens", "structural_tokens", "type_tokens"):
        for token in config.get(section, []):
            _append_token(tokens, seen, token, section)

    register_tokens = config.get("register_tokens", {})
    if "range" in register_tokens:
        for token in _expand_angle_range(register_tokens["range"]):
            _append_token(tokens, seen, token, "register_tokens.range")
    for token in register_tokens.get("named", []):
        _append_token(tokens, seen, token, "register_tokens.named")

    numeric_tokens = config.get("numeric_tokens", {})
    atomic = numeric_tokens.get("atomic_integer_range", {})
    if atomic:
        fmt = atomic["format"]
        for value in range(int(atomic["min"]), int(atomic["max"]) + 1):
            _append_token(tokens, seen, fmt.replace("{value}", str(value)), "numeric_tokens.atomic_integer_range")
    for token in numeric_tokens.get("optional_digit_tokens", {}).get("tokens", []):
        _append_token(tokens, seen, token, "numeric_tokens.optional_digit_tokens")

    for token in _flatten_strings(config.get("core_operator_tokens", {})):
        _append_token(tokens, seen, token, "core_operator_tokens")

    for token in _planned_operator_tokens(root, config):
        _append_token(tokens, seen, token, "planned_operator_lists")

    reserved = config.get("reserved_operator_slots", {})
    count = int(reserved.get("count", 0))
    fmt = reserved.get("format")
    if count and not fmt:
        raise ValueError("reserved_operator_slots.count is set but format is missing")
    for index in range(count):
        _append_token(tokens, seen, _format_reserved_token(fmt, index), "reserved_operator_slots")

    for token in config.get("fallback", {}).get("operator_spelling_tokens", []):
        _append_token(tokens, seen, token, "fallback.operator_spelling_tokens")

    return tokens


def build_vocab_map(config_path: str | Path, repo_root: str | Path | None = None) -> dict[str, int]:
    return {token: idx for idx, token in enumerate(build_vocab(config_path, repo_root=repo_root))}


def build_vocab_hash(config_path: str | Path, repo_root: str | Path | None = None) -> str:
    config = load_yaml(config_path)
    vocab = build_vocab(config_path, repo_root=repo_root)
    aliases = config.get("aliases", {}) or {}
    if aliases and not isinstance(aliases, dict):
        raise TypeError("tokenizer aliases must be a mapping")
    payload: object = vocab if not aliases else {"tokens": vocab, "aliases": {str(k): str(v) for k, v in aliases.items()}}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
