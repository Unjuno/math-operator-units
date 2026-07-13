from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .build_vocab import build_vocab


@dataclass(frozen=True)
class TokenizerMetadata:
    profile: str
    vocab_hash: str
    vocab_size: int


class FixedVocabTokenizer:
    """Whitespace/token-list tokenizer for controlled operator experiments.

    The generator emits complete token strings. The tokenizer never guesses
    lexical boundaries or grows the vocabulary at runtime. Optional aliases
    let implementation-facing names (for example ``<EQ_STEP>``) resolve to a
    canonical surface token (for example ``=``) without adding another model
    output class.
    """

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        profile: str = "operator_experiment_v1",
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        if len(tokens) != len(set(tokens)):
            raise ValueError("token vocabulary contains duplicates")
        self.tokens = list(tokens)
        self.aliases = dict(aliases or {})
        canonical = {token: index for index, token in enumerate(self.tokens)}
        for alias, target in self.aliases.items():
            if not isinstance(alias, str) or not alias:
                raise ValueError(f"invalid tokenizer alias: {alias!r}")
            if alias in canonical:
                raise ValueError(f"tokenizer alias collides with canonical token: {alias}")
            if target not in canonical:
                raise ValueError(f"tokenizer alias target is not in the vocabulary: {alias} -> {target}")
        self.token_to_id = dict(canonical)
        self.token_to_id.update({alias: canonical[target] for alias, target in self.aliases.items()})
        self.profile = profile
        hash_payload: object = self.tokens if not self.aliases else {"tokens": self.tokens, "aliases": self.aliases}
        self.vocab_hash = hashlib.sha256(
            json.dumps(hash_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        for required in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
            if required not in self.token_to_id:
                raise ValueError(f"missing required token: {required}")

    @classmethod
    def from_config(cls, path: str | Path) -> "FixedVocabTokenizer":
        path = Path(path)
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        profile = data.get("tokenizer", {}).get("profile", path.stem)
        aliases = data.get("aliases", {}) or {}
        if not isinstance(aliases, dict):
            raise TypeError("tokenizer aliases must be a mapping")
        return cls(build_vocab(path), profile=profile, aliases={str(k): str(v) for k, v in aliases.items()})

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata(self.profile, self.vocab_hash, self.vocab_size)

    def encode_tokens(
        self,
        tokens: Iterable[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        strict: bool = True,
    ) -> list[int]:
        ids: list[int] = [self.bos_id] if add_bos else []
        for token in tokens:
            if strict and token not in self.token_to_id:
                raise KeyError(f"token not in fixed vocabulary: {token}")
            ids.append(self.token_to_id.get(token, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return self.encode_tokens(text.split(), **kwargs)

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        special = {"<PAD>", "<BOS>", "<EOS>"}
        output: list[str] = []
        for index in ids:
            if index < 0 or index >= self.vocab_size:
                raise IndexError(f"token id out of range: {index}")
            token = self.tokens[index]
            if skip_special and token in special:
                continue
            output.append(token)
        return " ".join(output)

    def save_vocab(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile": self.profile,
            "vocab_hash": self.vocab_hash,
            "tokens": self.tokens,
            "aliases": self.aliases,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
