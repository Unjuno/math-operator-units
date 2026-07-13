from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from opfusion.io import load_yaml


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 9666
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1
    weight_tying: bool = True
    bias: bool = False

    @property
    def param_count_estimate(self) -> int:
        embed = self.vocab_size * self.d_model
        pos = self.max_seq_len * self.d_model
        qkv = self.d_model * 3 * self.d_model
        attn_proj = self.d_model * self.d_model
        attn = self.n_layers * (qkv + attn_proj)
        ff = self.n_layers * (2 * self.d_model * self.d_ff)
        norm = (self.n_layers * 2 + 1) * self.d_model
        head = 0 if self.weight_tying else self.vocab_size * self.d_model
        return embed + pos + attn + ff + norm + head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GPTConfig:
        keys = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in d.items() if k in keys}
        return cls(**filtered)


def load_config(path: str | Path) -> GPTConfig:
    data = load_yaml(path)
    raw = data.get("model", data)
    return GPTConfig.from_dict(raw)


def save_config(config: GPTConfig, path: str | Path) -> None:
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump({"model": config.to_dict()}, f, default_flow_style=False, sort_keys=False)
