from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypedDict


class BiasContribution(TypedDict):
    gate: float
    bias: Sequence[float]


def fuse_bias(base_logits: Sequence[float], contributions: Iterable[BiasContribution]) -> list[float]:
    result = [float(value) for value in base_logits]
    width = len(result)
    for contribution in contributions:
        gate = float(contribution["gate"])
        bias = list(contribution["bias"])
        if len(bias) != width:
            raise ValueError(f"bias width mismatch: expected {width}, got {len(bias)}")
        for idx, value in enumerate(bias):
            result[idx] += gate * float(value)
    return result


def inactive_leakage(gates: Sequence[float], active_indices: Iterable[int]) -> float:
    active = set(active_indices)
    return sum(float(gate) for idx, gate in enumerate(gates) if idx not in active)
