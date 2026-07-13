from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from opfusion.tokenizer import FixedVocabTokenizer


EXPERIMENT_OPERATORS: tuple[str, ...] = (
    "scalar.add",
    "aggregation.sum",
    "scalar.neg",
    "scalar.min",
    "scalar.max",
)

OPERATOR_TOKENS: dict[str, str] = {
    "scalar.add": "<OP_SCALAR_ADD>",
    "aggregation.sum": "<OP_AGG_SUM>",
    "scalar.neg": "<OP_SCALAR_NEG>",
    "scalar.min": "<OP_SCALAR_MIN>",
    "scalar.max": "<OP_SCALAR_MAX>",
}


@dataclass(frozen=True)
class SyntheticDataConfig:
    operand_min: int = -64
    operand_max: int = 64
    min_terms: int = 3
    max_terms: int = 8
    numeric_token_min: int = -1024
    numeric_token_max: int = 1024

    def validate(self) -> None:
        if self.operand_min > self.operand_max:
            raise ValueError("operand_min must not exceed operand_max")
        if self.min_terms < 2 or self.min_terms > self.max_terms:
            raise ValueError("term range must satisfy 2 <= min_terms <= max_terms")
        worst_sum = max(abs(self.operand_min), abs(self.operand_max)) * self.max_terms
        if worst_sum > max(abs(self.numeric_token_min), abs(self.numeric_token_max)):
            raise ValueError("numeric token range is too small for generated sums")


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _number_token(value: int) -> str:
    return f"<N_{value}>"


def _infix_state(values: Sequence[int]) -> list[str]:
    output: list[str] = []
    for index, value in enumerate(values):
        if index:
            output.append("<PLUS>")
        output.append(_number_token(value))
    return output


def _list_state(values: Sequence[int]) -> list[str]:
    output = ["<LBRACK>"]
    for index, value in enumerate(values):
        if index:
            output.append("<COMMA>")
        output.append(_number_token(value))
    output.append("<RBRACK>")
    return output


class SyntheticTraceFactory:
    """Deterministic equality-trace generator for five GPT operator models."""

    def __init__(self, tokenizer: FixedVocabTokenizer, config: SyntheticDataConfig) -> None:
        config.validate()
        self.tokenizer = tokenizer
        self.config = config
        missing = [token for token in OPERATOR_TOKENS.values() if token not in tokenizer.token_to_id]
        if missing:
            raise ValueError(f"operator tokens missing from tokenizer: {missing}")

    def _rng(self, *, seed: int, split: str, step: int, sample_index: int, operator_id: str) -> random.Random:
        return random.Random(_stable_seed(seed, split, step, sample_index, operator_id))

    def _values(self, rng: random.Random, count: int) -> list[int]:
        return [rng.randint(self.config.operand_min, self.config.operand_max) for _ in range(count)]

    def example_tokens(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> list[str]:
        if operator_id not in EXPERIMENT_OPERATORS:
            raise KeyError(f"unsupported experiment operator: {operator_id}")
        rng = self._rng(seed=seed, split=split, step=step, sample_index=sample_index, operator_id=operator_id)
        tokens = [OPERATOR_TOKENS[operator_id]]

        if operator_id == "scalar.add":
            left, right = self._values(rng, 2)
            tokens.extend([_number_token(left), "<PLUS>", _number_token(right)])
            tokens.extend(["<EQ_STEP>", _number_token(left + right), "<TRACE_STOP>"])
            return tokens

        if operator_id == "scalar.neg":
            value = self._values(rng, 1)[0]
            tokens.extend([_number_token(value), "<EQ_STEP>", _number_token(-value), "<TRACE_STOP>"])
            return tokens

        count = rng.randint(self.config.min_terms, self.config.max_terms)
        values = self._values(rng, count)

        if operator_id == "aggregation.sum":
            current = list(values)
            tokens.extend(_infix_state(current))
            while len(current) > 1:
                current = [current[0] + current[1], *current[2:]]
                tokens.append("<EQ_STEP>")
                tokens.extend(_infix_state(current))
            tokens.append("<TRACE_STOP>")
            return tokens

        reducer = min if operator_id == "scalar.min" else max
        current = list(values)
        tokens.extend(_list_state(current))
        while len(current) > 1:
            current = [reducer(current[0], current[1]), *current[2:]]
            tokens.append("<EQ_STEP>")
            if len(current) == 1:
                tokens.append(_number_token(current[0]))
            else:
                tokens.extend(_list_state(current))
        tokens.append("<TRACE_STOP>")
        return tokens

    def joint_operator(self, *, seed: int, split: str, step: int, sample_index: int) -> str:
        rng = random.Random(_stable_seed(seed, split, step, sample_index, "joint-operator"))
        return EXPERIMENT_OPERATORS[rng.randrange(len(EXPERIMENT_OPERATORS))]

    def encoded_example(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> list[int]:
        if operator_id == "joint.all_five":
            operator_id = self.joint_operator(seed=seed, split=split, step=step, sample_index=sample_index)
        tokens = self.example_tokens(
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
        )
        return self.tokenizer.encode_tokens(tokens, add_bos=True, add_eos=True)

    def batch(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        examples = [
            self.encoded_example(
                operator_id,
                seed=seed,
                split=split,
                step=step,
                sample_index=index,
            )
            for index in range(batch_size)
        ]
        max_length = max(len(example) for example in examples)
        input_ids = torch.full((batch_size, max_length - 1), self.tokenizer.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_length - 1), -100, dtype=torch.long)
        for row, example in enumerate(examples):
            source = example[:-1]
            target = example[1:]
            input_ids[row, : len(source)] = torch.tensor(source, dtype=torch.long)
            labels[row, : len(target)] = torch.tensor(target, dtype=torch.long)
        return input_ids.to(device), labels.to(device)

    def render(self, tokens: Iterable[str]) -> str:
        replacements = {
            "<PLUS>": "+",
            "<COMMA>": ",",
            "<LBRACK>": "[",
            "<RBRACK>": "]",
            "<EQ_STEP>": "=",
            "<TRACE_STOP>": "<STOP>",
        }
        rendered: list[str] = []
        for token in tokens:
            if token.startswith("<N_") and token.endswith(">"):
                rendered.append(token[3:-1])
            else:
                rendered.append(replacements.get(token, token))
        return " ".join(rendered)
