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
    value_ood_abs_min: int = 65
    value_ood_abs_max: int = 80
    length_ood_min_terms: int = 9
    length_ood_max_terms: int = 10

    def validate(self) -> None:
        if self.operand_min > self.operand_max:
            raise ValueError("operand_min must not exceed operand_max")
        if self.min_terms < 2 or self.min_terms > self.max_terms:
            raise ValueError("term range must satisfy 2 <= min_terms <= max_terms")
        if self.value_ood_abs_min <= max(abs(self.operand_min), abs(self.operand_max)):
            raise ValueError("value OOD range must begin outside the train operand range")
        if self.value_ood_abs_min > self.value_ood_abs_max:
            raise ValueError("value OOD range is invalid")
        if self.length_ood_min_terms <= self.max_terms:
            raise ValueError("length OOD range must begin above max_terms")
        if self.length_ood_min_terms > self.length_ood_max_terms:
            raise ValueError("length OOD range is invalid")
        maximum_abs_value = max(
            max(abs(self.operand_min), abs(self.operand_max)) * self.length_ood_max_terms,
            self.value_ood_abs_max * self.max_terms,
        )
        token_limit = max(abs(self.numeric_token_min), abs(self.numeric_token_max))
        if maximum_abs_value > token_limit:
            raise ValueError("numeric token range is too small for train/OOD generated sums")


@dataclass(frozen=True)
class TrainingExample:
    job_id: str
    operator_id: str
    prompt_tokens: tuple[str, ...]
    response_tokens: tuple[str, ...]
    final_value: int | None
    split: str
    task: str

    @property
    def all_tokens(self) -> tuple[str, ...]:
        return (*self.prompt_tokens, *self.response_tokens)


@dataclass(frozen=True)
class EncodedTrainingExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    response_length: int
    final_token_id: int | None
    operator_id: str
    task: str


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
    """Deterministic, exact trace generator for the controlled GPT experiment.

    The legacy ``example_tokens`` and default ``batch`` behavior remain available
    for v1 checkpoints/tests. v2 calls ``training_example`` and requests
    response-only supervision.
    """

    def __init__(self, tokenizer: FixedVocabTokenizer, config: SyntheticDataConfig) -> None:
        config.validate()
        self.tokenizer = tokenizer
        self.config = config
        missing = [token for token in OPERATOR_TOKENS.values() if token not in tokenizer.token_to_id]
        if missing:
            raise ValueError(f"operator tokens missing from tokenizer: {missing}")

    def _rng(self, *, seed: int, split: str, step: int, sample_index: int, operator_id: str) -> random.Random:
        return random.Random(_stable_seed(seed, split, step, sample_index, operator_id))

    def _values(self, rng: random.Random, count: int, split: str) -> list[int]:
        if split == "value_ood":
            values: list[int] = []
            for _ in range(count):
                magnitude = rng.randint(self.config.value_ood_abs_min, self.config.value_ood_abs_max)
                values.append(magnitude if rng.random() < 0.5 else -magnitude)
            return values
        return [rng.randint(self.config.operand_min, self.config.operand_max) for _ in range(count)]

    def _term_count(self, rng: random.Random, split: str) -> int:
        if split == "length_ood":
            return rng.randint(self.config.length_ood_min_terms, self.config.length_ood_max_terms)
        return rng.randint(self.config.min_terms, self.config.max_terms)

    def joint_operator(self, *, seed: int, split: str, step: int, sample_index: int, namespace: str = "joint") -> str:
        rng = random.Random(_stable_seed(seed, split, step, sample_index, f"{namespace}-operator"))
        return EXPERIMENT_OPERATORS[rng.randrange(len(EXPERIMENT_OPERATORS))]

    def _expression_and_response(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> tuple[list[str], list[str], int]:
        if operator_id not in EXPERIMENT_OPERATORS:
            raise KeyError(f"unsupported experiment operator: {operator_id}")
        rng = self._rng(seed=seed, split=split, step=step, sample_index=sample_index, operator_id=operator_id)

        if operator_id == "scalar.add":
            left, right = self._values(rng, 2, split)
            expression = [_number_token(left), "<PLUS>", _number_token(right)]
            value = left + right
            response = ["<EQ_STEP>", _number_token(value), "<TRACE_STOP>"]
            return expression, response, value

        if operator_id == "scalar.neg":
            value = self._values(rng, 1, split)[0]
            result = -value
            expression = [_number_token(value)]
            response = ["<EQ_STEP>", _number_token(result), "<TRACE_STOP>"]
            return expression, response, result

        count = self._term_count(rng, split)
        values = self._values(rng, count, split)

        if operator_id == "aggregation.sum":
            current = list(values)
            expression = _infix_state(current)
            response: list[str] = []
            while len(current) > 1:
                current = [current[0] + current[1], *current[2:]]
                response.append("<EQ_STEP>")
                response.extend(_infix_state(current))
            response.append("<TRACE_STOP>")
            return expression, response, current[0]

        reducer = min if operator_id == "scalar.min" else max
        current = list(values)
        expression = _list_state(current)
        response = []
        while len(current) > 1:
            current = [reducer(current[0], current[1]), *current[2:]]
            response.append("<EQ_STEP>")
            if len(current) == 1:
                response.append(_number_token(current[0]))
            else:
                response.extend(_list_state(current))
        response.append("<TRACE_STOP>")
        return expression, response, current[0]

    def example_tokens(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> list[str]:
        """Legacy v1 view: operator + prompt + response, without delimiter."""
        expression, response, _ = self._expression_and_response(
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
        )
        return [OPERATOR_TOKENS[operator_id], *expression, *response]

    def training_example(
        self,
        job_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
        forced_operator: str | None = None,
    ) -> TrainingExample:
        if forced_operator is not None:
            operator_id = forced_operator
        elif job_id in EXPERIMENT_OPERATORS:
            operator_id = job_id
        elif job_id == "base.common":
            operator_id = self.joint_operator(
                seed=seed,
                split=split,
                step=step,
                sample_index=sample_index,
                namespace="base",
            )
        elif job_id.startswith("joint."):
            operator_id = self.joint_operator(
                seed=seed,
                split=split,
                step=step,
                sample_index=sample_index,
                namespace=job_id,
            )
        else:
            raise KeyError(f"unsupported training job: {job_id}")

        expression, response, final_value = self._expression_and_response(
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
        )
        if job_id == "base.common":
            required = ("<TASK_COPY>", "<RESPONSE>")
            missing = [token for token in required if token not in self.tokenizer.token_to_id]
            if missing:
                raise ValueError(f"v2 base task tokens missing from tokenizer: {missing}")
            prompt = ["<TASK_COPY>", OPERATOR_TOKENS[operator_id], *expression, "<RESPONSE>"]
            response_tokens = [*expression, "<TRACE_STOP>"]
            return TrainingExample(
                job_id=job_id,
                operator_id=operator_id,
                prompt_tokens=tuple(prompt),
                response_tokens=tuple(response_tokens),
                final_value=None,
                split=split,
                task="copy_expression",
            )

        if "<RESPONSE>" not in self.tokenizer.token_to_id:
            raise ValueError("v2 response-only training requires <RESPONSE> in the tokenizer")
        prompt = [OPERATOR_TOKENS[operator_id], *expression, "<RESPONSE>"]
        return TrainingExample(
            job_id=job_id,
            operator_id=operator_id,
            prompt_tokens=tuple(prompt),
            response_tokens=tuple(response),
            final_value=final_value,
            split=split,
            task="equivalence_trace",
        )

    def encode_training_example(self, example: TrainingExample, *, response_only: bool) -> EncodedTrainingExample:
        prompt_ids = self.tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
        response_ids = self.tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
        sequence = [*prompt_ids, *response_ids]
        source = sequence[:-1]
        target = sequence[1:]
        labels = list(target)
        if response_only:
            # labels[position] predicts sequence[position + 1]. The first response
            # token is predicted from the final prompt token (<RESPONSE>).
            first_supervised_position = len(prompt_ids) - 1
            for index in range(first_supervised_position):
                labels[index] = -100
        final_token_id = None
        if example.final_value is not None:
            final_token_id = self.tokenizer.token_to_id[_number_token(example.final_value)]
        return EncodedTrainingExample(
            input_ids=tuple(source),
            labels=tuple(labels),
            prompt_length=len(prompt_ids),
            response_length=len(response_ids),
            final_token_id=final_token_id,
            operator_id=example.operator_id,
            task=example.task,
        )

    def encoded_example(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> list[int]:
        """Legacy v1 encoded view."""
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
        response_only: bool = False,
        sample_offset: int = 0,
        forced_operator: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if response_only:
            encoded = [
                self.encode_training_example(
                    self.training_example(
                        operator_id,
                        seed=seed,
                        split=split,
                        step=step,
                        sample_index=sample_offset + index,
                        forced_operator=forced_operator,
                    ),
                    response_only=True,
                )
                for index in range(batch_size)
            ]
            max_length = max(len(example.input_ids) for example in encoded)
            input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_id, dtype=torch.long)
            labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
            for row, example in enumerate(encoded):
                input_ids[row, : len(example.input_ids)] = torch.tensor(example.input_ids, dtype=torch.long)
                labels[row, : len(example.labels)] = torch.tensor(example.labels, dtype=torch.long)
            return input_ids.to(device), labels.to(device)

        legacy_operator_id = forced_operator or operator_id
        examples = [
            self.encoded_example(
                legacy_operator_id,
                seed=seed,
                split=split,
                step=step,
                sample_index=sample_offset + index,
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

    def prompt_and_expected_ids(
        self,
        job_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
        forced_operator: str | None = None,
    ) -> tuple[list[int], list[int], int | None, str]:
        example = self.training_example(
            job_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
            forced_operator=forced_operator,
        )
        prompt = self.tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
        expected = self.tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
        final_id = None if example.final_value is None else self.tokenizer.token_to_id[_number_token(example.final_value)]
        return prompt, expected, final_id, example.operator_id

    def render(self, tokens: Iterable[str]) -> str:
        replacements = {
            "<PLUS>": "+",
            "<COMMA>": ",",
            "<LBRACK>": "[",
            "<RBRACK>": "]",
            "<EQ_STEP>": "=",
            "<TRACE_STOP>": "<STOP>",
            "<RESPONSE>": "=>",
        }
        rendered: list[str] = []
        for token in tokens:
            if token.startswith("<N_") and token.endswith(">"):
                rendered.append(token[3:-1])
            else:
                rendered.append(replacements.get(token, token))
        return " ".join(rendered)
