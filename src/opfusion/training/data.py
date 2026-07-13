from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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

IID_SPLITS = {"train", "validation", "test", "iid_test"}
OPERAND_OOD_SPLITS = {"operand_ood", "value_ood"}


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
    partition_modulus: int = 100
    train_bucket_end: int = 70
    validation_bucket_end: int = 85
    full_trace_weight: int = 60
    continuation_weight: int = 25
    terminal_weight: int = 15
    max_partition_attempts: int = 20_000
    randomized_train_reduction: bool = True

    def validate(self) -> None:
        if self.operand_min > self.operand_max:
            raise ValueError("operand_min must not exceed operand_max")
        if self.min_terms < 2 or self.min_terms > self.max_terms:
            raise ValueError("term range must satisfy 2 <= min_terms <= max_terms")
        if self.value_ood_abs_min <= max(abs(self.operand_min), abs(self.operand_max)):
            raise ValueError("operand OOD range must begin outside the train operand range")
        if self.value_ood_abs_min > self.value_ood_abs_max:
            raise ValueError("operand OOD range is invalid")
        if self.length_ood_min_terms <= self.max_terms:
            raise ValueError("length OOD range must begin above max_terms")
        if self.length_ood_min_terms > self.length_ood_max_terms:
            raise ValueError("length OOD range is invalid")
        if self.partition_modulus < 3:
            raise ValueError("partition_modulus must be at least 3")
        if not 0 < self.train_bucket_end < self.validation_bucket_end < self.partition_modulus:
            raise ValueError("IID split bucket boundaries are invalid")
        if min(self.full_trace_weight, self.continuation_weight, self.terminal_weight) < 0:
            raise ValueError("trace-view weights must be nonnegative")
        if self.full_trace_weight + self.continuation_weight + self.terminal_weight <= 0:
            raise ValueError("at least one trace-view weight must be positive")
        if self.max_partition_attempts <= 0:
            raise ValueError("max_partition_attempts must be positive")
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
    initial_values: tuple[int, ...] = ()
    prompt_state_values: tuple[int, ...] = ()
    trace_states: tuple[tuple[int, ...], ...] = ()
    partition_bucket: int | None = None

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


@dataclass(frozen=True)
class TraceRecord:
    operator_id: str
    initial_values: tuple[int, ...]
    states: tuple[tuple[int, ...], ...]
    final_value: int
    partition_bucket: int | None


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _number_token(value: int) -> str:
    return f"<N_{value}>"


def _parse_number_token(token: str) -> int:
    if not token.startswith("<N_") or not token.endswith(">"):
        raise ValueError(f"not an atomic integer token: {token}")
    return int(token[3:-1])


def _infix_state(values: Sequence[int], plus_token: str = "<PLUS>") -> list[str]:
    output: list[str] = []
    for index, value in enumerate(values):
        if index:
            output.append(plus_token)
        output.append(_number_token(value))
    return output


def _list_state(
    values: Sequence[int],
    *,
    left: str = "<LBRACK>",
    comma: str = "<COMMA>",
    right: str = "<RBRACK>",
) -> list[str]:
    output = [left]
    for index, value in enumerate(values):
        if index:
            output.append(comma)
        output.append(_number_token(value))
    output.append(right)
    return output


class SyntheticTraceFactory:
    """Deterministic, split-safe, exactly verifiable operator data factory.

    V1 legacy views remain available. The active surface experiment uses the
    aliases ``<EQ_STEP> -> =`` and ``<TRACE_STOP> -> <EOS>`` so the Python code
    can share one implementation while the model predicts ordinary equality
    punctuation and EOS rather than task-specific trace-control output classes.
    """

    def __init__(self, tokenizer: FixedVocabTokenizer, config: SyntheticDataConfig) -> None:
        config.validate()
        self.tokenizer = tokenizer
        self.config = config
        missing = [token for token in OPERATOR_TOKENS.values() if token not in tokenizer.token_to_id]
        if missing:
            raise ValueError(f"operator tokens missing from tokenizer: {missing}")
        for required in ("<PLUS>", "<COMMA>", "<LBRACK>", "<RBRACK>"):
            if required not in tokenizer.token_to_id:
                raise ValueError(f"structural token missing from tokenizer: {required}")
        self.eq_token = "<EQ_STEP>" if "<EQ_STEP>" in tokenizer.token_to_id else "="
        if self.eq_token not in tokenizer.token_to_id:
            raise ValueError("tokenizer needs an equality token or <EQ_STEP> alias")
        self.eq_canonical = tokenizer.tokens[tokenizer.token_to_id[self.eq_token]]
        self.explicit_stop = "<TRACE_STOP>" in tokenizer.tokens
        self.plus_canonical = tokenizer.tokens[tokenizer.token_to_id["<PLUS>"]]
        self.comma_canonical = tokenizer.tokens[tokenizer.token_to_id["<COMMA>"]]
        self.left_canonical = tokenizer.tokens[tokenizer.token_to_id["<LBRACK>"]]
        self.right_canonical = tokenizer.tokens[tokenizer.token_to_id["<RBRACK>"]]

    def _rng(self, *, seed: int, split: str, step: int, sample_index: int, operator_id: str, namespace: str) -> random.Random:
        return random.Random(_stable_seed(seed, split, step, sample_index, operator_id, namespace))

    def _partition_bucket(self, operator_id: str, values: Sequence[int]) -> int:
        key = f"{operator_id}|" + ",".join(str(value) for value in values)
        return _stable_seed("iid-partition-v1", key) % self.config.partition_modulus

    def _bucket_matches(self, split: str, bucket: int) -> bool:
        if split == "train":
            return bucket < self.config.train_bucket_end
        if split == "validation":
            return self.config.train_bucket_end <= bucket < self.config.validation_bucket_end
        if split in {"test", "iid_test"}:
            return self.config.validation_bucket_end <= bucket < self.config.partition_modulus
        return True

    def _term_count(self, rng: random.Random, split: str, operator_id: str) -> int:
        if operator_id in {"scalar.add", "scalar.neg"}:
            return 2 if operator_id == "scalar.add" else 1
        if split == "length_ood":
            return rng.randint(self.config.length_ood_min_terms, self.config.length_ood_max_terms)
        return rng.randint(self.config.min_terms, self.config.max_terms)

    def _raw_values(self, rng: random.Random, count: int, split: str) -> list[int]:
        if split in OPERAND_OOD_SPLITS:
            values: list[int] = []
            for _ in range(count):
                magnitude = rng.randint(self.config.value_ood_abs_min, self.config.value_ood_abs_max)
                values.append(magnitude if rng.random() < 0.5 else -magnitude)
            return values
        return [rng.randint(self.config.operand_min, self.config.operand_max) for _ in range(count)]

    def _initial_values(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> tuple[tuple[int, ...], int | None]:
        if split not in IID_SPLITS and split not in OPERAND_OOD_SPLITS and split != "length_ood":
            raise KeyError(f"unsupported data split: {split}")
        rng = self._rng(
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
            operator_id=operator_id,
            namespace="initial-values",
        )
        for _ in range(self.config.max_partition_attempts):
            count = self._term_count(rng, split, operator_id)
            values = tuple(self._raw_values(rng, count, split))
            bucket = self._partition_bucket(operator_id, values) if split in IID_SPLITS else None
            if bucket is None or self._bucket_matches(split, bucket):
                return values, bucket
        raise RuntimeError(
            f"failed to sample {operator_id} for split={split} after "
            f"{self.config.max_partition_attempts} partition attempts"
        )

    def _state_tokens(self, operator_id: str, values: Sequence[int]) -> list[str]:
        if len(values) == 1:
            return [_number_token(values[0])]
        if operator_id in {"scalar.add", "aggregation.sum"}:
            return _infix_state(values)
        if operator_id in {"scalar.min", "scalar.max"}:
            return _list_state(values)
        if operator_id == "scalar.neg":
            if len(values) != 1:
                raise ValueError("NEG state must contain exactly one value")
            return [_number_token(values[0])]
        raise KeyError(operator_id)

    def _trace_record(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
        canonical: bool | None = None,
    ) -> TraceRecord:
        if operator_id not in EXPERIMENT_OPERATORS:
            raise KeyError(f"unsupported experiment operator: {operator_id}")
        values, bucket = self._initial_values(
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
        )
        states: list[tuple[int, ...]] = [values]
        if operator_id == "scalar.add":
            states.append((values[0] + values[1],))
        elif operator_id == "scalar.neg":
            states.append((-values[0],))
        else:
            reducer = (lambda a, b: a + b) if operator_id == "aggregation.sum" else (min if operator_id == "scalar.min" else max)
            current = list(values)
            if canonical is None:
                canonical = split != "train" or not self.config.randomized_train_reduction
            trace_rng = self._rng(
                seed=seed,
                split=split,
                step=step,
                sample_index=sample_index,
                operator_id=operator_id,
                namespace="reduction-order",
            )
            while len(current) > 1:
                index = 0 if canonical else trace_rng.randrange(len(current) - 1)
                reduced = reducer(current[index], current[index + 1])
                current = [*current[:index], reduced, *current[index + 2 :]]
                states.append(tuple(current))
        final_value = states[-1][0]
        record = TraceRecord(operator_id, values, tuple(states), final_value, bucket)
        verification = self.verify_trace_record(record)
        if not verification["valid"]:
            raise AssertionError(f"internal trace generator produced an invalid record: {verification}")
        return record

    def verify_trace_record(self, record: TraceRecord) -> dict[str, Any]:
        if not record.states or record.states[0] != record.initial_values:
            return {"valid": False, "reason": "initial_state_mismatch"}
        seen: set[tuple[int, ...]] = set()
        for index, state in enumerate(record.states):
            if state in seen and index != len(record.states) - 1:
                return {"valid": False, "reason": "repeated_state", "index": index}
            seen.add(state)
        for index, (before, after) in enumerate(zip(record.states, record.states[1:])):
            if len(after) != 1 and len(after) != len(before) - 1:
                return {"valid": False, "reason": "non_contractive", "index": index}
            if not self._valid_transition(record.operator_id, before, after):
                return {"valid": False, "reason": "invalid_transition", "index": index}
        if len(record.states[-1]) != 1 or record.states[-1][0] != record.final_value:
            return {"valid": False, "reason": "invalid_terminal"}
        return {"valid": True, "steps": len(record.states) - 1, "final_value": record.final_value}

    def _valid_transition(self, operator_id: str, before: Sequence[int], after: Sequence[int]) -> bool:
        if operator_id == "scalar.add":
            return len(before) == 2 and tuple(after) == (before[0] + before[1],)
        if operator_id == "scalar.neg":
            return len(before) == 1 and tuple(after) == (-before[0],)
        if len(after) != len(before) - 1:
            return False
        reducer = (lambda a, b: a + b) if operator_id == "aggregation.sum" else (min if operator_id == "scalar.min" else max)
        for index in range(len(before) - 1):
            candidate = (*before[:index], reducer(before[index], before[index + 1]), *before[index + 2 :])
            if tuple(candidate) == tuple(after):
                return True
        return False

    def joint_operator(self, *, seed: int, split: str, step: int, sample_index: int, namespace: str = "joint") -> str:
        rng = random.Random(_stable_seed(seed, split, step, sample_index, f"{namespace}-operator"))
        return EXPERIMENT_OPERATORS[rng.randrange(len(EXPERIMENT_OPERATORS))]

    def _append_stop(self, response: list[str]) -> None:
        if self.explicit_stop:
            response.append("<TRACE_STOP>")

    def _response_for_states(self, operator_id: str, states: Sequence[Sequence[int]]) -> list[str]:
        response: list[str] = []
        for state in states:
            response.append(self.eq_token)
            response.extend(self._state_tokens(operator_id, state))
        self._append_stop(response)
        return response

    def _trace_view(self, *, seed: int, step: int, sample_index: int, operator_id: str, state_count: int) -> str:
        if state_count <= 2:
            total = self.config.full_trace_weight + self.config.terminal_weight
            roll = _stable_seed(seed, step, sample_index, operator_id, "trace-view") % max(1, total)
            return "full_trace" if roll < self.config.full_trace_weight else "terminal_stop"
        total = self.config.full_trace_weight + self.config.continuation_weight + self.config.terminal_weight
        roll = _stable_seed(seed, step, sample_index, operator_id, "trace-view") % total
        if roll < self.config.full_trace_weight:
            return "full_trace"
        if roll < self.config.full_trace_weight + self.config.continuation_weight:
            return "continuation"
        return "terminal_stop"

    def example_tokens(
        self,
        operator_id: str,
        *,
        seed: int,
        split: str,
        step: int,
        sample_index: int,
    ) -> list[str]:
        """Legacy V1 view: canonical typed trace without a response delimiter."""
        record = self._trace_record(
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
            canonical=True,
        )
        expression = self._state_tokens(operator_id, record.states[0])
        response = self._response_for_states(operator_id, record.states[1:])
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

        record = self._trace_record(
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
        )
        initial_expression = self._state_tokens(operator_id, record.states[0])

        if job_id == "base.common":
            for required in ("<TASK_COPY>", "<RESPONSE>"):
                if required not in self.tokenizer.token_to_id:
                    raise ValueError(f"base task token missing from tokenizer: {required}")
            prompt = ["<TASK_COPY>", OPERATOR_TOKENS[operator_id], *initial_expression, "<RESPONSE>"]
            response = [self.eq_token, *initial_expression]
            self._append_stop(response)
            return TrainingExample(
                job_id=job_id,
                operator_id=operator_id,
                prompt_tokens=tuple(prompt),
                response_tokens=tuple(response),
                final_value=None,
                split=split,
                task="identity_equivalence",
                initial_values=record.initial_values,
                prompt_state_values=record.initial_values,
                trace_states=(record.initial_values, record.initial_values),
                partition_bucket=record.partition_bucket,
            )

        if "<RESPONSE>" not in self.tokenizer.token_to_id:
            raise ValueError("response-only training requires <RESPONSE> in the tokenizer")

        view = "full_trace" if split != "train" else self._trace_view(
            seed=seed,
            step=step,
            sample_index=sample_index,
            operator_id=operator_id,
            state_count=len(record.states),
        )
        start_index = 0
        if view == "continuation":
            rng = self._rng(
                seed=seed,
                split=split,
                step=step,
                sample_index=sample_index,
                operator_id=operator_id,
                namespace="continuation-start",
            )
            start_index = rng.randint(1, len(record.states) - 2)
        elif view == "terminal_stop":
            start_index = len(record.states) - 1

        prompt_state = record.states[start_index]
        prompt = [OPERATOR_TOKENS[operator_id], *self._state_tokens(operator_id, prompt_state), "<RESPONSE>"]
        if view == "terminal_stop":
            response: list[str] = []
            self._append_stop(response)
            final_value: int | None = None
        else:
            response = self._response_for_states(operator_id, record.states[start_index + 1 :])
            final_value = record.final_value
        return TrainingExample(
            job_id=job_id,
            operator_id=operator_id,
            prompt_tokens=tuple(prompt),
            response_tokens=tuple(response),
            final_value=final_value,
            split=split,
            task=view,
            initial_values=record.initial_values,
            prompt_state_values=tuple(prompt_state),
            trace_states=tuple(record.states[start_index:]),
            partition_bucket=record.partition_bucket,
        )

    def encode_training_example(self, example: TrainingExample, *, response_only: bool) -> EncodedTrainingExample:
        prompt_ids = self.tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
        response_ids = self.tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
        sequence = [*prompt_ids, *response_ids]
        source = sequence[:-1]
        target = sequence[1:]
        labels = list(target)
        if response_only:
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
        """Legacy V1 encoded view."""
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

    def _parse_state_tokens(self, operator_id: str, tokens: Sequence[str]) -> tuple[int, ...]:
        if not tokens:
            raise ValueError("empty state")
        if len(tokens) == 1:
            return (_parse_number_token(tokens[0]),)
        if operator_id in {"scalar.add", "aggregation.sum"}:
            values: list[int] = []
            expect_number = True
            for token in tokens:
                if expect_number:
                    values.append(_parse_number_token(token))
                elif token != self.plus_canonical:
                    raise ValueError(f"expected plus token, got {token}")
                expect_number = not expect_number
            if expect_number:
                raise ValueError("infix state ends with an operator")
            return tuple(values)
        if operator_id in {"scalar.min", "scalar.max"}:
            if tokens[0] != self.left_canonical or tokens[-1] != self.right_canonical:
                raise ValueError("list state has invalid brackets")
            inner = tokens[1:-1]
            values = []
            expect_number = True
            for token in inner:
                if expect_number:
                    values.append(_parse_number_token(token))
                elif token != self.comma_canonical:
                    raise ValueError(f"expected comma token, got {token}")
                expect_number = not expect_number
            if expect_number and inner:
                raise ValueError("list state ends with a comma")
            return tuple(values)
        raise ValueError(f"cannot parse multi-token state for {operator_id}")

    def verify_generated_ids(self, example: TrainingExample, generated_ids: Sequence[int]) -> dict[str, Any]:
        canonical = [self.tokenizer.tokens[token_id] for token_id in generated_ids]
        eos_positions = [index for index, token in enumerate(canonical) if token == "<EOS>"]
        stop_positions = [index for index, token in enumerate(canonical) if token == "<TRACE_STOP>"]
        first_stop = min([*eos_positions, *stop_positions], default=len(canonical))
        stop_correct = first_stop < len(canonical) and all(
            token in {"<EOS>", "<PAD>"} for token in canonical[first_stop:]
        )
        content = canonical[:first_stop]
        if example.task == "terminal_stop":
            return {
                "valid": len(content) == 0 and stop_correct,
                "stop_correct": stop_correct,
                "final_correct": None,
                "steps": 0,
                "reason": None if len(content) == 0 else "content_after_terminal_prompt",
            }
        if example.task == "identity_equivalence":
            expected = [self.eq_canonical, *self._state_tokens(example.operator_id, example.prompt_state_values)]
            expected = [self.tokenizer.tokens[self.tokenizer.token_to_id[token]] for token in expected]
            return {
                "valid": content == expected and stop_correct,
                "stop_correct": stop_correct,
                "final_correct": None,
                "steps": 1,
                "reason": None if content == expected else "identity_mismatch",
            }
        if not content or content[0] != self.eq_canonical:
            return {"valid": False, "stop_correct": stop_correct, "final_correct": False, "steps": 0, "reason": "missing_equality"}
        segments: list[list[str]] = []
        current: list[str] = []
        for token in content:
            if token == self.eq_canonical:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            segments.append(current)
        try:
            states = [tuple(example.prompt_state_values)] + [self._parse_state_tokens(example.operator_id, segment) for segment in segments]
        except (ValueError, IndexError) as exc:
            return {"valid": False, "stop_correct": stop_correct, "final_correct": False, "steps": 0, "reason": f"parse_error:{exc}"}
        seen: set[tuple[int, ...]] = set()
        for index, state in enumerate(states):
            if state in seen:
                return {"valid": False, "stop_correct": stop_correct, "final_correct": False, "steps": index, "reason": "repeated_state"}
            seen.add(state)
        for index, (before, after) in enumerate(zip(states, states[1:])):
            if not self._valid_transition(example.operator_id, before, after):
                return {"valid": False, "stop_correct": stop_correct, "final_correct": False, "steps": index, "reason": "invalid_transition"}
        terminal = states[-1]
        final_correct = len(terminal) == 1 and (example.final_value is None or terminal[0] == example.final_value)
        valid = bool(stop_correct and final_correct)
        return {
            "valid": valid,
            "stop_correct": stop_correct,
            "final_correct": final_correct,
            "steps": len(states) - 1,
            "reason": None if valid else "invalid_terminal_or_stop",
        }

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
