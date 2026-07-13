from __future__ import annotations

from typing import Any, Sequence

from .data import SyntheticTraceFactory, TrainingExample


def verify_generated_ids_strict(
    self: SyntheticTraceFactory,
    example: TrainingExample,
    generated_ids: Sequence[int],
) -> dict[str, Any]:
    """Verify generated equality traces without accepting malformed separators.

    This is installed on ``SyntheticTraceFactory`` for both typed and surface
    profiles. It deliberately permits the one legitimate repeated-state case,
    scalar negation of zero (``0 = 0``), while rejecting empty equality
    segments such as ``= = 3``.
    """

    try:
        canonical = [self.tokenizer.tokens[token_id] for token_id in generated_ids]
    except IndexError as exc:
        return {
            "valid": False,
            "stop_correct": False,
            "final_correct": False,
            "steps": 0,
            "reason": f"token_id_out_of_range:{exc}",
        }

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
        return {
            "valid": False,
            "stop_correct": stop_correct,
            "final_correct": False,
            "steps": 0,
            "reason": "missing_equality",
        }

    segments: list[list[str]] = []
    current: list[str] | None = None
    for token in content:
        if token == self.eq_canonical:
            if current is not None:
                if not current:
                    return {
                        "valid": False,
                        "stop_correct": stop_correct,
                        "final_correct": False,
                        "steps": len(segments),
                        "reason": "empty_equality_segment",
                    }
                segments.append(current)
            current = []
        else:
            if current is None:
                return {
                    "valid": False,
                    "stop_correct": stop_correct,
                    "final_correct": False,
                    "steps": 0,
                    "reason": "content_before_equality",
                }
            current.append(token)

    if current is None or not current:
        return {
            "valid": False,
            "stop_correct": stop_correct,
            "final_correct": False,
            "steps": len(segments),
            "reason": "empty_equality_segment",
        }
    segments.append(current)

    try:
        states = [tuple(example.prompt_state_values)] + [
            self._parse_state_tokens(example.operator_id, segment) for segment in segments
        ]
    except (ValueError, IndexError) as exc:
        return {
            "valid": False,
            "stop_correct": stop_correct,
            "final_correct": False,
            "steps": 0,
            "reason": f"parse_error:{exc}",
        }

    seen: set[tuple[int, ...]] = set()
    for index, state in enumerate(states):
        if state in seen:
            zero_negation = (
                example.operator_id == "scalar.neg"
                and len(states) == 2
                and index == 1
                and state == (0,)
            )
            if not zero_negation:
                return {
                    "valid": False,
                    "stop_correct": stop_correct,
                    "final_correct": False,
                    "steps": index,
                    "reason": "repeated_state",
                }
        seen.add(state)

    for index, (before, after) in enumerate(zip(states, states[1:])):
        if not self._valid_transition(example.operator_id, before, after):
            return {
                "valid": False,
                "stop_correct": stop_correct,
                "final_correct": False,
                "steps": index,
                "reason": "invalid_transition",
            }

    terminal = states[-1]
    final_correct = len(terminal) == 1 and (
        example.final_value is None or terminal[0] == example.final_value
    )
    valid = bool(stop_correct and final_correct)
    return {
        "valid": valid,
        "stop_correct": stop_correct,
        "final_correct": final_correct,
        "steps": len(states) - 1,
        "reason": None if valid else "invalid_terminal_or_stop",
    }


def install_strict_verifier() -> None:
    SyntheticTraceFactory.verify_generated_ids = verify_generated_ids_strict  # type: ignore[method-assign]
