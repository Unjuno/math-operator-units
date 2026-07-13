from pathlib import Path

from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import SyntheticTraceFactory


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml"


def _surface():
    run = load_run_config(CONFIG)
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    return tokenizer, SyntheticTraceFactory(tokenizer, run.data)


def test_zero_negation_is_a_valid_single_transition() -> None:
    tokenizer, factory = _surface()
    example = None
    # NEG(0) belongs to exactly one hash-partitioned IID split. Search all
    # partitions rather than assuming that its fixed bucket is validation.
    for split in ("train", "validation", "test"):
        for index in range(10_000):
            candidate = factory.training_example(
                "scalar.neg",
                seed=401,
                split=split,
                step=index,
                sample_index=index,
            )
            if candidate.prompt_state_values == (0,) and candidate.task != "terminal_stop":
                example = candidate
                break
        if example is not None:
            break
    assert example is not None
    expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
    result = factory.verify_generated_ids(example, expected)
    assert result["valid"], result


def test_consecutive_equals_are_rejected() -> None:
    tokenizer, factory = _surface()
    example = factory.training_example(
        "scalar.add",
        seed=402,
        split="validation",
        step=0,
        sample_index=0,
    )
    malformed = [
        tokenizer.token_to_id["="],
        tokenizer.token_to_id["="],
        tokenizer.token_to_id[f"<N_{example.final_value}>"],
        tokenizer.eos_id,
    ]
    result = factory.verify_generated_ids(example, malformed)
    assert not result["valid"]
    assert result["reason"] == "empty_equality_segment"
