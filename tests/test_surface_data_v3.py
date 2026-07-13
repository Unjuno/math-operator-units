from pathlib import Path

import torch

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer, build_vocab_hash
from opfusion.training.audit_data import audit_data
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml"


def _surface():
    run = load_run_config(CONFIG)
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    return run, tokenizer, factory


def test_surface_tokenizer_has_normal_equality_and_eos_policy() -> None:
    run, tokenizer, factory = _surface()
    assert tokenizer.vocab_size == 2065
    assert tokenizer.vocab_hash == build_vocab_hash(ROOT / run.tokenizer_config)
    assert "=" in tokenizer.tokens
    assert "<EQ_STEP>" not in tokenizer.tokens
    assert "<TRACE_STOP>" not in tokenizer.tokens
    assert tokenizer.token_to_id["<EQ_STEP>"] == tokenizer.token_to_id["="]
    assert tokenizer.token_to_id["<TRACE_STOP>"] == tokenizer.eos_id
    assert factory.eq_canonical == "="
    assert not factory.explicit_stop


def test_surface_model_stays_below_parameter_limit() -> None:
    run, tokenizer, _ = _surface()
    config = load_config(ROOT / run.model_config)
    model = GPTModel(config)
    assert config.vocab_size == tokenizer.vocab_size
    assert config.max_seq_len == 256
    assert model.param_count == 863_072
    assert model.param_count <= 1_000_000


def test_model_facing_trace_uses_equals_and_eos_not_typed_controls() -> None:
    _, tokenizer, factory = _surface()
    example = factory.training_example(
        "aggregation.sum",
        seed=0,
        split="validation",
        step=0,
        sample_index=0,
    )
    expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
    canonical = [tokenizer.tokens[token_id] for token_id in expected]
    assert canonical[0] == "="
    assert canonical[-1] == "<EOS>"
    assert "<EQ_STEP>" not in canonical
    assert "<TRACE_STOP>" not in canonical
    assert canonical.count("=") >= 2


def test_iid_hash_partitions_are_disjoint() -> None:
    _, _, factory = _surface()
    keys = {split: set() for split in ("train", "validation", "test")}
    for operator_id in EXPERIMENT_OPERATORS:
        for split in keys:
            for index in range(256):
                example = factory.training_example(
                    operator_id,
                    seed=3,
                    split=split,
                    step=index,
                    sample_index=index,
                )
                assert example.partition_bucket is not None
                assert factory._bucket_matches(split, example.partition_bucket)
                keys[split].add((operator_id, example.initial_values))
    assert not (keys["train"] & keys["validation"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["validation"] & keys["test"])


def test_train_contains_full_continuation_and_terminal_views() -> None:
    _, _, factory = _surface()
    observed = {
        factory.training_example(
            "aggregation.sum",
            seed=7,
            split="train",
            step=index,
            sample_index=index,
        ).task
        for index in range(1000)
    }
    assert observed == {"full_trace", "continuation", "terminal_stop"}


def test_train_reduction_order_is_not_only_left_fold() -> None:
    _, _, factory = _surface()
    found_non_left = False
    for index in range(256):
        record = factory._trace_record(
            "aggregation.sum",
            seed=11,
            split="train",
            step=index,
            sample_index=index,
        )
        before, after = record.states[0], record.states[1]
        left = (before[0] + before[1], *before[2:])
        if tuple(after) != tuple(left):
            found_non_left = True
            break
    assert found_non_left


def test_common_base_learns_identity_equality_not_arithmetic() -> None:
    _, tokenizer, factory = _surface()
    example = factory.training_example(
        "base.common",
        seed=0,
        split="train",
        step=4,
        sample_index=4,
    )
    assert example.task == "identity_equivalence"
    assert example.final_value is None
    assert example.trace_states[0] == example.trace_states[1]
    expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
    canonical = [tokenizer.tokens[token_id] for token_id in expected]
    assert canonical[0] == "="
    assert canonical[-1] == "<EOS>"


def test_expected_surface_traces_pass_exact_verifier() -> None:
    _, tokenizer, factory = _surface()
    for operator_id in EXPERIMENT_OPERATORS:
        for split in ("validation", "test", "operand_ood", "length_ood"):
            for index in range(32):
                example = factory.training_example(
                    operator_id,
                    seed=19,
                    split=split,
                    step=index,
                    sample_index=index,
                )
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                verification = factory.verify_generated_ids(example, expected)
                assert verification["valid"], (operator_id, split, example, verification)


def test_response_masking_and_context_for_surface_batch() -> None:
    run, _, factory = _surface()
    input_ids, labels = factory.batch(
        "aggregation.sum",
        seed=0,
        split="length_ood",
        step=0,
        batch_size=8,
        device=torch.device("cpu"),
        response_only=True,
    )
    model_config = load_config(ROOT / run.model_config)
    assert input_ids.shape == labels.shape
    assert input_ids.shape[1] <= model_config.max_seq_len
    assert torch.any(labels == -100)
    assert torch.any(labels != -100)


def test_data_audit_passes_surface_profile() -> None:
    report = audit_data(CONFIG, samples_per_operator=64)
    assert report["status"] == "passed", report["failures"]
    assert report["surface_policy"]["active"]
    assert report["iid_split_overlaps"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
