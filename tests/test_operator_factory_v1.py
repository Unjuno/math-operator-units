from pathlib import Path

import torch

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory

ROOT = Path(__file__).parents[1]


def test_operator_tokenizer_has_exact_fixed_vocab() -> None:
    tokenizer = FixedVocabTokenizer.from_config(ROOT / "configs/tokenizer/operator_experiment_v1.yaml")
    assert tokenizer.vocab_size == 2064
    assert not any(token.startswith("<OP_RESERVED_") for token in tokenizer.tokens)
    for token in (
        "<OP_SCALAR_ADD>",
        "<OP_AGG_SUM>",
        "<OP_SCALAR_NEG>",
        "<OP_SCALAR_MIN>",
        "<OP_SCALAR_MAX>",
    ):
        assert token in tokenizer.token_to_id


def test_operator_gpt_stays_below_one_million_parameters() -> None:
    config = load_config(ROOT / "configs/model/gpt_operator_1m_v1.yaml")
    model = GPTModel(config)
    assert model.param_count == 848_624
    assert model.param_count <= 1_000_000


def test_trace_factory_is_deterministic_and_contractive() -> None:
    run = load_run_config(ROOT / "configs/experiments/gpt_operator_factory_v1.yaml")
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    first = factory.example_tokens("aggregation.sum", seed=2, split="train", step=7, sample_index=3)
    second = factory.example_tokens("aggregation.sum", seed=2, split="train", step=7, sample_index=3)
    assert first == second
    assert first.count("<EQ_STEP>") >= 2
    assert first[-1] == "<TRACE_STOP>"
    states = []
    current = []
    for token in first[1:]:
        if token in {"<EQ_STEP>", "<TRACE_STOP>"}:
            states.append(current)
            current = []
        else:
            current.append(token)
    unresolved_counts = [state.count("<PLUS>") for state in states]
    assert unresolved_counts == sorted(unresolved_counts, reverse=True)
    assert unresolved_counts[-1] == 0


def test_joint_job_balances_only_the_five_operator_families() -> None:
    run = load_run_config(ROOT / "configs/experiments/gpt_operator_factory_v1.yaml")
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    observed = {
        factory.joint_operator(seed=0, split="train", step=step, sample_index=index)
        for step in range(10)
        for index in range(20)
    }
    assert observed == set(EXPERIMENT_OPERATORS)


def test_batch_shapes_fit_model_context() -> None:
    run = load_run_config(ROOT / "configs/experiments/gpt_operator_factory_v1.yaml")
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    input_ids, labels = factory.batch(
        "joint.all_five",
        seed=0,
        split="train",
        step=0,
        batch_size=8,
        device=torch.device("cpu"),
    )
    model_config = load_config(ROOT / run.model_config)
    assert input_ids.shape == labels.shape
    assert input_ids.shape[1] <= model_config.max_seq_len
