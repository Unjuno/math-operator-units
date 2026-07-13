from pathlib import Path

import torch

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.batch import _plan, _write_subset_directory
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


ROOT = Path(__file__).parents[1]


def _v2():
    run = load_run_config(ROOT / "configs/experiments/gpt_bias_fusion_factory_v2.yaml")
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    return run, tokenizer, SyntheticTraceFactory(tokenizer, run.data)


def test_v2_tokenizer_and_model_remain_under_parameter_limit() -> None:
    run, tokenizer, _ = _v2()
    assert tokenizer.vocab_size == 2066
    assert "<RESPONSE>" in tokenizer.token_to_id
    assert "<TASK_COPY>" in tokenizer.token_to_id
    config = load_config(ROOT / run.model_config)
    model = GPTModel(config)
    assert config.max_seq_len == 256
    assert model.param_count == 863_184
    assert model.param_count <= 1_000_000


def test_response_only_labels_mask_the_prompt() -> None:
    _, _, factory = _v2()
    example = factory.training_example(
        "aggregation.sum",
        seed=0,
        split="train",
        step=0,
        sample_index=0,
    )
    encoded = factory.encode_training_example(example, response_only=True)
    first_supervised = encoded.prompt_length - 1
    assert all(value == -100 for value in encoded.labels[:first_supervised])
    assert encoded.labels[first_supervised] != -100
    assert encoded.response_length >= 1


def test_common_base_learns_identity_equality_not_arithmetic_answers() -> None:
    _, _, factory = _v2()
    example = factory.training_example(
        "base.common",
        seed=0,
        split="train",
        step=0,
        sample_index=3,
    )
    assert example.prompt_tokens[0] == "<TASK_COPY>"
    assert example.prompt_tokens[-1] == "<RESPONSE>"
    assert example.response_tokens[0] == "<EQ_STEP>"
    assert example.response_tokens[-1] == "<TRACE_STOP>"
    assert example.trace_states[0] == example.trace_states[1]
    assert example.final_value is None


def test_v2_plan_has_minimum_seven_models_and_sixteen_checkpoints() -> None:
    run, _, _ = _v2()
    plan = _plan(run)
    assert run.jobs[0] == "base.common"
    assert tuple(run.jobs[1:6]) == EXPERIMENT_OPERATORS
    assert run.jobs[-1] == "joint.all_five.exposure_matched"
    assert plan["trained_models_per_seed"] == 7
    assert plan["total_trained_models"] == 21
    assert plan["logical_checkpoints_per_model"] == 16
    assert plan["exposure_multiplier_by_job"][run.primary_joint_model_id] == 5


def test_length_ood_batch_fits_v2_context() -> None:
    run, _, factory = _v2()
    input_ids, labels = factory.batch(
        "aggregation.sum",
        seed=0,
        split="length_ood",
        step=0,
        batch_size=4,
        device=torch.device("cpu"),
        response_only=True,
    )
    model_config = load_config(ROOT / run.model_config)
    assert input_ids.shape == labels.shape
    assert input_ids.shape[1] <= model_config.max_seq_len


def test_v2_subset_manifest_uses_trained_base(tmp_path: Path) -> None:
    _, tokenizer, _ = _v2()
    checkpoints = {operator: tmp_path / f"{operator}.pt" for operator in EXPERIMENT_OPERATORS}
    base = tmp_path / "base.pt"
    index_path = _write_subset_directory(
        target=tmp_path / "subsets",
        checkpoints=checkpoints,
        joint_checkpoint=tmp_path / "joint.pt",
        initial=tmp_path / "random.pt",
        base_checkpoint=base,
        tokenizer=tokenizer,
    )
    import json

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["count"] == 32
    assert {item["base_checkpoint"] for item in payload["subsets"]} == {str(base)}
    assert payload["subsets"][0]["unit_checkpoints"] == {}
