from pathlib import Path

import torch

from opfusion.training.batch import _write_subset_directory
from opfusion.training.data import EXPERIMENT_OPERATORS
from opfusion.training.trainer import _delta_summary


class _TokenizerStub:
    profile = "operator_experiment_v1"
    vocab_hash = "v" * 64


def test_subset_directory_writes_all_32_combinations(tmp_path: Path) -> None:
    checkpoints = {operator: tmp_path / f"{operator}.pt" for operator in EXPERIMENT_OPERATORS}
    index_path = _write_subset_directory(
        target=tmp_path / "subsets",
        checkpoints=checkpoints,
        joint_checkpoint=tmp_path / "joint.pt",
        initial=tmp_path / "initial.pt",
        tokenizer=_TokenizerStub(),
    )
    import json

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["count"] == 32
    assert len(payload["subsets"]) == 32
    assert payload["subsets"][0]["operators"] == []
    assert payload["subsets"][-1]["operators"] == list(EXPERIMENT_OPERATORS)


def test_parameter_delta_does_not_double_count_tied_head() -> None:
    initial_embedding = torch.zeros(2, 2)
    current_embedding = torch.ones(2, 2)
    initial = {
        "token_embedding.weight": initial_embedding.clone(),
        "lm_head.weight": initial_embedding.clone(),
    }
    current = {
        "token_embedding.weight": current_embedding.clone(),
        "lm_head.weight": current_embedding.clone(),
    }
    summary = _delta_summary(initial, current)
    assert summary["initial_to_current_l2"] == 2.0
