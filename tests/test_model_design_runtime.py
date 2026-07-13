from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import SyntheticTraceFactory
from opfusion.training.design_config import (
    ModelDesignConfig,
    attach_model_design,
    load_design_run_config,
    model_design,
)
from opfusion.training.experiment_contract import ensure_experiment_contract
from opfusion.training.trainer_design import (
    _masked_teacher_kl,
    _parent_checkpoint_selected,
    _select_validation_checkpoint,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml"


def _temporary_run(tmp_path: Path):
    original = load_design_run_config(CONFIG)
    run = replace(original, output_dir=str(tmp_path / "run"))
    return attach_model_design(run, model_design(original))


def test_weak_multitask_base_is_restricted_and_deterministic() -> None:
    run = load_design_run_config(CONFIG)
    design = model_design(run)
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    kwargs = dict(
        job_id="base.common",
        seed=7,
        split="train",
        step=91,
        sample_index=13,
        forced_operator="aggregation.sum",
    )
    first = factory.training_example(**kwargs)
    second = factory.training_example(**kwargs)
    assert first == second
    assert first.job_id == "base.common"
    assert first.task in {"full_trace", "continuation", "terminal_stop"}
    assert len(first.initial_values) <= design.base_weak_max_terms
    assert all(abs(value) <= design.base_weak_operand_abs_max for value in first.initial_values)
    assert first.final_value is not None or first.task == "terminal_stop"


def test_experiment_contract_rejects_design_changes_in_existing_output(tmp_path: Path) -> None:
    run = _temporary_run(tmp_path)
    first = ensure_experiment_contract(ROOT, run)
    assert len(first["fingerprint"]) == 64

    changed_design = replace(model_design(run), specialist_retention_kl_weight=0.20)
    changed = replace(run)
    attach_model_design(changed, changed_design)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        ensure_experiment_contract(ROOT, changed)


def test_validation_selection_uses_specialist_metric_not_final_step(tmp_path: Path) -> None:
    run = _temporary_run(tmp_path)
    job_id = "scalar.add"
    job_dir = Path(run.output_dir) / "seed_0" / "scalar_add"
    checkpoints = job_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    early = checkpoints / "step_000000100.pt"
    late = checkpoints / "step_000000200.pt"
    torch.save({"model_state_dict": {}, "step": 100}, early)
    torch.save({"model_state_dict": {}, "step": 200}, late)
    rows = [
        {
            "step": 100,
            "checkpoint": str(early),
            "validation_loss": {"scalar.add": 0.4, "mean": 1.0},
        },
        {
            "step": 200,
            "checkpoint": str(late),
            "validation_loss": {"scalar.add": 0.7, "mean": 0.5},
        },
    ]
    (job_dir / "checkpoint_index.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (job_dir / "complete.json").write_text(
        json.dumps({"job_id": job_id, "seed": 0, "final_checkpoint": str(late), "completed_step": 200}),
        encoding="utf-8",
    )
    selected = _select_validation_checkpoint(
        repo_root=ROOT,
        config=run,
        job_id=job_id,
        seed=0,
        fingerprint="a" * 64,
    )
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    complete = json.loads((job_dir / "complete.json").read_text(encoding="utf-8"))
    assert payload["selection"]["step"] == 100
    assert complete["selected_step"] == 100
    assert complete["final_checkpoint_is_selected"] is False


def test_dependency_parent_prefers_validation_selected_base(tmp_path: Path) -> None:
    run = _temporary_run(tmp_path)
    base_dir = Path(run.output_dir) / "seed_0" / "base_common"
    base_dir.mkdir(parents=True)
    final = base_dir / "final.pt"
    selected = base_dir / "selected.pt"
    final.write_bytes(b"final")
    selected.write_bytes(b"selected")
    (base_dir / "complete.json").write_text(
        json.dumps({"final_checkpoint": str(final), "selected_checkpoint": str(selected), "completed_step": 1}),
        encoding="utf-8",
    )
    parent = _parent_checkpoint_selected(
        ROOT,
        run,
        0,
        "scalar.add",
        Path(run.output_dir) / "seed_0" / "shared_initial.pt",
    )
    assert parent == selected


def test_teacher_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, -100], [3, -100, -100]])
    value = _masked_teacher_kl(logits, logits.clone(), labels)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


def test_invalid_retention_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="retention examples"):
        ModelDesignConfig(
            specialist_retention_kl_weight=0.1,
            specialist_retention_examples_per_operator=0,
        ).validate()
