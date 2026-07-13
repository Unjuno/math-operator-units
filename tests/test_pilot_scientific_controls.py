from __future__ import annotations

from pathlib import Path

import torch

from opfusion.fusion_diagnostics import _base_to_unit_kl
from opfusion.fusion_eval_seeded import DEFAULT_FINAL_EVALUATION_SEED, DEFAULT_PILOT_EVALUATION_SEED, _default_evaluation_seed
from opfusion.training import trainer_design_hardened as hardened
from opfusion.training.audit_pilot_pairs import _state_hash
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.training.design_config import load_design_run_config


ROOT = Path(__file__).parents[1]
CONDITIONS = (
    "identity_unanchored",
    "identity_retention",
    "weak_unanchored",
    "weak_retention",
)


def test_pilot_configs_use_deterministic_math_and_reserve_final_splits() -> None:
    for condition in CONDITIONS:
        config = load_design_run_config(ROOT / f"configs/experiments/model_design_pilot_{condition}.yaml")
        assert config.deterministic_algorithms
        assert not config.allow_tf32
        assert config.seeds == (0,)

    launcher = (ROOT / "scripts/run_model_design_pilot.sh").read_text(encoding="utf-8")
    assert "evaluation_splits=(validation)" in launcher
    assert "opfusion-evaluate-unit-diagnostics" in launcher
    assert "opfusion-audit-pilot-pairs" in launcher
    assert "--split test" not in launcher
    declared = launcher.split("evaluation_splits=", 1)[1].split("for condition", 1)[0]
    assert "operand_ood" not in declared
    assert "length_ood" not in declared

    trainer_surface = (ROOT / "src/opfusion/training/trainer_surface.py").read_text(encoding="utf-8")
    assert 'splits = ("validation",)' in trainer_surface
    assert 'splits = ("validation", "operand_ood", "length_ood")' not in trainer_surface


def test_pilot_and_final_evaluation_seed_namespaces_are_separate() -> None:
    pilot = ROOT / "configs/experiments/model_design_pilot_weak_retention.yaml"
    production = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml"
    assert _default_evaluation_seed(pilot) == DEFAULT_PILOT_EVALUATION_SEED
    assert _default_evaluation_seed(production) == DEFAULT_FINAL_EVALUATION_SEED
    assert DEFAULT_PILOT_EVALUATION_SEED != DEFAULT_FINAL_EVALUATION_SEED
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'opfusion-evaluate-fusion = "opfusion.fusion_eval_seeded:main"' in pyproject


def test_retention_redirects_base_batches_to_full_domain_inactive_operator(monkeypatch) -> None:
    config = load_design_run_config(ROOT / "configs/experiments/model_design_pilot_weak_retention.yaml")
    calls: list[tuple[str, str | None]] = []
    original_class_batch = SyntheticTraceFactory.batch

    def fake_batch(self, operator_id: str, *args, **kwargs):
        calls.append((operator_id, kwargs.get("forced_operator")))
        return "sentinel"

    monkeypatch.setattr(hardened, "_ORIGINAL_BATCH", fake_batch)
    factory = object.__new__(SyntheticTraceFactory)
    try:
        with hardened._full_domain_inactive_retention("scalar.add", config):
            result = SyntheticTraceFactory.batch(factory, "base.common", forced_operator="scalar.neg")
        assert result == "sentinel"
        assert calls == [("scalar.neg", "scalar.neg")]
    finally:
        SyntheticTraceFactory.batch = original_class_batch  # type: ignore[method-assign]


def test_pair_state_hash_detects_exact_equality(tmp_path: Path) -> None:
    state = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    changed = tmp_path / "changed.pt"
    torch.save({"model_state_dict": state}, left)
    torch.save({"model_state_dict": {"weight": state["weight"].clone()}}, right)
    altered = state["weight"].clone()
    altered[0, 0] = altered[0, 0] + 1
    torch.save({"model_state_dict": {"weight": altered}}, changed)
    assert _state_hash(left) == _state_hash(right)
    assert _state_hash(left) != _state_hash(changed)


def test_unit_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 5, 11)
    value = _base_to_unit_kl(logits, logits)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-7)


def test_all_experiment_operators_remain_present() -> None:
    assert EXPERIMENT_OPERATORS == (
        "scalar.add",
        "aggregation.sum",
        "scalar.neg",
        "scalar.min",
        "scalar.max",
    )
