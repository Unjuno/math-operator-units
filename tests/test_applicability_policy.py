from __future__ import annotations

from pathlib import Path

from opfusion.operators import level_rank, load_applicability_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "operators" / "applicability_policy.yaml"


def test_level_ordering() -> None:
    assert level_rank("L0") < level_rank("L1") < level_rank("L2") < level_rank("L3") < level_rank("L4")


def test_core_rule_blocks_unevaluable_units() -> None:
    policy = load_applicability_policy(POLICY)
    assert "No rule" in policy.core_rule
    assert policy.supervision_is_trainable("none") is False


def test_unit_requires_l3_and_runtime_requires_l4() -> None:
    policy = load_applicability_policy(POLICY)
    assert policy.can_enter_mode("L2", "unit") is False
    assert policy.can_enter_mode("L3", "unit") is True
    assert policy.can_enter_runtime_fusion("L3") is False
    assert policy.can_enter_runtime_fusion("L4") is True


def test_program_is_evaluable_but_not_directly_trainable_by_default() -> None:
    policy = load_applicability_policy(POLICY)
    assert policy.can_enter_mode("L2", "program") is True
    assert policy.supervision_is_trainable("program") is False
