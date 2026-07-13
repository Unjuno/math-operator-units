from pathlib import Path

import yaml

from opfusion.training.design_config import load_design_run_config

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/experiments/experiment_plan_v1.yaml"


def test_experiment_plan_matches_configs() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))["plan"]
    assert plan["status"] == "prospective_before_target_gpu_pilot"

    smoke = load_design_run_config(ROOT / plan["smoke"]["config"])
    assert smoke.require_cuda and smoke.seeds == (0,)
    assert 0 < smoke.max_steps <= 3

    pilots = [load_design_run_config(ROOT / p) for p in plan["pilot"]["configs"]]
    assert len(pilots) == 4
    assert all(run.seeds == (0,) for run in pilots)
    assert all(run.require_cuda for run in pilots)
    assert all(run.deterministic_algorithms and not run.allow_tf32 for run in pilots)
    assert len({run.max_steps for run in pilots}) == 1
    assert plan["pilot"]["splits"] == ["validation"]

    production = load_design_run_config(ROOT / plan["production"]["config"])
    assert list(production.seeds) == plan["production"]["seeds"] == [0, 1, 2]
    assert plan["production"]["models_per_seed"] == 7

    final = plan["final"]
    assert final["evaluation_seed"] == 700000
    assert final["splits"] == ["iid_test", "operand_ood", "length_ood"]
    assert final["examples_per_operator"] == 64
    assert final["primary_subset"] == 31
    assert final["primary_condition"] == "raw_sum"
    assert final["primary_alpha"] == 1.0
    assert plan["reserved_until_frozen"] == final["splits"]


def test_plan_precommits_ambiguity_and_reporting_rules() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))["plan"]
    assert plan["pilot"]["ambiguous_action"].startswith("repeat_all_four_conditions")
    assert plan["pilot"]["no_eligible_action"] == "stop_and_version_new_plan"
    assert plan["reporting"]["replication_unit"] == "training_seed"
    assert plan["reporting"]["formal_p_values"] is False
    assert plan["reporting"]["missing_seed_policy"] == "no_imputation"
