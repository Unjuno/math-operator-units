from pathlib import Path

import yaml

from opfusion.training.design_config import load_design_run_config

ROOT = Path(__file__).parents[1]
PLAN_V1 = ROOT / "configs/experiments/experiment_plan_v1.yaml"
PLAN_V2 = ROOT / "configs/experiments/experiment_plan_v2.yaml"


def load_plan(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["plan"]


def test_experiment_plan_v2_matches_configs() -> None:
    v1 = load_plan(PLAN_V1)
    plan = load_plan(PLAN_V2)
    assert plan["id"] == "bias_fusion_surface_v4_v2"
    assert plan["supersedes"] == v1["id"]
    assert plan["status"] == "prospective_before_target_gpu_pilot"
    assert (ROOT / plan["document"]).is_file()

    smoke = load_design_run_config(ROOT / plan["smoke"]["config"])
    assert smoke.require_cuda and smoke.seeds == (0,)
    assert 0 < smoke.max_steps <= 3

    pilots = [load_design_run_config(ROOT / path) for path in plan["pilot"]["configs"]]
    assert len(pilots) == 4
    assert all(run.seeds == (0,) for run in pilots)
    assert all(run.require_cuda for run in pilots)
    assert all(run.deterministic_algorithms and not run.allow_tf32 for run in pilots)
    assert len({run.max_steps for run in pilots}) == 1
    assert plan["pilot"]["splits"] == ["validation"]
    assert plan["pilot"]["eligibility"] == v1["pilot"]["eligibility"]
    for key in (
        "clear_winner_gap",
        "near_tie_tolerance",
        "near_tie_inactive_drift_reduction",
        "retention_minimum_drift_reduction",
        "retention_maximum_relevant_regression",
    ):
        assert plan["pilot"][key] == v1["pilot"][key]

    production = load_design_run_config(ROOT / plan["production"]["config"])
    assert list(production.seeds) == plan["production"]["seeds"] == [0, 1, 2]
    assert plan["production"]["models_per_seed"] == 7

    final = plan["final"]
    assert final["evaluation_seed"] == 700000
    assert final["splits"] == ["iid_test", "operand_ood", "length_ood"]
    assert final["primary_condition"] == "raw_sum"
    assert final["primary_alpha"] == 1.0
    for key in (
        "preserved_mean_gap",
        "preserved_worst_operator_gap",
        "material_failure_mean_gap",
        "material_failure_eos_gap",
    ):
        assert final[key] == v1["final"][key]
    assert final["selected_rescue_is_secondary"] is True
    assert plan["reserved_until_frozen"] == final["splits"]


def test_plan_precommits_fallback_mixing_ladder() -> None:
    plan = load_plan(PLAN_V2)
    fallback = plan["contingency"]

    assert fallback["calibration_split"] == "validation"
    assert fallback["calibration_evaluation_seed"] == 703000
    assert fallback["calibration_examples_per_operator"] == 128
    assert fallback["activation"]["final_data_may_trigger_tuning"] is False
    assert fallback["validation"] == {
        "method": "leave_one_training_seed_out",
        "folds": 3,
        "tune_per_seed": False,
        "tune_per_operator": False,
    }
    assert fallback["fixed_baselines"] == ["raw_sum", "bias_mean"]
    assert fallback["stage_order"] == [
        "global_shrinkage",
        "norm_controlled",
        "static_weighted_mean",
        "consensus_tempered",
    ]
    assert fallback["global_shrinkage"]["alpha_grid"] == [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]
    for family in ("norm_controlled", "static_weighted_mean", "consensus_tempered"):
        assert fallback[family]["alpha_grid_from"] == "global_shrinkage"
    assert fallback["selection"]["choose_first_eligible_family"] is True
    assert fallback["selection"]["freeze_mixer_contract"] is True
    assert fallback["learned_router_followup"]["part_of_confirmatory_final"] is False
    assert fallback["learned_router_followup"]["requires_new_plan_and_output_roots"] is True
    assert plan["reporting"]["preserve_raw_result_as_confirmatory"] is True
