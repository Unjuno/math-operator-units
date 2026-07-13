from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_experiment_plan_exists() -> None:
    plan = yaml.safe_load((ROOT / "configs/experiments/experiment_plan_v1.yaml").read_text())["plan"]
    assert plan["id"] == "bias_fusion_surface_v4_v1"
