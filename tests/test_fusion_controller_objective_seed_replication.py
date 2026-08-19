from __future__ import annotations

from opfusion import fusion_controller_objective_ablation as objective
from opfusion import fusion_controller_objective_seed_replication as replication
from opfusion import fusion_self_tuned_operator_controller as self_tuned


def test_training_data_seed_is_fixed_while_init_seed_varies(monkeypatch) -> None:
    seen_data_seeds: list[int] = []

    def fake_collect(*args, **kwargs):
        seen_data_seeds.append(int(kwargs["data_seed"]))
        return "fixed-batch"

    monkeypatch.setattr(self_tuned, "collect_teacher_forced_control_batch", fake_collect)

    def fake_run_experiment(**kwargs):
        assert int(kwargs["controller_seed"]) == 739003
        value = self_tuned.collect_teacher_forced_control_batch(
            None, data_seed=kwargs["controller_seed"]
        )
        assert value == "fixed-batch"
        return {"status": "completed"}

    monkeypatch.setattr(objective, "run_experiment", fake_run_experiment)
    original = self_tuned.collect_teacher_forced_control_batch
    report = replication.run_with_fixed_training_data_seed(
        controller_data_seed=739000,
        controller_seed=739003,
    )

    assert seen_data_seeds == [739000]
    assert report["controller_data_seed"] == 739000
    assert report["replication_axis"] == "controller_initialization_seed_only"
    assert self_tuned.collect_teacher_forced_control_batch is original
