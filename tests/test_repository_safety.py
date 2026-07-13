from pathlib import Path

from opfusion.audit import audit_repo
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.training.design_config import load_design_run_config, model_design


ROOT = Path(__file__).parents[1]
PRIMARY = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml"


def test_repository_audit_passes_for_the_guarded_surface_v4_experiment() -> None:
    report = audit_repo(ROOT, data_samples_per_operator=16)
    assert report["status"] == "passed", report["errors"]
    assert all(report["shared_prefix_checks"].values())
    assert all(report["weak_base_checks"].values())


def test_weak_common_base_and_specialists_share_model_facing_prefix_schema() -> None:
    run = load_design_run_config(PRIMARY)
    design = model_design(run)
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    assert design.base_target_mode == "weak_multitask"
    for index, operator_id in enumerate(EXPERIMENT_OPERATORS):
        kwargs = dict(
            seed=123,
            split="validation",
            step=index,
            sample_index=index,
            forced_operator=operator_id,
        )
        base = factory.training_example("base.common", **kwargs)
        specialist = factory.training_example(operator_id, **kwargs)
        assert base.prompt_tokens[0] == specialist.prompt_tokens[0]
        assert base.prompt_tokens[-1] == specialist.prompt_tokens[-1] == "<RESPONSE>"
        assert "<TASK_COPY>" not in base.prompt_tokens
        assert base.job_id == "base.common"
        assert base.task in {"full_trace", "continuation", "terminal_stop"}
        assert len(base.initial_values) <= design.base_weak_max_terms
        assert all(abs(value) <= design.base_weak_operand_abs_max for value in base.initial_values)
        assert base.final_value is not None
        assert specialist.final_value is not None


def test_legacy_launchers_require_explicit_opt_in() -> None:
    typed = (ROOT / "scripts/run_bias_fusion_factory_v2.sh").read_text(encoding="utf-8")
    surface_v3 = (ROOT / "scripts/run_bias_fusion_factory_surface_v3.sh").read_text(encoding="utf-8")
    assert "OPFUSION_ALLOW_TYPED_V2" in typed
    assert "OPFUSION_ALLOW_LEGACY_SURFACE_V3" in surface_v3


def test_arch_bootstrap_advertises_pilot_and_guarded_v4() -> None:
    bootstrap = (ROOT / "scripts/bootstrap_arch_linux.sh").read_text(encoding="utf-8")
    assert "run_model_design_pilot.sh" in bootstrap
    assert "run_bias_fusion_factory_surface_v4.sh" in bootstrap
    assert "gpt_bias_fusion_factory_surface_v4.yaml" in bootstrap
    assert "OPFUSION_ALLOW_V4_PRODUCTION" in bootstrap
