from __future__ import annotations

from pathlib import Path

from opfusion.training.design_config import load_design_run_config, model_design


ROOT = Path(__file__).parents[1]
PRODUCTION = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml"
SMOKE = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v4_smoke.yaml"


def test_cuda_smoke_matches_production_runtime_and_model_design() -> None:
    production = load_design_run_config(PRODUCTION)
    smoke = load_design_run_config(SMOKE)

    assert smoke.experiment_id.endswith("_smoke")
    assert smoke.output_dir != production.output_dir
    assert smoke.seeds == (0,)
    assert smoke.require_cuda
    assert smoke.deterministic_algorithms == production.deterministic_algorithms is True
    assert smoke.allow_tf32 == production.allow_tf32 is False
    assert smoke.precision == production.precision
    assert smoke.model_config == production.model_config
    assert smoke.tokenizer_config == production.tokenizer_config
    assert smoke.base_model_id == production.base_model_id
    assert smoke.operators == production.operators
    assert smoke.joint_model_ids == production.joint_model_ids
    assert smoke.max_parameters == production.max_parameters
    assert smoke.response_only_loss == production.response_only_loss
    assert smoke.effective_batch_size == production.effective_batch_size
    assert smoke.micro_batch_size == production.micro_batch_size == 0
    assert smoke.micro_batch_candidates == production.micro_batch_candidates
    assert smoke.data == production.data
    assert model_design(smoke).to_dict() == model_design(production).to_dict()

    assert 0 < smoke.max_steps <= 3
    assert smoke.eval_examples < production.eval_examples
    assert smoke.generation_eval_examples <= production.generation_eval_examples
    assert smoke.resolved_checkpoint_steps == (0, 1, 2)


def test_cuda_smoke_is_standalone_verified_and_used_by_production() -> None:
    script = (ROOT / "scripts/run_surface_v4_cuda_smoke.sh").read_text(encoding="utf-8")
    production_launcher = (ROOT / "scripts/run_bias_fusion_factory_surface_v4.sh").read_text(
        encoding="utf-8"
    )
    bootstrap = (ROOT / "scripts/bootstrap_arch_linux.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "nvidia-smi" in script
    assert "torch.cuda.is_available()" in script
    assert "CUBLAS_WORKSPACE_CONFIG" in script
    assert "opfusion-train-batch-design" in script
    assert "opfusion-evaluate-fusion" in script
    assert "opfusion-evaluate-unit-diagnostics" in script
    assert "--split validation" in script
    assert "SMOKE_EVALUATION_SEED" in script
    assert "cuda_smoke_complete.json" in script
    assert "expected seven completed smoke models" in script
    assert "nvidia_driver_version" in script
    assert "torch_cuda_runtime" in script
    assert 'gpu.get("name") == props.name' in script
    assert "smoke output_dir must be a relative path below runs/" in script
    assert "OPFUSION_ALLOW_V4_PRODUCTION" not in script

    assert "run_surface_v4_cuda_smoke.sh" in production_launcher
    assert 'SKIP_STATIC_PREFLIGHT=1 bash "$CUDA_SMOKE" "$SMOKE_CONFIG"' in production_launcher
    assert "run_surface_v4_cuda_smoke.sh" in bootstrap
    assert "TORCH_INDEX_URL" in bootstrap
    assert "--upgrade --force-reinstall torch --index-url" in bootstrap
    assert "scripts/run_surface_v4_cuda_smoke.sh" in workflow
    assert "Validate CUDA smoke plan" in workflow
