from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.audit_data_design import audit_design_data
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.training.design_config import load_design_run_config, model_design


PRIMARY_CONFIG = Path("configs/experiments/gpt_bias_fusion_factory_surface_v4.yaml")
PRIMARY_LAUNCHER = Path("scripts/run_bias_fusion_factory_surface_v4.sh")
PILOT_LAUNCHER = Path("scripts/run_model_design_pilot.sh")
PILOT_WATCHER = Path("scripts/watch_model_design_pilot.sh")
PILOT_STATUS = Path("scripts/status_model_design_pilot.sh")
BATCH_DESIGN = Path("src/opfusion/training/batch_design.py")
HARDENED_TRAINER = Path("src/opfusion/training/trainer_design_hardened.py")
SEEDED_EVALUATOR = Path("src/opfusion/fusion_eval_seeded.py")
UNIT_DIAGNOSTICS = Path("src/opfusion/fusion_diagnostics.py")
PAIR_AUDIT = Path("src/opfusion/training/audit_pilot_pairs.py")
EXPERIMENT_CONTRACT = Path("src/opfusion/training/experiment_contract.py")
PYPROJECT = Path("pyproject.toml")
LEGACY_V3_LAUNCHER = Path("scripts/run_bias_fusion_factory_surface_v3.sh")
TYPED_V2_LAUNCHER = Path("scripts/run_bias_fusion_factory_v2.sh")
ARCH_BOOTSTRAP = Path("scripts/bootstrap_arch_linux.sh")
ARCH_RUNBOOK = Path("docs/arch_linux_runbook.md")
PILOT_CONDITIONS = (
    "identity_unanchored",
    "identity_retention",
    "weak_unanchored",
    "weak_retention",
)


def _read(root: Path, path: Path) -> str:
    target = root / path
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_text(encoding="utf-8")


def _shared_prompt_schema(base: Any, specialist: Any) -> bool:
    return bool(
        base.prompt_tokens
        and specialist.prompt_tokens
        and base.prompt_tokens[0] == specialist.prompt_tokens[0]
        and base.prompt_tokens[-1] == "<RESPONSE>"
        and specialist.prompt_tokens[-1] == "<RESPONSE>"
        and "<TASK_COPY>" not in base.prompt_tokens
        and "<TASK_COPY>" not in specialist.prompt_tokens
    )


def audit_repo(repo_root: str | Path, *, data_samples_per_operator: int = 32) -> dict[str, Any]:
    """Audit the guarded production candidate and model-design safety gates."""

    root = Path(repo_root).resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def check(ok: bool, kind: str, **details: Any) -> None:
        if not ok:
            errors.append({"kind": kind, **details})

    config_path = root / PRIMARY_CONFIG
    check(config_path.is_file(), "missing_primary_config", path=str(PRIMARY_CONFIG))
    if not config_path.is_file():
        return {"status": "failed", "errors": errors, "warnings": warnings}

    run = load_design_run_config(config_path)
    design = model_design(run)
    tokenizer_path = root / run.tokenizer_config
    model_path = root / run.model_config
    tokenizer = FixedVocabTokenizer.from_config(tokenizer_path)
    model_config = load_config(model_path)
    model = GPTModel(model_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)

    check(run.experiment_id == "gpt_bias_fusion_factory_surface_v4", "unexpected_primary_experiment_id", value=run.experiment_id)
    check(run.base_model_id == "base.common", "missing_trained_common_base", value=run.base_model_id)
    check(tuple(run.operators) == EXPERIMENT_OPERATORS, "unexpected_operator_set", value=list(run.operators))
    check(run.response_only_loss, "prompt_loss_not_masked")
    check(run.require_cuda, "production_does_not_require_cuda")
    check(tokenizer.profile == "operator_experiment_surface_v3", "unexpected_tokenizer_profile", value=tokenizer.profile)
    check(factory.eq_canonical == "=", "surface_equality_not_active", value=factory.eq_canonical)
    check(not factory.explicit_stop, "surface_profile_uses_explicit_trace_stop")
    check("<EQ_STEP>" not in tokenizer.tokens, "typed_eq_token_in_surface_vocab")
    check("<TRACE_STOP>" not in tokenizer.tokens, "typed_stop_token_in_surface_vocab")
    check(tokenizer.token_to_id.get("<EQ_STEP>") == tokenizer.token_to_id.get("="), "equality_alias_mismatch")
    check(tokenizer.token_to_id.get("<TRACE_STOP>") == tokenizer.eos_id, "stop_alias_not_eos")
    check(model_config.vocab_size == tokenizer.vocab_size, "model_tokenizer_vocab_mismatch", model=model_config.vocab_size, tokenizer=tokenizer.vocab_size)
    check(model.param_count <= run.max_parameters <= 1_000_000, "parameter_limit_violation", parameters=model.param_count, limit=run.max_parameters)

    check(design.base_target_mode == "weak_multitask", "production_base_is_not_weak_multitask", value=design.base_target_mode)
    check(design.specialist_retention_kl_weight > 0.0, "production_retention_kl_disabled")
    check(design.specialist_retention_examples_per_operator > 0, "production_retention_examples_missing")
    check(design.specialist_parameter_anchor_weight > 0.0, "production_parameter_anchor_disabled")
    check(design.strict_experiment_fingerprint, "strict_experiment_fingerprint_disabled")

    shared_prefix_checks: dict[str, bool] = {}
    weak_base_checks: dict[str, bool] = {}
    for index, operator_id in enumerate(EXPERIMENT_OPERATORS):
        kwargs = dict(
            seed=91,
            split="validation",
            step=index,
            sample_index=index,
            forced_operator=operator_id,
        )
        base = factory.training_example("base.common", **kwargs)
        specialist = factory.training_example(operator_id, **kwargs)
        matched = _shared_prompt_schema(base, specialist)
        shared_prefix_checks[operator_id] = matched
        weak_ok = (
            base.job_id == "base.common"
            and base.task in {"full_trace", "continuation", "terminal_stop"}
            and len(base.initial_values) <= design.base_weak_max_terms
            and all(abs(value) <= design.base_weak_operand_abs_max for value in base.initial_values)
        )
        weak_base_checks[operator_id] = weak_ok
        check(matched, "base_specialist_prefix_schema_mismatch", operator=operator_id, base=list(base.prompt_tokens), specialist=list(specialist.prompt_tokens))
        check(weak_ok, "weak_base_contract_violation", operator=operator_id, task=base.task, values=base.initial_values)
        check(specialist.final_value is not None, "specialist_missing_arithmetic_target", operator=operator_id)

    pilot_configs: dict[str, dict[str, Any]] = {}
    for condition in PILOT_CONDITIONS:
        path = root / f"configs/experiments/model_design_pilot_{condition}.yaml"
        check(path.is_file(), "missing_pilot_config", condition=condition, path=str(path.relative_to(root)))
        if not path.is_file():
            continue
        pilot_run = load_design_run_config(path)
        pilot_configs[condition] = {
            "deterministic_algorithms": pilot_run.deterministic_algorithms,
            "allow_tf32": pilot_run.allow_tf32,
            "max_steps": pilot_run.max_steps,
            "effective_batch_size": pilot_run.effective_batch_size,
            "seed": list(pilot_run.seeds),
        }
        check(pilot_run.deterministic_algorithms, "pilot_not_deterministic", condition=condition)
        check(not pilot_run.allow_tf32, "pilot_tf32_enabled", condition=condition)
        check(tuple(pilot_run.seeds) == (0,), "pilot_seed_mismatch", condition=condition, seeds=list(pilot_run.seeds))

    try:
        launcher = _read(root, PRIMARY_LAUNCHER)
        pilot = _read(root, PILOT_LAUNCHER)
        pilot_watcher = _read(root, PILOT_WATCHER)
        pilot_status = _read(root, PILOT_STATUS)
        batch_design = _read(root, BATCH_DESIGN)
        hardened = _read(root, HARDENED_TRAINER)
        seeded_evaluator = _read(root, SEEDED_EVALUATOR)
        diagnostics = _read(root, UNIT_DIAGNOSTICS)
        pair_audit = _read(root, PAIR_AUDIT)
        contract = _read(root, EXPERIMENT_CONTRACT)
        pyproject = _read(root, PYPROJECT)
        bootstrap = _read(root, ARCH_BOOTSTRAP)
        runbook = _read(root, ARCH_RUNBOOK)
        legacy_v3 = _read(root, LEGACY_V3_LAUNCHER)
        typed_v2 = _read(root, TYPED_V2_LAUNCHER)
    except FileNotFoundError as exc:
        errors.append({"kind": "missing_operational_file", "path": str(exc)})
        launcher = pilot = pilot_watcher = pilot_status = batch_design = hardened = seeded_evaluator = diagnostics = pair_audit = contract = pyproject = bootstrap = runbook = legacy_v3 = typed_v2 = ""

    check(str(PRIMARY_CONFIG) in launcher, "primary_launcher_default_config_mismatch")
    check("opfusion-train-batch-design" in launcher, "primary_launcher_uses_legacy_batch_runner")
    check("opfusion-audit-data-design" in launcher, "primary_launcher_skips_design_data_audit")
    check("OPFUSION_ALLOW_V4_PRODUCTION" in launcher, "production_launcher_not_gated")
    for condition in PILOT_CONDITIONS:
        check(condition in pilot, f"pilot_missing_{condition}")
    check("watch_model_design_pilot.sh" in pilot, "pilot_detach_does_not_use_watchdog")
    check("status_model_design_pilot.sh" in pilot, "pilot_does_not_advertise_status_command")
    check("CUBLAS_WORKSPACE_CONFIG" in pilot, "pilot_missing_deterministic_cublas_workspace")
    check("evaluation_splits=(validation)" in pilot, "pilot_evaluation_split_contract_missing")
    declared_splits = pilot.split("evaluation_splits=", 1)[-1].split("for condition", 1)[0]
    check("test" not in declared_splits, "pilot_consumes_reserved_iid_test")
    check("operand_ood" not in declared_splits, "pilot_consumes_reserved_operand_ood")
    check("length_ood" not in declared_splits, "pilot_consumes_reserved_length_ood")
    check("--split \"$split\"" in pilot, "pilot_does_not_iterate_declared_splits")
    check("opfusion-evaluate-unit-diagnostics" in pilot, "pilot_missing_unit_diagnostics")
    check("opfusion-audit-pilot-pairs" in pilot, "pilot_missing_pair_consistency_audit")
    check("MAX_RESTARTS" in pilot_watcher, "pilot_watchdog_has_no_retry_limit")
    check("OPFUSION_PILOT_WATCHED" in pilot_watcher, "pilot_watchdog_does_not_mark_worker")
    check("pilot_state.json" in pilot_watcher, "pilot_watchdog_has_no_state_file")
    check("64|65|66|67|73" in pilot_watcher, "pilot_watchdog_retries_scientific_pair_failure")
    check("completed_models" in pilot_status, "pilot_status_omits_model_progress")
    check("fusion_reports={reports}/1" in pilot_status, "pilot_status_report_count_mismatch")
    check("pair_consistency" in pilot_status, "pilot_status_omits_pair_consistency")
    check("trainer_design_hardened import train_job" in batch_design, "batch_design_bypasses_hardened_trainer")
    check("full_domain_inactive_retention" in hardened, "inactive_retention_not_full_domain")
    check("enable_flash_sdp(False)" in hardened, "deterministic_pilot_keeps_flash_attention")
    check("DEFAULT_PILOT_EVALUATION_SEED" in seeded_evaluator, "seeded_evaluator_missing_pilot_namespace")
    check("DEFAULT_FINAL_EVALUATION_SEED" in seeded_evaluator, "seeded_evaluator_missing_final_namespace")
    check("evaluation_seed" in diagnostics, "unit_diagnostics_omit_evaluation_seed")
    check("inactive_summary" in diagnostics, "unit_diagnostics_missing_inactive_summary")
    check("paired_shared_endpoint_mismatch" in pair_audit, "pair_audit_missing_exact_state_check")
    check("paired_specialist_runtime_mismatch" in pair_audit, "pair_audit_missing_runtime_warning")
    check("trainer_design_hardened.py" in contract, "fingerprint_omits_hardened_trainer")
    check("fusion_eval_seeded.py" in contract, "fingerprint_omits_seeded_evaluator")
    check("fusion_diagnostics.py" in contract, "fingerprint_omits_unit_diagnostics")
    check('opfusion-evaluate-fusion = "opfusion.fusion_eval_seeded:main"' in pyproject, "canonical_evaluator_bypasses_seed_recording")
    check("run_model_design_pilot.sh" in bootstrap, "bootstrap_does_not_advertise_pilot")
    check("gpt_bias_fusion_factory_surface_v4" in bootstrap, "bootstrap_omits_surface_v4")
    check("gpt_bias_fusion_factory_surface_v4" in runbook, "runbook_omits_surface_v4")
    check("OPFUSION_ALLOW_LEGACY_SURFACE_V3" in legacy_v3, "legacy_surface_v3_not_guarded")
    check("OPFUSION_ALLOW_TYPED_V2" in typed_v2, "typed_v2_launcher_not_guarded")

    data_report = audit_design_data(config_path, samples_per_operator=data_samples_per_operator)
    if data_report["status"] != "passed":
        errors.append({"kind": "generated_data_audit_failed", "failures": data_report["failures"]})
    warnings.extend(data_report.get("warnings", []))

    return {
        "status": "passed" if not errors else "failed",
        "primary_config": str(PRIMARY_CONFIG),
        "experiment_id": run.experiment_id,
        "model_design": design.to_dict(),
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "vocab_size": tokenizer.vocab_size,
        "model_parameters": model.param_count,
        "jobs": list(run.jobs),
        "seeds": list(run.seeds),
        "shared_prefix_checks": shared_prefix_checks,
        "weak_base_checks": weak_base_checks,
        "pilot_configs": pilot_configs,
        "pilot_reserved_final_splits": ["iid_test", "operand_ood", "length_ood"],
        "data_audit_status": data_report["status"],
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit the guarded model-design and bias-fusion repository contract.")
    parser.add_argument("repo_root", nargs="?", default=".")
    parser.add_argument("--data-samples-per-operator", type=int, default=32)
    args = parser.parse_args()
    if args.data_samples_per_operator <= 0:
        parser.error("--data-samples-per-operator must be positive")
    report = audit_repo(args.repo_root, data_samples_per_operator=args.data_samples_per_operator)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
