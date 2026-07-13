from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opfusion.model import GPTModel, load_config
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.audit_data import audit_data
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


PRIMARY_CONFIG = Path("configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml")
PRIMARY_LAUNCHER = Path("scripts/run_bias_fusion_factory_surface_v3.sh")
TYPED_V2_LAUNCHER = Path("scripts/run_bias_fusion_factory_v2.sh")
ARCH_BOOTSTRAP = Path("scripts/bootstrap_arch_linux.sh")
ARCH_RUNBOOK = Path("docs/arch_linux_runbook.md")


def _read(root: Path, path: Path) -> str:
    target = root / path
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_text(encoding="utf-8")


def audit_repo(repo_root: str | Path, *, data_samples_per_operator: int = 32) -> dict[str, Any]:
    """Audit the one supported production path and its data/model ABI.

    This intentionally treats surface-v3 as canonical. Typed v2 may remain in
    the repository, but it must require an explicit opt-in and must not appear
    as the default command in the Arch bootstrap or runbook.
    """

    root = Path(repo_root).resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def check(condition: bool, kind: str, **details: Any) -> None:
        if not condition:
            errors.append({"kind": kind, **details})

    config_path = root / PRIMARY_CONFIG
    check(config_path.is_file(), "missing_primary_config", path=str(PRIMARY_CONFIG))
    if not config_path.is_file():
        return {"status": "failed", "errors": errors, "warnings": warnings}

    run = load_run_config(config_path)
    tokenizer_path = root / run.tokenizer_config
    model_path = root / run.model_config
    tokenizer = FixedVocabTokenizer.from_config(tokenizer_path)
    model_config = load_config(model_path)
    model = GPTModel(model_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)

    check(run.experiment_id == "gpt_bias_fusion_factory_surface_v3", "unexpected_primary_experiment_id", value=run.experiment_id)
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

    # The base and specialists must define logits on the same production prefix.
    # Validation forces a full arithmetic trace, avoiding the train-only
    # terminal/continuation mixture in this structural check.
    shared_prefix_checks: dict[str, bool] = {}
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
        matched = base.prompt_tokens == specialist.prompt_tokens
        shared_prefix_checks[operator_id] = matched
        check(matched, "base_specialist_prefix_mismatch", operator=operator_id, base=list(base.prompt_tokens), specialist=list(specialist.prompt_tokens))
        check(base.task == "identity_equivalence" and base.final_value is None, "base_target_is_not_neutral_identity", operator=operator_id)
        check(specialist.final_value is not None, "specialist_missing_arithmetic_target", operator=operator_id)

    try:
        launcher = _read(root, PRIMARY_LAUNCHER)
        bootstrap = _read(root, ARCH_BOOTSTRAP)
        runbook = _read(root, ARCH_RUNBOOK)
        typed_launcher = _read(root, TYPED_V2_LAUNCHER)
    except FileNotFoundError as exc:
        errors.append({"kind": "missing_operational_file", "path": str(exc)})
        launcher = bootstrap = runbook = typed_launcher = ""

    check(str(PRIMARY_CONFIG) in launcher, "primary_launcher_default_config_mismatch")
    check("opfusion-audit" in launcher, "primary_launcher_skips_repository_audit")
    check("opfusion-audit-data" in launcher, "primary_launcher_skips_data_audit")
    check("gpt_bias_fusion_factory_surface_v3" in bootstrap, "bootstrap_points_to_noncanonical_experiment")
    check("run_bias_fusion_factory_v2.sh" not in bootstrap, "bootstrap_advertises_typed_v2")
    check("gpt_bias_fusion_factory_surface_v3" in runbook, "runbook_omits_canonical_experiment")
    check("OPFUSION_ALLOW_TYPED_V2" in typed_launcher, "typed_v2_launcher_not_guarded")

    data_report = audit_data(config_path, samples_per_operator=data_samples_per_operator)
    if data_report["status"] != "passed":
        errors.append({"kind": "generated_data_audit_failed", "failures": data_report["failures"]})
    warnings.extend(data_report.get("warnings", []))

    return {
        "status": "passed" if not errors else "failed",
        "primary_config": str(PRIMARY_CONFIG),
        "experiment_id": run.experiment_id,
        "tokenizer_profile": tokenizer.profile,
        "vocab_hash": tokenizer.vocab_hash,
        "vocab_size": tokenizer.vocab_size,
        "model_parameters": model.param_count,
        "jobs": list(run.jobs),
        "seeds": list(run.seeds),
        "shared_prefix_checks": shared_prefix_checks,
        "data_audit_status": data_report["status"],
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit the canonical bias-fusion experiment and repository safety contract.")
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
