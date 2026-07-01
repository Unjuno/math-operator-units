from __future__ import annotations

import pytest

from opfusion.fusion import fuse_bias, inactive_leakage, validate_manifest_dict


BASE_MANIFEST = {
    "fusion_set_id": "test_scalar_core_v1",
    "dispatch": False,
    "tokenizer_profile": "tokenizer_core_v1",
    "vocab_hash": "v" * 64,
    "registry_assignment_hash": "r" * 64,
    "output_space_id": "core_logits_v1",
    "units": [
        {
            "operator_id": "scalar.add",
            "applicability_level": "L4",
            "trained_unit": "units/scalar.add/main.safetensors",
            "corrector_or_no_corrector_justification": "units/scalar.add/corrector.safetensors",
            "metrics": {"inactive_leakage_mean": 0.0},
            "tokenizer_profile": "tokenizer_core_v1",
            "vocab_hash": "v" * 64,
            "registry_assignment_hash": "r" * 64,
            "output_space_id": "core_logits_v1",
        }
    ],
}


def test_runtime_manifest_accepts_dispatch_false() -> None:
    validate_manifest_dict(BASE_MANIFEST)


def test_runtime_manifest_rejects_dispatch_true() -> None:
    manifest = dict(BASE_MANIFEST)
    manifest["dispatch"] = True
    with pytest.raises(ValueError, match="dispatch"):
        validate_manifest_dict(manifest)


def test_runtime_manifest_rejects_non_l4_units() -> None:
    manifest = dict(BASE_MANIFEST)
    manifest["units"] = [dict(BASE_MANIFEST["units"][0], applicability_level="L3")]
    with pytest.raises(ValueError, match="not L4"):
        validate_manifest_dict(manifest)


def test_runtime_manifest_rejects_compatibility_mismatch() -> None:
    manifest = dict(BASE_MANIFEST)
    manifest["units"] = [dict(BASE_MANIFEST["units"][0], vocab_hash="x" * 64)]
    with pytest.raises(ValueError, match="compatibility mismatch"):
        validate_manifest_dict(manifest)


def test_fuse_bias_adds_gated_biases() -> None:
    assert fuse_bias([1.0, 2.0], [{"gate": 0.5, "bias": [2.0, -2.0]}]) == [2.0, 1.0]


def test_fuse_bias_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="width mismatch"):
        fuse_bias([0.0, 0.0], [{"gate": 1.0, "bias": [1.0]}])


def test_inactive_leakage() -> None:
    assert inactive_leakage([0.1, 0.2, 0.3], active_indices=[1]) == pytest.approx(0.4)
