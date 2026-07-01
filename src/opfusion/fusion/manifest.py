from __future__ import annotations

from pathlib import Path
from typing import Any

from opfusion.io import load_yaml

REQUIRED_COMPATIBILITY_FIELDS = (
    "tokenizer_profile",
    "vocab_hash",
    "registry_assignment_hash",
    "output_space_id",
)

REQUIRED_L4_UNIT_FIELDS = (
    "operator_id",
    "applicability_level",
    "trained_unit",
    "corrector_or_no_corrector_justification",
    "metrics",
    "tokenizer_profile",
    "vocab_hash",
    "registry_assignment_hash",
    "output_space_id",
)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_manifest_dict(data: dict[str, Any], *, enforce_l4: bool = True) -> None:
    _require_mapping(data, "manifest")

    if data.get("dispatch", False) is not False:
        raise ValueError("runtime dispatch must always be false")

    for field in ("fusion_set_id", *REQUIRED_COMPATIBILITY_FIELDS, "units"):
        if field not in data:
            raise ValueError(f"fusion manifest missing required field: {field}")

    units = data["units"]
    if not isinstance(units, list):
        raise ValueError("fusion manifest units must be a list")

    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            if enforce_l4:
                raise ValueError(f"unit {index} must be a mapping for L4 runtime fusion")
            continue
        if enforce_l4 and unit.get("applicability_level") != "L4":
            raise ValueError(f"unit {index} is not L4 and cannot enter runtime fusion")
        if enforce_l4:
            for field in REQUIRED_L4_UNIT_FIELDS:
                if field not in unit:
                    raise ValueError(f"unit {index} missing required L4 field: {field}")
        for field in REQUIRED_COMPATIBILITY_FIELDS:
            if field in unit and unit[field] != data[field]:
                raise ValueError(f"unit {index} compatibility mismatch for {field}")


def load_and_validate_manifest(path: str | Path, *, enforce_l4: bool = True) -> dict[str, Any]:
    data = load_yaml(path)
    validate_manifest_dict(data, enforce_l4=enforce_l4)
    return data
