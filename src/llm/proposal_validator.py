import json
from pathlib import Path

from jsonschema import Draft202012Validator

from src.data.registry import DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "proposal.schema.json"
)


SUPPORTED_TARGET_FIELDS = {
    "material": {"material_id"},
    "process": {"process_id"},
    "raw_stock": {"raw_stock"},
    "geometry": {"geometry", "geometry.finished_volume_mm3"},
    "fastener": {"fasteners", "fastener_id"},
}


def _load_schema() -> dict:
    with PROPOSAL_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_proposal_schema(
    proposal: dict,
) -> dict:
    validator = Draft202012Validator(
        _load_schema()
    )
    errors = sorted(
        validator.iter_errors(proposal),
        key=lambda error: list(error.path),
    )

    return {
        "schema_valid": not errors,
        "errors": [
            {
                "path": list(error.path),
                "message": error.message,
            }
            for error in errors
        ],
    }


def _bom_part_ids(
    bom: dict,
) -> set[str]:
    return {
        part["part_id"]
        for part in bom.get("parts", [])
        if part.get("part_id")
    }


def _fastener_ids_from_value(
    value,
) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        fastener_id = value.get(
            "fastener_id"
        )
        return [fastener_id] if fastener_id else []

    if isinstance(value, list):
        ids = []
        for item in value:
            ids.extend(
                _fastener_ids_from_value(item)
            )
        return ids

    return []


def validate_proposal_authority(
    proposal: dict | None,
    bom: dict,
    registry: DataRegistry | None = None,
) -> dict:
    registry = registry or DataRegistry()
    unknown = []
    errors = []

    if not proposal:
        return {
            "authority_valid": False,
            "unknown_identifiers": [],
            "errors": [
                {
                    "category": "NO_PROPOSAL",
                    "message": "No parsed proposal was provided",
                }
            ],
        }

    part_id = proposal.get("part_id")

    if part_id not in _bom_part_ids(bom):
        unknown.append({
            "category": "UNKNOWN_PART_ID",
            "identifier": part_id,
        })

    change_type = proposal.get(
        "change_type"
    )
    target_field = proposal.get(
        "target_field"
    )

    supported_fields = SUPPORTED_TARGET_FIELDS.get(
        change_type,
        set(),
    )

    if (
        target_field
        and supported_fields
        and target_field not in supported_fields
    ):
        errors.append({
            "category": "UNSUPPORTED_TARGET_FIELD",
            "identifier": target_field,
        })

    new_value = proposal.get(
        "new_value"
    )

    if change_type == "material":
        if not isinstance(new_value, str):
            errors.append({
                "category": "INVALID_MATERIAL_ID_TYPE",
                "identifier": str(type(new_value).__name__),
            })
        elif new_value not in registry.materials:
            unknown.append({
                "category": "UNKNOWN_MATERIAL_ID",
                "identifier": new_value,
            })

    elif change_type == "process":
        if not isinstance(new_value, str):
            errors.append({
                "category": "INVALID_PROCESS_ID_TYPE",
                "identifier": str(type(new_value).__name__),
            })
        elif new_value not in registry.processes:
            unknown.append({
                "category": "UNKNOWN_PROCESS_ID",
                "identifier": new_value,
            })

    elif change_type == "fastener":
        for fastener_id in _fastener_ids_from_value(
            new_value
        ):
            if fastener_id not in registry.fasteners:
                unknown.append({
                    "category": "UNKNOWN_FASTENER_ID",
                    "identifier": fastener_id,
                })

    return {
        "authority_valid": not unknown and not errors,
        "unknown_identifiers": unknown,
        "errors": errors,
    }


def classify_hallucination(
    parse_valid: bool,
    schema_result: dict,
    authority_result: dict,
) -> dict:
    categories = []

    if not parse_valid:
        categories.append("PARSE_ERROR")

    if not schema_result.get(
        "schema_valid",
        False,
    ):
        categories.append("SCHEMA_ERROR")

    for item in authority_result.get(
        "unknown_identifiers",
        [],
    ):
        categories.append(item["category"])

    for item in authority_result.get(
        "errors",
        [],
    ):
        category = item.get("category")
        if category and category != "NO_PROPOSAL":
            categories.append(category)

    categories = sorted(set(categories))

    return {
        "hallucinated": bool(categories),
        "categories": categories,
    }
