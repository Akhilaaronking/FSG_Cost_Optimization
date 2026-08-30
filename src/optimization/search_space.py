import json
from pathlib import Path

from src.data.registry import DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_SEARCH_SPACE_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "real_search_space.json"
)
DEFAULT_B4_BOM_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "pilot_10_parts_ground_truth.json"
)


VARIABLE_TYPES = {
    "continuous",
}


def _duplicates(values: list[str]) -> list[str]:
    seen = set()
    duplicate_ids = []

    for value in values:
        if value in seen and value not in duplicate_ids:
            duplicate_ids.append(value)
        seen.add(value)

    return duplicate_ids


def _baseline_part_ids(
    baseline_bom: dict,
) -> set[str]:
    parts = baseline_bom.get("parts")

    if not isinstance(parts, list) or not parts:
        raise ValueError(
            "Baseline BOM must contain a non-empty parts list"
        )

    part_ids = [
        part.get("part_id")
        for part in parts
    ]

    if any(not part_id for part_id in part_ids):
        raise ValueError(
            "Every baseline part must have part_id"
        )

    duplicates = _duplicates(part_ids)

    if duplicates:
        raise ValueError(
            f"Duplicate baseline part IDs: {duplicates}"
        )

    return set(part_ids)


def _baseline_parts_by_id(
    baseline_bom: dict,
) -> dict:
    return {
        part["part_id"]: part
        for part in baseline_bom.get("parts", [])
        if part.get("part_id")
    }


def _load_json(
    path: Path | str,
) -> dict:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _choice_ids(
    choices,
    field_name: str,
) -> list[str]:
    if choices is None:
        return []

    if not isinstance(choices, dict):
        raise ValueError(
            f"{field_name} must be an object mapping IDs to rationale"
        )

    return list(choices)


def _append_unique(
    values: list[str],
    value: str,
) -> list[str]:
    result = list(values)

    if value not in result:
        result.append(value)

    return result


def _validate_option_a_id_set(
    *,
    part_id: str,
    field_name: str,
    ids: list[str],
    registry_records: dict,
    label: str,
    errors: list[str],
):
    duplicates = _duplicates(ids)

    if duplicates:
        errors.append(
            f"{part_id}: duplicate {field_name}: {duplicates}"
        )

    unknown = [
        item
        for item in ids
        if item not in registry_records
    ]

    if unknown:
        errors.append(
            f"{part_id}: unknown {label} IDs in "
            f"{field_name}: {unknown}"
        )


def _numeric_verified_geometry_variables(
    dimension_envelope,
) -> dict:
    """
    Geometry is activated only for explicit numeric bounds
    that also carry an engineering verification flag.

    Current real_search_space.json uses descriptive envelopes
    or bare ranges, so this intentionally returns no active
    geometry variables for that dataset.
    """

    if not isinstance(dimension_envelope, dict):
        return {}

    variables = {}

    for field_name, spec in dimension_envelope.items():
        if (
            isinstance(spec, dict)
            and spec.get("engineering_verified") is True
            and isinstance(spec.get("min"), (int, float))
            and isinstance(spec.get("max"), (int, float))
        ):
            variables[field_name] = {
                "type": "continuous",
                "min": spec["min"],
                "max": spec["max"],
                "engineering_verified": True,
            }

    return variables


def convert_option_a_search_space(
    option_a: dict,
    baseline_bom: dict,
    registry: DataRegistry | None = None,
    include_unverified: bool = False,
) -> dict:
    """
    Convert the approved Option A real search-space format
    into the optimiser SearchSpace representation.

    The global registry is used only to validate identifier
    existence. It is never expanded into extra alternatives.
    """

    registry = registry or DataRegistry()

    if not isinstance(option_a, dict):
        raise ValueError(
            "Option A search space must be a dictionary"
        )

    parts = option_a.get("parts")

    if not isinstance(parts, list) or not parts:
        raise ValueError(
            "Option A search space must contain a non-empty parts list"
        )

    baseline_parts = _baseline_parts_by_id(
        baseline_bom
    )
    output_parts = []
    errors = []

    for option_part in parts:
        part_id = option_part.get("part_id")

        if not part_id:
            errors.append(
                "Option A part is missing part_id"
            )
            continue

        baseline_part = baseline_parts.get(
            part_id
        )

        if baseline_part is None:
            errors.append(
                f"{part_id}: not found in baseline BOM"
            )
            continue

        if option_part.get("engineering_verified") is not True:
            if include_unverified:
                errors.append(
                    f"{part_id}: engineering_verified must be true"
                )
            continue

        current = option_part.get(
            "current",
            {},
        )
        baseline_material = baseline_part.get(
            "material_id"
        )
        baseline_process = baseline_part.get(
            "process_id"
        )

        if current.get("material_id") != baseline_material:
            errors.append(
                f"{part_id}: current.material_id "
                f"{current.get('material_id')!r} does not match "
                f"baseline material_id {baseline_material!r}"
            )

        if current.get("process_id") != baseline_process:
            errors.append(
                f"{part_id}: current.process_id "
                f"{current.get('process_id')!r} does not match "
                f"baseline process_id {baseline_process!r}"
            )

        material_choices = _append_unique(
            _choice_ids(
                option_part.get(
                    "admissible_materials"
                ),
                "admissible_materials",
            ),
            baseline_material,
        )
        process_choices = _append_unique(
            _choice_ids(
                option_part.get(
                    "admissible_processes"
                ),
                "admissible_processes",
            ),
            baseline_process,
        )

        _validate_option_a_id_set(
            part_id=part_id,
            field_name="material_choices",
            ids=material_choices,
            registry_records=registry.materials,
            label="material",
            errors=errors,
        )
        _validate_option_a_id_set(
            part_id=part_id,
            field_name="process_choices",
            ids=process_choices,
            registry_records=registry.processes,
            label="process",
            errors=errors,
        )

        converted_part = {
            "part_id": part_id,
            "engineering_verified": True,
            "material_choices": material_choices,
            "process_choices": process_choices,
            "baseline_material_id": baseline_material,
            "baseline_process_id": baseline_process,
            "metadata": {
                "part_name": option_part.get(
                    "part_name"
                ),
                "engineering_context": option_part.get(
                    "engineering_context"
                ),
                "verification_status": option_part.get(
                    "verification_status"
                ),
                "admissible_material_rationale": option_part.get(
                    "admissible_materials",
                    {},
                ),
                "admissible_process_rationale": option_part.get(
                    "admissible_processes",
                    {},
                ),
                "inadmissible_materials_recorded_but_excluded": sorted(
                    option_part.get(
                        "inadmissible_materials",
                        {},
                    )
                ),
                "inadmissible_processes_recorded_but_excluded": sorted(
                    option_part.get(
                        "inadmissible_processes",
                        {},
                    )
                ),
                "dimension_envelope_recorded_but_not_activated": option_part.get(
                    "dimension_envelope",
                    {},
                ),
            },
        }

        geometry_variables = _numeric_verified_geometry_variables(
            option_part.get("dimension_envelope")
        )

        if geometry_variables:
            converted_part["geometry_variables"] = geometry_variables

        output_parts.append(converted_part)

    if errors:
        raise ValueError(
            "Invalid real search space:\n"
            + "\n".join(errors)
        )

    if not output_parts:
        raise ValueError(
            "Invalid real search space: no parts with "
            "engineering_verified == true"
        )

    search_space = {
        "search_space_id": option_a.get(
            "search_space_id",
            "REAL_B4_OPTION_A_MATERIAL_PROCESS",
        ),
        "engineering_verified": True,
        "purpose": "REAL_C5_OPTIMISATION_SEARCH_SPACE",
        "parts": output_parts,
        "metadata": {
            "source_schema_version": option_a.get(
                "schema_version"
            ),
            "source_status": option_a.get(
                "status"
            ),
            "source_note": option_a.get(
                "note"
            ),
            "verification": option_a.get(
                "verification",
                {},
            ),
            "active_scope": [
                "material",
                "process",
            ],
            "registry_validation_meaning": (
                "Identifier existence only; not engineering compatibility."
            ),
        },
    }

    validate_search_space(
        search_space,
        baseline_bom,
        registry=registry,
    )

    return search_space


def load_verified_real_search_space(
    real_search_space_path: Path | str = DEFAULT_REAL_SEARCH_SPACE_PATH,
    baseline_bom_path: Path | str = DEFAULT_B4_BOM_PATH,
    registry: DataRegistry | None = None,
) -> dict:
    option_a = _load_json(
        real_search_space_path
    )
    baseline_bom = _load_json(
        baseline_bom_path
    )

    return convert_option_a_search_space(
        option_a,
        baseline_bom,
        registry=registry,
    )


def validate_candidate_within_search_space(
    candidate_bom: dict,
    search_space: dict,
) -> dict:
    parts_by_id = _baseline_parts_by_id(
        candidate_bom
    )
    errors = []

    for part_space in search_space.get(
        "parts",
        [],
    ):
        part_id = part_space["part_id"]
        part = parts_by_id.get(part_id)

        if part is None:
            errors.append(
                f"{part_id}: missing from candidate BOM"
            )
            continue

        material_id = part.get("material_id")
        process_id = part.get("process_id")

        if (
            "material_choices" in part_space
            and material_id
            not in part_space["material_choices"]
        ):
            errors.append(
                f"{part_id}: material_id {material_id!r} "
                "outside approved material_choices"
            )

        if (
            "process_choices" in part_space
            and process_id
            not in part_space["process_choices"]
        ):
            errors.append(
                f"{part_id}: process_id {process_id!r} "
                "outside approved process_choices"
            )

    return {
        "valid": not errors,
        "errors": errors,
    }


def validate_search_space(
    search_space: dict,
    baseline_bom: dict,
    registry: DataRegistry | None = None,
) -> dict:
    registry = registry or DataRegistry()

    if not isinstance(search_space, dict):
        raise ValueError(
            "Search space must be a dictionary"
        )

    if not search_space.get("search_space_id"):
        raise ValueError(
            "Search space is missing search_space_id"
        )

    if "engineering_verified" not in search_space:
        raise ValueError(
            "Search space is missing engineering_verified"
        )

    parts = search_space.get("parts")

    if not isinstance(parts, list) or not parts:
        raise ValueError(
            "Search space must contain a non-empty parts list"
        )

    baseline_ids = _baseline_part_ids(
        baseline_bom
    )
    search_part_ids = [
        part.get("part_id")
        for part in parts
    ]

    if any(not part_id for part_id in search_part_ids):
        raise ValueError(
            "Every search-space part must have part_id"
        )

    duplicates = _duplicates(search_part_ids)

    if duplicates:
        raise ValueError(
            f"Duplicate search-space part IDs: {duplicates}"
        )

    unknown_parts = (
        set(search_part_ids) - baseline_ids
    )

    if unknown_parts:
        raise ValueError(
            "Search-space part IDs not found in baseline BOM: "
            f"{sorted(unknown_parts)}"
        )

    variable_count = 0

    for part in parts:
        for field, records, label in [
            (
                "material_choices",
                registry.materials,
                "material",
            ),
            (
                "process_choices",
                registry.processes,
                "process",
            ),
        ]:
            choices = part.get(field)

            if choices is None:
                continue

            if (
                not isinstance(choices, list)
                or not choices
            ):
                raise ValueError(
                    f"{field} must be a non-empty array"
                )

            duplicates = _duplicates(choices)

            if duplicates:
                raise ValueError(
                    f"Duplicate {field}: {duplicates}"
                )

            unknown = [
                choice
                for choice in choices
                if choice not in records
            ]

            if unknown:
                raise ValueError(
                    f"Unknown {label} IDs in {field}: "
                    f"{unknown}"
                )

            variable_count += 1

        geometry_variables = part.get(
            "geometry_variables",
            {},
        )

        if not isinstance(
            geometry_variables,
            dict,
        ):
            raise ValueError(
                "geometry_variables must be an object"
            )

        for field_name, spec in geometry_variables.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Geometry variable {field_name} must be an object"
                )

            if spec.get("type") not in VARIABLE_TYPES:
                raise ValueError(
                    f"Unknown variable type for {field_name}: "
                    f"{spec.get('type')}"
                )

            if "min" not in spec or "max" not in spec:
                raise ValueError(
                    f"Geometry variable {field_name} requires min and max"
                )

            if float(spec["min"]) > float(spec["max"]):
                raise ValueError(
                    f"Geometry variable {field_name} has min > max"
                )

            variable_count += 1

        fastener_choices = part.get(
            "fastener_choices",
            {},
        )

        if not isinstance(fastener_choices, dict):
            raise ValueError(
                "fastener_choices must be an object"
            )

        for field_name, choices in fastener_choices.items():
            if (
                not isinstance(choices, list)
                or not choices
            ):
                raise ValueError(
                    f"fastener_choices.{field_name} must be non-empty"
                )

            for choice in choices:
                fastener_id = choice.get(
                    "fastener_id"
                )

                if fastener_id not in registry.fasteners:
                    raise ValueError(
                        "Unknown fastener ID in "
                        f"fastener_choices.{field_name}: "
                        f"{fastener_id}"
                    )

                quantity = choice.get(
                    "quantity"
                )

                if (
                    not isinstance(quantity, int)
                    or quantity < 0
                ):
                    raise ValueError(
                        "Fastener choice quantity must be "
                        "a non-negative integer"
                    )

            variable_count += 1

    return {
        "search_space_id": search_space[
            "search_space_id"
        ],
        "engineering_verified": search_space[
            "engineering_verified"
        ],
        "part_count": len(parts),
        "variable_count": variable_count,
        "registry_validation_meaning": (
            "Identifier existence only; not engineering "
            "compatibility."
        ),
        "active_decision_variable_types": [
            variable_type
            for variable_type in [
                "material"
                if any(
                    "material_choices" in part
                    for part in parts
                )
                else None,
                "process"
                if any(
                    "process_choices" in part
                    for part in parts
                )
                else None,
                "geometry"
                if any(
                    part.get("geometry_variables")
                    for part in parts
                )
                else None,
                "fastener"
                if any(
                    part.get("fastener_choices")
                    for part in parts
                )
                else None,
            ]
            if variable_type
        ],
    }


def part_space_by_id(
    search_space: dict,
) -> dict:
    return {
        part["part_id"]: part
        for part in search_space["parts"]
    }
