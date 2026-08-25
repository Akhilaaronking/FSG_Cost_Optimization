from src.data.registry import DataRegistry


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
    }


def part_space_by_id(
    search_space: dict,
) -> dict:
    return {
        part["part_id"]: part
        for part in search_space["parts"]
    }
