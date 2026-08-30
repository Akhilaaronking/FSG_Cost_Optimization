import json
from pathlib import Path

from src.data.registry import DataRegistry
from src.optimization.search_space import (
    load_verified_real_search_space,
)


BOM_PATH = Path(
    "data/benchmark/pilot_10_parts_ground_truth.json"
)
REAL_SEARCH_SPACE_PATH = Path(
    "data/benchmark/real_search_space.json"
)


def _load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _parts_by_id(bom: dict) -> dict:
    return {
        part["part_id"]: part
        for part in bom["parts"]
    }


def _real_parts_by_id(real: dict) -> dict:
    return {
        part["part_id"]: part
        for part in real["parts"]
    }


def _assert(
    condition: bool,
    message: str,
):
    if not condition:
        raise ValueError(message)


def main():
    bom = _load_json(BOM_PATH)
    real = _load_json(REAL_SEARCH_SPACE_PATH)
    registry = DataRegistry()

    print("=" * 80)
    print("A8 REAL SEARCH SPACE VALIDATION")
    print("=" * 80)

    search_space = load_verified_real_search_space(
        REAL_SEARCH_SPACE_PATH,
        BOM_PATH,
        registry=registry,
    )

    baseline_parts = _parts_by_id(bom)
    real_parts = _real_parts_by_id(real)

    _assert(
        len(search_space["parts"]) == 10,
        "Expected all 10 approved parts to be represented",
    )

    for part_space in search_space["parts"]:
        part_id = part_space["part_id"]
        baseline = baseline_parts[part_id]
        real_part = real_parts[part_id]

        _assert(
            baseline["material_id"]
            in part_space["material_choices"],
            f"{part_id}: baseline material missing from choices",
        )
        _assert(
            baseline["process_id"]
            in part_space["process_choices"],
            f"{part_id}: baseline process missing from choices",
        )

        inadmissible_materials = set(
            real_part.get(
                "inadmissible_materials",
                {},
            )
        )
        inadmissible_processes = set(
            real_part.get(
                "inadmissible_processes",
                {},
            )
        )

        _assert(
            not inadmissible_materials
            & set(part_space["material_choices"]),
            f"{part_id}: inadmissible material entered optimiser choices",
        )
        _assert(
            not inadmissible_processes
            & set(part_space["process_choices"]),
            f"{part_id}: inadmissible process entered optimiser choices",
        )
        _assert(
            "geometry_variables" not in part_space,
            f"{part_id}: geometry optimisation unexpectedly active",
        )
        _assert(
            "fastener_choices" not in part_space,
            f"{part_id}: fastener optimisation unexpectedly active",
        )

        print("-" * 80)
        print("part_id:", part_id)
        print(
            "baseline material:",
            part_space["baseline_material_id"],
        )
        print(
            "allowed materials:",
            part_space["material_choices"],
        )
        print(
            "baseline process:",
            part_space["baseline_process_id"],
        )
        print(
            "allowed processes:",
            part_space["process_choices"],
        )

    active_types = search_space["metadata"][
        "active_scope"
    ]
    _assert(
        active_types == ["material", "process"],
        f"Expected material/process active scope, got {active_types}",
    )

    print("-" * 80)
    print(
        "Real parts loaded:",
        len(search_space["parts"]),
    )
    print(
        "Active decision-variable types:",
        active_types,
    )
    print()
    print("A8 REAL SEARCH SPACE VALIDATION: PASS")


if __name__ == "__main__":
    main()
