from copy import deepcopy
import random

from src.optimization.candidate import (
    OptimizationCandidate,
)
from src.optimization.search_space import (
    part_space_by_id,
    validate_search_space,
)


def _rng(
    rng: random.Random | None = None,
    seed: int | None = None,
) -> random.Random:
    if rng is not None:
        return rng

    return random.Random(seed)


def _parts_by_id(
    bom: dict,
) -> dict:
    return {
        part["part_id"]: part
        for part in bom["parts"]
    }


def _normalise_fasteners(
    choices: list[dict],
) -> list[dict]:
    fasteners = []

    for choice in choices:
        fasteners.append({
            "fastener_id": choice[
                "fastener_id"
            ],
            "quantity": choice[
                "quantity"
            ],
        })

    return fasteners


def initialize_candidate(
    baseline_bom: dict,
    search_space: dict,
    candidate_id: str,
    rng: random.Random | None = None,
    seed: int | None = None,
    generation: int = 0,
) -> OptimizationCandidate:
    validate_search_space(
        search_space,
        baseline_bom,
    )

    generator = _rng(rng, seed)
    bom = deepcopy(baseline_bom)
    parts = _parts_by_id(bom)

    for part_space in search_space["parts"]:
        part = parts[part_space["part_id"]]

        if "material_choices" in part_space:
            part["material_id"] = generator.choice(
                part_space[
                    "material_choices"
                ]
            )

        if "process_choices" in part_space:
            part["process_id"] = generator.choice(
                part_space[
                    "process_choices"
                ]
            )

        for field_name, spec in part_space.get(
            "geometry_variables",
            {},
        ).items():
            part.setdefault(
                "geometry",
                {},
            )[field_name] = generator.uniform(
                float(spec["min"]),
                float(spec["max"]),
            )

        for field_name, choices in part_space.get(
            "fastener_choices",
            {},
        ).items():
            selected = generator.choice(
                choices
            )

            if field_name == "fasteners":
                part["fasteners"] = _normalise_fasteners(
                    selected
                    if isinstance(selected, list)
                    else [selected]
                )
            else:
                part[field_name] = deepcopy(
                    selected
                )

    return OptimizationCandidate(
        candidate_id=candidate_id,
        bom=bom,
        generation=generation,
    )


def mutate_candidate(
    parent: OptimizationCandidate,
    search_space: dict,
    mutation_rate: float = 0.2,
    rng: random.Random | None = None,
    seed: int | None = None,
    candidate_id: str | None = None,
) -> OptimizationCandidate:
    if mutation_rate < 0 or mutation_rate > 1:
        raise ValueError(
            "mutation_rate must be between 0 and 1"
        )

    generator = _rng(rng, seed)
    child = parent.copy_with(
        candidate_id=(
            candidate_id
            or f"{parent.candidate_id}_mut"
        ),
        generation=parent.generation + 1,
        parent_ids=(parent.candidate_id,),
    )

    parts = _parts_by_id(child.bom)

    for part_space in search_space["parts"]:
        part = parts[part_space["part_id"]]

        if (
            "material_choices" in part_space
            and generator.random() < mutation_rate
        ):
            choices = part_space[
                "material_choices"
            ]
            alternatives = [
                choice
                for choice in choices
                if choice != part.get("material_id")
            ]
            part["material_id"] = generator.choice(
                alternatives or choices
            )

        if (
            "process_choices" in part_space
            and generator.random() < mutation_rate
        ):
            choices = part_space[
                "process_choices"
            ]
            alternatives = [
                choice
                for choice in choices
                if choice != part.get("process_id")
            ]
            part["process_id"] = generator.choice(
                alternatives or choices
            )

        for field_name, spec in part_space.get(
            "geometry_variables",
            {},
        ).items():
            if generator.random() >= mutation_rate:
                continue

            part.setdefault(
                "geometry",
                {},
            )[field_name] = generator.uniform(
                float(spec["min"]),
                float(spec["max"]),
            )

    return child


def crossover_candidates(
    parent_a: OptimizationCandidate,
    parent_b: OptimizationCandidate,
    search_space: dict,
    candidate_id: str,
    rng: random.Random | None = None,
    seed: int | None = None,
) -> OptimizationCandidate:
    generator = _rng(rng, seed)
    child_bom = deepcopy(parent_a.bom)
    parts_child = _parts_by_id(child_bom)
    parts_a = _parts_by_id(parent_a.bom)
    parts_b = _parts_by_id(parent_b.bom)
    spaces = part_space_by_id(search_space)

    for part_id, part_space in spaces.items():
        child_part = parts_child[part_id]
        source_part = (
            parts_a[part_id]
            if generator.random() < 0.5
            else parts_b[part_id]
        )

        if "material_choices" in part_space:
            value = source_part["material_id"]
            if value not in part_space[
                "material_choices"
            ]:
                raise ValueError(
                    "Crossover produced material outside "
                    "the explicit search space"
                )
            child_part["material_id"] = value

        if "process_choices" in part_space:
            value = source_part["process_id"]
            if value not in part_space[
                "process_choices"
            ]:
                raise ValueError(
                    "Crossover produced process outside "
                    "the explicit search space"
                )
            child_part["process_id"] = value

        for field_name, spec in part_space.get(
            "geometry_variables",
            {},
        ).items():
            value = (
                parts_a[part_id]
                .get("geometry", {})
                .get(field_name)
                if generator.random() < 0.5
                else parts_b[part_id]
                .get("geometry", {})
                .get(field_name)
            )

            if (
                value < spec["min"]
                or value > spec["max"]
            ):
                raise ValueError(
                    "Crossover produced geometry outside "
                    "the explicit search-space bounds"
                )

            child_part.setdefault(
                "geometry",
                {},
            )[field_name] = value

    return OptimizationCandidate(
        candidate_id=candidate_id,
        bom=child_bom,
        generation=max(
            parent_a.generation,
            parent_b.generation,
        )
        + 1,
        parent_ids=(
            parent_a.candidate_id,
            parent_b.candidate_id,
        ),
    )
