from copy import deepcopy
from math import isclose


def _validate_vector(
    vector,
    name: str,
) -> list[float]:
    if (
        not isinstance(vector, (list, tuple))
        or len(vector) != 2
    ):
        raise ValueError(
            f"{name} must contain [cost_eur, mass_kg]"
        )

    return [
        float(vector[0]),
        float(vector[1]),
    ]


def normalized_objectives(
    objective_vector,
    baseline_vector,
) -> list[float]:
    objective = _validate_vector(
        objective_vector,
        "objective_vector",
    )
    baseline = _validate_vector(
        baseline_vector,
        "baseline_vector",
    )

    if baseline[0] <= 0 or baseline[1] <= 0:
        raise ValueError(
            "Baseline objectives must be positive"
        )

    return [
        objective[0] / baseline[0],
        objective[1] / baseline[1],
    ]


def _validate_weights(
    weight_cost: float,
    weight_mass: float,
):
    if weight_cost < 0 or weight_mass < 0:
        raise ValueError(
            "Weights must be non-negative"
        )

    if not isclose(
        weight_cost + weight_mass,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Weights must sum to one"
        )


def weighted_score(
    objective_vector,
    baseline_vector,
    weight_cost: float,
    weight_mass: float,
) -> float:
    _validate_weights(
        weight_cost,
        weight_mass,
    )

    cost_norm, mass_norm = normalized_objectives(
        objective_vector,
        baseline_vector,
    )

    return (
        weight_cost * cost_norm
        + weight_mass * mass_norm
    )


def rank_weighted_candidates(
    candidates: list[dict],
    baseline_vector,
    weight_cost: float,
    weight_mass: float,
) -> list[dict]:
    _validate_weights(
        weight_cost,
        weight_mass,
    )

    ranked = []

    for index, candidate in enumerate(candidates):
        item = deepcopy(candidate)
        item["weighted_score"] = weighted_score(
            item["objective_vector"],
            baseline_vector,
            weight_cost,
            weight_mass,
        )
        item["_input_order"] = index
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            item["weighted_score"],
            item.get("candidate_id", ""),
            item["_input_order"],
        )
    )

    for item in ranked:
        del item["_input_order"]

    return ranked
