from src.optimization.pareto import (
    non_dominated,
)


def hypervolume_2d(
    candidates: list[dict],
    reference_point: list[float],
) -> float:
    if len(reference_point) != 2:
        raise ValueError(
            "2D hypervolume requires a two-value reference point"
        )

    ref_x, ref_y = reference_point

    points = [
        {
            "candidate_id": candidate.get(
                "candidate_id",
                "",
            ),
            "objective_vector": candidate[
                "objective_vector"
            ],
        }
        for candidate in candidates
        if (
            len(candidate["objective_vector"]) == 2
            and candidate["objective_vector"][0] <= ref_x
            and candidate["objective_vector"][1] <= ref_y
        )
    ]

    front = sorted(
        non_dominated(points),
        key=lambda item: (
            item["objective_vector"][0],
            item["objective_vector"][1],
            item.get("candidate_id", ""),
        ),
    )

    hv = 0.0
    previous_y = ref_y

    for candidate in front:
        x, y = candidate["objective_vector"]

        width = max(0.0, ref_x - x)
        height = max(0.0, previous_y - y)
        hv += width * height
        previous_y = min(previous_y, y)

    return hv
