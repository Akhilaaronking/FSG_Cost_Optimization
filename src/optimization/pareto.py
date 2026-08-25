def dominates(
    objective_a: list[float],
    objective_b: list[float],
) -> bool:
    if len(objective_a) != len(objective_b):
        raise ValueError(
            "Objective vectors must have equal length"
        )

    return (
        all(
            a <= b
            for a, b in zip(
                objective_a,
                objective_b,
            )
        )
        and any(
            a < b
            for a, b in zip(
                objective_a,
                objective_b,
            )
        )
    )


def non_dominated(
    candidates: list[dict],
) -> list[dict]:
    archive = []

    for candidate in candidates:
        objective = candidate[
            "objective_vector"
        ]

        if any(
            dominates(
                other["objective_vector"],
                objective,
            )
            for other in candidates
            if other is not candidate
        ):
            continue

        archive.append(candidate)

    return sorted(
        archive,
        key=lambda item: item.get(
            "candidate_id",
            "",
        ),
    )


def update_archive(
    archive: list[dict],
    candidates: list[dict],
) -> list[dict]:
    by_id = {
        candidate.get("candidate_id"): candidate
        for candidate in archive + candidates
    }

    return non_dominated(
        list(by_id.values())
    )
