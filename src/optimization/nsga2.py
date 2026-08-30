import math
import random

from src.optimization.candidate import (
    OptimizationCandidate,
    design_fingerprint,
)
from src.optimization.hypervolume import (
    hypervolume_2d,
)
from src.optimization.operators import (
    crossover_candidates,
    initialize_candidate,
    mutate_candidate,
)
from src.optimization.pareto import (
    dominates,
    non_dominated,
)


def _constraints(
    candidate: dict,
) -> dict:
    return candidate.get(
        "constraints",
        {},
    )


def _feasible(
    candidate: dict,
):
    return _constraints(candidate).get(
        "feasible"
    )


def _violation_count(
    candidate: dict,
) -> float:
    value = _constraints(candidate).get(
        "violation_count"
    )

    if value is None:
        return math.inf

    return float(value)


def constrained_dominates(
    candidate_a: dict,
    candidate_b: dict,
) -> bool:
    feasible_a = _feasible(candidate_a)
    feasible_b = _feasible(candidate_b)

    if feasible_a is True and feasible_b is not True:
        return True

    if feasible_a is not True and feasible_b is True:
        return False

    if feasible_a is True and feasible_b is True:
        return dominates(
            candidate_a["objective_vector"],
            candidate_b["objective_vector"],
        )

    violations_a = _violation_count(
        candidate_a
    )
    violations_b = _violation_count(
        candidate_b
    )

    if violations_a != violations_b:
        return violations_a < violations_b

    if feasible_a is False and feasible_b is None:
        return True

    if feasible_a is None and feasible_b is False:
        return False

    return dominates(
        candidate_a["objective_vector"],
        candidate_b["objective_vector"],
    )


def fast_non_dominated_sort(
    candidates: list[dict],
    constrained: bool = False,
) -> list[list[dict]]:
    domination_sets = {}
    domination_counts = {}
    fronts = [[]]

    for candidate in candidates:
        candidate_id = candidate[
            "candidate_id"
        ]
        domination_sets[candidate_id] = []
        domination_counts[candidate_id] = 0

        for other in candidates:
            if other is candidate:
                continue

            if constrained:
                candidate_dominates = constrained_dominates(
                    candidate,
                    other,
                )
                other_dominates = constrained_dominates(
                    other,
                    candidate,
                )
            else:
                candidate_dominates = dominates(
                    candidate[
                        "objective_vector"
                    ],
                    other[
                        "objective_vector"
                    ],
                )
                other_dominates = dominates(
                    other[
                        "objective_vector"
                    ],
                    candidate[
                        "objective_vector"
                    ],
                )

            if candidate_dominates:
                domination_sets[
                    candidate_id
                ].append(other)

            elif other_dominates:
                domination_counts[
                    candidate_id
                ] += 1

        if domination_counts[candidate_id] == 0:
            candidate["rank"] = 0
            fronts[0].append(candidate)

    index = 0

    while fronts[index]:
        next_front = []

        for candidate in fronts[index]:
            candidate_id = candidate[
                "candidate_id"
            ]

            for dominated_candidate in domination_sets[
                candidate_id
            ]:
                dominated_id = dominated_candidate[
                    "candidate_id"
                ]
                domination_counts[dominated_id] -= 1

                if domination_counts[dominated_id] == 0:
                    dominated_candidate["rank"] = index + 1
                    next_front.append(
                        dominated_candidate
                    )

        index += 1
        fronts.append(next_front)

    return [
        sorted(
            front,
            key=lambda item: item[
                "candidate_id"
            ],
        )
        for front in fronts
        if front
    ]


def crowding_distance(
    front: list[dict],
) -> dict[str, float]:
    distances = {
        candidate["candidate_id"]: 0.0
        for candidate in front
    }

    if not front:
        return distances

    if len(front) <= 2:
        for candidate in front:
            distances[
                candidate["candidate_id"]
            ] = math.inf
            candidate["crowding_distance"] = math.inf
        return distances

    objective_count = len(
        front[0]["objective_vector"]
    )

    for objective_index in range(
        objective_count
    ):
        ordered = sorted(
            front,
            key=lambda item: (
                item["objective_vector"][
                    objective_index
                ],
                item["candidate_id"],
            ),
        )

        distances[
            ordered[0]["candidate_id"]
        ] = math.inf
        distances[
            ordered[-1]["candidate_id"]
        ] = math.inf

        minimum = ordered[0][
            "objective_vector"
        ][objective_index]
        maximum = ordered[-1][
            "objective_vector"
        ][objective_index]
        span = maximum - minimum

        if span == 0:
            continue

        for index in range(
            1,
            len(ordered) - 1,
        ):
            candidate_id = ordered[index][
                "candidate_id"
            ]

            if math.isinf(
                distances[candidate_id]
            ):
                continue

            previous_value = ordered[
                index - 1
            ]["objective_vector"][
                objective_index
            ]
            next_value = ordered[
                index + 1
            ]["objective_vector"][
                objective_index
            ]

            distances[candidate_id] += (
                next_value - previous_value
            ) / span

    for candidate in front:
        candidate["crowding_distance"] = distances[
            candidate["candidate_id"]
        ]

    return distances


def binary_tournament(
    candidate_a: dict,
    candidate_b: dict,
) -> dict:
    rank_a = candidate_a.get(
        "rank",
        math.inf,
    )
    rank_b = candidate_b.get(
        "rank",
        math.inf,
    )

    if rank_a != rank_b:
        return (
            candidate_a
            if rank_a < rank_b
            else candidate_b
        )

    distance_a = candidate_a.get(
        "crowding_distance",
        0.0,
    )
    distance_b = candidate_b.get(
        "crowding_distance",
        0.0,
    )

    if distance_a != distance_b:
        return (
            candidate_a
            if distance_a > distance_b
            else candidate_b
        )

    return min(
        candidate_a,
        candidate_b,
        key=lambda item: item[
            "candidate_id"
        ],
    )


def environmental_selection(
    candidates: list[dict],
    population_size: int,
    constrained: bool = False,
) -> list[dict]:
    selected = []
    fronts = fast_non_dominated_sort(
        candidates,
        constrained=constrained,
    )

    for front in fronts:
        crowding_distance(front)

        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue

        remaining = population_size - len(selected)
        selected.extend(
            sorted(
                front,
                key=lambda item: (
                    -item.get(
                        "crowding_distance",
                        0.0,
                    ),
                    item["candidate_id"],
                ),
            )[:remaining]
        )
        break

    return selected


def _evaluation_record(
    candidate: OptimizationCandidate,
    evaluation: dict,
    fingerprint: str,
) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate,
        "generation": candidate.generation,
        "parent_ids": candidate.parent_ids,
        "objective_vector": evaluation[
            "objective_vector"
        ],
        "objectives": evaluation.get(
            "objectives",
            {},
        ),
        "constraints": evaluation.get(
            "constraints",
            {},
        ),
        "evaluation": evaluation,
        "fingerprint": fingerprint,
    }


def _evaluate_candidate(
    candidate: OptimizationCandidate,
    search_space: dict,
    evaluator,
    cache: dict,
    counters: dict,
) -> dict:
    fingerprint = design_fingerprint(
        candidate.bom,
        search_space=search_space,
    )

    if fingerprint in cache:
        counters["cache_hits"] += 1
        cached = cache[fingerprint].copy()
        cached["candidate_id"] = candidate.candidate_id
        cached["candidate"] = candidate
        cached["generation"] = candidate.generation
        cached["parent_ids"] = candidate.parent_ids
        return cached

    evaluation = evaluator(candidate.bom)
    counters["evaluation_count"] += 1
    record = _evaluation_record(
        candidate,
        evaluation,
        fingerprint,
    )
    cache[fingerprint] = record.copy()
    return record


def _archive(
    evaluated: list[dict],
) -> list[dict]:
    established = [
        candidate
        for candidate in evaluated
        if _feasible(candidate) is not False
    ]

    by_fingerprint = {}

    for candidate in sorted(
        established,
        key=lambda item: item["candidate_id"],
    ):
        key = candidate.get(
            "fingerprint",
            candidate["candidate_id"],
        )
        by_fingerprint.setdefault(
            key,
            candidate,
        )

    return non_dominated(
        list(by_fingerprint.values())
    )


def _record_history(
    history: list[dict],
    generation: int,
    evaluation_count: int,
    archive: list[dict],
    reference_point: list[float],
):
    history.append({
        "generation": generation,
        "evaluation_count": evaluation_count,
        "archive_size": len(archive),
        "hypervolume": hypervolume_2d(
            archive,
            reference_point,
        ),
    })


def nsga2_optimize(
    baseline_bom: dict,
    search_space: dict,
    evaluator,
    population_size: int,
    generations: int,
    seed: int,
    evaluation_budget: int | None = None,
    mutation_rate: float = 0.2,
    reference_point: list[float] | None = None,
) -> dict:
    if population_size < 1:
        raise ValueError(
            "population_size must be positive"
        )

    if generations < 0:
        raise ValueError(
            "generations must be non-negative"
        )

    if (
        evaluation_budget is not None
        and evaluation_budget < 1
    ):
        raise ValueError(
            "evaluation_budget must be positive"
        )

    generator = random.Random(seed)
    cache = {}
    counters = {
        "evaluation_count": 0,
        "cache_hits": 0,
    }
    history = []
    population = []
    serial = 0
    termination_reason = "GENERATION_LIMIT"

    def budget_available():
        return (
            evaluation_budget is None
            or counters["evaluation_count"]
            < evaluation_budget
        )

    while (
        len(population) < population_size
        and budget_available()
    ):
        candidate = initialize_candidate(
            baseline_bom,
            search_space,
            candidate_id=f"cand_{serial:05d}",
            rng=generator,
            generation=0,
        )
        serial += 1
        population.append(
            _evaluate_candidate(
                candidate,
                search_space,
                evaluator,
                cache,
                counters,
            )
        )

    if len(population) < population_size:
        termination_reason = "EVALUATION_BUDGET"

    fronts = fast_non_dominated_sort(
        population,
        constrained=True,
    )
    for front in fronts:
        crowding_distance(front)

    archive = _archive(population)

    if reference_point is None:
        maxima = [
            max(
                candidate["objective_vector"][index]
                for candidate in population
            )
            for index in range(2)
        ]
        reference_point = [
            maxima[0] * 1.2
            if maxima[0] > 0
            else 1.0,
            maxima[1] * 1.2
            if maxima[1] > 0
            else 1.0,
        ]

    _record_history(
        history,
        0,
        counters["evaluation_count"],
        archive,
        reference_point,
    )

    completed = 0

    for generation in range(
        1,
        generations + 1,
    ):
        if not budget_available():
            termination_reason = "EVALUATION_BUDGET"
            break

        offspring = []

        while (
            len(offspring) < population_size
            and budget_available()
        ):
            parent_a = generator.choice(
                population
            )
            parent_b = generator.choice(
                population
            )
            winner_a = binary_tournament(
                parent_a,
                parent_b,
            )["candidate"]

            parent_c = generator.choice(
                population
            )
            parent_d = generator.choice(
                population
            )
            winner_b = binary_tournament(
                parent_c,
                parent_d,
            )["candidate"]

            child = crossover_candidates(
                winner_a,
                winner_b,
                search_space,
                candidate_id=f"cand_{serial:05d}",
                rng=generator,
            )
            serial += 1
            child = mutate_candidate(
                child,
                search_space,
                mutation_rate=mutation_rate,
                rng=generator,
                candidate_id=f"cand_{serial:05d}",
            )
            serial += 1

            offspring.append(
                _evaluate_candidate(
                    child,
                    search_space,
                    evaluator,
                    cache,
                    counters,
                )
            )

        if len(offspring) < population_size:
            termination_reason = "EVALUATION_BUDGET"

        population = environmental_selection(
            population + offspring,
            population_size,
            constrained=True,
        )
        archive = _archive(
            population + archive
        )
        completed = generation

        _record_history(
            history,
            generation,
            counters["evaluation_count"],
            archive,
            reference_point,
        )

        if termination_reason == "EVALUATION_BUDGET":
            break

    return {
        "seed": seed,
        "population_size": population_size,
        "generations_completed": completed,
        "evaluation_count": counters[
            "evaluation_count"
        ],
        "cache_hits": counters["cache_hits"],
        "termination_reason": termination_reason,
        "final_population": sorted(
            population,
            key=lambda item: item[
                "candidate_id"
            ],
        ),
        "pareto_archive": archive,
        "history": history,
        "reference_point": reference_point,
    }
