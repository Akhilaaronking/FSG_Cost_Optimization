import math

from src.optimization.nsga2 import (
    binary_tournament,
    constrained_dominates,
    crowding_distance,
    environmental_selection,
    fast_non_dominated_sort,
    nsga2_optimize,
)
from src.optimization.pareto import (
    non_dominated,
)


def record(
    candidate_id,
    objective_vector,
    feasible=True,
    violation_count=0,
):
    return {
        "candidate_id": candidate_id,
        "objective_vector": objective_vector,
        "constraints": {
            "feasible": feasible,
            "violation_count": violation_count,
        },
    }


def baseline_bom(x=0.5):
    return {
        "parts": [
            {
                "part_id": "TEST_PART",
                "geometry": {"x": x},
            }
        ]
    }


def search_space(minimum=0.0, maximum=1.0):
    return {
        "search_space_id": "TEST_SPACE",
        "engineering_verified": False,
        "purpose": "UNIT_TEST_ONLY",
        "parts": [
            {
                "part_id": "TEST_PART",
                "geometry_variables": {
                    "x": {
                        "type": "continuous",
                        "min": minimum,
                        "max": maximum,
                    }
                },
            }
        ],
    }


def synthetic_evaluator(bom):
    x = bom["parts"][0]["geometry"]["x"]

    return {
        "objective_vector": [
            x * x,
            (1.0 - x) * (1.0 - x),
        ],
        "objectives": {
            "cost_eur": x * x,
            "mass_kg": (1.0 - x) * (1.0 - x),
        },
        "constraints": {
            "feasible": True,
            "violation_count": 0,
        },
    }


def archive_ids(result):
    return [
        candidate["candidate_id"]
        for candidate in result["pareto_archive"]
    ]


def test_non_dominated_sorting():
    candidates = [
        record("A", [1.0, 1.0]),
        record("B", [2.0, 2.0]),
    ]

    fronts = fast_non_dominated_sort(candidates)

    assert [
        item["candidate_id"]
        for item in fronts[0]
    ] == ["A"]


def test_multiple_fronts():
    candidates = [
        record("A", [1.0, 1.0]),
        record("B", [2.0, 2.0]),
        record("C", [3.0, 3.0]),
    ]

    fronts = fast_non_dominated_sort(candidates)

    assert len(fronts) == 3


def test_crowding_boundary_is_infinity():
    front = [
        record("A", [0.0, 1.0]),
        record("B", [0.5, 0.5]),
        record("C", [1.0, 0.0]),
    ]

    distances = crowding_distance(front)

    assert math.isinf(distances["A"])
    assert math.isinf(distances["C"])


def test_crowding_interior_values():
    front = [
        record("A", [0.0, 1.0]),
        record("B", [0.5, 0.5]),
        record("C", [1.0, 0.0]),
    ]

    distances = crowding_distance(front)

    assert distances["B"] == 2.0


def test_zero_objective_range_handled():
    front = [
        record("A", [1.0, 1.0]),
        record("B", [1.0, 1.0]),
        record("C", [1.0, 1.0]),
    ]

    distances = crowding_distance(front)

    assert distances["B"] == 0.0


def test_tournament_prefers_lower_rank():
    winner = binary_tournament(
        {"candidate_id": "A", "rank": 0},
        {"candidate_id": "B", "rank": 1},
    )

    assert winner["candidate_id"] == "A"


def test_tournament_prefers_larger_crowding_distance():
    winner = binary_tournament(
        {
            "candidate_id": "A",
            "rank": 0,
            "crowding_distance": 1.0,
        },
        {
            "candidate_id": "B",
            "rank": 0,
            "crowding_distance": 2.0,
        },
    )

    assert winner["candidate_id"] == "B"


def test_tournament_tie_breaks_by_candidate_id():
    winner = binary_tournament(
        {
            "candidate_id": "B",
            "rank": 0,
            "crowding_distance": 1.0,
        },
        {
            "candidate_id": "A",
            "rank": 0,
            "crowding_distance": 1.0,
        },
    )

    assert winner["candidate_id"] == "A"


def test_environmental_selection_respects_population_size():
    selected = environmental_selection(
        [
            record("A", [0.0, 1.0]),
            record("B", [0.5, 0.5]),
            record("C", [1.0, 0.0]),
        ],
        2,
    )

    assert len(selected) == 2


def test_same_seed_gives_identical_result():
    first = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        population_size=6,
        generations=3,
        seed=42,
    )
    second = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        population_size=6,
        generations=3,
        seed=42,
    )

    assert [
        item["objective_vector"]
        for item in first["final_population"]
    ] == [
        item["objective_vector"]
        for item in second["final_population"]
    ]
    assert archive_ids(first) == archive_ids(second)


def test_different_seeds_are_allowed_to_differ():
    first = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        6,
        2,
        seed=1,
    )
    second = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        6,
        2,
        seed=2,
    )

    assert [
        item["objective_vector"]
        for item in first["final_population"]
    ] != [
        item["objective_vector"]
        for item in second["final_population"]
    ]


def test_evaluation_budget_is_never_exceeded():
    result = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        population_size=6,
        generations=10,
        seed=3,
        evaluation_budget=8,
    )

    assert result["evaluation_count"] <= 8
    assert result["termination_reason"] == "EVALUATION_BUDGET"


def test_generation_limit_works():
    result = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        population_size=6,
        generations=2,
        seed=4,
    )

    assert result["generations_completed"] == 2
    assert result["termination_reason"] == "GENERATION_LIMIT"


def test_cache_avoids_duplicate_re_evaluation():
    result = nsga2_optimize(
        baseline_bom(0.25),
        search_space(0.25, 0.25),
        synthetic_evaluator,
        population_size=4,
        generations=2,
        seed=5,
        evaluation_budget=10,
    )

    assert result["evaluation_count"] == 1
    assert result["cache_hits"] > 0


def test_feasible_candidate_preferred_over_infeasible():
    assert constrained_dominates(
        record("A", [2.0, 2.0], True, 0),
        record("B", [1.0, 1.0], False, 1),
    )


def test_lower_violation_count_preferred_among_infeasible():
    assert constrained_dominates(
        record("A", [2.0, 2.0], False, 1),
        record("B", [1.0, 1.0], False, 2),
    )


def test_feasibility_none_is_not_confirmed_feasible():
    assert not constrained_dominates(
        record("A", [1.0, 1.0], None, None),
        record("B", [2.0, 2.0], True, 0),
    )


def test_final_archive_is_non_dominated():
    result = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        population_size=8,
        generations=3,
        seed=6,
    )

    assert result["pareto_archive"] == non_dominated(
        result["pareto_archive"]
    )


def test_history_records_generation_evaluation_count_and_hv():
    result = nsga2_optimize(
        baseline_bom(),
        search_space(),
        synthetic_evaluator,
        population_size=6,
        generations=2,
        seed=7,
    )

    assert result["history"]
    assert {
        "generation",
        "evaluation_count",
        "archive_size",
        "hypervolume",
    }.issubset(result["history"][0])
    assert result["history"][-1]["hypervolume"] >= 0.0
