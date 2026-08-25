from src.optimization.nsga2 import (
    nsga2_optimize,
)
from src.optimization.pareto import (
    non_dominated,
)


def baseline_bom():
    return {
        "parts": [
            {
                "part_id": "TEST_PART",
                "geometry": {"x": 0.5},
            }
        ]
    }


def search_space():
    return {
        "search_space_id": "TEST_SYNTHETIC_NSGA2",
        "engineering_verified": False,
        "purpose": "UNIT_TEST_ONLY",
        "parts": [
            {
                "part_id": "TEST_PART",
                "geometry_variables": {
                    "x": {
                        "type": "continuous",
                        "min": 0.0,
                        "max": 1.0,
                    }
                },
            }
        ],
    }


def evaluator(bom):
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


def main():
    print("=" * 70)
    print("A8 — NSGA-II ALGORITHM VALIDATION")
    print("=" * 70)
    print("SYNTHETIC ALGORITHM VALIDATION ONLY")
    print("NOT FORMULA STUDENT EXPERIMENTAL RESULTS")
    print("Synthetic benchmark: f1(x)=x^2, f2(x)=(1-x)^2")
    print("-" * 70)

    result = nsga2_optimize(
        baseline_bom(),
        search_space(),
        evaluator,
        population_size=20,
        generations=8,
        seed=42,
        evaluation_budget=180,
        reference_point=[1.2, 1.2],
    )
    replay = nsga2_optimize(
        baseline_bom(),
        search_space(),
        evaluator,
        population_size=20,
        generations=8,
        seed=42,
        evaluation_budget=180,
        reference_point=[1.2, 1.2],
    )

    archive = result["pareto_archive"]
    nondominated_ok = archive == non_dominated(
        archive
    )
    budget_ok = result["evaluation_count"] <= 180
    hv = result["history"][-1]["hypervolume"]
    hv_ok = hv >= 0.0
    replay_ok = [
        item["objective_vector"]
        for item in result["final_population"]
    ] == [
        item["objective_vector"]
        for item in replay["final_population"]
    ]

    rounded_tradeoffs = {
        (
            round(item["objective_vector"][0], 4),
            round(item["objective_vector"][1], 4),
        )
        for item in archive
    }
    tradeoff_ok = len(rounded_tradeoffs) > 1

    print(
        "Synthetic benchmark:",
        "PASS" if tradeoff_ok else "FAIL",
    )
    print(
        "Deterministic seed replay:",
        "PASS" if replay_ok else "FAIL",
    )
    print(
        "Evaluation budget:",
        "PASS" if budget_ok else "FAIL",
    )
    print(
        "Pareto archive:",
        "PASS" if nondominated_ok else "FAIL",
    )
    print(
        "Hypervolume tracking:",
        "PASS" if hv_ok else "FAIL",
    )
    print(
        "Evaluation count:",
        result["evaluation_count"],
    )
    print(
        "Cache hits:",
        result["cache_hits"],
    )
    print("Final hypervolume:", hv)

    if all([
        tradeoff_ok,
        replay_ok,
        budget_ok,
        nondominated_ok,
        hv_ok,
    ]):
        print("\nA8 NSGA-II CORE VALIDATION: PASS")
    else:
        print("\nA8 NSGA-II CORE VALIDATION: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
