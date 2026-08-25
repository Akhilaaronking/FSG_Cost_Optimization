from src.optimization.hypervolume import (
    hypervolume_2d,
)
from src.optimization.pareto import (
    dominates,
    non_dominated,
)


def main():
    candidates = [
        {
            "candidate_id": "A",
            "objective_vector": [1.0, 4.0],
        },
        {
            "candidate_id": "B",
            "objective_vector": [2.0, 2.0],
        },
        {
            "candidate_id": "C",
            "objective_vector": [4.0, 1.0],
        },
        {
            "candidate_id": "D",
            "objective_vector": [4.0, 4.0],
        },
    ]

    archive = non_dominated(candidates)
    hv = hypervolume_2d(
        archive,
        [5.0, 5.0],
    )

    print("=" * 70)
    print("A7 — PARETO ENGINE VALIDATION")
    print("=" * 70)
    print(
        "Archive IDs:",
        [item["candidate_id"] for item in archive],
    )
    print("Hypervolume:", hv)

    if (
        dominates([1.0, 1.0], [1.0, 2.0])
        and [item["candidate_id"] for item in archive]
        == ["A", "B", "C"]
        and hv > 0
    ):
        print("\nA7 PARETO ENGINE VALIDATION: PASS")
    else:
        print("\nA7 PARETO ENGINE VALIDATION: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
