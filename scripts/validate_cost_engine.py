import json
from pathlib import Path

from src.cost_engine.cost_calculator import (
    calculate_bom_cost,
    calculate_part_cost,
)
from src.data.registry import DataRegistry


BOM_PATH = Path(
    "data/benchmark/pilot_10_parts_ground_truth.json"
)

KNOWN_GROUND_TRUTH_ISSUES = set()


def main():
    with BOM_PATH.open("r", encoding="utf-8") as file:
        bom = json.load(file)

    registry = DataRegistry()

    matches = 0
    known_issues = 0
    unexpected_mismatches = 0

    print("=" * 90)
    print("A4 — DETERMINISTIC COST ENGINE VALIDATION")
    print("=" * 90)

    for part in bom["parts"]:
        result = calculate_part_cost(
            part,
            registry=registry,
        )

        calculated = result["total_cost_eur"]

        expected = float(
            part["manual_calculation"][
                "total_cost_eur"
            ]
        )

        difference = round(
            calculated - expected,
            2,
        )

        if abs(difference) < 0.01:
            status = "PASS"
            matches += 1

        elif part["part_id"] in KNOWN_GROUND_TRUTH_ISSUES:
            status = "KNOWN DATA ISSUE"
            known_issues += 1

        else:
            status = "FAIL"
            unexpected_mismatches += 1

        print(
            f"{part['part_id']:<12} "
            f"calculated=€{calculated:>7.2f}   "
            f"expected=€{expected:>7.2f}   "
            f"{status}"
        )

    result = calculate_bom_cost(
        bom,
        registry=registry,
    )

    expected_total = sum(
        float(
            part["manual_calculation"][
                "total_cost_eur"
            ]
        )
        for part in bom["parts"]
    )

    print("-" * 90)
    print(
        f"Calculated BOM total : "
        f"€{result['total_cost_eur']:.2f}"
    )
    print(
        f"Ground-truth total   : "
        f"€{expected_total:.2f}"
    )
    print()
    print(f"Exact matches        : {matches}")
    print(f"Known data issues    : {known_issues}")
    print(
        f"Unexpected mismatches: "
        f"{unexpected_mismatches}"
    )

    if unexpected_mismatches == 0:
        print("\nA4 SOFTWARE VALIDATION: PASS")
    else:
        print("\nA4 SOFTWARE VALIDATION: FAIL")
        raise SystemExit(1)

    if known_issues:
        print(
            "A4 BENCHMARK ALIGNMENT: "
            "PENDING BENCHMARK-DATA CORRECTION"
        )
    else:
        print(
            "A4 BENCHMARK ALIGNMENT: PASS"
        )


if __name__ == "__main__":
    main()
