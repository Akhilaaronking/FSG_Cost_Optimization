import json
from pathlib import Path

import pytest

from src.evaluator import evaluate_bom


BOM_PATH = Path(
    "data/benchmark/pilot_10_parts_ground_truth.json"
)


def main():
    with BOM_PATH.open("r", encoding="utf-8") as file:
        bom = json.load(file)

    result = evaluate_bom(bom)

    print("=" * 70)
    print("A6 — UNIFIED EVALUATOR VALIDATION")
    print("=" * 70)
    print(
        "Cost:",
        result["objectives"]["cost_eur"],
    )
    print(
        "Mass:",
        result["objectives"]["mass_kg"],
    )
    print(
        "Constraints:",
        result["constraints"]["status"],
    )
    print(
        "DERIVED_QG_001:",
        result["quality_gates"][
            "DERIVED_QG_001"
        ]["status"],
    )

    checks = [
        result["objectives"]["cost_eur"] == 312.02,
        result["objectives"]["mass_kg"]
        == pytest.approx(0.650706, abs=1e-5),
        result["objective_vector"][0]
        == result["objectives"]["cost_eur"],
        result["objective_vector"][1]
        == result["objectives"]["mass_kg"],
        result["constraints"]["feasible"] is None,
        result["quality_gates"][
            "DERIVED_QG_001"
        ]["fsg_compliance_claim"]
        is False,
    ]

    if all(checks):
        print("\nA6 UNIFIED EVALUATOR VALIDATION: PASS")
    else:
        print("\nA6 UNIFIED EVALUATOR VALIDATION: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
