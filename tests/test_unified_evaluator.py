import json
from pathlib import Path

import pytest

from src.evaluator import evaluate_bom


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "pilot_10_parts_ground_truth.json"
)


def load_bom():
    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_unified_evaluator_matches_b4_benchmark():
    result = evaluate_bom(load_bom())

    assert result["evaluator_version"] == "A6.1"
    assert result["part_count"] == 10

    assert result["objectives"]["cost_eur"] == 312.02
    assert result["objectives"]["mass_kg"] == pytest.approx(
        0.650706,
        abs=1e-5,
    )

    assert result["objective_vector"] == [
        result["objectives"]["cost_eur"],
        result["objectives"]["mass_kg"],
    ]


def test_default_constraints_do_not_claim_feasibility():
    result = evaluate_bom(load_bom())

    constraints = result["constraints"]

    assert constraints["status"] == "NOT_EVALUATED"
    assert constraints["feasible"] is None
    assert constraints["violation_count"] is None
    assert "not claimed" in constraints["reason"]


def test_provenance_quality_gate_is_separate():
    result = evaluate_bom(load_bom())

    gate = result["quality_gates"][
        "DERIVED_QG_001"
    ]

    assert gate["status"] == "PASS"
    assert gate["passed"] is True
    assert gate["checked_records"] == 31
    assert gate["fsg_compliance_claim"] is False
    assert "Traceability only" in gate["meaning"]


def test_trace_exposes_cost_and_rule_counts():
    result = evaluate_bom(load_bom())

    trace = result["trace"]

    assert len(trace["per_part_costs"]) == 10
    assert trace["per_part_costs"][0]["part_id"] == "PILOT_001"
    assert trace["provenance_passed"] is True
    assert trace["objective_vector_order"] == [
        "cost_eur",
        "mass_kg",
    ]

    assert trace["rule_counts"]["optimizer_rules"] == 7
    assert (
        trace["rule_counts"][
            "special_deterministic_rules"
        ]
        == 1
    )
    assert trace["rule_counts"]["review_rules"] == 9


def test_constraint_evaluation_uses_routed_rules():
    bom = load_bom()

    bom.update({
        "system_assignment": "Suspension System",
        "unit_system": "kg",
        "bought_or_made": "made",
        "part_breakdown_fields": [
            "materials",
            "processes",
            "fasteners",
            "tooling",
            "overhead",
        ],
        "tool_type": "fixture",
        "catalog_availability": True,
        "currency": "EUR",
    })

    result = evaluate_bom(
        bom,
        evaluate_constraints=True,
    )

    constraints = result["constraints"]

    assert constraints["status"] == "EVALUATED"
    assert constraints["feasible"] is True
    assert constraints["violation_count"] == 0
    assert len(constraints["results"]) == 7
    assert (
        constraints["special_results"][0]["status"]
        == "NOT_EVALUATED_SPECIAL"
    )
