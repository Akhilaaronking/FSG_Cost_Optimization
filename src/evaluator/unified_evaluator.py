from src.constraint_engine.provenance_gate import (
    evaluate_cost_provenance,
)
from src.constraint_engine.evaluator import (
    evaluate_rules,
)
from src.constraint_engine.rule_router import (
    route_b5_rules,
)
from src.cost_engine.cost_calculator import (
    calculate_bom_cost,
)
from src.data.registry import DataRegistry
from src.mass_engine.mass_calculator import (
    calculate_bom_mass,
)


EVALUATOR_VERSION = "A6.1"


def _rule_counts(routed: dict) -> dict:
    return {
        "optimizer_rules": len(
            routed["optimizer_rules"]
        ),
        "special_deterministic_rules": len(
            routed[
                "special_deterministic_rules"
            ]
        ),
        "compliance_rules": len(
            routed["compliance_rules"]
        ),
        "no_active_constraints": len(
            routed["no_active_constraints"]
        ),
        "review_rules": len(
            routed["review_rules"]
        ),
        "quality_gates": len(
            routed["quality_gates"]
        ),
        "total_entries": routed["total_entries"],
    }


def _not_evaluated_constraints(
    routed: dict,
) -> dict:
    counts = _rule_counts(routed)

    return {
        "status": "NOT_EVALUATED",
        "feasible": None,
        "violation_count": None,
        "reason": (
            "Constraint evaluation was not requested. "
            "The frozen B4 benchmark does not yet expose "
            "all canonical optimisation-constraint fields, "
            "so feasibility is intentionally not claimed."
        ),
        "available_optimizer_rules": counts[
            "optimizer_rules"
        ],
        "special_deterministic_rules": counts[
            "special_deterministic_rules"
        ],
        "review_rules": counts["review_rules"],
    }


def _evaluate_constraints(
    bom: dict,
    routed: dict,
) -> dict:
    optimizer_result = evaluate_rules(
        bom,
        routed["optimizer_rules"],
    )

    special_results = [
        {
            "rule_id": rule["rule_id"],
            "status": "NOT_EVALUATED_SPECIAL",
            "passed": None,
            "reason": (
                "Special deterministic implication "
                "logic is routed separately and is not "
                "implemented in the generic scalar "
                "constraint evaluator."
            ),
        }
        for rule in routed[
            "special_deterministic_rules"
        ]
    ]

    violation_count = optimizer_result[
        "violation_count"
    ]

    return {
        "status": "EVALUATED",
        "feasible": violation_count == 0,
        "violation_count": violation_count,
        "available_optimizer_rules": len(
            routed["optimizer_rules"]
        ),
        "special_deterministic_rules": len(
            routed[
                "special_deterministic_rules"
            ]
        ),
        "review_rules": len(
            routed["review_rules"]
        ),
        "results": optimizer_result["results"],
        "special_results": special_results,
    }


def evaluate_bom(
    bom: dict,
    registry: DataRegistry | None = None,
    evaluate_constraints: bool = False,
) -> dict:
    """
    Common deterministic evaluator.

    Objectives:
        1. minimise total cost
        2. minimise total mass

    Objective-vector order is always:
        [cost_eur, mass_kg]

    By default, constraints are not evaluated and
    feasibility is not claimed. This is required because
    the frozen B4 benchmark does not yet contain every
    canonical field required by the optimiser-facing
    deterministic FSG rules.
    """

    registry = registry or DataRegistry()

    parts = bom.get("parts")

    if not isinstance(parts, list) or not parts:
        raise ValueError(
            "BOM must contain a non-empty parts list"
        )

    total_mass_kg = calculate_bom_mass(
        bom,
        registry=registry,
    )

    cost_result = calculate_bom_cost(
        bom,
        registry=registry,
    )

    total_cost_eur = cost_result[
        "total_cost_eur"
    ]

    provenance = evaluate_cost_provenance(
        registry=registry,
    )

    routed = route_b5_rules()
    rule_counts = _rule_counts(routed)

    if evaluate_constraints:
        constraints = _evaluate_constraints(
            bom,
            routed,
        )
    else:
        constraints = _not_evaluated_constraints(
            routed,
        )

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "part_count": len(parts),

        "objectives": {
            "cost_eur": total_cost_eur,
            "mass_kg": total_mass_kg,
        },

        "objective_vector": [
            total_cost_eur,
            total_mass_kg,
        ],

        "quality_gates": {
            "DERIVED_QG_001": provenance,
        },

        "constraints": constraints,

        "trace": {
            "per_part_costs": cost_result["parts"],
            "cost_parts": cost_result["parts"],
            "provenance_passed": provenance["passed"],
            "provenance_status": provenance["status"],
            "rule_counts": rule_counts,
            "objective_sense": {
                "cost_eur": "minimize",
                "mass_kg": "minimize",
            },
            "objective_vector_order": [
                "cost_eur",
                "mass_kg",
            ],
        },
    }
