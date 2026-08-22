from src.constraint_engine.provenance_gate import (
    evaluate_cost_provenance,
)
from src.constraint_engine.rule_router import (
    route_b5_rules,
)


def main():
    routed = route_b5_rules()
    provenance = evaluate_cost_provenance()

    print("=" * 70)
    print("A5 — B5 CONSTRAINT ROUTING VALIDATION")
    print("=" * 70)

    print(
        "Optimizer rules:",
        len(routed["optimizer_rules"]),
    )
    print(
        "Special deterministic rules:",
        len(
            routed[
                "special_deterministic_rules"
            ]
        ),
    )
    print(
        "Submission/compliance rules:",
        len(routed["compliance_rules"]),
    )
    print(
        "No-active-constraint rules:",
        len(
            routed["no_active_constraints"]
        ),
    )
    print(
        "Review rules:",
        len(routed["review_rules"]),
    )
    print(
        "Derived quality gates:",
        len(routed["quality_gates"]),
    )

    print("-" * 70)
    print(
        "Total B5 entries:",
        routed["total_entries"],
    )
    print(
        "DERIVED_QG_001:",
        provenance["status"],
    )
    print(
        "Provenance records checked:",
        provenance["checked_records"],
    )
    print(
        "FSG compliance claimed by gate:",
        provenance[
            "fsg_compliance_claim"
        ],
    )

    if (
        routed["total_entries"] == 27
        and provenance["passed"]
    ):
        print("\nA5 ROUTING VALIDATION: PASS")
    else:
        print("\nA5 ROUTING VALIDATION: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
