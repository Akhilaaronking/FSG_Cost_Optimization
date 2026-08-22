from src.constraint_engine.b5_adapter import load_b5_rules


# Rules that can directly constrain candidate BOM/design records.
OPTIMIZER_RULE_IDS = {
    "S_3.4.7",   # system assignment
    "S_3.4.9",   # metric/unit system
    "S_3.5.4",   # bought or made
    "S_3.5.5",   # part cost breakdown fields
    "S_3.5.7",   # excluded tooling types
    "S_3.5.10",  # bought-part catalog availability
    "S_3.5.12",  # EUR currency
}


# Deterministic, but requires custom implication logic rather
# than the generic scalar operator evaluator.
SPECIAL_DETERMINISTIC_RULE_IDS = {
    "S_3.4.6",
}


# Deterministic FSG rules concerning submission package,
# documentation, completeness, ordering or process reporting.
COMPLIANCE_RULE_IDS = {
    "S_3.2.1",
    "S_3.3.1",
    "S_3.3.6",
    "S_3.4.1",
    "S_3.5.2",
    "S_3.5.14",
    "S_3.7.2",
    "S_3.8.3",
}


# FSG explicitly states no maximum cost constraint here.
NO_ACTIVE_CONSTRAINT_IDS = {
    "S_3.5.13",
}


def route_b5_rules() -> dict:
    data = load_b5_rules()

    deterministic = data["deterministic_rules"]

    by_id = {
        rule["rule_id"]: rule
        for rule in deterministic
    }

    expected_ids = (
        OPTIMIZER_RULE_IDS
        | SPECIAL_DETERMINISTIC_RULE_IDS
        | COMPLIANCE_RULE_IDS
        | NO_ACTIVE_CONSTRAINT_IDS
    )

    actual_ids = set(by_id)

    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        unexpected = actual_ids - expected_ids

        raise ValueError(
            "B5 deterministic routing mismatch. "
            f"Missing={sorted(missing)}, "
            f"Unexpected={sorted(unexpected)}"
        )

    optimizer_rules = [
        by_id[rule_id]
        for rule_id in sorted(
            OPTIMIZER_RULE_IDS
        )
    ]

    special_rules = [
        by_id[rule_id]
        for rule_id in sorted(
            SPECIAL_DETERMINISTIC_RULE_IDS
        )
    ]

    compliance_rules = [
        by_id[rule_id]
        for rule_id in sorted(
            COMPLIANCE_RULE_IDS
        )
    ]

    no_active_constraints = [
        by_id[rule_id]
        for rule_id in sorted(
            NO_ACTIVE_CONSTRAINT_IDS
        )
    ]

    return {
        "optimizer_rules": optimizer_rules,
        "special_deterministic_rules":
            special_rules,
        "compliance_rules":
            compliance_rules,
        "no_active_constraints":
            no_active_constraints,
        "review_rules":
            data["review_rules"],
        "quality_gates":
            data["quality_gates"],
        "total_entries":
            data["total_entries"],
    }
