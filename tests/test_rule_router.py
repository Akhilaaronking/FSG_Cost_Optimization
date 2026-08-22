from src.constraint_engine.rule_router import (
    route_b5_rules,
)


def test_all_27_entries_are_accounted_for():
    routed = route_b5_rules()

    total = (
        len(routed["optimizer_rules"])
        + len(
            routed[
                "special_deterministic_rules"
            ]
        )
        + len(routed["compliance_rules"])
        + len(
            routed["no_active_constraints"]
        )
        + len(routed["review_rules"])
        + len(routed["quality_gates"])
    )

    assert total == 27
    assert routed["total_entries"] == 27


def test_optimizer_receives_only_direct_rules():
    routed = route_b5_rules()

    assert len(
        routed["optimizer_rules"]
    ) == 7

    ids = {
        rule["rule_id"]
        for rule in routed["optimizer_rules"]
    }

    assert "S_3.5.12" in ids
    assert "S_3.2.1" not in ids
    assert "S_3.3.1" not in ids


def test_submission_rules_are_separate():
    routed = route_b5_rules()

    assert len(
        routed["compliance_rules"]
    ) == 8


def test_implication_rule_is_special():
    routed = route_b5_rules()

    rules = routed[
        "special_deterministic_rules"
    ]

    assert len(rules) == 1
    assert rules[0]["rule_id"] == "S_3.4.6"


def test_no_maximum_cost_not_used_as_constraint():
    routed = route_b5_rules()

    rules = routed[
        "no_active_constraints"
    ]

    assert len(rules) == 1
    assert rules[0]["rule_id"] == "S_3.5.13"


def test_review_rules_never_enter_optimizer():
    routed = route_b5_rules()

    optimizer_ids = {
        rule["rule_id"]
        for rule in routed["optimizer_rules"]
    }

    review_ids = {
        rule["rule_id"]
        for rule in routed["review_rules"]
    }

    assert optimizer_ids.isdisjoint(
        review_ids
    )

    assert len(review_ids) == 9
