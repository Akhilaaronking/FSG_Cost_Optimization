from src.constraint_engine.b5_adapter import (
    load_b5_rules,
)
from src.constraint_engine.evaluator import (
    evaluate_rule,
    evaluate_operator,
)


def test_b5_dataset_routes_correctly():
    data = load_b5_rules()

    assert data["total_entries"] == 27
    assert len(
        data["deterministic_rules"]
    ) == 17
    assert len(
        data["review_rules"]
    ) == 9
    assert len(
        data["quality_gates"]
    ) == 1


def test_currency_rule_is_normalized():
    data = load_b5_rules()

    rule = next(
        rule
        for rule
        in data["deterministic_rules"]
        if rule["rule_id"] == "S_3.5.12"
    )

    assert rule["operator"] == "eq"
    assert rule["limit_value"] == "EUR"
    assert rule["deterministic"] is True


def test_includes_rule_is_normalized():
    data = load_b5_rules()

    rule = next(
        rule
        for rule
        in data["deterministic_rules"]
        if rule["rule_id"] == "S_3.5.2"
    )

    assert rule["operator"] == "includes"
    assert (
        rule["limit_value"]
        == "Engine and Tractive System"
    )


def test_no_maximum_rule_is_no_constraint():
    data = load_b5_rules()

    rule = next(
        rule
        for rule
        in data["deterministic_rules"]
        if rule["rule_id"] == "S_3.5.13"
    )

    assert rule["operator"] == "no_constraint"


def test_review_rules_are_not_deterministic():
    data = load_b5_rules()

    assert all(
        rule["deterministic"] is False
        for rule in data["review_rules"]
    )


def test_quality_gate_is_separate_from_fsg():
    data = load_b5_rules()

    gate = data["quality_gates"][0]

    assert gate["rule_id"] == "DERIVED_QG_001"
    assert (
        gate["rule_category"]
        == "derived_quality_gate"
    )

    assert gate.get("source_id", "") == ""


def test_includes_operator():
    actual = [
        "Brake System",
        "Engine and Tractive System",
    ]

    assert evaluate_operator(
        actual,
        "includes",
        "Engine and Tractive System",
    ) is True


def test_non_deterministic_rule_is_skipped():
    data = load_b5_rules()

    rule = next(
        rule
        for rule
        in data["review_rules"]
        if rule["rule_id"] == "S_3.5.11"
    )

    result = evaluate_rule(
        {},
        rule,
    )

    assert result["status"] == "SKIPPED_REVIEW"
    assert result["passed"] is None
