from typing import Any


def resolve_field(
    record: dict,
    field_path: str,
) -> Any:
    current = record

    for key in field_path.split("."):
        if not isinstance(current, dict):
            raise KeyError(field_path)

        if key not in current:
            raise KeyError(field_path)

        current = current[key]

    return current


def evaluate_operator(
    actual,
    operator: str,
    limit,
) -> bool:

    if operator == "eq":
        return actual == limit

    if operator == "neq":
        return actual != limit

    if operator == "lt":
        return actual < limit

    if operator == "lte":
        return actual <= limit

    if operator == "gt":
        return actual > limit

    if operator == "gte":
        return actual >= limit

    if operator == "in":
        return actual in limit

    if operator == "not_in":
        return actual not in limit

    if operator == "includes":
        if isinstance(limit, list):
            return all(
                item in actual
                for item in limit
            )

        return limit in actual

    if operator == "exists":
        return actual is not None

    if operator == "no_constraint":
        return True

    if operator == "subjective_review":
        raise ValueError(
            "subjective_review cannot be "
            "evaluated deterministically"
        )

    raise ValueError(
        f"Unsupported operator: {operator}"
    )


def evaluate_rule(
    record: dict,
    rule: dict,
) -> dict:

    rule_id = rule["rule_id"]

    if not rule["deterministic"]:
        return {
            "rule_id": rule_id,
            "status": "SKIPPED_REVIEW",
            "passed": None,
        }

    if rule["operator"] == "no_constraint":
        return {
            "rule_id": rule_id,
            "status": "NO_ACTIVE_CONSTRAINT",
            "passed": True,
        }

    field_path = rule["target_field"]

    try:
        actual = resolve_field(
            record,
            field_path,
        )
    except KeyError:
        return {
            "rule_id": rule_id,
            "status": "MISSING_FIELD",
            "passed": False,
            "target_field": field_path,
        }

    passed = evaluate_operator(
        actual,
        rule["operator"],
        rule.get("limit_value"),
    )

    return {
        "rule_id": rule_id,
        "status": (
            "PASS"
            if passed
            else "VIOLATION"
        ),
        "passed": passed,
        "target_field": field_path,
        "actual_value": actual,
        "limit_value": rule.get(
            "limit_value"
        ),
    }


def evaluate_rules(
    record: dict,
    rules: list[dict],
) -> dict:

    results = [
        evaluate_rule(
            record,
            rule,
        )
        for rule in rules
    ]

    violations = [
        result
        for result in results
        if result["passed"] is False
    ]

    skipped = [
        result
        for result in results
        if result["passed"] is None
    ]

    return {
        "passed": len(violations) == 0,
        "violation_count": len(violations),
        "skipped_review_count": len(skipped),
        "results": results,
    }
