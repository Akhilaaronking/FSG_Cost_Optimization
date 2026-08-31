import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

B5_CONSTRAINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deterministic_constraints_B5.json"
)

B5_CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rule_classification_B5.json"
)


OPERATOR_MAP = {
    "==": "eq",
    "!=": "neq",
    "<": "lt",
    "<=": "lte",
    ">": "gt",
    ">=": "gte",
    "in": "in",
    "not_in": "not_in",
    "includes": "includes",
    "subjective_review": "subjective_review",
    "N/A": "no_constraint",
}


def _read_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _parse_limit_value(value):
    if not isinstance(value, str):
        return value

    text = value.strip()

    if text == "N/A":
        return None

    if text.lower() == "true":
        return True

    if text.lower() == "false":
        return False

    if (
        text.startswith("{")
        and text.endswith("}")
    ):
        inner = text[1:-1]

        return [
            item.strip()
            for item in inner.split(",")
            if item.strip()
        ]

    return value


def _classification_text_map():
    data = _read_json(
        B5_CLASSIFICATION_PATH
    )

    return {
        rule["rule_id"]: rule["text"]
        for rule in data["rules"]
    }


def _to_canonical(
    raw_rule: dict,
    text_map: dict,
) -> dict:

    raw_operator = raw_rule["operator"]

    if raw_operator not in OPERATOR_MAP:
        raise ValueError(
            f"Unsupported B5 operator "
            f"'{raw_operator}' in "
            f"{raw_rule['rule_id']}"
        )

    unit = raw_rule.get("units")

    if unit == "N/A":
        unit = None

    return {
        "rule_id": raw_rule["rule_id"],
        "rule_category": raw_rule[
            "rule_category"
        ],
        "description": text_map.get(
            raw_rule["rule_id"],
            raw_rule.get(
                "notes",
                raw_rule["parameter_field"],
            ),
        ),
        "applies_to": raw_rule.get(
            "affected_part_category"
        ),
        "target_field": raw_rule[
            "parameter_field"
        ],
        "operator": OPERATOR_MAP[
            raw_operator
        ],
        "limit_value": _parse_limit_value(
            raw_rule.get("limit_value")
        ),
        "unit": unit,
        "source_id": raw_rule.get(
            "source_id",
            "",
        ),
        "source_reference": raw_rule.get(
            "fsg_reference"
        ),
        "deterministic": bool(
            raw_rule["deterministic"]
        ),
        "notes": (
            raw_rule.get("notes")
            or None
        ),
    }


def load_b5_rules() -> dict:
    """
    Load the verified B5 dataset and route it into:

    1. deterministic FSG rules
    2. non-deterministic review rules
    3. internal derived quality gates

    The B5 source files are never modified.
    """

    data = _read_json(
        B5_CONSTRAINT_PATH
    )

    raw_rules = data["constraints"]

    if len(raw_rules) != 27:
        raise ValueError(
            "Expected 27 B5 entries, "
            f"found {len(raw_rules)}"
        )

    ids = [
        rule["rule_id"]
        for rule in raw_rules
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate rule_id found "
            "in B5 dataset"
        )

    text_map = _classification_text_map()

    deterministic_rules = []
    review_rules = []
    quality_gates = []

    for raw_rule in raw_rules:

        if (
            raw_rule["rule_category"]
            == "derived_quality_gate"
        ):
            quality_gates.append(
                raw_rule.copy()
            )
            continue

        canonical = _to_canonical(
            raw_rule,
            text_map,
        )

        if canonical["deterministic"]:
            deterministic_rules.append(
                canonical
            )
        else:
            review_rules.append(
                canonical
            )

    if len(deterministic_rules) != 17:
        raise ValueError(
            "Expected 17 deterministic "
            "FSG rules"
        )

    if len(review_rules) != 9:
        raise ValueError(
            "Expected 9 review rules"
        )

    if len(quality_gates) != 1:
        raise ValueError(
            "Expected 1 derived "
            "quality gate"
        )

    return {
        "deterministic_rules":
            deterministic_rules,
        "review_rules":
            review_rules,
        "quality_gates":
            quality_gates,
        "total_entries":
            len(raw_rules),
    }
