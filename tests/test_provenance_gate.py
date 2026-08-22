import csv

from src.constraint_engine.provenance_gate import (
    evaluate_cost_provenance,
    load_source_ids,
)
from src.data.registry import DataRegistry


def test_source_register_loads():
    source_ids = load_source_ids()

    assert "FSG_RULES_2026" in source_ids
    assert len(source_ids) > 0


def test_real_cost_provenance_passes():
    result = evaluate_cost_provenance()

    assert result["rule_id"] == "DERIVED_QG_001"
    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert result["failure_count"] == 0

    # 11 materials + 10 processes + 10 fasteners
    assert result["checked_records"] == 31

    assert (
        result["fsg_compliance_claim"]
        is False
    )


def test_unknown_source_is_detected(
    tmp_path,
):
    registry = DataRegistry()

    registry.materials[
        "AL_6061_T6"
    ]["source_id"] = "DOES_NOT_EXIST"

    result = evaluate_cost_provenance(
        registry=registry,
    )

    assert result["passed"] is False
    assert result["failure_count"] == 1

    failure = result["failures"][0]

    assert (
        failure["reason"]
        == "SOURCE_ID_NOT_IN_REGISTER"
    )


def test_blank_source_is_detected():
    registry = DataRegistry()

    registry.fasteners[
        "BOLT_M6X20"
    ]["source_id"] = ""

    result = evaluate_cost_provenance(
        registry=registry,
    )

    assert result["passed"] is False

    assert (
        result["failures"][0]["reason"]
        == "MISSING_SOURCE_ID"
    )


def test_duplicate_source_register_ids_rejected(
    tmp_path,
):
    path = tmp_path / "source_register.csv"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "source_id",
            "title",
        ])

        writer.writerow([
            "SOURCE_A",
            "First",
        ])

        writer.writerow([
            "SOURCE_A",
            "Duplicate",
        ])

    try:
        load_source_ids(path)
        assert False, (
            "Expected duplicate source_id "
            "to raise ValueError"
        )
    except ValueError as exc:
        assert "Duplicate source_id" in str(exc)
