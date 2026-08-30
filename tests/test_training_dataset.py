import json

from jsonschema import Draft202012Validator

from src.data.registry import DataRegistry
from src.training.dataset_builder import (
    build_dataset_records,
)
from src.training.split import grouped_part_split
from src.training.validation import (
    PROJECT_ROOT,
)


def test_grouped_split_keeps_part_ids_disjoint():
    split = grouped_part_split(
        [f"PILOT_{index:03d}" for index in range(1, 11)]
    )

    assert list(split.values()).count("train") == 8
    assert list(split.values()).count("validation") == 1
    assert list(split.values()).count("test") == 1
    assert split["PILOT_001"] == "train"
    assert split["PILOT_009"] == "validation"
    assert split["PILOT_010"] == "test"


def test_generated_positive_records_are_schema_valid_and_admissible():
    records, _ = build_dataset_records()
    schema = json.loads(
        (
            PROJECT_ROOT
            / "schemas"
            / "proposal.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    registry = DataRegistry()

    positives = [
        record
        for record in records
        if record["metadata"]["example_type"] == "positive"
    ]

    assert positives

    for record in positives:
        proposal = json.loads(
            record["messages"][2]["content"]
        )
        assert list(
            validator.iter_errors(proposal)
        ) == []
        assert proposal["change_type"] in {
            "material",
            "process",
        }
        assert isinstance(
            proposal["new_value"],
            str,
        )
        if proposal["change_type"] == "material":
            assert proposal["target_field"] == "material_id"
            assert proposal["new_value"] in registry.materials
        else:
            assert proposal["target_field"] == "process_id"
            assert proposal["new_value"] in registry.processes


def test_generated_dataset_has_no_test_placeholders_or_fasteners():
    records, manifest = build_dataset_records()
    text = "\n".join(
        json.dumps(record, sort_keys=True)
        for record in records
    )

    assert "TEST_" not in text
    assert "CONFIRM" not in json.dumps(
        [
            record["metadata"]["target"]
            for record in records
        ],
        sort_keys=True,
    )
    assert set(manifest["change_type_counts"]) == {
        "material",
        "process",
    }
