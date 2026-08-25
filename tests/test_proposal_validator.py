from src.data.registry import DataRegistry
from src.llm.proposal_validator import (
    classify_hallucination,
    validate_proposal_authority,
    validate_proposal_schema,
)


def bom():
    return {
        "parts": [
            {
                "part_id": "PILOT_001",
                "material_id": "AL_6061_T6",
                "process_id": "CNC_MILLING",
            }
        ]
    }


def proposal(**overrides):
    base = {
        "proposal_id": "PROP_001",
        "part_id": "PILOT_001",
        "change_type": "material",
        "target_field": "material_id",
        "new_value": "AL_7075_T6",
        "reasoning_summary": "Concise summary.",
    }
    base.update(overrides)
    return base


def test_valid_proposal_passes_schema():
    assert validate_proposal_schema(
        proposal()
    )["schema_valid"]


def test_missing_required_property_fails_schema():
    item = proposal()
    del item["new_value"]

    assert not validate_proposal_schema(
        item
    )["schema_valid"]


def test_known_material_id_passes_authority():
    result = validate_proposal_authority(
        proposal(),
        bom(),
        registry=DataRegistry(),
    )

    assert result["authority_valid"]


def test_fake_material_id_detected():
    result = validate_proposal_authority(
        proposal(new_value="FAKE_MATERIAL"),
        bom(),
        registry=DataRegistry(),
    )

    assert result["unknown_identifiers"][0][
        "category"
    ] == "UNKNOWN_MATERIAL_ID"


def test_fake_process_id_detected():
    result = validate_proposal_authority(
        proposal(
            change_type="process",
            target_field="process_id",
            new_value="FAKE_PROCESS",
        ),
        bom(),
        registry=DataRegistry(),
    )

    assert result["unknown_identifiers"][0][
        "category"
    ] == "UNKNOWN_PROCESS_ID"


def test_fake_fastener_id_detected():
    result = validate_proposal_authority(
        proposal(
            change_type="fastener",
            target_field="fasteners",
            new_value={
                "fastener_id": "FAKE_FASTENER",
                "quantity": 1,
            },
        ),
        bom(),
        registry=DataRegistry(),
    )

    assert result["unknown_identifiers"][0][
        "category"
    ] == "UNKNOWN_FASTENER_ID"


def test_fake_part_id_detected():
    result = validate_proposal_authority(
        proposal(part_id="FAKE_PART"),
        bom(),
        registry=DataRegistry(),
    )

    assert result["unknown_identifiers"][0][
        "category"
    ] == "UNKNOWN_PART_ID"


def test_unknown_identifier_classified_hallucination():
    authority = validate_proposal_authority(
        proposal(new_value="FAKE_MATERIAL"),
        bom(),
        registry=DataRegistry(),
    )
    result = classify_hallucination(
        True,
        {"schema_valid": True},
        authority,
    )

    assert result["hallucinated"]
    assert "UNKNOWN_MATERIAL_ID" in result["categories"]


def test_schema_failure_classified():
    result = classify_hallucination(
        True,
        {"schema_valid": False},
        {"authority_valid": True},
    )

    assert result["categories"] == ["SCHEMA_ERROR"]


def test_valid_canonical_proposal_not_hallucinated():
    result = classify_hallucination(
        True,
        {"schema_valid": True},
        {"authority_valid": True},
    )

    assert not result["hallucinated"]
    assert result["categories"] == []


def test_material_dict_new_value_fails_safely():
    result = validate_proposal_authority(
        proposal(
            new_value={
                "material_id": "AL_7075_T6"
            }
        ),
        bom(),
        registry=DataRegistry(),
    )

    assert not result["authority_valid"]
    assert result["errors"][0][
        "category"
    ] == "INVALID_MATERIAL_ID_TYPE"


def test_material_dict_new_value_fails_schema():
    result = validate_proposal_schema(
        proposal(
            change_type="material",
            target_field="material_id",
            new_value={
                "material_id": "AL_7075_T6"
            },
        )
    )

    assert not result["schema_valid"]


def test_material_string_new_value_passes_schema():
    result = validate_proposal_schema(
        proposal(
            change_type="material",
            target_field="material_id",
            new_value="AL_7075_T6",
        )
    )

    assert result["schema_valid"]


def test_material_requires_material_target_field():
    result = validate_proposal_schema(
        proposal(
            change_type="material",
            target_field="process_id",
            new_value="AL_7075_T6",
        )
    )

    assert not result["schema_valid"]


def test_process_requires_string_new_value():
    result = validate_proposal_schema(
        proposal(
            change_type="process",
            target_field="process_id",
            new_value={
                "process_id": "CNC_MILLING"
            },
        )
    )

    assert not result["schema_valid"]
