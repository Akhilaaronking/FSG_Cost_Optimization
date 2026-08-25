from src.rag.query_builder import (
    build_engineering_query,
)


def test_part_information_inserted_when_present():
    query = build_engineering_query(
        {
            "part_id": "TEST_PART",
            "name": "Suspension bracket",
            "material_id": "AL_6061_T6",
            "process_id": "CNC_MILLING",
        },
        change_type="material",
        target_field="material_id",
    )

    assert "TEST_PART" in query
    assert "Suspension bracket" in query
    assert "AL_6061_T6" in query
    assert "CNC_MILLING" in query


def test_absent_fields_do_not_generate_fake_values():
    query = build_engineering_query({})

    assert "Current material:" not in query
    assert "Current process:" not in query


def test_query_deterministic():
    part = {"part_id": "TEST_PART"}

    assert build_engineering_query(part) == build_engineering_query(part)
