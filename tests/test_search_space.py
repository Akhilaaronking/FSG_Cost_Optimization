from copy import deepcopy

import pytest

from src.optimization.operators import (
    initialize_candidate,
    mutate_candidate,
)
from src.optimization.search_space import (
    convert_option_a_search_space,
    validate_candidate_within_search_space,
    validate_search_space,
)


def baseline_bom():
    return {
        "parts": [
            {
                "part_id": "TEST_PART",
                "material_id": "AL_6061_T6",
                "process_id": "CNC_MILLING",
                "geometry": {
                    "x": 0.5,
                },
                "fasteners": [],
            }
        ]
    }


def search_space():
    return {
        "search_space_id": "TEST_SPACE",
        "engineering_verified": False,
        "purpose": "UNIT_TEST_ONLY",
        "parts": [
            {
                "part_id": "TEST_PART",
                "material_choices": [
                    "AL_6061_T6",
                    "AL_7075_T6",
                ],
                "process_choices": [
                    "CNC_MILLING",
                ],
                "geometry_variables": {
                    "x": {
                        "type": "continuous",
                        "min": 0.0,
                        "max": 1.0,
                    }
                },
            }
        ],
    }


def test_search_space_validates_identifier_existence_only():
    result = validate_search_space(
        search_space(),
        baseline_bom(),
    )

    assert result["engineering_verified"] is False
    assert "Identifier existence only" in result[
        "registry_validation_meaning"
    ]


def test_unknown_part_fails_clearly():
    space = search_space()
    space["parts"][0]["part_id"] = "TEST_UNKNOWN"

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        validate_search_space(
            space,
            baseline_bom(),
        )


def test_empty_categorical_choices_fail():
    space = search_space()
    space["parts"][0]["material_choices"] = []

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        validate_search_space(
            space,
            baseline_bom(),
        )


def test_unknown_variable_type_fails():
    space = search_space()
    space["parts"][0]["geometry_variables"][
        "x"
    ]["type"] = "integer"

    with pytest.raises(
        ValueError,
        match="Unknown variable type",
    ):
        validate_search_space(
            space,
            baseline_bom(),
        )


def test_mutation_does_not_modify_parent():
    parent = initialize_candidate(
        baseline_bom(),
        search_space(),
        "TEST_PARENT",
        seed=1,
    )
    original = deepcopy(parent.bom)

    child = mutate_candidate(
        parent,
        search_space(),
        mutation_rate=1.0,
        seed=2,
        candidate_id="TEST_CHILD",
    )

    assert parent.bom == original
    assert 0.0 <= child.bom["parts"][0][
        "geometry"
    ]["x"] <= 1.0


def option_a_space():
    return {
        "schema_version": "OptionA_v1_verified",
        "status": "engineering_verified",
        "parts": [
            {
                "part_id": "TEST_PART",
                "part_name": "Test part",
                "current": {
                    "material_id": "AL_6061_T6",
                    "process_id": "CNC_MILLING",
                },
                "admissible_materials": {
                    "AL_7075_T6": "test alternative"
                },
                "inadmissible_materials": {
                    "CF_PLATE_3K": "test exclusion"
                },
                "admissible_processes": {
                    "CNC_TURNING": "test alternative"
                },
                "inadmissible_processes": {
                    "WATERJET_CUT": "test exclusion"
                },
                "dimension_envelope": {
                    "thickness_mm": "CONFIRM"
                },
                "engineering_verified": True,
                "verification_status": "APPROVED_BY_PERSON_B",
            }
        ],
    }


def test_approved_option_a_conversion_works():
    converted = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )

    assert converted["engineering_verified"] is True
    assert converted["parts"][0]["part_id"] == "TEST_PART"
    assert converted["metadata"]["active_scope"] == [
        "material",
        "process",
    ]


def test_baseline_choices_retained():
    converted = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )
    part = converted["parts"][0]

    assert "AL_6061_T6" in part["material_choices"]
    assert "CNC_MILLING" in part["process_choices"]
    assert part["baseline_material_id"] == "AL_6061_T6"
    assert part["baseline_process_id"] == "CNC_MILLING"


def test_inadmissible_choices_excluded():
    converted = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )
    part = converted["parts"][0]

    assert "CF_PLATE_3K" not in part["material_choices"]
    assert "WATERJET_CUT" not in part["process_choices"]
    assert "CF_PLATE_3K" in part["metadata"][
        "inadmissible_materials_recorded_but_excluded"
    ]


def test_option_a_unknown_material_rejected():
    space = option_a_space()
    space["parts"][0]["admissible_materials"] = {
        "FAKE_MATERIAL": "bad"
    }

    with pytest.raises(
        ValueError,
        match="unknown material IDs",
    ):
        convert_option_a_search_space(
            space,
            baseline_bom(),
        )


def test_option_a_unknown_process_rejected():
    space = option_a_space()
    space["parts"][0]["admissible_processes"] = {
        "FAKE_PROCESS": "bad"
    }

    with pytest.raises(
        ValueError,
        match="unknown process IDs",
    ):
        convert_option_a_search_space(
            space,
            baseline_bom(),
        )


def test_engineering_verified_false_excluded():
    space = option_a_space()
    space["parts"][0]["engineering_verified"] = False

    with pytest.raises(
        ValueError,
        match="engineering_verified",
    ):
        convert_option_a_search_space(
            space,
            baseline_bom(),
        )


def test_confirm_geometry_bounds_not_activated():
    converted = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )

    assert "geometry_variables" not in converted["parts"][0]


def test_option_a_loading_is_deterministic():
    first = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )
    second = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )

    assert first == second


def test_candidate_outside_material_choices_rejected():
    converted = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )
    candidate = baseline_bom()
    candidate["parts"][0]["material_id"] = "CF_PLATE_3K"

    result = validate_candidate_within_search_space(
        candidate,
        converted,
    )

    assert result["valid"] is False
    assert "outside approved material_choices" in result[
        "errors"
    ][0]


def test_candidate_outside_process_choices_rejected():
    converted = convert_option_a_search_space(
        option_a_space(),
        baseline_bom(),
    )
    candidate = baseline_bom()
    candidate["parts"][0]["process_id"] = "WATERJET_CUT"

    result = validate_candidate_within_search_space(
        candidate,
        converted,
    )

    assert result["valid"] is False
    assert "outside approved process_choices" in result[
        "errors"
    ][0]
