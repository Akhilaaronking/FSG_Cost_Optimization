from copy import deepcopy

import pytest

from src.optimization.operators import (
    initialize_candidate,
    mutate_candidate,
)
from src.optimization.search_space import (
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
