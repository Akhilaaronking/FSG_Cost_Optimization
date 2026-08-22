import json
from pathlib import Path

import pytest

from src.data.registry import DataRegistry
from src.mass_engine.mass_calculator import (
    calculate_unit_mass,
    calculate_part_mass,
    calculate_bom_mass,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "pilot_10_parts_ground_truth.json"
)


def load_ground_truth():
    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_all_10_ground_truth_masses():
    """
    Every calculated mass must agree with Person B's
    independently prepared manual ground truth.
    """

    registry = DataRegistry()
    bom = load_ground_truth()

    assert len(bom["parts"]) == 10

    for part in bom["parts"]:
        calculated = calculate_unit_mass(
            part,
            registry=registry,
        )

        expected = float(part["mass_kg"])

        assert calculated == pytest.approx(
            expected,
            abs=1e-5,
        ), (
            f"{part['part_id']} mass mismatch: "
            f"calculated={calculated}, "
            f"expected={expected}"
        )


def test_ground_truth_total_mass():
    registry = DataRegistry()
    bom = load_ground_truth()

    calculated_total = calculate_bom_mass(
        bom,
        registry=registry,
    )

    expected_total = sum(
        float(part["mass_kg"])
        for part in bom["parts"]
    )

    assert calculated_total == pytest.approx(
        expected_total,
        abs=1e-5,
    )


def test_canonical_mm3_volume_format():
    """
    Verify compatibility with Person A's canonical BOM schema.
    """

    registry = DataRegistry()

    part = {
        "part_id": "TEST_CANONICAL",
        "material_id": "AL_6061_T6",
        "quantity": 2,
        "geometry": {
            "finished_volume_mm3": 24000
        },
    }

    unit_mass = calculate_unit_mass(
        part,
        registry=registry,
    )

    total_mass = calculate_part_mass(
        part,
        registry=registry,
    )

    assert unit_mass == pytest.approx(
        0.0648,
        abs=1e-9,
    )

    assert total_mass == pytest.approx(
        0.1296,
        abs=1e-9,
    )


def test_stored_mass_is_not_trusted():
    """
    A wrong stored mass must not influence calculation.
    """

    registry = DataRegistry()

    part = {
        "material_id": "AL_6061_T6",
        "volume_m3": 0.000024,
        "mass_kg": 999999.0,
    }

    calculated = calculate_unit_mass(
        part,
        registry=registry,
    )

    assert calculated == pytest.approx(
        0.0648,
        abs=1e-9,
    )


def test_unknown_material_is_rejected():
    registry = DataRegistry()

    part = {
        "material_id": "FAKE_MATERIAL",
        "volume_m3": 0.001,
    }

    with pytest.raises(
        KeyError,
        match="Unknown material ID",
    ):
        calculate_unit_mass(
            part,
            registry=registry,
        )


def test_zero_volume_is_rejected():
    registry = DataRegistry()

    part = {
        "material_id": "AL_6061_T6",
        "volume_m3": 0,
    }

    with pytest.raises(
        ValueError,
        match="volume must be greater than zero",
    ):
        calculate_unit_mass(
            part,
            registry=registry,
        )
