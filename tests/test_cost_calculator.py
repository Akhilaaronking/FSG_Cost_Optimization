import json
from pathlib import Path

import pytest

from src.cost_engine.cost_calculator import (
    calculate_part_cost,
    calculate_bom_cost,
)
from src.data.registry import DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "pilot_10_parts_ground_truth.json"
)


def load_bom():
    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_all_10_verified_pilot_costs_match():
    registry = DataRegistry()
    bom = load_bom()

    assert len(bom["parts"]) == 10

    checked = 0

    for part in bom["parts"]:
        result = calculate_part_cost(
            part,
            registry,
        )

        expected = float(
            part["manual_calculation"][
                "total_cost_eur"
            ]
        )

        assert result[
            "total_cost_eur"
        ] == pytest.approx(
            expected,
            abs=0.01,
        )

        checked += 1

    assert checked == 10


def test_fdm_hour_rate_is_converted_to_minutes():
    registry = DataRegistry()
    bom = load_bom()

    part = next(
        part
        for part in bom["parts"]
        if part["part_id"] == "PILOT_009"
    )

    result = calculate_part_cost(
        part,
        registry,
    )

    assert result["process_cost_eur"] == 5.00


def test_waterjet_uses_cut_length():
    registry = DataRegistry()
    bom = load_bom()

    part = next(
        part
        for part in bom["parts"]
        if part["part_id"] == "PILOT_008"
    )

    result = calculate_part_cost(
        part,
        registry,
    )

    assert result["process_cost_eur"] == 47.50


def test_bom_total_uses_full_precision():
    registry = DataRegistry()
    bom = load_bom()

    result = calculate_bom_cost(
        bom,
        registry,
    )

    # Corrected B4 v2 benchmark and deterministic
    # recomputation must agree exactly.
    assert result["total_cost_eur"] == 312.02
