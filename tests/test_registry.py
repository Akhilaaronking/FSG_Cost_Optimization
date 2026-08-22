import csv

import pytest

from src.data.registry import DataRegistry


def test_real_registry_loads():
    registry = DataRegistry()

    assert len(registry.materials) == 11
    assert len(registry.processes) == 10
    assert len(registry.fasteners) == 10


def test_real_material_lookup():
    registry = DataRegistry()

    material = registry.get_material("AL_6061_T6")

    assert material["name"] == "Aluminium 6061-T6"
    assert material["density_kg_m3"] == 2700.0
    assert material["yield_strength_mpa"] == 276.0
    assert material["source_id"]


def test_real_process_lookup():
    registry = DataRegistry()

    process = registry.get_process("CNC_MILLING")

    assert process["rate"] == 1.42
    assert process["rate_unit"] == "EUR_per_min"
    assert process["source_id"]


def test_real_fastener_lookup():
    registry = DataRegistry()

    fastener = registry.get_fastener("BOLT_M6X20")

    assert fastener["unit_price_eur"] == 0.37
    assert fastener["source_id"]


def test_unknown_material_is_rejected():
    registry = DataRegistry()

    with pytest.raises(
        KeyError,
        match="Unknown material ID"
    ):
        registry.get_material("NOT_A_REAL_MATERIAL")


def test_unknown_process_is_rejected():
    registry = DataRegistry()

    with pytest.raises(
        KeyError,
        match="Unknown process ID"
    ):
        registry.get_process("NOT_A_REAL_PROCESS")


def test_duplicate_ids_are_rejected(tmp_path):
    materials = tmp_path / "materials.csv"
    processes = tmp_path / "processes.csv"
    fasteners = tmp_path / "fasteners.csv"

    with materials.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material_id",
            "name",
            "density_kg_m3",
            "source_id",
        ])
        writer.writerow([
            "DUPLICATE",
            "Material A",
            "2700",
            "SRC_A",
        ])
        writer.writerow([
            "DUPLICATE",
            "Material B",
            "2800",
            "SRC_B",
        ])

    with processes.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "process_id",
            "name",
            "rate",
            "rate_unit",
            "source_id",
        ])
        writer.writerow([
            "PROC_1",
            "Process",
            "1.0",
            "EUR_per_min",
            "SRC",
        ])

    with fasteners.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fastener_id",
            "name",
            "unit_price_eur",
            "source_id",
        ])
        writer.writerow([
            "FAST_1",
            "Fastener",
            "1.0",
            "SRC",
        ])

    with pytest.raises(
        ValueError,
        match="duplicate ID"
    ):
        DataRegistry(data_dir=tmp_path)
