import pytest

from src.data.registry import DataRegistry


def test_registry_loads_all_tables():
    registry = DataRegistry()

    assert registry.counts() == {
        "materials": 2,
        "processes": 2,
        "fasteners": 1,
        "suppliers": 1,
    }


def test_known_material_is_found():
    registry = DataRegistry()

    material = registry.get_material("TEST_MATERIAL_01")

    assert material["material_id"] == "TEST_MATERIAL_01"
    assert material["verified"] is False


def test_known_process_is_found():
    registry = DataRegistry()

    process = registry.get_process("TEST_PROCESS_01")

    assert process["process_id"] == "TEST_PROCESS_01"
    assert process["verified"] is False


def test_known_fastener_is_found():
    registry = DataRegistry()

    fastener = registry.get_fastener("TEST_FASTENER_01")

    assert fastener["fastener_id"] == "TEST_FASTENER_01"
    assert fastener["verified"] is False


def test_known_supplier_item_is_found():
    registry = DataRegistry()

    supplier = registry.get_supplier_item("TEST_SUPPLIER_ITEM_01")

    assert supplier["supplier_item_id"] == "TEST_SUPPLIER_ITEM_01"
    assert supplier["verified"] is False


def test_unknown_material_is_rejected():
    registry = DataRegistry()

    with pytest.raises(KeyError, match="Unknown material ID"):
        registry.get_material("DOES_NOT_EXIST")


def test_unknown_process_is_rejected():
    registry = DataRegistry()

    with pytest.raises(KeyError, match="Unknown process ID"):
        registry.get_process("DOES_NOT_EXIST")
