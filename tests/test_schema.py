import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_validator():
    schema = load_json(ROOT / "schemas" / "bom.schema.json")
    return Draft202012Validator(schema)


def get_valid_bom():
    return load_json(ROOT / "data" / "benchmark" / "pilot_bom.json")


def test_valid_pilot_bom_passes():
    errors = list(get_validator().iter_errors(get_valid_bom()))
    assert errors == []


def test_missing_material_id_fails():
    bom = copy.deepcopy(get_valid_bom())
    del bom["parts"][0]["material_id"]
    errors = list(get_validator().iter_errors(bom))
    assert errors


def test_bad_quantity_type_fails():
    bom = copy.deepcopy(get_valid_bom())
    bom["parts"][0]["quantity"] = "two"
    errors = list(get_validator().iter_errors(bom))
    assert errors


def test_zero_volume_fails():
    bom = copy.deepcopy(get_valid_bom())
    bom["parts"][0]["geometry"]["finished_volume_mm3"] = 0
    errors = list(get_validator().iter_errors(bom))
    assert errors
