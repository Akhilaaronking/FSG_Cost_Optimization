import json
from pathlib import Path
from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_bom(bom_path: Path | None = None):
    if bom_path is None:
        bom_path = PROJECT_ROOT / "data" / "benchmark" / "pilot_bom.json"

    schema_path = PROJECT_ROOT / "schemas" / "bom.schema.json"

    bom = load_json(bom_path)
    schema = load_json(schema_path)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(bom), key=lambda e: list(e.path))

    return bom, errors


def print_validation_report(bom, errors):
    print("=" * 60)
    print("FSG THESIS — STEP 1 BOM VALIDATION")
    print("=" * 60)
    print(f"BOM ID:          {bom.get('bom_id', '<missing>')}")
    print(f"Schema version:  {bom.get('schema_version', '<missing>')}")
    print(f"Vehicle class:   {bom.get('vehicle_class', '<missing>')}")
    print(f"Parts loaded:    {len(bom.get('parts', []))}")
    print()

    if not errors:
        for part in bom.get("parts", []):
            print(f"[PASS] {part['part_id']} — {part['part_name']}")
        print()
        print("BOM VALIDATION: PASS")
        return True

    for i, error in enumerate(errors, start=1):
        path = ".".join(str(x) for x in error.path) or "<root>"
        print(f"[FAIL {i}] {path}: {error.message}")

    print()
    print("BOM VALIDATION: FAIL")
    return False
