import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _load_csv(filename: str, id_field: str, numeric_fields=None):
    numeric_fields = numeric_fields or []

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    records = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row_number, row in enumerate(reader, start=2):
            record_id = row.get(id_field, "").strip()

            if not record_id:
                raise ValueError(
                    f"{filename}: missing {id_field} at row {row_number}"
                )

            if record_id in records:
                raise ValueError(
                    f"{filename}: duplicate ID '{record_id}' at row {row_number}"
                )

            for field in numeric_fields:
                value = row.get(field, "").strip()

                if value == "":
                    row[field] = None
                else:
                    try:
                        row[field] = float(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"{filename}: invalid numeric value for "
                            f"{field} at row {row_number}: {value}"
                        ) from exc

            if "verified" in row:
                row["verified"] = _parse_bool(row["verified"])

            records[record_id] = row

    return records


class DataRegistry:
    def __init__(self):
        self.materials = _load_csv(
            "materials.csv",
            "material_id",
            numeric_fields=["density_kg_m3"],
        )

        self.processes = _load_csv(
            "processes.csv",
            "process_id",
            numeric_fields=["rate_value"],
        )

        self.fasteners = _load_csv(
            "fasteners.csv",
            "fastener_id",
            numeric_fields=["unit_cost"],
        )

        self.suppliers = _load_csv(
            "suppliers.csv",
            "supplier_item_id",
            numeric_fields=["unit_price"],
        )

    def get_material(self, material_id: str):
        try:
            return self.materials[material_id]
        except KeyError as exc:
            raise KeyError(f"Unknown material ID: {material_id}") from exc

    def get_process(self, process_id: str):
        try:
            return self.processes[process_id]
        except KeyError as exc:
            raise KeyError(f"Unknown process ID: {process_id}") from exc

    def get_fastener(self, fastener_id: str):
        try:
            return self.fasteners[fastener_id]
        except KeyError as exc:
            raise KeyError(f"Unknown fastener ID: {fastener_id}") from exc

    def get_supplier_item(self, supplier_item_id: str):
        try:
            return self.suppliers[supplier_item_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown supplier item ID: {supplier_item_id}"
            ) from exc

    def counts(self):
        return {
            "materials": len(self.materials),
            "processes": len(self.processes),
            "fasteners": len(self.fasteners),
            "suppliers": len(self.suppliers),
        }
