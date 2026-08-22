import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _parse_bool(value: str):
    text = value.strip().lower()

    if text == "true":
        return True
    if text == "false":
        return False

    return None


def _load_csv(
    path: Path,
    id_field: str,
    numeric_fields=None,
    required_fields=None,
):
    numeric_fields = numeric_fields or []
    required_fields = required_fields or [id_field]

    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    records = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: CSV has no header")

        missing_columns = [
            field for field in required_fields
            if field not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"{path.name}: missing required columns: "
                f"{', '.join(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            record_id = row.get(id_field, "").strip()

            if not record_id:
                raise ValueError(
                    f"{path.name}: missing {id_field} "
                    f"at row {row_number}"
                )

            if record_id in records:
                raise ValueError(
                    f"{path.name}: duplicate ID "
                    f"'{record_id}' at row {row_number}"
                )

            for field in numeric_fields:
                if field not in row:
                    continue

                raw_value = row[field].strip()

                if raw_value == "":
                    row[field] = None
                else:
                    try:
                        row[field] = float(raw_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"{path.name}: invalid numeric value "
                            f"for {field} at row {row_number}: "
                            f"{raw_value}"
                        ) from exc

            if "verified" in row:
                row["verified"] = _parse_bool(row["verified"])

            records[record_id] = row

    return records


class DataRegistry:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)

        self.materials = _load_csv(
            self.data_dir / "materials.csv",
            id_field="material_id",
            numeric_fields=[
                "density_kg_m3",
                "yield_strength_mpa",
                "unit_price_eur_per_kg",
            ],
            required_fields=[
                "material_id",
                "name",
                "density_kg_m3",
                "source_id",
            ],
        )

        self.processes = _load_csv(
            self.data_dir / "processes.csv",
            id_field="process_id",
            numeric_fields=[
                "rate",
            ],
            required_fields=[
                "process_id",
                "name",
                "rate",
                "rate_unit",
                "source_id",
            ],
        )

        self.fasteners = _load_csv(
            self.data_dir / "fasteners.csv",
            id_field="fastener_id",
            numeric_fields=[
                "unit_price_eur",
            ],
            required_fields=[
                "fastener_id",
                "name",
                "unit_price_eur",
                "source_id",
            ],
        )

        supplier_path = self.data_dir / "suppliers.csv"

        if supplier_path.exists():
            self.suppliers = _load_csv(
                supplier_path,
                id_field="supplier_item_id",
                numeric_fields=[
                    "unit_price",
                ],
                required_fields=[
                    "supplier_item_id",
                    "supplier_name",
                    "manufacturer_part_number",
                    "unit_price",
                    "currency",
                    "source_id",
                ],
            )
        else:
            self.suppliers = {}

    def get_material(self, material_id: str):
        try:
            return self.materials[material_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown material ID: {material_id}"
            ) from exc

    def get_process(self, process_id: str):
        try:
            return self.processes[process_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown process ID: {process_id}"
            ) from exc

    def get_fastener(self, fastener_id: str):
        try:
            return self.fasteners[fastener_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown fastener ID: {fastener_id}"
            ) from exc

    def get_supplier_item(self, supplier_item_id: str):
        try:
            return self.suppliers[supplier_item_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown supplier item ID: "
                f"{supplier_item_id}"
            ) from exc

    def counts(self):
        return {
            "materials": len(self.materials),
            "processes": len(self.processes),
            "fasteners": len(self.fasteners),
            "suppliers": len(self.suppliers),
        }
