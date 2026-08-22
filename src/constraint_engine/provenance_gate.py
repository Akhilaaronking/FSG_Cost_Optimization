import csv
from pathlib import Path

from src.data.registry import DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_REGISTER_PATH = (
    PROJECT_ROOT
    / "docs"
    / "person_b_b4"
    / "source_register.csv"
)


def load_source_ids(
    source_register_path: Path = SOURCE_REGISTER_PATH,
) -> set[str]:
    if not source_register_path.exists():
        raise FileNotFoundError(
            f"Source register not found: "
            f"{source_register_path}"
        )

    source_ids = set()

    with source_register_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if "source_id" not in reader.fieldnames:
            raise ValueError(
                "Source register is missing source_id column"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            source_id = row.get(
                "source_id",
                "",
            ).strip()

            if not source_id:
                raise ValueError(
                    f"Blank source_id in source register "
                    f"at row {row_number}"
                )

            if source_id in source_ids:
                raise ValueError(
                    f"Duplicate source_id in source register: "
                    f"{source_id}"
                )

            source_ids.add(source_id)

    return source_ids


def evaluate_cost_provenance(
    registry: DataRegistry | None = None,
    source_register_path: Path = SOURCE_REGISTER_PATH,
) -> dict:
    """
    DERIVED_QG_001

    Internal quality gate only.

    Checks that each material/process/fastener row used
    for deterministic cost calculation has a non-empty
    source_id that exists in the source register.

    Passing this gate proves traceability only.
    It does NOT prove that a cost is realistic and does
    NOT constitute compliance with FSG S_3.5.11.
    """

    registry = registry or DataRegistry()

    valid_source_ids = load_source_ids(
        source_register_path
    )

    failures = []
    checked = 0

    datasets = [
        (
            "material",
            registry.materials,
            "material_id",
            "unit_price_eur_per_kg",
        ),
        (
            "process",
            registry.processes,
            "process_id",
            "rate",
        ),
        (
            "fastener",
            registry.fasteners,
            "fastener_id",
            "unit_price_eur",
        ),
    ]

    for (
        record_type,
        records,
        id_field,
        cost_field,
    ) in datasets:

        for record_id, record in records.items():
            cost_value = record.get(cost_field)

            # Only rows participating in cost evaluation
            # are subject to this quality gate.
            if cost_value is None:
                continue

            checked += 1

            source_id = (
                record.get("source_id")
                or ""
            ).strip()

            if not source_id:
                failures.append({
                    "record_type": record_type,
                    "record_id": record_id,
                    "reason": "MISSING_SOURCE_ID",
                })
                continue

            if source_id not in valid_source_ids:
                failures.append({
                    "record_type": record_type,
                    "record_id": record_id,
                    "source_id": source_id,
                    "reason": (
                        "SOURCE_ID_NOT_IN_REGISTER"
                    ),
                })

    return {
        "rule_id": "DERIVED_QG_001",
        "rule_category": "derived_quality_gate",
        "status": (
            "PASS"
            if not failures
            else "FAIL"
        ),
        "passed": not failures,
        "checked_records": checked,
        "failure_count": len(failures),
        "failures": failures,
        "fsg_compliance_claim": False,
        "meaning": (
            "Traceability only; does not prove "
            "cost realism or S_3.5.11 compliance."
        ),
    }
