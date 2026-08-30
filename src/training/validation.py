from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

from src.data.registry import DataRegistry
from src.optimization.search_space import (
    DEFAULT_B4_BOM_PATH,
    DEFAULT_REAL_SEARCH_SPACE_PATH,
)
from src.training.dataset_builder import (
    build_dataset_records,
    canonical_json_line,
    dataset_hash,
    load_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "data" / "training"
PROPOSAL_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "proposal.schema.json"
)
JSONL_FILES = {
    "train": "c3_train.jsonl",
    "validation": "c3_validation.jsonl",
    "test": "c3_test.jsonl",
}


def load_jsonl(
    path: Path,
) -> list[dict]:
    records = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue
            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc
    return records


def _load_records(
    training_dir: Path,
) -> list[dict]:
    records = []
    for expected_split, filename in JSONL_FILES.items():
        path = training_dir / filename
        split_records = load_jsonl(
            path
        )
        for record in split_records:
            split = record.get(
                "metadata",
                {},
            ).get("split")
            if split != expected_split:
                raise ValueError(
                    f"{path}: record split {split!r} does not match file"
                )
        records.extend(
            split_records
        )
    return records


def _assistant_json(
    record: dict,
) -> dict:
    messages = record.get(
        "messages",
        [],
    )
    if len(messages) != 3:
        raise ValueError(
            "Each record must contain exactly three messages"
        )
    if [
        message.get("role")
        for message in messages
    ] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError(
            "Message roles must be system, user, assistant"
        )
    return json.loads(
        messages[2]["content"]
    )


def _validate_no_split_leakage(
    records: list[dict],
):
    splits_by_part = defaultdict(set)
    for record in records:
        metadata = record["metadata"]
        splits_by_part[metadata["part_id"]].add(
            metadata["split"]
        )

    leaked = {
        part_id: sorted(splits)
        for part_id, splits in splits_by_part.items()
        if len(splits) > 1
    }
    if leaked:
        raise ValueError(
            f"Split leakage by part_id: {leaked}"
        )


def _validate_positive(
    *,
    record: dict,
    proposal: dict,
    proposal_validator: Draft202012Validator,
    registry: DataRegistry,
    option_parts: dict[str, dict],
):
    schema_errors = sorted(
        proposal_validator.iter_errors(proposal),
        key=lambda error: list(error.path),
    )
    if schema_errors:
        raise ValueError(
            "Positive proposal schema error: "
            + "; ".join(
                error.message
                for error in schema_errors
            )
        )

    metadata = record["metadata"]
    part_id = metadata["part_id"]
    option_part = option_parts[part_id]
    change_type = proposal["change_type"]
    target_field = proposal["target_field"]
    new_value = proposal["new_value"]

    if change_type not in {
        "material",
        "process",
    }:
        raise ValueError(
            f"Unsupported positive change_type: {change_type}"
        )

    if not isinstance(new_value, str):
        raise ValueError(
            "Positive material/process new_value must be a string"
        )

    if change_type == "material":
        if target_field != "material_id":
            raise ValueError(
                "Material positive must target material_id"
            )
        if new_value not in registry.materials:
            raise ValueError(
                f"Unknown positive material ID: {new_value}"
            )
        approved = option_part.get(
            "admissible_materials",
            {},
        )
        inadmissible = option_part.get(
            "inadmissible_materials",
            {},
        )
        current = option_part["current"]["material_id"]
    else:
        if target_field != "process_id":
            raise ValueError(
                "Process positive must target process_id"
            )
        if new_value not in registry.processes:
            raise ValueError(
                f"Unknown positive process ID: {new_value}"
            )
        approved = option_part.get(
            "admissible_processes",
            {},
        )
        inadmissible = option_part.get(
            "inadmissible_processes",
            {},
        )
        current = option_part["current"]["process_id"]

    if new_value == current:
        raise ValueError(
            "Baseline value used as positive target"
        )
    if new_value not in approved:
        raise ValueError(
            f"Positive target {new_value} is not explicitly approved"
        )
    if new_value in inadmissible:
        raise ValueError(
            f"Inadmissible target {new_value} appears as positive"
        )


def _validate_metadata(
    record: dict,
):
    metadata = record.get(
        "metadata",
        {},
    )
    required = {
        "example_id",
        "part_id",
        "split",
        "example_type",
        "engineering_source",
        "engineering_verified",
        "generated_programmatically",
        "change_type",
        "target",
        "source_hash",
        "search_space_hash",
        "benchmark_hash",
    }
    missing = sorted(
        required - set(metadata)
    )
    if missing:
        raise ValueError(
            f"Missing metadata fields: {missing}"
        )
    if metadata["engineering_source"] != "real_search_space":
        raise ValueError(
            "engineering_source must be real_search_space"
        )
    if metadata["engineering_verified"] is not True:
        raise ValueError(
            "engineering_verified must be true"
        )
    if metadata["generated_programmatically"] is not True:
        raise ValueError(
            "generated_programmatically must be true"
        )
    if "TEST_" in canonical_json_line(record):
        raise ValueError(
            "TEST_* placeholder ID detected"
        )
    target = metadata["target"]
    if isinstance(target, dict):
        for key, value in target.items():
            if "CONFIRM" in str(key) or "CONFIRM" in str(value):
                raise ValueError(
                    "CONFIRM field used as target"
                )


def _validate_reproduction(
    records: list[dict],
    manifest: dict,
):
    regenerated_records, regenerated_manifest = build_dataset_records()
    canonical_records = sorted(
        records,
        key=lambda record: record["metadata"]["example_id"],
    )

    actual_hash = dataset_hash(
        canonical_records
    )
    if actual_hash != manifest["hashes"]["dataset_hash"]:
        raise ValueError(
            "Manifest dataset_hash does not match JSONL records"
        )
    if actual_hash != dataset_hash(
        regenerated_records
    ):
        raise ValueError(
            "Dataset is not reproduced deterministically"
        )
    if (
        manifest["hashes"]["dataset_hash"]
        != regenerated_manifest["hashes"]["dataset_hash"]
    ):
        raise ValueError(
            "Manifest hash differs from regenerated manifest"
        )


def validate_training_dataset(
    training_dir: Path | str = DEFAULT_TRAINING_DIR,
) -> dict:
    training_dir = Path(training_dir)
    records = _load_records(
        training_dir
    )
    manifest = load_json(
        training_dir / "c3_dataset_manifest.json"
    )
    option_a = load_json(
        DEFAULT_REAL_SEARCH_SPACE_PATH
    )
    option_parts = {
        part["part_id"]: part
        for part in option_a.get("parts", [])
    }
    proposal_schema = load_json(
        PROPOSAL_SCHEMA_PATH
    )
    proposal_validator = Draft202012Validator(
        proposal_schema
    )
    registry = DataRegistry()

    _validate_no_split_leakage(
        records
    )

    counts = Counter()
    for record in records:
        _validate_metadata(
            record
        )
        metadata = record["metadata"]
        counts[
            f'{metadata["split"]}:{metadata["example_type"]}'
        ] += 1
        assistant = _assistant_json(
            record
        )
        if metadata["example_type"] == "positive":
            _validate_positive(
                record=record,
                proposal=assistant,
                proposal_validator=proposal_validator,
                registry=registry,
                option_parts=option_parts,
            )
        elif metadata["example_type"] == "negative":
            if assistant.get("rejected") is not True:
                raise ValueError(
                    "Negative examples must be explicit rejections"
                )
        else:
            raise ValueError(
                f'Unknown example_type: {metadata["example_type"]}'
            )

    _validate_reproduction(
        records,
        manifest,
    )

    if len(records) != manifest["record_count"]:
        raise ValueError(
            "Manifest record_count does not match JSONL records"
        )

    return {
        "valid": True,
        "record_count": len(records),
        "counts": dict(
            sorted(counts.items())
        ),
        "dataset_hash": manifest["hashes"]["dataset_hash"],
        "part_ids_by_split": manifest["part_ids_by_split"],
    }
