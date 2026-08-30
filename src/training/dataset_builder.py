from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.llm.prompt_builder import sha256_json, sha256_text
from src.optimization.search_space import (
    DEFAULT_B4_BOM_PATH,
    DEFAULT_REAL_SEARCH_SPACE_PATH,
    load_verified_real_search_space,
)
from src.training.split import (
    DEFAULT_SPLIT_SEED,
    grouped_part_split,
    part_ids_by_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "training"

SYSTEM_MESSAGES = [
    (
        "You are a controlled C3 training-data assistant. Return only the "
        "requested JSON object, use canonical IDs, and make no unsupported "
        "engineering, cost, mass, or FSG-compliance claims."
    ),
    (
        "You generate atomic Formula Student BOM proposals for deterministic "
        "evaluation. Use only the part-specific approved material/process "
        "search space and reject inadmissible requests."
    ),
    (
        "You are preparing schema-constrained proposal outputs. Material and "
        "process new_value fields must be strings, never nested objects."
    ),
]

POSITIVE_USER_TEMPLATES = [
    (
        "For {part_id} ({part_name}), propose the approved {change_type} "
        "alternative {new_value}. Baseline {target_field} is {old_value}."
    ),
    (
        "Use the engineering-reviewed search space to create one {change_type} "
        "proposal for {part_id}. The requested target is {new_value}."
    ),
    (
        "Produce one valid JSON proposal for {part_id} changing only "
        "{target_field} from {old_value} to {new_value}."
    ),
    (
        "Retrieved context: {rationale} Create a conservative {change_type} "
        "proposal for {part_id} using {new_value}."
    ),
    (
        "The deterministic evaluator will calculate cost and mass later. "
        "Return only the atomic {change_type} modification for {part_id}: "
        "{target_field} = {new_value}."
    ),
    (
        "Training example: select an approved non-baseline {change_type} "
        "choice for {part_id}. Candidate ID: {new_value}."
    ),
]

NEGATIVE_USER_TEMPLATES = [
    (
        "For {part_id} ({part_name}), propose {change_type} change "
        "{target_field} = {new_value}. Reject it if it is not admissible."
    ),
    (
        "Check this requested C3 output for {part_id}: {target_field} should "
        "be {new_value}. Return a rejection JSON if it violates the approved "
        "search space or schema discipline."
    ),
    (
        "Use the engineering-reviewed space only. Is {new_value} valid for "
        "{part_id} as {target_field}? Return the correct rejection JSON when "
        "it is not."
    ),
    (
        "Contrastive example for {part_id}: do not silently accept "
        "{target_field} = {new_value}. Explain the rejection without creating "
        "a positive proposal."
    ),
]

CONSERVATIVE_REASONING = (
    "Engineering-reviewed candidate selected from the approved search "
    "space for deterministic evaluation."
)


def load_json(
    path: Path | str,
) -> dict:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def canonical_json_line(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def dataset_hash(
    records: list[dict],
) -> str:
    return sha256_text(
        "\n".join(
            canonical_json_line(record)
            for record in records
        )
        + "\n"
    )


def _parts_by_id(
    bom: dict,
) -> dict[str, dict]:
    return {
        part["part_id"]: part
        for part in bom.get("parts", [])
    }


def _option_parts_by_id(
    option_a: dict,
) -> dict[str, dict]:
    return {
        part["part_id"]: part
        for part in option_a.get("parts", [])
    }


def _metadata_base(
    *,
    part_id: str,
    split: str,
    example_type: str,
    change_type: str,
    target: dict,
    source_hash: str,
    search_space_hash: str,
    benchmark_hash: str,
) -> dict:
    return {
        "part_id": part_id,
        "split": split,
        "example_type": example_type,
        "engineering_source": "real_search_space",
        "engineering_verified": True,
        "generated_programmatically": True,
        "change_type": change_type,
        "target": target,
        "source_hash": source_hash,
        "search_space_hash": search_space_hash,
        "benchmark_hash": benchmark_hash,
    }


def _example_id(
    record_without_id: dict,
) -> str:
    digest = sha256_json(
        record_without_id
    )[:16]
    return f"C3_A11_{digest}"


def _positive_assistant(
    *,
    proposal_id: str,
    part_id: str,
    change_type: str,
    target_field: str,
    old_value: str,
    new_value: str,
) -> str:
    return canonical_json_line({
        "proposal_id": proposal_id,
        "part_id": part_id,
        "change_type": change_type,
        "target_field": target_field,
        "old_value": old_value,
        "new_value": new_value,
        "reasoning_summary": CONSERVATIVE_REASONING,
    })


def _negative_assistant(
    *,
    part_id: str,
    rejection_code: str,
    reason: str,
) -> str:
    return canonical_json_line({
        "rejected": True,
        "part_id": part_id,
        "rejection_code": rejection_code,
        "reasoning_summary": reason,
    })


def _with_example_id(
    record: dict,
) -> dict:
    example_id = _example_id(
        record
    )
    record["metadata"] = {
        "example_id": example_id,
        **record["metadata"],
    }
    return record


def _positive_records_for_choice(
    *,
    part: dict,
    option_part: dict,
    split: str,
    change_type: str,
    target_field: str,
    old_value: str,
    new_value: str,
    rationale: str,
    hashes: dict[str, str],
) -> list[dict]:
    records = []
    part_id = part["part_id"]
    part_name = option_part.get(
        "part_name",
        part.get("name", part_id),
    )

    for system_index, system_message in enumerate(
        SYSTEM_MESSAGES
    ):
        for template_index, template in enumerate(
            POSITIVE_USER_TEMPLATES
        ):
            target = {
                "target_field": target_field,
                "new_value": new_value,
                "old_value": old_value,
            }
            metadata = _metadata_base(
                part_id=part_id,
                split=split,
                example_type="positive",
                change_type=change_type,
                target=target,
                **hashes,
            )
            metadata["prompt_variant"] = {
                "system_index": system_index,
                "user_template_index": template_index,
            }
            proposal_id = (
                f"C3_{part_id}_{change_type}_{new_value}_"
                f"s{system_index}_u{template_index}"
            )
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": template.format(
                            part_id=part_id,
                            part_name=part_name,
                            change_type=change_type,
                            target_field=target_field,
                            old_value=old_value,
                            new_value=new_value,
                            rationale=rationale,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": _positive_assistant(
                            proposal_id=proposal_id,
                            part_id=part_id,
                            change_type=change_type,
                            target_field=target_field,
                            old_value=old_value,
                            new_value=new_value,
                        ),
                    },
                ],
                "metadata": metadata,
            }
            records.append(
                _with_example_id(record)
            )

    return records


def _negative_specs(
    option_part: dict,
    baseline_part: dict,
) -> list[dict]:
    part_id = option_part["part_id"]
    specs = []

    for material_id, reason in option_part.get(
        "inadmissible_materials",
        {},
    ).items():
        specs.append({
            "change_type": "material",
            "target_field": "material_id",
            "new_value": material_id,
            "rejection_code": "INADMISSIBLE_MATERIAL",
            "reason": reason,
        })

    for process_id, reason in option_part.get(
        "inadmissible_processes",
        {},
    ).items():
        specs.append({
            "change_type": "process",
            "target_field": "process_id",
            "new_value": process_id,
            "rejection_code": "INADMISSIBLE_PROCESS",
            "reason": reason,
        })

    specs.extend([
        {
            "change_type": "material",
            "target_field": "material_id",
            "new_value": "UNKNOWN_MATERIAL_ID",
            "rejection_code": "UNKNOWN_MATERIAL_ID",
            "reason": "Material ID is not a canonical registry identifier.",
        },
        {
            "change_type": "process",
            "target_field": "process_id",
            "new_value": "UNKNOWN_PROCESS_ID",
            "rejection_code": "UNKNOWN_PROCESS_ID",
            "reason": "Process ID is not a canonical registry identifier.",
        },
        {
            "change_type": "material",
            "target_field": "process_id",
            "new_value": baseline_part["material_id"],
            "rejection_code": "INVALID_TARGET_FIELD",
            "reason": "Material changes must target material_id.",
        },
        {
            "change_type": "process",
            "target_field": "material_id",
            "new_value": baseline_part["process_id"],
            "rejection_code": "INVALID_TARGET_FIELD",
            "reason": "Process changes must target process_id.",
        },
        {
            "change_type": "material",
            "target_field": "material_id",
            "new_value": {
                "material_id": baseline_part["material_id"],
            },
            "rejection_code": "OBJECT_NEW_VALUE",
            "reason": "Material new_value must be a canonical string.",
        },
        {
            "change_type": "process",
            "target_field": "process_id",
            "new_value": {
                "process_id": baseline_part["process_id"],
            },
            "rejection_code": "OBJECT_NEW_VALUE",
            "reason": "Process new_value must be a canonical string.",
        },
    ])

    if part_id in {
        "PILOT_009",
        "PILOT_010",
    }:
        specs.append({
            "change_type": "material",
            "target_field": "material_id",
            "new_value": "PA6_SHEET",
            "rejection_code": "PA6_FORM_CONFLATION",
            "reason": (
                "Do not conflate PA6 sheet stock with PA6 filament "
                "for FDM-printing examples."
            ),
        })

    return specs


def _negative_records_for_spec(
    *,
    part: dict,
    option_part: dict,
    split: str,
    spec: dict,
    hashes: dict[str, str],
) -> list[dict]:
    records = []
    part_id = part["part_id"]
    part_name = option_part.get(
        "part_name",
        part.get("name", part_id),
    )

    for system_index, system_message in enumerate(
        SYSTEM_MESSAGES
    ):
        for template_index, template in enumerate(
            NEGATIVE_USER_TEMPLATES
        ):
            metadata = _metadata_base(
                part_id=part_id,
                split=split,
                example_type="negative",
                change_type=spec["change_type"],
                target={
                    "target_field": spec["target_field"],
                    "new_value": spec["new_value"],
                },
                **hashes,
            )
            metadata["rejection_code"] = spec[
                "rejection_code"
            ]
            metadata["prompt_variant"] = {
                "system_index": system_index,
                "user_template_index": template_index,
            }
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": template.format(
                            part_id=part_id,
                            part_name=part_name,
                            change_type=spec["change_type"],
                            target_field=spec["target_field"],
                            new_value=spec["new_value"],
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": _negative_assistant(
                            part_id=part_id,
                            rejection_code=spec[
                                "rejection_code"
                            ],
                            reason=spec["reason"],
                        ),
                    },
                ],
                "metadata": metadata,
            }
            records.append(
                _with_example_id(record)
            )

    return records


def build_dataset_records(
    *,
    real_search_space_path: Path | str = DEFAULT_REAL_SEARCH_SPACE_PATH,
    benchmark_path: Path | str = DEFAULT_B4_BOM_PATH,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict], dict]:
    option_a = load_json(
        real_search_space_path
    )
    benchmark = load_json(
        benchmark_path
    )
    search_space = load_verified_real_search_space(
        real_search_space_path=real_search_space_path,
        baseline_bom_path=benchmark_path,
    )

    hashes = {
        "source_hash": sha256_json(
            option_a
        ),
        "search_space_hash": sha256_json(
            search_space
        ),
        "benchmark_hash": sha256_json(
            benchmark
        ),
    }

    parts = _parts_by_id(
        benchmark
    )
    option_parts = _option_parts_by_id(
        option_a
    )
    split_by_part_id = grouped_part_split(
        list(parts),
        seed=split_seed,
    )

    records = []

    for part_id in sorted(parts):
        part = parts[part_id]
        option_part = option_parts[part_id]
        split = split_by_part_id[part_id]
        current = option_part.get(
            "current",
            {},
        )

        for material_id, rationale in sorted(
            option_part.get(
                "admissible_materials",
                {},
            ).items()
        ):
            if material_id == current.get("material_id"):
                continue
            records.extend(
                _positive_records_for_choice(
                    part=part,
                    option_part=option_part,
                    split=split,
                    change_type="material",
                    target_field="material_id",
                    old_value=current["material_id"],
                    new_value=material_id,
                    rationale=rationale,
                    hashes=hashes,
                )
            )

        for process_id, rationale in sorted(
            option_part.get(
                "admissible_processes",
                {},
            ).items()
        ):
            if process_id == current.get("process_id"):
                continue
            records.extend(
                _positive_records_for_choice(
                    part=part,
                    option_part=option_part,
                    split=split,
                    change_type="process",
                    target_field="process_id",
                    old_value=current["process_id"],
                    new_value=process_id,
                    rationale=rationale,
                    hashes=hashes,
                )
            )

        for spec in _negative_specs(
            option_part,
            part,
        ):
            records.extend(
                _negative_records_for_spec(
                    part=part,
                    option_part=option_part,
                    split=split,
                    spec=spec,
                    hashes=hashes,
                )
            )

    records = sorted(
        records,
        key=lambda record: record["metadata"]["example_id"],
    )
    manifest = build_manifest(
        records,
        split_by_part_id,
        hashes,
        split_seed=split_seed,
    )
    return records, manifest


def build_manifest(
    records: list[dict],
    split_by_part_id: dict[str, str],
    hashes: dict[str, str],
    split_seed: int,
) -> dict:
    split_counts = Counter(
        record["metadata"]["split"]
        for record in records
    )
    example_type_counts = Counter(
        record["metadata"]["example_type"]
        for record in records
    )
    change_type_counts = Counter(
        record["metadata"]["change_type"]
        for record in records
    )
    by_split_and_type = Counter(
        (
            record["metadata"]["split"],
            record["metadata"]["example_type"],
        )
        for record in records
    )

    return {
        "label": "A11 C3 training data preparation",
        "dataset_description": (
            "Programmatically generated instruction examples derived "
            "from an engineering-reviewed admissible search space."
        ),
        "not_training_run": True,
        "not_c3_experimental_results": True,
        "supported_change_types": [
            "material",
            "process",
        ],
        "split_seed": split_seed,
        "part_ids_by_split": part_ids_by_split(
            split_by_part_id
        ),
        "record_count": len(records),
        "split_counts": dict(
            sorted(split_counts.items())
        ),
        "example_type_counts": dict(
            sorted(example_type_counts.items())
        ),
        "change_type_counts": dict(
            sorted(change_type_counts.items())
        ),
        "split_example_type_counts": {
            f"{split}:{example_type}": count
            for (
                split,
                example_type,
            ), count in sorted(by_split_and_type.items())
        },
        "hashes": {
            **hashes,
            "dataset_hash": dataset_hash(
                records
            ),
        },
        "limitations": [
            (
                "Examples are generated from approved search-space "
                "entries, not independently verified optimal designs."
            ),
            (
                "No global cost, mass, strength, or FSG-compliance "
                "claims are encoded in assistant positives."
            ),
            (
                "Only material and process changes are active for this "
                "initial C3 dataset."
            ),
        ],
    }


def records_by_split(
    records: list[dict],
) -> dict[str, list[dict]]:
    return {
        split: [
            record
            for record in records
            if record["metadata"]["split"] == split
        ]
        for split in [
            "train",
            "validation",
            "test",
        ]
    }


def write_jsonl(
    path: Path,
    records: list[dict],
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                canonical_json_line(record)
                + "\n"
            )


def write_manifest(
    path: Path,
    manifest: dict,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def status_markdown(
    manifest: dict,
) -> str:
    splits = manifest["part_ids_by_split"]
    return f"""# A11 Training Data Status

A11 prepares C3 training-data files only. It does not train a model, install MLX, run C3, or produce C3 experimental results.

## Source

- `data/benchmark/real_search_space.json`
- `data/benchmark/pilot_10_parts_ground_truth.json`
- `schemas/proposal.schema.json`
- `schemas/ollama_proposal_output.schema.json`

The dataset is programmatically generated instruction data derived from an engineering-reviewed admissible search space. It is not a hand-written engineering judgement dataset.

## Methodology

Positive examples use only explicit, non-baseline material/process alternatives approved for the target part in the real search space. Negative examples teach rejection of inadmissible identifiers, unknown identifiers, target-field mistakes, object-valued material/process `new_value` payloads, PA6 sheet/filament conflation, and recorded turning/non-axisymmetric conflicts.

Assistant positives make no numerical cost, mass, strength, or Formula Student compliance claims. They only select an engineering-reviewed candidate for later deterministic evaluation.

## Split

Grouped split by `part_id`, deterministic with split seed `{manifest["split_seed"]}`:

- Train: {", ".join(splits["train"])}
- Validation: {", ".join(splits["validation"])}
- Test: {", ".join(splits["test"])}

No part ID appears in more than one split.

## Counts

- Total examples: {manifest["record_count"]}
- Train examples: {manifest["split_counts"].get("train", 0)}
- Validation examples: {manifest["split_counts"].get("validation", 0)}
- Test examples: {manifest["split_counts"].get("test", 0)}
- Positive examples: {manifest["example_type_counts"].get("positive", 0)}
- Negative examples: {manifest["example_type_counts"].get("negative", 0)}
- Material examples: {manifest["change_type_counts"].get("material", 0)}
- Process examples: {manifest["change_type_counts"].get("process", 0)}

## Hashes

- Source hash: `{manifest["hashes"]["source_hash"]}`
- Converted search-space hash: `{manifest["hashes"]["search_space_hash"]}`
- Benchmark hash: `{manifest["hashes"]["benchmark_hash"]}`
- Dataset hash: `{manifest["hashes"]["dataset_hash"]}`

## Limitations

- Generated examples are not independently verified optimal designs.
- They do not establish global optimality, final cost/mass performance, or H2 conclusions.
- Only material and process changes are represented.
- Interpretive Formula Student rule compliance is not encoded as a deterministic training target.
"""


def write_status_doc(
    path: Path,
    manifest: dict,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            status_markdown(manifest)
        )


def build_and_write_dataset(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    real_search_space_path: Path | str = DEFAULT_REAL_SEARCH_SPACE_PATH,
    benchmark_path: Path | str = DEFAULT_B4_BOM_PATH,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict], dict]:
    records, manifest = build_dataset_records(
        real_search_space_path=real_search_space_path,
        benchmark_path=benchmark_path,
        split_seed=split_seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    by_split = records_by_split(
        records
    )
    write_jsonl(
        output_dir / "c3_train.jsonl",
        by_split["train"],
    )
    write_jsonl(
        output_dir / "c3_validation.jsonl",
        by_split["validation"],
    )
    write_jsonl(
        output_dir / "c3_test.jsonl",
        by_split["test"],
    )
    write_manifest(
        output_dir / "c3_dataset_manifest.json",
        manifest,
    )
    write_status_doc(
        PROJECT_ROOT / "docs" / "A11_TRAINING_DATA_STATUS.md",
        manifest,
    )
    return records, manifest

