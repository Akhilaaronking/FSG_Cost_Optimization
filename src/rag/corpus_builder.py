import csv
import json
from collections import Counter
from pathlib import Path

from src.rag.chunker import (
    chunk_documents,
    deterministic_id,
)
from src.rag.models import (
    RagDocument,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
B5_CONSTRAINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "deterministic_constraints_B5.json"
)
B5_CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rule_classification_B5.json"
)
SOURCE_REGISTER_PATHS = [
    PROJECT_ROOT / "docs" / "source_register.csv",
    PROJECT_ROOT
    / "docs"
    / "b4_engineering_data"
    / "source_register.csv",
]
MARKDOWN_INPUTS = [
    (
        PROJECT_ROOT
        / "docs"
        / "b4_engineering_data"
        / "handoff_B5_to_A.md",
        "handoff_document",
    ),
    (
        PROJECT_ROOT
        / "docs"
        / "b4_engineering_data"
        / "provenance_notes.md",
        "provenance_document",
    ),
]


def _read_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _document_id(
    source_type: str,
    stable_key: str,
) -> str:
    return deterministic_id(
        source_type,
        stable_key,
        prefix="doc",
    )


def _classification_by_rule_id() -> dict:
    data = _read_json(
        B5_CLASSIFICATION_PATH
    )

    return {
        rule["rule_id"]: rule
        for rule in data["rules"]
    }


def _rule_text(
    rule: dict,
    classification: dict | None,
) -> str:
    lines = [
        f"Rule ID: {rule['rule_id']}",
        f"Source type: {rule['rule_category']}",
        f"Affected part/category: {rule.get('affected_part_category', '')}",
        f"Parameter field: {rule.get('parameter_field', '')}",
        f"Operator: {rule.get('operator', '')}",
        f"Limit value: {rule.get('limit_value', '')}",
        f"Units: {rule.get('units', '')}",
        f"FSG reference: {rule.get('fsg_reference', '')}",
        f"Deterministic: {rule.get('deterministic')}",
    ]

    if classification:
        lines.extend([
            f"Rule text: {classification.get('text', '')}",
            f"Classification: {classification.get('classification', '')}",
            f"Classification rationale: {classification.get('rationale', '')}",
        ])

    if rule.get("notes"):
        lines.append(f"Notes: {rule['notes']}")

    return "\n".join(lines)


def build_rule_documents() -> list[RagDocument]:
    data = _read_json(
        B5_CONSTRAINT_PATH
    )
    classifications = _classification_by_rule_id()
    documents = []

    for rule in data["constraints"]:
        rule_id = rule["rule_id"]
        is_quality_gate = (
            rule.get("rule_category")
            == "derived_quality_gate"
        )

        source_type = (
            "derived_quality_gate"
            if is_quality_gate
            else "fsg_rule"
        )

        source_id = (
            None
            if is_quality_gate
            else rule.get("source_id")
        )

        document = RagDocument(
            document_id=_document_id(
                source_type,
                rule_id,
            ),
            text=_rule_text(
                rule,
                classifications.get(rule_id),
            ),
            source_type=source_type,
            source_id=source_id,
            source_reference=rule.get(
                "fsg_reference"
            )
            or None,
            metadata={
                "rule_id": rule_id,
                "rule_category": rule.get(
                    "rule_category"
                ),
                "deterministic": rule.get(
                    "deterministic"
                ),
                "parameter_field": rule.get(
                    "parameter_field"
                ),
                "source_file": str(
                    B5_CONSTRAINT_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
            },
        )
        documents.append(document)

    return documents


def _source_register_path() -> Path | None:
    for path in SOURCE_REGISTER_PATHS:
        if path.exists():
            return path

    return None


def build_source_documents() -> list[RagDocument]:
    path = _source_register_path()

    if path is None:
        return []

    documents = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            source_id = (
                row.get("source_id")
                or ""
            ).strip()

            if not source_id:
                continue

            lines = []

            for field, value in row.items():
                value = (value or "").strip()
                if value:
                    lines.append(
                        f"{field}: {value}"
                    )

            documents.append(
                RagDocument(
                    document_id=_document_id(
                        "source_register",
                        source_id,
                    ),
                    text="\n".join(lines),
                    source_type="source_register",
                    source_id=source_id,
                    source_reference=(
                        row.get("location")
                        or None
                    ),
                    metadata={
                        "source_file": str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "source_type_label": row.get(
                            "source_type"
                        ),
                        "title": row.get(
                            "title"
                        ),
                    },
                )
            )

    return documents


def build_markdown_documents() -> list[RagDocument]:
    documents = []

    for path, source_type in MARKDOWN_INPUTS:
        if not path.exists():
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        documents.append(
            RagDocument(
                document_id=_document_id(
                    source_type,
                    str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                ),
                text=text,
                source_type=source_type,
                source_id=None,
                source_reference=str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                metadata={
                    "source_file": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    )
                },
            )
        )

    return documents


def build_documents() -> list[RagDocument]:
    return (
        build_rule_documents()
        + build_source_documents()
        + build_markdown_documents()
    )


def build_corpus(
    chunk_size_words: int = 180,
    overlap_words: int = 30,
) -> tuple[list[RagDocument], list]:
    documents = build_documents()
    chunks = chunk_documents(
        documents,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    return documents, chunks


def corpus_manifest(
    documents: list[RagDocument],
    chunks: list,
    chunk_size_words: int,
    overlap_words: int,
) -> dict:
    return {
        "corpus_version": "A9.1",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_type_counts": dict(
            Counter(
                document.source_type
                for document in documents
            )
        ),
        "source_id_count": len({
            document.source_id
            for document in documents
            if document.source_id
        }),
        "chunking": {
            "chunk_size_words": chunk_size_words,
            "overlap_words": overlap_words,
        },
        "build_inputs": [
            str(
                B5_CONSTRAINT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                B5_CLASSIFICATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            *[
                str(path.relative_to(PROJECT_ROOT))
                for path in SOURCE_REGISTER_PATHS
                if path.exists()
            ],
            *[
                str(path.relative_to(PROJECT_ROOT))
                for path, _ in MARKDOWN_INPUTS
                if path.exists()
            ],
        ],
        "actual_fsg_pdf_or_text_found": False,
    }
