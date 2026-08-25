from src.rag.corpus_builder import (
    build_documents,
)


def test_corpus_contains_actual_fsg_b5_rules():
    documents = build_documents()
    rule_ids = {
        document.metadata.get("rule_id")
        for document in documents
        if document.source_type == "fsg_rule"
    }

    assert "S_3.5.12" in rule_ids


def test_s_3_5_11_remains_interpretive():
    document = next(
        document
        for document in build_documents()
        if document.metadata.get("rule_id")
        == "S_3.5.11"
    )

    assert document.source_type == "fsg_rule"
    assert (
        document.metadata["rule_category"]
        == "interpretive"
    )
    assert "as realistic as possible" in document.text


def test_derived_qg_remains_quality_gate():
    document = next(
        document
        for document in build_documents()
        if document.metadata.get("rule_id")
        == "DERIVED_QG_001"
    )

    assert document.source_type == "derived_quality_gate"
    assert document.source_id is None


def test_provenance_metadata_preserved():
    document = next(
        document
        for document in build_documents()
        if document.metadata.get("rule_id")
        == "S_3.5.12"
    )

    assert document.source_id == "FSG_RULES_2026"
    assert document.source_reference == "S 3.5.12, p.115"
    assert document.metadata["source_file"]


def test_document_ids_are_deterministic():
    first = [
        document.document_id
        for document in build_documents()
    ]
    second = [
        document.document_id
        for document in build_documents()
    ]

    assert first == second
