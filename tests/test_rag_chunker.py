from src.rag.chunker import (
    chunk_document,
)
from src.rag.models import (
    RagDocument,
)


def document(text):
    return RagDocument(
        document_id="doc_TEST",
        text=text,
        source_type="fsg_rule",
        source_id="TEST_SOURCE",
        source_reference="TEST_REF",
        metadata={"rule_id": "TEST_RULE"},
    )


def test_short_rule_remains_one_chunk():
    chunks = chunk_document(
        document("short rule text")
    )

    assert len(chunks) == 1


def test_long_text_chunks_with_overlap():
    text = " ".join(
        f"word{i}"
        for i in range(12)
    )
    chunks = chunk_document(
        document(text),
        chunk_size_words=5,
        overlap_words=2,
    )

    assert len(chunks) > 1
    assert "word3" in chunks[1].text


def test_chunk_ids_are_deterministic():
    first = chunk_document(
        document(" ".join(["x"] * 20)),
        chunk_size_words=6,
        overlap_words=1,
    )
    second = chunk_document(
        document(" ".join(["x"] * 20)),
        chunk_size_words=6,
        overlap_words=1,
    )

    assert [
        chunk.chunk_id
        for chunk in first
    ] == [
        chunk.chunk_id
        for chunk in second
    ]


def test_no_empty_chunks():
    chunks = chunk_document(
        document(" ".join(["x"] * 20)),
        chunk_size_words=6,
        overlap_words=1,
    )

    assert all(
        chunk.text
        for chunk in chunks
    )
