from src.rag.context_formatter import (
    format_retrieval_context,
)
from src.rag.models import (
    RagChunk,
    RetrievalResult,
)


def result():
    chunk = RagChunk(
        chunk_id="chunk_TEST",
        document_id="doc_TEST",
        text="Evidence text stays intact.",
        source_type="fsg_rule",
        source_id="FSG_RULES_2026",
        source_reference="S 3.5.12, p.115",
        metadata={},
    )
    return RetrievalResult(
        chunk=chunk,
        score=0.9,
        rank=1,
    )


def test_source_id_visible():
    context = format_retrieval_context(
        [result()]
    )

    assert "[SOURCE_ID] FSG_RULES_2026" in context


def test_source_reference_visible():
    context = format_retrieval_context(
        [result()]
    )

    assert "[SOURCE_REFERENCE] S 3.5.12, p.115" in context


def test_ranking_visible():
    context = format_retrieval_context(
        [result()]
    )

    assert "[RANK] 1" in context


def test_text_preserved():
    context = format_retrieval_context(
        [result()]
    )

    assert "Evidence text stays intact." in context
