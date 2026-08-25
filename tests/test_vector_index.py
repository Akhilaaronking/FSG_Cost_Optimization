import numpy as np
import pytest

from src.rag.models import RagChunk
from src.rag.vector_index import VectorIndex


def chunk(
    chunk_id,
    source_type="fsg_rule",
    rule_category="hard",
):
    return RagChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=chunk_id,
        source_type=source_type,
        source_id="TEST_SOURCE",
        source_reference="TEST_REF",
        metadata={
            "rule_category": rule_category
        },
    )


def test_highest_cosine_similarity_ranks_first():
    index = VectorIndex().build(
        [chunk("A"), chunk("B")],
        np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    results = index.search(
        np.array([0.9, 0.1], dtype=np.float32)
    )

    assert results[0].chunk.chunk_id == "A"


def test_top_k_respected():
    index = VectorIndex().build(
        [chunk("A"), chunk("B")],
        np.eye(2, dtype=np.float32),
    )

    assert len(
        index.search(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=1,
        )
    ) == 1


def test_filtering_works():
    index = VectorIndex().build(
        [
            chunk("A", rule_category="hard"),
            chunk("B", rule_category="interpretive"),
        ],
        np.eye(2, dtype=np.float32),
    )

    results = index.search(
        np.array([1.0, 0.0], dtype=np.float32),
        filters={"rule_category": "interpretive"},
    )

    assert results[0].chunk.chunk_id == "B"


def test_dimension_mismatch_rejected():
    index = VectorIndex().build(
        [chunk("A")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        index.search(
            np.array([1.0], dtype=np.float32)
        )


def test_duplicate_scores_tie_break_deterministically():
    index = VectorIndex().build(
        [chunk("B"), chunk("A")],
        np.array(
            [[1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    results = index.search(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    assert [
        result.chunk.chunk_id
        for result in results
    ] == ["A", "B"]
