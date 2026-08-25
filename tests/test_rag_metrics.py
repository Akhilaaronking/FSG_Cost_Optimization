import pytest

from src.rag.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_calculation():
    assert recall_at_k(
        ["A", "B"],
        ["B", "C"],
        2,
    ) == 0.5


def test_precision_at_k_calculation():
    assert precision_at_k(
        ["A", "B"],
        ["B"],
        2,
    ) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(
        ["A", "B", "C"],
        ["C"],
    ) == pytest.approx(1 / 3)


def test_mrr():
    assert mean_reciprocal_rank([
        {
            "retrieved_ids": ["A", "B"],
            "relevant_ids": ["B"],
        },
        {
            "retrieved_ids": ["C", "D"],
            "relevant_ids": ["C"],
        },
    ]) == pytest.approx(0.75)


def test_empty_relevant_set_handled_explicitly():
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        recall_at_k(
            ["A"],
            [],
            1,
        )
