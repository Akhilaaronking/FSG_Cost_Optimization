from copy import deepcopy

import pytest

from src.optimization.weighted_sum import (
    normalized_objectives,
    rank_weighted_candidates,
    weighted_score,
)


BASELINE = [312.02, 0.650706]


def test_baseline_normalises_to_one():
    assert normalized_objectives(
        BASELINE,
        BASELINE,
    ) == pytest.approx([1.0, 1.0])


def test_equal_weights_give_baseline_score_one():
    assert weighted_score(
        BASELINE,
        BASELINE,
        0.5,
        0.5,
    ) == pytest.approx(1.0)


def test_lower_cost_with_equal_mass_improves_score():
    assert weighted_score(
        [300.0, BASELINE[1]],
        BASELINE,
        0.5,
        0.5,
    ) < 1.0


def test_lower_mass_with_equal_cost_improves_score():
    assert weighted_score(
        [BASELINE[0], 0.6],
        BASELINE,
        0.5,
        0.5,
    ) < 1.0


def test_negative_weights_rejected():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        weighted_score(
            BASELINE,
            BASELINE,
            -0.1,
            1.1,
        )


def test_weights_must_sum_to_one():
    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        weighted_score(
            BASELINE,
            BASELINE,
            0.6,
            0.5,
        )


def test_candidate_ranking_is_deterministic():
    candidates = [
        {
            "candidate_id": "B",
            "objective_vector": [300.0, 0.7],
        },
        {
            "candidate_id": "A",
            "objective_vector": [300.0, 0.7],
        },
    ]

    ranked = rank_weighted_candidates(
        candidates,
        BASELINE,
        0.5,
        0.5,
    )

    assert [
        item["candidate_id"]
        for item in ranked
    ] == ["A", "B"]


def test_inputs_are_not_mutated():
    candidates = [
        {
            "candidate_id": "A",
            "objective_vector": [300.0, 0.7],
            "metadata": {"tag": "original"},
        }
    ]
    original = deepcopy(candidates)

    rank_weighted_candidates(
        candidates,
        BASELINE,
        0.5,
        0.5,
    )

    assert candidates == original
