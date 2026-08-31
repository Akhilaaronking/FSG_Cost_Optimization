import pytest

from src.experiment.ledger import (
    EqualBudgetLedger,
    canonical_bom_hash,
)


def bom(material="AL_6061_T6", process="CNC_MILLING"):
    return {
        "parts": [
            {
                "part_id": "PILOT_001",
                "material_id": material,
                "process_id": process,
            }
        ]
    }


class CountingEvaluator:
    """Records every call; returns a distinct marker per BOM."""

    def __init__(self):
        self.calls = []

    def __call__(self, candidate_bom):
        self.calls.append(candidate_bom)
        part = candidate_bom["parts"][0]
        return {
            "objective_vector": [len(self.calls), 0.5],
            "marker": (part["material_id"], part["process_id"]),
        }


def ledger(n_eval=5, cap=None):
    return EqualBudgetLedger(
        CountingEvaluator(),
        n_eval=n_eval,
        proposal_attempt_cap=cap,
    )


# -- canonical hash --------------------------------------------------


def test_canonical_hash_is_key_order_independent():
    a = {"parts": [{"part_id": "P1", "material_id": "M", "process_id": "Q"}]}
    b = {"parts": [{"process_id": "Q", "part_id": "P1", "material_id": "M"}]}
    assert canonical_bom_hash(a) == canonical_bom_hash(b)


def test_canonical_hash_distinguishes_different_boms():
    assert canonical_bom_hash(bom("AL_6061_T6")) != canonical_bom_hash(
        bom("AL_7075_T6")
    )


# -- construction validation --------------------------------------


def test_rejects_non_positive_n_eval():
    with pytest.raises(ValueError, match="n_eval"):
        EqualBudgetLedger(CountingEvaluator(), n_eval=0)


def test_rejects_non_positive_attempt_cap():
    with pytest.raises(ValueError, match="proposal_attempt_cap"):
        EqualBudgetLedger(
            CountingEvaluator(), n_eval=5, proposal_attempt_cap=0
        )


# -- fresh evaluation vs cache hit -------------------------------


def test_fresh_evaluation_consumes_budget():
    led = ledger()
    outcome = led.consume(bom("AL_7075_T6"))

    assert outcome.cache_hit is False
    assert outcome.consumed_budget is True
    assert outcome.bom_hash.startswith("sha256:")
    assert led.objective_evaluations == 1
    assert led.objective_eval_cache_hits == 0
    assert len(led._evaluator.calls) == 1


def test_repeat_candidate_is_a_cache_hit_and_does_not_consume_budget():
    led = ledger()
    first = led.consume(bom("AL_7075_T6"))
    second = led.consume(bom("AL_7075_T6"))

    assert second.cache_hit is True
    assert second.consumed_budget is False
    assert second.result == first.result
    assert led.objective_evaluations == 1
    assert led.objective_eval_cache_hits == 1
    assert len(led._evaluator.calls) == 1  # evaluator not called again


def test_distinct_candidates_each_consume_one_unit():
    led = ledger()
    led.consume(bom("AL_6061_T6", "CNC_MILLING"))
    led.consume(bom("AL_7075_T6", "CNC_MILLING"))
    led.consume(bom("AL_6061_T6", "TIG_WELDING"))

    assert led.objective_evaluations == 3
    assert led.distinct_candidates_evaluated == 3


def test_budget_invariant_sum_of_consumed_flags_equals_objective_evaluations():
    led = ledger(n_eval=10)
    outcomes = [
        led.consume(bom("AL_6061_T6")),
        led.consume(bom("AL_7075_T6")),
        led.consume(bom("AL_6061_T6")),  # cache hit
        led.consume(bom("STEEL_S235")),
        led.consume(bom("AL_7075_T6")),  # cache hit
    ]
    consumed = sum(1 for o in outcomes if o.consumed_budget)
    assert consumed == led.objective_evaluations == 3


# -- proposal-attempt accounting -------------------------------


def test_record_proposal_attempt_increments():
    led = ledger()
    assert led.record_proposal_attempt() == 1
    assert led.record_proposal_attempt() == 2
    assert led.proposal_attempts == 2


# -- exhaustion predicates ------------------------------------


def test_budget_exhausted_flips_at_n_eval():
    led = ledger(n_eval=2)
    assert led.budget_exhausted is False
    led.consume(bom("A"))
    assert led.budget_exhausted is False
    led.consume(bom("B"))
    assert led.budget_exhausted is True


def test_cache_hits_do_not_advance_budget_exhaustion():
    led = ledger(n_eval=2)
    led.consume(bom("A"))
    for _ in range(5):
        led.consume(bom("A"))  # all cache hits
    assert led.budget_exhausted is False
    assert led.objective_eval_cache_hits == 5


def test_attempts_exhausted_respects_cap():
    led = ledger(n_eval=99, cap=3)
    assert led.attempts_exhausted is False
    led.record_proposal_attempt()
    led.record_proposal_attempt()
    assert led.attempts_exhausted is False
    led.record_proposal_attempt()
    assert led.attempts_exhausted is True


def test_attempts_never_exhausted_without_a_cap():
    led = ledger(n_eval=99, cap=None)
    for _ in range(1000):
        led.record_proposal_attempt()
    assert led.attempts_exhausted is False


# -- reporting / cross-check --------------------------------


def test_snapshot_shape_matches_metrics_budget_block():
    led = ledger(n_eval=7)
    led.record_proposal_attempt()
    led.consume(bom("A"))
    led.consume(bom("A"))  # cache hit

    assert led.snapshot() == {
        "n_eval_target": 7,
        "n_eval_consumed": 1,
        "proposal_attempts": 1,
        "objective_eval_cache_hits": 1,
    }


def test_has_evaluated_reflects_cache():
    led = ledger()
    assert led.has_evaluated(bom("A")) is False
    led.consume(bom("A"))
    assert led.has_evaluated(bom("A")) is True


def test_assert_consistent_with_passes_when_counts_match():
    led = ledger(n_eval=5)
    led.consume(bom("A"))
    led.consume(bom("B"))
    led.assert_consistent_with(2)  # no raise


def test_assert_consistent_with_raises_on_drift():
    led = ledger(n_eval=5)
    led.consume(bom("A"))
    with pytest.raises(AssertionError, match="drift"):
        led.assert_consistent_with(4)
