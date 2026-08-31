"""
Equal-budget ledger for the A12 harness (thesis 11.6).

The budget for every condition is the number of *fresh* deterministic
objective evaluations: calls to the frozen evaluator that actually run.
It does NOT count

  - proposals that fail the validity funnel before evaluation
    (they are proposal-generation statistics, 11.7), or
  - repeat candidates served from the per-run cache.

Wall-clock time and token usage are recorded elsewhere and never gate a
run. Both drivers -- GenerativeDriver for C1/C2/C3 and Nsga2Driver for
C5 -- share one ledger instance, so "same N" means the same thing in
both (11.6 equal-budget principle; 11.5 pairing).
"""

from dataclasses import dataclass

from src.llm.prompt_builder import sha256_json


def canonical_bom_hash(bom: dict) -> str:
    """
    Stable content hash of a candidate BOM, used as the objective-eval
    cache key and reported as ``evaluation.bom_hash`` in events.jsonl.
    Independent of key order.
    """
    return "sha256:" + sha256_json(bom)


@dataclass(frozen=True)
class ConsumeOutcome:
    """Result of offering one candidate BOM to the ledger."""

    result: dict
    bom_hash: str
    cache_hit: bool
    consumed_budget: bool


class EqualBudgetLedger:
    """
    Single source of truth for how much of the equal budget a run has
    spent. Evaluator-agnostic: it calls whatever callable it is given
    (``evaluate_bom`` for the generative path, the search-space-guarded
    evaluator for C5).
    """

    def __init__(
        self,
        evaluator,
        *,
        n_eval: int,
        proposal_attempt_cap: int | None = None,
    ):
        if n_eval < 1:
            raise ValueError("n_eval must be >= 1")
        if (
            proposal_attempt_cap is not None
            and proposal_attempt_cap < 1
        ):
            raise ValueError(
                "proposal_attempt_cap must be >= 1 or None"
            )

        self._evaluator = evaluator
        self.n_eval = int(n_eval)
        self.proposal_attempt_cap = proposal_attempt_cap

        self.objective_evaluations = 0
        self.proposal_attempts = 0
        self.objective_eval_cache_hits = 0

        self._cache: dict[str, dict] = {}

    # -- generation-side accounting (C1/C2/C3) --------------------------

    def record_proposal_attempt(self) -> int:
        """
        Count one generation attempt, valid or not (11.6 ``N_prop``).
        Returns the running attempt count.
        """
        self.proposal_attempts += 1
        return self.proposal_attempts

    # -- evaluation-side accounting (all conditions) ------------------

    def consume(self, candidate_bom: dict) -> ConsumeOutcome:
        """
        Offer one candidate BOM for deterministic objective evaluation.

        Cache hit  -> returns the cached result, no budget consumed.
        Cache miss -> runs the evaluator once, caches it, consumes one
                      unit of budget.
        """
        key = canonical_bom_hash(candidate_bom)

        cached = self._cache.get(key)
        if cached is not None:
            self.objective_eval_cache_hits += 1
            return ConsumeOutcome(
                result=cached,
                bom_hash=key,
                cache_hit=True,
                consumed_budget=False,
            )

        result = self._evaluator(candidate_bom)
        self._cache[key] = result
        self.objective_evaluations += 1
        return ConsumeOutcome(
            result=result,
            bom_hash=key,
            cache_hit=False,
            consumed_budget=True,
        )

    # -- predicates the drivers loop on ------------------------------

    @property
    def budget_exhausted(self) -> bool:
        """True once ``n_eval`` fresh objective evaluations have run."""
        return self.objective_evaluations >= self.n_eval

    @property
    def attempts_exhausted(self) -> bool:
        """
        True once the generative attempt cap is hit. Always False when
        no cap is set (C5 has none).
        """
        return (
            self.proposal_attempt_cap is not None
            and self.proposal_attempts >= self.proposal_attempt_cap
        )

    @property
    def distinct_candidates_evaluated(self) -> int:
        """Number of unique candidate BOMs that consumed budget."""
        return len(self._cache)

    def has_evaluated(self, candidate_bom: dict) -> bool:
        """True if this exact candidate is already in the run cache."""
        return canonical_bom_hash(candidate_bom) in self._cache

    # -- reporting -------------------------------------------------

    def snapshot(self) -> dict:
        """The ``budget`` block for metrics.json (docs/A12 section 5)."""
        return {
            "n_eval_target": self.n_eval,
            "n_eval_consumed": self.objective_evaluations,
            "proposal_attempts": self.proposal_attempts,
            "objective_eval_cache_hits": self.objective_eval_cache_hits,
        }

    def assert_consistent_with(self, external_evaluation_count: int) -> None:
        """
        Cross-check against a driver that keeps its own counter (the C5
        NSGA-II loop reports ``evaluation_count``). Raises on drift so a
        logging bug cannot silently corrupt the equal-budget claim.
        """
        if self.objective_evaluations != external_evaluation_count:
            raise AssertionError(
                "equal-budget ledger drift: ledger counted "
                f"{self.objective_evaluations} fresh evaluations, "
                f"driver reported {external_evaluation_count}"
            )
