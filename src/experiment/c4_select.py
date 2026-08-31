"""
C4 selection policy ``archive_guided_v1`` (docs/A13 section 5, decision A).

Deterministic given the run seed. ``Select(A_t, history)`` -> a
:class:`Selection` ``{part_id, intent}`` with
``intent in {reduce_cost, reduce_mass, fix_violation, diversify}`` (thesis
eq 10.30):

  repair   -- the previous step's deterministic evaluation had a
              constraint violation -> target that part, ``fix_violation``
              (overrides everything else).
  explore  -- no ``pareto_improving`` archive event for ``explore_after``
              steps -> ``diversify`` an unseen part.
  exploit  -- otherwise: push the current cost-min / mass-min corner of
              the archive (alternating by step parity) on an unseen part,
              ``reduce_cost`` / ``reduce_mass``.

"Unseen this pass" walks every target part once (in a seed-shuffled
order) before repeating.

Note: on the frozen B4 benchmark the evaluator runs with
``evaluate_constraints=False`` (matching C1/C2/C3/C5), so
``proposal_level_violation`` is never True and the *repair* branch stays
dormant. The code path is kept for when constraint fields become
available.
"""

import random
from dataclasses import dataclass


INTENTS = ("reduce_cost", "reduce_mass", "fix_violation", "diversify")


@dataclass(frozen=True)
class Selection:
    part_id: str
    intent: str
    policy_reason: str


def _corner_entry(archive_entries: list, corner: str):
    """The archive member with the smallest cost (or mass), ties broken
    by the other objective then candidate_id. None if the archive is
    empty apart from the baseline."""
    if not archive_entries:
        return None
    primary = 0 if corner == "cost" else 1
    other = 1 - primary
    return min(
        archive_entries,
        key=lambda e: (
            e["objective_vector"][primary],
            e["objective_vector"][other],
            e.get("candidate_id", ""),
        ),
    )


def _violation_part(last_evaluation: dict | None, target_part_ids: list) -> str | None:
    if not last_evaluation:
        return None
    constraints = last_evaluation.get("constraints") or {}
    if not constraints.get("proposal_level_violation"):
        return None
    for err in constraints.get("errors") or []:
        for part_id in target_part_ids:
            if part_id in str(err):
                return part_id
    return last_evaluation.get("modified_part_id")


class ArchiveGuidedSelector:
    def __init__(
        self,
        target_part_ids: list,
        *,
        seed: int,
        explore_after: int = 10,
    ):
        if not target_part_ids:
            raise ValueError("target_part_ids must be non-empty")
        if explore_after < 1:
            raise ValueError("explore_after must be >= 1")

        self.explore_after = explore_after
        self._order = list(target_part_ids)
        random.Random(seed).shuffle(self._order)

        self._cursor = 0
        self._pass_selected: set[str] = set()
        self._step = 0
        self._since_improve = 0

    # -- the driver calls this after each step's archive outcome is known --

    def note_step(
        self,
        selection: Selection,
        archive_status: str | None,
        accepted: bool,
    ) -> None:
        if archive_status == "pareto_improving":
            self._since_improve = 0
        else:
            self._since_improve += 1

    # -- selection --

    def _next_unseen(self) -> str:
        if len(self._pass_selected) >= len(self._order):
            self._pass_selected.clear()
        for _ in range(len(self._order)):
            part = self._order[self._cursor % len(self._order)]
            self._cursor += 1
            if part not in self._pass_selected:
                self._pass_selected.add(part)
                return part
        # unreachable given the reset above, but stay total
        self._pass_selected.clear()
        part = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        self._pass_selected.add(part)
        return part

    def select(
        self,
        *,
        archive_entries: list,
        last_evaluation: dict | None = None,
    ) -> Selection:
        step = self._step
        self._step += 1

        # 1. repair
        viol_part = _violation_part(last_evaluation, self._order)
        if viol_part is not None:
            return Selection(
                viol_part,
                "fix_violation",
                f"repair: deterministic violation on {viol_part} "
                f"at step {step - 1}",
            )

        # 2. explore
        if self._since_improve >= self.explore_after:
            part = self._next_unseen()
            return Selection(
                part,
                "diversify",
                f"explore: {self._since_improve} steps without a "
                f"pareto-improving move",
            )

        # 3. exploit
        corner = "cost" if step % 2 == 0 else "mass"
        entry = _corner_entry(archive_entries, corner)
        part = self._next_unseen()
        intent = "reduce_cost" if corner == "cost" else "reduce_mass"
        candidate = entry.get("candidate_id", "?") if entry else "baseline"
        return Selection(
            part,
            intent,
            f"exploit: {corner}-min corner ({candidate}), "
            f"part unseen this pass",
        )
