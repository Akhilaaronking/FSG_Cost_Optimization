import pytest

from src.experiment.c4_select import (
    INTENTS,
    ArchiveGuidedSelector,
    Selection,
    _corner_entry,
)


PARTS = [f"PILOT_{i:03d}" for i in range(1, 11)]


def entry(cid, cost, mass, mods=None):
    return {
        "candidate_id": cid,
        "objective_vector": [cost, mass],
        "modifications": mods or [],
    }


ARCHIVE = [
    entry("c_cheap", 150.0, 0.80),
    entry("c_light", 300.0, 0.55),
    entry("c_mid", 220.0, 0.65),
]


# -- construction --------------------------------------------------


def test_rejects_empty_parts():
    with pytest.raises(ValueError, match="non-empty"):
        ArchiveGuidedSelector([], seed=0)


def test_rejects_bad_explore_after():
    with pytest.raises(ValueError, match="explore_after"):
        ArchiveGuidedSelector(PARTS, seed=0, explore_after=0)


# -- exploit corner alternation ---------------------------------


def test_exploit_alternates_cost_and_mass_by_step_parity():
    sel = ArchiveGuidedSelector(PARTS, seed=1)
    s0 = sel.select(archive_entries=ARCHIVE)
    s1 = sel.select(archive_entries=ARCHIVE)
    s2 = sel.select(archive_entries=ARCHIVE)
    assert s0.intent == "reduce_cost"
    assert s1.intent == "reduce_mass"
    assert s2.intent == "reduce_cost"
    assert "c_cheap" in s0.policy_reason  # cost-min corner
    assert "c_light" in s1.policy_reason  # mass-min corner


def test_selection_shape_is_valid():
    sel = ArchiveGuidedSelector(PARTS, seed=3)
    s = sel.select(archive_entries=ARCHIVE)
    assert isinstance(s, Selection)
    assert s.part_id in PARTS
    assert s.intent in INTENTS
    assert s.policy_reason


def test_exploit_with_empty_archive_uses_baseline_label():
    sel = ArchiveGuidedSelector(PARTS, seed=0)
    s = sel.select(archive_entries=[])
    assert "baseline" in s.policy_reason
    assert s.intent in ("reduce_cost", "reduce_mass")


# -- round-robin coverage -------------------------------------


def test_covers_every_part_once_before_repeating():
    sel = ArchiveGuidedSelector(PARTS, seed=7)
    picked = [
        sel.select(archive_entries=ARCHIVE).part_id
        for _ in range(len(PARTS))
    ]
    assert sorted(picked) == sorted(PARTS)  # each exactly once
    # next pass starts over, still a permutation
    picked2 = [
        sel.select(archive_entries=ARCHIVE).part_id
        for _ in range(len(PARTS))
    ]
    assert sorted(picked2) == sorted(PARTS)


# -- determinism / seed ------------------------------------


def _run(seed, n=25):
    sel = ArchiveGuidedSelector(PARTS, seed=seed)
    out = []
    for _ in range(n):
        s = sel.select(archive_entries=ARCHIVE)
        sel.note_step(s, "non_dominated", accepted=True)
        out.append((s.part_id, s.intent))
    return out


def test_same_seed_same_trajectory():
    assert _run(2) == _run(2)


def test_different_seed_different_part_order():
    assert _run(0) != _run(1)


# -- explore trigger -------------------------------------


def test_explore_fires_after_explore_after_non_improving_steps():
    sel = ArchiveGuidedSelector(PARTS, seed=1, explore_after=3)
    intents = []
    for _ in range(6):
        s = sel.select(archive_entries=ARCHIVE)
        sel.note_step(s, "dominated", accepted=False)  # never improving
        intents.append(s.intent)
    # first 3 are exploit, then explore kicks in
    assert intents[0] in ("reduce_cost", "reduce_mass")
    assert "diversify" in intents[3:]


def test_pareto_improving_resets_the_explore_counter():
    sel = ArchiveGuidedSelector(PARTS, seed=1, explore_after=3)
    for i in range(5):
        s = sel.select(archive_entries=ARCHIVE)
        status = "pareto_improving" if i == 2 else "dominated"
        sel.note_step(s, status, accepted=(i == 2))
        # counter reset at i==2 -> never reaches explore_after within 5
        assert s.intent != "diversify"


# -- repair override -----------------------------------


def test_violation_forces_fix_violation_on_that_part():
    sel = ArchiveGuidedSelector(PARTS, seed=4)
    last_eval = {
        "constraints": {
            "proposal_level_violation": True,
            "errors": ["rule S_3.4.9 failed on PILOT_006"],
        },
        "modified_part_id": "PILOT_002",
    }
    s = sel.select(archive_entries=ARCHIVE, last_evaluation=last_eval)
    assert s.intent == "fix_violation"
    assert s.part_id == "PILOT_006"  # from the error text, not modified_part_id
    assert "repair" in s.policy_reason


def test_violation_without_error_text_falls_back_to_modified_part():
    sel = ArchiveGuidedSelector(PARTS, seed=4)
    last_eval = {
        "constraints": {"proposal_level_violation": True, "errors": []},
        "modified_part_id": "PILOT_002",
    }
    s = sel.select(archive_entries=ARCHIVE, last_evaluation=last_eval)
    assert s.intent == "fix_violation"
    assert s.part_id == "PILOT_002"


def test_not_evaluated_constraints_do_not_trigger_repair():
    sel = ArchiveGuidedSelector(PARTS, seed=4)
    last_eval = {
        "constraints": {"status": "NOT_EVALUATED", "proposal_level_violation": None},
        "modified_part_id": "PILOT_002",
    }
    s = sel.select(archive_entries=ARCHIVE, last_evaluation=last_eval)
    assert s.intent != "fix_violation"


# -- helper --------------------------------------------


def test_corner_entry_picks_min_and_breaks_ties():
    e = _corner_entry(ARCHIVE, "cost")
    assert e["candidate_id"] == "c_cheap"
    e = _corner_entry(ARCHIVE, "mass")
    assert e["candidate_id"] == "c_light"
    assert _corner_entry([], "cost") is None
