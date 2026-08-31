import json
from pathlib import Path

import pytest

from src.data.registry import DataRegistry
from src.experiment.c4_driver import C4Driver
from src.experiment.events import EventLog, read_events
from src.llm.generator import ProposalGenerator


PILOT_BOM_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"


def baseline():
    # full 10-part BOM: fake_evaluator's numbers (BASE_C/BASE_M + deltas)
    # must be consistent with evaluate_bom(baseline) and the 1.2x
    # reference point the driver computes.
    return json.loads(Path(PILOT_BOM_PATH).read_text())


def c4_run_config(seed=0, *, n_eval=3, L=3, eps=0.1, K=3, ablation=None,
                  parts=None):
    parts = parts or [f"PILOT_{i:03d}" for i in range(1, 11)]
    return {
        "run_id": f"sha256:c4{seed}{ablation}",
        "condition": "C4_base",
        "seed": seed,
        "identity": {"budget": {"n_eval": n_eval}},
        "condition_spec": {
            "driver": "C4Driver",
            "target_parts": parts,
            "c4_loop": {
                "n_eval": n_eval,
                "convergence": {"look_back_L": L, "epsilon_hv": eps,
                                "variant": "delta"},
                "retry_cap_K": K,
                "proposal_attempt_cap": 200,
                "select_policy": "archive_guided_v1",
                "feedback_mode": "prev_eval+archive+rejection",
                "ablation": ablation,
            },
        },
    }


class CyclingBackend:
    backend_name = "stub"
    model_name = "cycling-stub"

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def generate(self, prompt, *, seed=None, temperature=0.2, max_tokens=512):
        r = self.responses[self.index % len(self.responses)]
        self.index += 1
        return r


def _proposal(part_id, material, schema_ok=True):
    obj = {
        "part_id": part_id,
        "change_type": "material",
        "target_field": "material_id",
        "old_value": "AL_6061_T6",
        "new_value": material,
    }
    if schema_ok:
        obj["proposal_id"] = f"C4_{part_id}_{material}"
    return json.dumps(obj)


# per-(part, material) objective delta vs the frozen baseline.
# baseline materials: 001 AL_6061_T6, 002 AL_7075_T6, 003 STEEL_S235,
# 004 STEEL_4130_CRMO, 005 POM_DELRIN -- so every key below is a real swap.
PART_DELTAS = {
    ("PILOT_001", "AL_7075_T6"): (-62.0, -0.05),        # big improvement
    ("PILOT_002", "STEEL_S235"): (+30.0, +0.10),        # worse on both
    ("PILOT_003", "AL_6061_T6"): (+30.0, +0.10),
    ("PILOT_004", "AL_6061_T6"): (+30.0, +0.10),
    ("PILOT_005", "AL_6061_T6"): (+30.0, +0.10),
    ("PILOT_002", "STEEL_4130_CRMO"): (-15.0, +0.03),   # non-dominated vs P1
    ("PILOT_003", "STEEL_4130_CRMO"): (+5.0, -0.02),    # non-dominated vs P1
}
BASE_C, BASE_M = 312.02, 0.6507108


def fake_evaluator(bom):
    cost, mass = BASE_C, BASE_M
    for part in bom["parts"]:
        d = PART_DELTAS.get((part["part_id"], part["material_id"]))
        if d:
            cost += d[0]
            mass += d[1]
    return {
        "objective_vector": [cost, mass],
        "objectives": {"cost_eur": cost, "mass_kg": mass},
        "constraints": {"status": "NOT_EVALUATED", "feasible": None},
    }


def driver(responses, cfg, *, retriever=None):
    return C4Driver(
        cfg,
        generator=ProposalGenerator(
            CyclingBackend(responses), registry=DataRegistry()
        ),
        baseline_bom=baseline(),
        retriever=retriever,
        evaluator=fake_evaluator,
    )


def run(responses, cfg, tmp_path, *, retriever=None):
    d = driver(responses, cfg, retriever=retriever)
    log = EventLog(
        tmp_path / "events.jsonl",
        run_id=cfg["run_id"],
        condition=cfg["condition"],
        seed=cfg["seed"],
    )
    outcome = d.run(log, pareto_archive_path=tmp_path / "pareto_archive.json")
    log.close()
    return outcome, read_events(tmp_path / "events.jsonl")


# ----------------------------------------------------------------------


GOOD_TRIO = [
    _proposal("PILOT_001", "AL_7075_T6"),
    _proposal("PILOT_002", "STEEL_4130_CRMO"),
    _proposal("PILOT_003", "STEEL_4130_CRMO"),
]


def test_state_compounds_across_accepted_steps(tmp_path):
    outcome, events = run(GOOD_TRIO, c4_run_config(n_eval=3), tmp_path)

    assert outcome.terminal_status == "COMPLETE"
    assert outcome.ledger["n_eval_consumed"] == 3
    # the final archive contains a candidate whose modifications span >1 part
    payload = json.loads((tmp_path / "pareto_archive.json").read_text())
    max_parts = max(
        len({m["part_id"] for m in e["modifications"]})
        for e in payload["entries"]
    )
    assert max_parts >= 2
    # working_state_hash advanced on accepted steps
    accepted = [e for e in events if e["agentic"]["accepted"]]
    assert len(accepted) >= 2
    assert accepted[0]["agentic"]["working_state_hash_after"] != accepted[0][
        "agentic"
    ]["working_state_hash_before"]


def test_budget_invariant(tmp_path):
    outcome, events = run(GOOD_TRIO, c4_run_config(n_eval=3), tmp_path)
    fresh = sum(
        1
        for e in events
        if (e.get("evaluation") or {}).get("consumed_objective_budget")
    )
    assert fresh == outcome.ledger["n_eval_consumed"] == 3


def test_events_are_agentic_steps_with_the_agentic_block(tmp_path):
    responses = [_proposal("PILOT_001", "AL_7075_T6")]
    _, events = run(responses, c4_run_config(n_eval=1), tmp_path)
    e = events[0]
    assert e["event_type"] == "agentic_step"
    a = e["agentic"]
    assert set(a) >= {
        "step_index",
        "selection",
        "working_state_hash_before",
        "working_state_hash_after",
        "accepted",
        "retry_of_selection",
        "hv_after",
        "delta_hv_recent",
    }
    assert a["selection"]["intent"] in (
        "reduce_cost",
        "reduce_mass",
        "fix_violation",
        "diversify",
    )


def test_convergence_early_stop(tmp_path):
    # P1 improves; then distinct BOMs all dominated by P1 -> HV plateaus
    responses = [
        _proposal("PILOT_001", "AL_7075_T6"),
        _proposal("PILOT_002", "STEEL_S235"),
        _proposal("PILOT_003", "AL_6061_T6"),
        _proposal("PILOT_004", "AL_6061_T6"),
        _proposal("PILOT_005", "AL_6061_T6"),
    ]
    cfg = c4_run_config(n_eval=20, L=3, eps=5.0)
    outcome, events = run(responses, cfg, tmp_path)

    assert outcome.terminal_status == "COMPLETE_CONVERGED"
    assert outcome.ledger["n_eval_consumed"] < 20
    assert outcome.extra["c4"]["converged"] is True
    assert outcome.extra["c4"]["stop_rule"] == "convergence"
    assert outcome.extra["c4"]["convergence_reason"] == "hv_plateau"


def test_converged_no_longer_gated_on_positive_hv(tmp_path):
    # regression: a working state with HV pinned at 0 for L evaluations
    # used to be unable to converge (old `hv[-1] <= 0.0` guard), so the
    # loop ground on to the proposal-attempt cap. It must now converge.
    d = driver(GOOD_TRIO, c4_run_config(L=3, eps=0.1))
    ids = [frozenset({"a"})]
    assert d._converged([0.0, 0.0, 0.0], ids * 3) == (False, None)  # len <= L
    assert d._converged([0.0, 0.0, 0.0, 0.0], ids * 4) == (True, "hv_plateau")
    assert d._converged([5.0, 5.0, 5.0, 5.0], ids * 4) == (True, "hv_plateau")


def test_converged_archive_unchanged_branch(tmp_path):
    d = driver(GOOD_TRIO, c4_run_config(L=3, eps=0.1))
    # HV number swings above epsilon (no plateau) but the archive
    # membership has not moved for L evaluations -> stalled search.
    same = [frozenset({"a", "b"})] * 4
    assert d._converged([0.0, 10.0, 0.0, 10.0], same) == (
        True,
        "archive_unchanged",
    )
    # genuinely still exploring: HV climbing, archive growing each eval
    growing = [frozenset({"a"}), frozenset({"a", "b"}), frozenset({"a", "b", "c"}),
               frozenset({"a", "b", "c", "d"})]
    assert d._converged([1.0, 3.0, 6.0, 10.0], growing) == (False, None)


def test_retry_cap_then_selector_advances(tmp_path):
    # P1 improves (accepted); P2/P3/P4 are distinct dominated -> each
    # selection is retried K times then the selector moves on.
    responses = [
        _proposal("PILOT_001", "AL_7075_T6"),
        _proposal("PILOT_002", "STEEL_S235"),
        _proposal("PILOT_003", "AL_6061_T6"),
        _proposal("PILOT_004", "AL_6061_T6"),
        _proposal("PILOT_005", "AL_6061_T6"),
    ]
    cfg = c4_run_config(n_eval=8, K=3, L=50, eps=0.0)
    outcome, events = run(responses, cfg, tmp_path)

    accepted = [e for e in events if e["agentic"]["accepted"]]
    assert len(accepted) == 1  # only the first improving proposal
    retries = {e["agentic"]["retry_of_selection"] for e in events}
    assert retries & {1, 2}
    assert outcome.extra["c4"]["mean_retries_per_selection"] >= 1


def test_no_rag_ablation_skips_retrieval(tmp_path):
    class BoomRetriever:
        def retrieve(self, *a, **k):
            raise AssertionError("retriever must not be called for no_rag")

    responses = [_proposal("PILOT_001", "AL_7075_T6")]
    cfg = c4_run_config(n_eval=1, ablation="no_rag")
    _, events = run(responses, cfg, tmp_path, retriever=BoomRetriever())
    assert events[0]["retrieval"] == {"rag_enabled": False}


def test_no_validator_ablation_accepts_dominated(tmp_path):
    responses = [
        _proposal("PILOT_001", "AL_7075_T6"),   # improves -> archive point
        _proposal("PILOT_002", "STEEL_S235"),   # dominated
        _proposal("PILOT_003", "AL_6061_T6"),   # dominated
    ]
    cfg = c4_run_config(n_eval=3, ablation="no_validator")
    outcome, events = run(responses, cfg, tmp_path)
    # dominated steps are still accepted -> x_t keeps advancing
    assert all(e["agentic"]["accepted"] for e in events)
    assert outcome.extra["c4"]["ablation"] == "no_validator"


def test_no_schema_ablation_lets_schema_invalid_through_the_gate(tmp_path):
    # missing proposal_id -> schema_valid False; no_schema must still apply it
    responses = [_proposal("PILOT_001", "AL_7075_T6", schema_ok=False)]
    cfg = c4_run_config(n_eval=1, ablation="no_schema")
    outcome, events = run(responses, cfg, tmp_path)
    e = events[0]
    assert e["validity"]["schema_valid"] is False
    # it reached evaluation despite the schema failure
    assert e.get("evaluation") is not None


def test_provider_error_aborts(tmp_path):
    class Boom:
        backend_name = "stub"
        model_name = "boom"

        def generate(self, *a, **k):
            raise RuntimeError("ollama down")

    cfg = c4_run_config(n_eval=5)
    d = C4Driver(
        cfg,
        generator=ProposalGenerator(Boom(), registry=DataRegistry()),
        baseline_bom=baseline(),
        evaluator=fake_evaluator,
    )
    log = EventLog(tmp_path / "e.jsonl", run_id=cfg["run_id"],
                   condition="C4_base", seed=0)
    outcome = d.run(log)
    log.close()
    assert outcome.terminal_status == "ABORTED_PROVIDER"
    assert "ollama down" in outcome.detail


def test_runoutcome_c4_extra_shape(tmp_path):
    outcome, _ = run(GOOD_TRIO, c4_run_config(n_eval=3), tmp_path)
    c4 = outcome.extra["c4"]
    assert set(c4) >= {
        "steps",
        "accepted_steps",
        "acceptance_rate",
        "selection_intent_counts",
        "mean_retries_per_selection",
        "hv_trajectory",
        "converged",
        "convergence_reason",
        "stop_rule",
        "ablation",
    }
    assert c4["steps"] == outcome.events_written
    assert 0.0 <= c4["acceptance_rate"] <= 1.0


def test_evaluator_exception_is_penalised_not_crashed(tmp_path):
    calls = {"n": 0}

    def flaky_evaluator(bom):
        calls["n"] += 1
        if bom["parts"][0]["material_id"] == "STEEL_S235":
            raise ValueError("Missing process input: time_min")
        return fake_evaluator(bom)

    responses = [
        _proposal("PILOT_001", "AL_7075_T6"),   # ok -> improving
        _proposal("PILOT_002", "STEEL_S235"),   # part[0] still AL_7075_T6 after P1
        _proposal("PILOT_001", "STEEL_S235"),   # part[0] -> STEEL_S235 -> raises
    ]
    cfg = c4_run_config(n_eval=6, K=2, L=50, eps=0.0)
    d = C4Driver(
        cfg,
        generator=ProposalGenerator(
            CyclingBackend(responses), registry=DataRegistry()
        ),
        baseline_bom=baseline(),
        evaluator=flaky_evaluator,
    )
    log = EventLog(tmp_path / "e.jsonl", run_id=cfg["run_id"],
                   condition="C4_base", seed=0)
    outcome = d.run(log, pareto_archive_path=tmp_path / "p.json")
    log.close()

    assert outcome.terminal_status in (
        "COMPLETE",
        "COMPLETE_CONVERGED",
        "ABORTED_BUDGET_UNREACHED",
    )
    events = read_events(tmp_path / "e.jsonl")
    failed = [
        e
        for e in events
        if (e.get("evaluation") or {}).get("constraints", {}).get("status")
        == "DETERMINISTIC_EVALUATION_FAILED"
    ]
    assert failed, "a failed-eval candidate should be recorded, not crash"
    assert all(e["agentic"]["accepted"] is False for e in failed)
