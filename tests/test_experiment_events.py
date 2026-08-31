import json

import pytest

from src.experiment.apply_proposal import ProposalApplication
from src.experiment.ledger import ConsumeOutcome
from src.experiment.events import (
    EventLog,
    attach_applicability,
    attach_archive,
    attach_evaluation,
    build_evaluation_block,
    derive_funnel_stage,
    event_from,
    nsga2_event,
    read_events,
)


# -- fixtures ------------------------------------------------------


def gen_rec(**overrides):
    rec = {
        "condition": "C1",
        "raw_output": '{"proposal_id": "P1", "part_id": "PILOT_001"}',
        "prompt_hash": "abc123",
        "proposal": {
            "proposal_id": "P1",
            "part_id": "PILOT_001",
            "change_type": "material",
            "target_field": "material_id",
            "new_value": "AL_7075_T6",
        },
        "parse_valid": True,
        "schema_valid": True,
        "authority_valid": True,
        "hallucinated": False,
        "hallucination_categories": [],
        "unknown_identifiers": [],
        "runtime_sec": 4.2,
        "retrieval": {"rag_enabled": False, "top_k": None},
    }
    rec.update(overrides)
    return rec


def c2_rec():
    return gen_rec(
        condition="C2",
        retrieval={
            "rag_enabled": True,
            "top_k": 5,
            "query_text": "aluminium bracket substitution",
            "retrieved_chunk_ids": ["chunk_a", "chunk_b"],
            "retrieved_source_ids": ["SR_1", "SR_2"],
            "retrieved_source_references": ["S 3.5.12", "S 3.4.9"],
            "similarity_scores": [0.71, 0.55],
        },
    )


def eval_result(cost=305.0, mass=0.64, status="EVALUATED", violations=0):
    return {
        "objectives": {"cost_eur": cost, "mass_kg": mass},
        "objective_vector": [cost, mass],
        "constraints": {
            "status": status,
            "feasible": violations == 0,
            "violation_count": violations,
            "available_optimizer_rules": 17,
        },
    }


def outcome(result=None, cache_hit=False):
    return ConsumeOutcome(
        result=result if result is not None else eval_result(),
        bom_hash="sha256:deadbeef",
        cache_hit=cache_hit,
        consumed_budget=not cache_hit,
    )


BASELINE_VECTOR = [312.02, 0.6507108]


# -- event_from -------------------------------------------------


def test_event_from_c1_has_no_retrieval_payload():
    event = event_from(gen_rec())
    assert event["event_type"] == "proposal"
    assert event["retrieval"] == {"rag_enabled": False}
    assert event["generation"]["target_part_id"] == "PILOT_001"
    assert event["generation"]["parsed_proposal"]["new_value"] == "AL_7075_T6"
    assert event["generation"]["raw_output_sha256"]
    assert event["validity"]["parse_valid"] is True
    assert event["validity"]["applicability_valid"] is False
    assert event["hallucination"] == {"hallucinated": False, "categories": []}
    assert event["efficiency"]["gen_runtime_sec"] == 4.2


def test_event_from_c2_carries_retrieval_ids_and_scores():
    event = event_from(c2_rec())
    assert event["retrieval"]["rag_enabled"] is True
    assert event["retrieval"]["retrieved_chunk_ids"] == ["chunk_a", "chunk_b"]
    assert event["retrieval"]["retrieved_source_ids"] == ["SR_1", "SR_2"]
    assert event["retrieval"]["similarity_scores"] == [0.71, 0.55]
    assert event["retrieval"]["query_text"]


def test_event_from_unparseable_proposal():
    event = event_from(
        gen_rec(
            proposal=None,
            parse_valid=False,
            schema_valid=False,
            authority_valid=False,
            hallucinated=True,
            hallucination_categories=["PARSE_ERROR"],
        ),
        target_part_id="PILOT_003",
    )
    assert event["generation"]["parsed_proposal"] is None
    assert event["generation"]["target_part_id"] == "PILOT_003"
    assert event["hallucination"]["categories"] == ["PARSE_ERROR"]


# -- attach_applicability -----------------------------------


def test_attach_applicability_folds_in_modifications_and_flags():
    event = event_from(gen_rec())
    application = ProposalApplication(
        applicability_valid=True,
        bom={"parts": []},
        target_part_id="PILOT_001",
        target_field="material_id",
        change_type="material",
        baseline_value="AL_6061_T6",
        new_value="AL_7075_T6",
        modifications=[
            {
                "part_id": "PILOT_001",
                "field": "material_id",
                "baseline": "AL_6061_T6",
                "candidate": "AL_7075_T6",
            }
        ],
    )
    attach_applicability(event, application)
    assert event["validity"]["applicability_valid"] is True
    assert event["generation"]["modifications"][0]["field"] == "material_id"


def test_attach_applicability_records_protected_writes_and_noop():
    blocked = ProposalApplication(
        applicability_valid=False,
        bom=None,
        target_part_id="PILOT_001",
        target_field="mass_kg",
        change_type="geometry",
        baseline_value=0.0648,
        new_value="0.01",
        protected_field_writes=["mass_kg"],
    )
    event = attach_applicability(event_from(gen_rec()), blocked)
    assert event["validity"]["applicability_valid"] is False
    assert event["validity"]["protected_field_writes"] == ["mass_kg"]

    noop = ProposalApplication(
        applicability_valid=True,
        bom={"parts": []},
        target_part_id="PILOT_001",
        target_field="material_id",
        change_type="material",
        baseline_value="AL_6061_T6",
        new_value="AL_6061_T6",
        modifications=[],
        is_noop=True,
    )
    event2 = attach_applicability(event_from(gen_rec()), noop)
    assert event2["generation"]["is_noop"] is True


# -- evaluation block -------------------------------------


def test_build_evaluation_block_computes_deltas_and_normalises_constraints():
    block = build_evaluation_block(
        outcome(eval_result(cost=300.0, mass=0.60)),
        BASELINE_VECTOR,
        eval_runtime_sec=0.01,
    )
    assert block["consumed_objective_budget"] is True
    assert block["objective_eval_cache_hit"] is False
    assert block["bom_hash"] == "sha256:deadbeef"
    assert block["objectives"] == {"cost_eur": 300.0, "mass_kg": 0.60}
    assert block["baseline_delta"]["cost_eur"] == pytest.approx(12.02)
    assert block["baseline_delta"]["cost_improvement_pct"] == pytest.approx(
        12.02 / 312.02 * 100.0
    )
    assert block["constraints"]["status"] == "EVALUATED"
    assert block["constraints"]["evaluated"] is True
    assert block["constraints"]["rule_level_checks"] == 17
    assert block["constraints"]["proposal_level_violation"] is False


def test_build_evaluation_block_handles_not_evaluated_constraints():
    result = eval_result(status="NOT_EVALUATED", violations=None)
    result["constraints"] = {
        "status": "NOT_EVALUATED",
        "feasible": None,
        "violation_count": None,
    }
    block = build_evaluation_block(outcome(result), BASELINE_VECTOR)
    assert block["constraints"]["evaluated"] is False
    assert block["constraints"]["proposal_level_violation"] is None


def test_attach_evaluation_sets_block_and_eval_runtime():
    event = event_from(gen_rec())
    attach_evaluation(
        event, outcome(cache_hit=True), BASELINE_VECTOR, eval_runtime_sec=0.02
    )
    assert event["evaluation"]["objective_eval_cache_hit"] is True
    assert event["evaluation"]["consumed_objective_budget"] is False
    assert "_eval_runtime_sec" not in event["evaluation"]
    assert event["efficiency"]["eval_runtime_sec"] == 0.02


# -- archive -------------------------------------------


def test_attach_archive_rejects_unknown_status():
    with pytest.raises(ValueError, match="archive status"):
        attach_archive(event_from(gen_rec()), "totally_new", 3)


def test_attach_archive_sets_block():
    event = attach_archive(event_from(gen_rec()), "pareto_improving", 4)
    assert event["archive"] == {
        "status": "pareto_improving",
        "archive_size_after": 4,
    }


# -- funnel stage ------------------------------------


@pytest.mark.parametrize(
    "flags,expected",
    [
        (dict(parse_valid=False), "parse"),
        (dict(schema_valid=False), "schema"),
        (dict(authority_valid=False), "identifier"),
        ({}, "applicability"),  # all valid but applicability not attached
    ],
)
def test_derive_funnel_stage_from_validity(flags, expected):
    event = event_from(gen_rec(**flags))
    assert derive_funnel_stage(event) == expected


def test_derive_funnel_stage_progression_to_archive():
    event = event_from(gen_rec())
    event["validity"]["applicability_valid"] = True
    assert derive_funnel_stage(event) == "feasibility"

    attach_evaluation(event, outcome(), BASELINE_VECTOR)
    assert derive_funnel_stage(event) == "objective_evaluation"

    attach_archive(event, "dominated", 2)
    assert derive_funnel_stage(event) == "archive"


def test_derive_funnel_stage_eval_failure_is_feasibility():
    event = event_from(gen_rec())
    event["validity"]["applicability_valid"] = True
    failed = eval_result(status="DETERMINISTIC_EVALUATION_FAILED")
    attach_evaluation(event, outcome(failed), BASELINE_VECTOR)
    assert derive_funnel_stage(event) == "feasibility"


def test_derive_funnel_stage_for_nsga2_event():
    event = nsga2_event(candidate_id="cand_0001")
    assert derive_funnel_stage(event) == "objective_evaluation"
    attach_archive(event, "non_dominated", 5)
    assert derive_funnel_stage(event) == "archive"


# -- EventLog --------------------------------------


def test_event_log_stamps_identity_index_and_software(tmp_path):
    path = tmp_path / "C1" / "seed_00" / "events.jsonl"
    with EventLog(path, run_id="sha256:r1", condition="C1", seed=0) as log:
        r0 = log.write(event_from(gen_rec()))
        r1 = log.write(event_from(gen_rec()))

    assert r0["run_id"] == "sha256:r1"
    assert r0["condition"] == "C1"
    assert r0["seed"] == 0
    assert r0["event_index"] == 0
    assert r1["event_index"] == 1
    assert r0["ts_utc"].endswith("Z")
    assert r0["software"]["harness_version"]
    assert r0["software"]["evaluator_version"]
    assert r0["validity"]["funnel_stage_reached"] == "applicability"


def test_event_log_derives_funnel_stage_only_when_unset(tmp_path):
    path = tmp_path / "events.jsonl"
    event = event_from(gen_rec())
    event["validity"]["funnel_stage_reached"] = "schema"  # caller pinned it
    with EventLog(path, run_id="r", condition="C1", seed=0) as log:
        written = log.write(event)
    assert written["validity"]["funnel_stage_reached"] == "schema"


def test_event_log_nsga2_event_has_null_validity(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path, run_id="r", condition="C5", seed=0) as log:
        written = log.write(
            nsga2_event(candidate_id="cand_0001", modifications=[])
        )
    assert written["validity"] is None
    assert written["hallucination"] is None
    assert "retrieval" not in written
    assert written["event_index"] == 0


def test_event_log_refuses_to_clobber_nonempty_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"existing": true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="11.18"):
        EventLog(path, run_id="r", condition="C1", seed=0)


def test_event_log_overwrite_mode_replaces_file(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"stale": true}\n', encoding="utf-8")
    with EventLog(
        path, run_id="r", condition="C1", seed=0, if_exists="overwrite"
    ) as log:
        log.write(event_from(gen_rec()))
    events = read_events(path)
    assert len(events) == 1
    assert "stale" not in events[0]


def test_event_log_validates_evaluation_block(tmp_path):
    path = tmp_path / "events.jsonl"
    bad = event_from(gen_rec())
    bad["evaluation"] = {"objectives": {}}  # missing bom_hash + consumed flag
    with EventLog(path, run_id="r", condition="C1", seed=0) as log:
        with pytest.raises(ValueError, match="evaluation block missing"):
            log.write(bad)


# -- read_events ----------------------------------


def test_read_events_round_trips(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path, run_id="r", condition="C1", seed=0) as log:
        for _ in range(3):
            log.write(event_from(gen_rec()))
    events = read_events(path)
    assert [e["event_index"] for e in events] == [0, 1, 2]


def test_read_events_flags_corrupt_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt events.jsonl line"):
        read_events(path)


def test_budget_flag_survives_round_trip(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path, run_id="r", condition="C1", seed=0) as log:
        e1 = event_from(gen_rec())
        e1["validity"]["applicability_valid"] = True
        attach_evaluation(e1, outcome(), BASELINE_VECTOR)
        log.write(e1)

        e2 = event_from(gen_rec())
        e2["validity"]["applicability_valid"] = True
        attach_evaluation(e2, outcome(cache_hit=True), BASELINE_VECTOR)
        log.write(e2)

    events = read_events(path)
    fresh = [
        e
        for e in events
        if e.get("evaluation", {}).get("consumed_objective_budget")
    ]
    assert len(fresh) == 1
