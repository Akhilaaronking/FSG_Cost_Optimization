import json
from pathlib import Path

import pytest

from src.data.registry import DataRegistry
from src.experiment.drivers import (
    GenerativeDriver,
    Nsga2Driver,
    ParetoArchive,
    bom_modifications,
    reference_point,
)
from src.experiment.events import EventLog, read_events
from src.experiment.identity import build_run_config
from src.llm.generator import ProposalGenerator
from src.optimization.search_space import load_verified_real_search_space


PILOT_BOM_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"
SEARCH_SPACE_PATH = "data/benchmark/real_search_space.json"

# Whole-BOM objective vectors keyed on part[0]'s material. Tuned to sit
# inside the 1.2x reference box of the real 2-part pilot slice
# (baseline ~ cost 63.14 / mass 0.22216) so hypervolume is non-zero and
# the three non-baseline materials form a 3-point front.
MATERIAL_WEIGHTS = {
    "AL_6061_T6": (63.14, 0.222),
    "AL_7075_T6": (55.0, 0.20),
    "STEEL_S235": (48.0, 0.25),
    "TI_GRADE5": (68.0, 0.13),
}


# -- fixtures -----------------------------------------------------


def two_part_baseline():
    bom = json.loads(Path(PILOT_BOM_PATH).read_text())
    bom["parts"] = bom["parts"][:2]
    return bom


def run_config(condition, seed=0, n_eval=6, parts=("PILOT_001", "PILOT_002")):
    cfg = build_run_config(
        condition, seed=seed, n_eval=n_eval, target_parts=list(parts)
    )
    return cfg


class CyclingBackend:
    backend_name = "stub"
    model_name = "cycling-stub"

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def generate(self, prompt, *, seed=None, temperature=0.2, max_tokens=512):
        response = self.responses[self.index % len(self.responses)]
        self.index += 1
        return response


class SeedRecordingBackend:
    backend_name = "stub"
    model_name = "seed-recording"

    def __init__(self, response):
        self.response = response
        self.seeds = []

    def generate(self, prompt, *, seed=None, temperature=0.2, max_tokens=512):
        self.seeds.append(seed)
        return self.response


def material_proposal(part_id, new_value, old_value="AL_6061_T6"):
    return json.dumps(
        {
            "proposal_id": f"P_{part_id}_{new_value}",
            "part_id": part_id,
            "change_type": "material",
            "target_field": "material_id",
            "old_value": old_value,
            "new_value": new_value,
        }
    )


def fake_evaluator():
    def evaluate(bom):
        material = bom["parts"][0]["material_id"]
        cost, mass = MATERIAL_WEIGHTS.get(material, (140.0, 1.4))
        return {
            "objective_vector": [cost, mass],
            "objectives": {"cost_eur": cost, "mass_kg": mass},
            "constraints": {
                "status": "NOT_EVALUATED",
                "feasible": None,
                "violation_count": None,
                "available_optimizer_rules": 7,
            },
        }

    return evaluate


def generator_with(responses):
    return ProposalGenerator(
        CyclingBackend(responses), registry=DataRegistry()
    )


def make_generative_driver(responses, *, condition="C1", n_eval=6, **kw):
    cfg = run_config(condition, n_eval=n_eval)
    return GenerativeDriver(
        cfg,
        generator=generator_with(responses),
        baseline_bom=two_part_baseline(),
        evaluator=fake_evaluator(),
        **kw,
    )


# -- helpers ---------------------------------------------------


def test_reference_point_is_1_2x_baseline():
    assert reference_point([100.0, 2.0]) == [120.0, 2.4]


def test_bom_modifications_reports_material_and_process_deltas():
    base = two_part_baseline()
    cand = json.loads(json.dumps(base))
    cand["parts"][0]["material_id"] = "STEEL_S235"
    cand["parts"][1]["process_id"] = "TIG_WELDING"
    mods = bom_modifications(base, cand)
    assert {m["field"] for m in mods} == {"material_id", "process_id"}


# -- ParetoArchive -------------------------------------------


def test_archive_first_entry_is_pareto_improving():
    archive = ParetoArchive()
    status, size = archive.offer(
        candidate_id="a",
        objective_vector=[100.0, 1.0],
        bom_hash="h_a",
        modifications=[],
    )
    assert (status, size) == ("pareto_improving", 1)


def test_archive_classifies_dominated_and_improving():
    archive = ParetoArchive()
    archive.offer(
        candidate_id="a",
        objective_vector=[100.0, 1.0],
        bom_hash="h_a",
        modifications=[],
    )
    dominated = archive.offer(
        candidate_id="b",
        objective_vector=[110.0, 1.2],
        bom_hash="h_b",
        modifications=[],
    )
    improving = archive.offer(
        candidate_id="c",
        objective_vector=[80.0, 0.9],
        bom_hash="h_c",
        modifications=[],
    )
    assert dominated[0] == "dominated"
    assert improving[0] == "pareto_improving"
    assert improving[1] == 1  # c dominates a, a is dropped


def test_archive_duplicate_by_hash():
    archive = ParetoArchive()
    archive.offer(
        candidate_id="a",
        objective_vector=[100.0, 1.0],
        bom_hash="h",
        modifications=[],
    )
    status, _ = archive.offer(
        candidate_id="a2",
        objective_vector=[100.0, 1.0],
        bom_hash="h",
        modifications=[],
    )
    assert status == "duplicate"


def test_archive_non_dominated_extends_front():
    archive = ParetoArchive()
    archive.offer(
        candidate_id="a",
        objective_vector=[100.0, 1.0],
        bom_hash="h_a",
        modifications=[],
    )
    status, size = archive.offer(
        candidate_id="b",
        objective_vector=[80.0, 2.0],  # cheaper, heavier -> neither dominates
        bom_hash="h_b",
        modifications=[],
    )
    assert (status, size) == ("non_dominated", 2)


# -- GenerativeDriver: happy path --------------------------


def test_generative_run_reaches_complete_with_distinct_candidates(tmp_path):
    responses = [
        material_proposal("PILOT_001", "AL_7075_T6"),
        material_proposal("PILOT_001", "STEEL_S235"),
        material_proposal("PILOT_001", "TI_GRADE5"),
    ]
    driver = make_generative_driver(responses, n_eval=3)
    log = EventLog(
        tmp_path / "events.jsonl", run_id="r", condition="C1", seed=0
    )
    outcome = driver.run(
        log, pareto_archive_path=tmp_path / "pareto_archive.json"
    )
    log.close()

    assert outcome.terminal_status == "COMPLETE"
    assert outcome.ledger["n_eval_consumed"] == 3
    assert outcome.archive_size >= 1
    assert outcome.hypervolume > 0

    events = read_events(tmp_path / "events.jsonl")
    fresh = [
        e
        for e in events
        if e.get("evaluation", {}).get("consumed_objective_budget")
    ]
    assert len(fresh) == outcome.ledger["n_eval_consumed"] == 3
    assert all(e["event_type"] == "proposal" for e in events)
    assert events[0]["run_id"] == "r"


def test_generative_budget_invariant_holds_with_cache_hits(tmp_path):
    # only two distinct candidates, but n_eval target is 2 -> COMPLETE
    responses = [
        material_proposal("PILOT_001", "AL_7075_T6"),
        material_proposal("PILOT_001", "STEEL_S235"),
    ]
    driver = make_generative_driver(responses, n_eval=2)
    log = EventLog(tmp_path / "e.jsonl", run_id="r", condition="C1", seed=0)
    outcome = driver.run(log)
    log.close()

    assert outcome.terminal_status == "COMPLETE"
    events = read_events(tmp_path / "e.jsonl")
    consumed = sum(
        1
        for e in events
        if e.get("evaluation", {}).get("consumed_objective_budget")
    )
    assert consumed == outcome.ledger["n_eval_consumed"] == 2


# -- GenerativeDriver: space exhaustion --------------------


def test_generative_space_exhaustion_terminal(tmp_path):
    # single repeating response -> 1 distinct candidate, target 6
    driver = make_generative_driver(
        [material_proposal("PILOT_001", "AL_7075_T6")], n_eval=6
    )
    log = EventLog(tmp_path / "e.jsonl", run_id="r", condition="C1", seed=0)
    outcome = driver.run(log)
    log.close()

    assert outcome.terminal_status == "COMPLETE_SPACE_EXHAUSTED"
    assert outcome.ledger["n_eval_consumed"] == 1
    assert outcome.ledger["objective_eval_cache_hits"] >= 1
    assert "no new distinct candidate" in outcome.detail


# -- GenerativeDriver: attempt cap ------------------------


def test_generative_attempt_cap_aborts(tmp_path):
    # A proposal targeting a non-optimisation field is rejected by the
    # generator's own authority check (UNSUPPORTED_TARGET_FIELD) before
    # apply_proposal runs, so it dies at the 'identifier' funnel stage.
    # It never progresses, so the attempt cap is what stops the run.
    bad = json.dumps(
        {
            "proposal_id": "P_bad",
            "part_id": "PILOT_001",
            "change_type": "geometry",
            "target_field": "mass_kg",
            "new_value": "0.01",
        }
    )
    cfg = run_config("C1", n_eval=6)
    cfg["identity"]["budget"]["proposal_attempt_cap"] = 4
    driver = GenerativeDriver(
        cfg,
        generator=generator_with([bad]),
        baseline_bom=two_part_baseline(),
        evaluator=fake_evaluator(),
        stall_limit=999,  # disable stall path so the cap path is exercised
    )
    log = EventLog(tmp_path / "e.jsonl", run_id="r", condition="C1", seed=0)
    outcome = driver.run(log)
    log.close()

    assert outcome.terminal_status == "ABORTED_BUDGET_UNREACHED"
    assert outcome.ledger["proposal_attempts"] == 4
    assert outcome.ledger["n_eval_consumed"] == 0

    events = read_events(tmp_path / "e.jsonl")
    assert len(events) == 4
    assert all(
        e["validity"]["applicability_valid"] is False for e in events
    )
    assert all(
        e["validity"]["funnel_stage_reached"] == "identifier"
        for e in events
    )


# -- GenerativeDriver: provider abort --------------------


def test_generative_provider_failure_aborts(tmp_path):
    class Boom:
        backend_name = "stub"
        model_name = "boom"

        def generate(self, *a, **k):
            raise RuntimeError("ollama connection refused")

    cfg = run_config("C1", n_eval=6)
    driver = GenerativeDriver(
        cfg,
        generator=ProposalGenerator(Boom(), registry=DataRegistry()),
        baseline_bom=two_part_baseline(),
        evaluator=fake_evaluator(),
    )
    log = EventLog(tmp_path / "e.jsonl", run_id="r", condition="C1", seed=0)
    outcome = driver.run(log)
    log.close()

    assert outcome.terminal_status == "ABORTED_PROVIDER"
    assert "ollama connection refused" in outcome.detail
    assert outcome.events_written == 0


# -- GenerativeDriver: C2 retrieval flows into events ----


def test_c2_driver_records_retrieval_block(tmp_path):
    from src.rag.models import RagChunk, RetrievalResult

    class OneHitRetriever:
        def retrieve(self, query, top_k=5, filters=None):
            chunk = RagChunk(
                chunk_id="chunk_X",
                document_id="doc_X",
                text="evidence",
                source_type="fsg_rule",
                source_id="SRC_X",
                source_reference="S 3.5.12",
                metadata={},
            )
            return [RetrievalResult(chunk=chunk, score=0.8, rank=1)]

    cfg = run_config("C2", n_eval=1)
    driver = GenerativeDriver(
        cfg,
        generator=generator_with(
            [material_proposal("PILOT_001", "AL_7075_T6")]
        ),
        baseline_bom=two_part_baseline(),
        retriever=OneHitRetriever(),
        evaluator=fake_evaluator(),
    )
    log = EventLog(tmp_path / "e.jsonl", run_id="r", condition="C2", seed=0)
    driver.run(log)
    log.close()

    events = read_events(tmp_path / "e.jsonl")
    assert events[0]["retrieval"]["rag_enabled"] is True
    assert events[0]["retrieval"]["retrieved_chunk_ids"] == ["chunk_X"]


# -- GenerativeDriver: seed changes part order -----------


def test_seed_rotates_target_part_order(tmp_path):
    responses = [material_proposal("PILOT_001", "AL_7075_T6")]

    d0 = make_generative_driver(responses)
    d0.seed = 0
    assert list(_first_n(d0._part_cycle(), 2)) == ["PILOT_001", "PILOT_002"]

    d1 = make_generative_driver(responses)
    d1.seed = 1
    assert list(_first_n(d1._part_cycle(), 2)) == ["PILOT_002", "PILOT_001"]


def _first_n(iterator, n):
    return [next(iterator) for _ in range(n)]


# -- GenerativeDriver: per-attempt decode seed ----------


def test_attempt_seed_formula():
    from src.experiment.drivers import ATTEMPT_SEED_STRIDE

    driver = make_generative_driver([material_proposal("PILOT_001", "AL_7075_T6")])
    driver.seed = 3
    assert ATTEMPT_SEED_STRIDE == 10_000
    assert driver._attempt_seed(1) == 30_001
    assert driver._attempt_seed(7) == 30_007


def _run_seed_recording(run_seed, n_eval, tmp_path, tag):
    backend = SeedRecordingBackend(
        material_proposal("PILOT_001", "AL_7075_T6")
    )
    cfg = run_config("C1", seed=run_seed, n_eval=n_eval)
    driver = GenerativeDriver(
        cfg,
        generator=ProposalGenerator(backend, registry=DataRegistry()),
        baseline_bom=two_part_baseline(),
        evaluator=fake_evaluator(),
    )
    log = EventLog(
        tmp_path / f"{tag}.jsonl", run_id="r", condition="C1", seed=run_seed
    )
    driver.run(log)
    log.close()
    return backend.seeds


def test_decode_seed_varies_per_attempt_and_is_monotonic(tmp_path):
    seeds = _run_seed_recording(2, 6, tmp_path, "s2")
    # run seed 2, stride 10_000, 1-based attempt index
    assert seeds[:3] == [20_001, 20_002, 20_003]
    assert seeds == sorted(seeds)
    assert len(set(seeds)) == len(seeds)  # no repeats within a run


def test_attempt_seed_sequence_is_reproducible_from_run_seed(tmp_path):
    first = _run_seed_recording(1, 5, tmp_path, "run_a")
    second = _run_seed_recording(1, 5, tmp_path, "run_b")
    assert first == second


def test_different_run_seeds_get_disjoint_attempt_seed_ranges(tmp_path):
    s0 = set(_run_seed_recording(0, 8, tmp_path, "s0"))
    s1 = set(_run_seed_recording(1, 8, tmp_path, "s1"))
    assert s0.isdisjoint(s1)
    assert max(s0) < min(s1)


# -- Nsga2Driver: real integration ----------------------


def test_nsga2_driver_runs_and_logs_one_event_per_evaluation(tmp_path):
    cfg = build_run_config("C5", seed=0, n_eval=12, target_parts=[])
    baseline = json.loads(Path(PILOT_BOM_PATH).read_text())
    search_space = load_verified_real_search_space(
        SEARCH_SPACE_PATH, PILOT_BOM_PATH
    )
    driver = Nsga2Driver(
        cfg, baseline_bom=baseline, search_space=search_space
    )
    log = EventLog(
        tmp_path / "events.jsonl", run_id="c5", condition="C5", seed=0
    )
    outcome = driver.run(
        log, pareto_archive_path=tmp_path / "pareto_archive.json"
    )
    log.close()

    assert outcome.terminal_status == "COMPLETE"
    assert "ledger_drift" not in outcome.extra
    assert outcome.ledger["n_eval_consumed"] == 12
    assert outcome.extra["nsga2_evaluation_count"] == 12
    assert outcome.archive_size >= 1
    assert outcome.hypervolume > 0

    events = read_events(tmp_path / "events.jsonl")
    assert len(events) == 12
    assert all(e["event_type"] == "nsga2_evaluation" for e in events)
    assert all(e["validity"] is None for e in events)
    assert all(
        "bom_hash" in e["evaluation"] for e in events
    )

    payload = json.loads(
        (tmp_path / "pareto_archive.json").read_text()
    )
    assert payload["archive_size"] == outcome.archive_size
    assert payload["reference_point"][0] == pytest.approx(
        outcome.baseline_vector[0] * 1.2
    )


def test_nsga2_driver_rejects_non_c5_config():
    cfg = build_run_config("C1", seed=0, n_eval=5, target_parts=["PILOT_001"])
    with pytest.raises(ValueError, match="only handles C5"):
        Nsga2Driver(cfg, baseline_bom={}, search_space={})
