import json
from pathlib import Path

import pytest

from src.data.registry import DataRegistry
from src.llm.generator import ProposalGenerator
from scripts.run_experiment import (
    findings,
    parse_conditions,
    parse_parts,
    parse_seeds,
    run_sweep,
)


PILOT_BOM_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"


# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------


def test_parse_seeds_range_and_list():
    assert parse_seeds("0-9") == list(range(10))
    assert parse_seeds("0,1,2") == [0, 1, 2]
    assert parse_seeds("3") == [3]


def test_parse_conditions_all_and_order():
    assert parse_conditions("all") == ["C1", "C2", "C3", "C5"]
    assert parse_conditions("C5,C1") == ["C1", "C5"]  # canonical order
    with pytest.raises(ValueError, match="unknown condition"):
        parse_conditions("C1,C9")


def test_parse_parts():
    bom = json.loads(Path(PILOT_BOM_PATH).read_text())
    assert len(parse_parts("all", bom)) == 10
    assert parse_parts("PILOT_001,PILOT_003", bom) == [
        "PILOT_001",
        "PILOT_003",
    ]
    with pytest.raises(ValueError, match="unknown part"):
        parse_parts("PILOT_999", bom)


# ----------------------------------------------------------------------
# findings -- emitted only when the data shows them
# ----------------------------------------------------------------------


def _metrics(condition, seed, *, status, n_eval, n_prop, hr, hv=5.0, counts=None):
    return {
        "condition": condition,
        "seed": seed,
        "terminal_status": status,
        "budget": {
            "n_eval_target": 50,
            "n_eval_consumed": n_eval,
            "proposal_attempts": n_prop,
            "objective_eval_cache_hits": 0,
        },
        "validity_funnel": {
            "n_prop": n_prop,
            "counts": counts or {},
            "rates": {},
        },
        "hallucination": {"hr_all_proposals": hr},
        "multiobjective": {"hypervolume": hv},
    }


def test_findings_flags_diversity_ceiling():
    metrics = [
        _metrics("C1", s, status="COMPLETE_SPACE_EXHAUSTED",
                 n_eval=11, n_prop=37, hr=0.0)
        for s in range(3)
    ]
    notes = findings(metrics, [])
    assert any("CANDIDATE-DIVERSITY CEILING" in n for n in notes)
    assert any("median of 11" in n for n in notes)


def test_findings_calls_out_dead_condition_separately():
    # C1 explores; C3 produces nothing valid on every seed
    metrics = []
    for s in range(3):
        metrics.append(
            _metrics("C1", s, status="COMPLETE_SPACE_EXHAUSTED",
                     n_eval=12, n_prop=40, hr=0.0)
        )
        metrics.append(
            _metrics(
                "C3", s, status="COMPLETE_SPACE_EXHAUSTED",
                n_eval=0, n_prop=20, hr=1.0,
                counts={"parse": 20, "schema": 0, "identifier": 0,
                        "applicability": 0, "objective_evaluated": 0},
            )
        )
    notes = findings(metrics, [])

    dead = next(n for n in notes if "PRODUCED NO VALID PROPOSALS" in n)
    assert dead.startswith("C3 ")
    assert "'schema' stage" in dead
    assert "hr_all = 1.0" in dead

    ceiling = next(n for n in notes if "CANDIDATE-DIVERSITY CEILING" in n)
    assert "range 12-12" in ceiling          # C3's 0 is NOT in the range
    assert "6/6 C1 runs" not in ceiling      # C3 excluded from the count
    assert "3/3 C1 runs" in ceiling
    assert "C3 excluded" in ceiling


def test_findings_flags_zero_hallucination_baseline():
    metrics = [
        _metrics("C1", s, status="COMPLETE", n_eval=40, n_prop=40, hr=0.0)
        for s in range(3)
    ]
    notes = findings(metrics, [])
    assert any("ZERO-HALLUCINATION BASELINE" in n for n in notes)


def test_findings_silent_when_neither_condition_holds():
    metrics = [
        _metrics("C1", 0, status="COMPLETE", n_eval=50, n_prop=50, hr=0.05),
    ]
    assert findings(metrics, []) == []


def test_findings_reports_blocked_and_aborted():
    metrics = [
        _metrics("C2", 0, status="ABORTED_PROVIDER", n_eval=3, n_prop=5, hr=0.0)
    ]
    blocked = [{"condition": "C3", "probe": {"detail": "mlx import failed"}}]
    notes = findings(metrics, blocked)
    assert any("C3 BLOCKED (environment)" in n for n in notes)
    assert any("ABORTED RUNS" in n for n in notes)


# ----------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------


def test_dry_run_writes_only_run_configs(tmp_path):
    lines = []
    run_sweep(
        conditions=["C1", "C5"],
        seeds=[0, 1],
        budget=50,
        parts_spec="all",
        out_root=tmp_path,
        overwrite=False,
        dry_run=True,
        log=lines.append,
    )
    for condition in ("C1", "C5"):
        for seed in (0, 1):
            d = tmp_path / "runs" / condition / f"seed_{seed:02d}"
            assert (d / "run_config.json").is_file()
            assert not (d / "events.jsonl").exists()
    index = json.loads((tmp_path / "run_index.json").read_text())
    assert len(index["runs"]) == 4
    assert not (tmp_path / "results" / "seed_summary.csv").exists()


# ----------------------------------------------------------------------
# end-to-end with injected fakes (C1 + C5)
# ----------------------------------------------------------------------


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


def _proposal(part_id, material):
    return json.dumps(
        {
            "proposal_id": f"P_{material}",
            "part_id": part_id,
            "change_type": "material",
            "target_field": "material_id",
            "old_value": "AL_6061_T6",
            "new_value": material,
        }
    )


def fake_generator_factory(condition):
    responses = [
        _proposal("PILOT_001", "AL_7075_T6"),
        _proposal("PILOT_001", "STEEL_S235"),
        _proposal("PILOT_002", "AL_6061_T6"),
    ]
    return ProposalGenerator(
        CyclingBackend(responses), registry=DataRegistry()
    )


def fake_evaluator(bom):
    weights = {
        "AL_7075_T6": (55.0, 0.20),
        "STEEL_S235": (48.0, 0.25),
    }
    cost, mass = weights.get(bom["parts"][0]["material_id"], (63.0, 0.22))
    return {
        "objective_vector": [cost, mass],
        "objectives": {"cost_eur": cost, "mass_kg": mass},
        "constraints": {"status": "NOT_EVALUATED", "feasible": None},
    }


def test_sweep_end_to_end_c1_and_c5(tmp_path):
    lines = []
    result = run_sweep(
        conditions=["C1", "C5"],
        seeds=[0, 1],
        budget=12,
        parts_spec="all",
        out_root=tmp_path,
        overwrite=False,
        dry_run=False,
        skip_c3_probe=True,
        generator_factory=fake_generator_factory,
        retriever_factory=lambda c: None,
        generative_evaluator=fake_evaluator,
        log=lines.append,
    )

    # per-run artifacts
    for condition in ("C1", "C5"):
        for seed in (0, 1):
            d = tmp_path / "runs" / condition / f"seed_{seed:02d}"
            assert (d / "run_config.json").is_file()
            assert (d / "events.jsonl").is_file()
            assert (d / "metrics.json").is_file()
            assert (d / "pareto_archive.json").is_file()

    # rollups
    results = tmp_path / "results"
    assert (results / "seed_summary.csv").is_file()
    assert (results / "condition_summary.csv").is_file()
    assert (results / "hypothesis_tests.csv").is_file()
    assert (results / "figures").is_dir()
    assert (results / "RUN_NOTES.md").is_file()

    # C1 exhausts its tiny fake space -> diversity-ceiling finding present
    notes = result["notes"]
    assert any("CANDIDATE-DIVERSITY CEILING" in n for n in notes)
    assert any("ZERO-HALLUCINATION BASELINE" in n for n in notes)

    run_notes = (results / "RUN_NOTES.md").read_text()
    assert "CANDIDATE-DIVERSITY CEILING" in run_notes
    assert "| C1 |" in run_notes and "| C5 |" in run_notes

    summary = json.loads((tmp_path / "run_index.json").read_text())
    assert len(summary["runs"]) == 4
    assert summary["deviations"] == []

    # seed_summary has one row per condition x seed
    import csv

    with (results / "seed_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {r["condition"] for r in rows} == {"C1", "C5"}


def test_sweep_refuses_run_id_mismatch_on_rerun(tmp_path):
    kwargs = dict(
        conditions=["C1"],
        seeds=[0],
        parts_spec="all",
        out_root=tmp_path,
        overwrite=True,
        dry_run=False,
        skip_c3_probe=True,
        generator_factory=fake_generator_factory,
        retriever_factory=lambda c: None,
        generative_evaluator=fake_evaluator,
        log=lambda *_: None,
    )
    run_sweep(budget=12, **kwargs)
    # same identity except budget -> different run_id -> refused (11.4)
    with pytest.raises(ValueError, match="11.4"):
        run_sweep(budget=20, **kwargs)


# ----------------------------------------------------------------------
# C4 wiring (A13 step 5)
# ----------------------------------------------------------------------


def test_parse_conditions_accepts_c4_labels_and_orders_after_c3():
    assert parse_conditions("C4_base") == ["C4_base"]
    assert parse_conditions("C5,C4_base,C1") == ["C1", "C4_base", "C5"]
    assert parse_conditions("all") == ["C1", "C2", "C3", "C5"]  # C4 opt-in
    with pytest.raises(ValueError, match="unknown condition"):
        parse_conditions("C4_bogus")


def test_dry_run_writes_c4_loop_spec(tmp_path):
    run_sweep(
        conditions=["C4_base", "C4_base_no_rag"],
        seeds=[0],
        budget=50,
        parts_spec="all",
        out_root=tmp_path,
        overwrite=False,
        dry_run=True,
        log=lambda *_: None,
    )
    cfg = json.loads(
        (tmp_path / "runs" / "C4_base" / "seed_00" / "run_config.json").read_text()
    )
    loop = cfg["condition_spec"]["c4_loop"]
    assert cfg["condition_spec"]["driver"] == "C4Driver"
    assert loop["n_eval"] == 50
    assert loop["convergence"] == {
        "look_back_L": 10,
        "epsilon_hv": 0.1,
        "variant": "delta",
    }
    assert loop["retry_cap_K"] == 3
    assert loop["ablation"] is None
    # C4 tool-loop cap is max(150, 6*N), far tighter than the 1500
    # generative cap (a >2 h/seed landmine for a non-functional loop).
    assert loop["proposal_attempt_cap"] == 300
    assert cfg["identity"]["budget"]["proposal_attempt_cap"] == 300

    abl = json.loads(
        (
            tmp_path / "runs" / "C4_base_no_rag" / "seed_00" / "run_config.json"
        ).read_text()
    )
    assert abl["condition_spec"]["c4_loop"]["ablation"] == "no_rag"
    assert abl["identity"]["retrieval"] == {"rag_enabled": False}
    assert abl["run_id"] != cfg["run_id"]


def _c4_generator_factory(condition):
    responses = [
        _proposal("PILOT_001", "AL_7075_T6"),
        _proposal("PILOT_002", "STEEL_4130_CRMO"),
        _proposal("PILOT_003", "AL_6061_T6"),
    ]
    return ProposalGenerator(
        CyclingBackend(responses), registry=DataRegistry()
    )


def _c4_evaluator(bom):
    deltas = {
        ("PILOT_001", "AL_7075_T6"): (-60.0, -0.04),
        ("PILOT_002", "STEEL_4130_CRMO"): (-10.0, +0.02),
        ("PILOT_003", "AL_6061_T6"): (+8.0, -0.03),
    }
    cost, mass = 312.02, 0.6507
    for p in bom["parts"]:
        d = deltas.get((p["part_id"], p["material_id"]))
        if d:
            cost += d[0]
            mass += d[1]
    return {
        "objective_vector": [cost, mass],
        "objectives": {"cost_eur": cost, "mass_kg": mass},
        "constraints": {"status": "NOT_EVALUATED", "feasible": None},
    }


def test_sweep_runs_c4_base_end_to_end(tmp_path):
    result = run_sweep(
        conditions=["C4_base"],
        seeds=[0, 1],
        budget=6,
        parts_spec="all",
        out_root=tmp_path,
        overwrite=False,
        dry_run=False,
        skip_c3_probe=True,
        generator_factory=_c4_generator_factory,
        retriever_factory=lambda c: None,
        generative_evaluator=_c4_evaluator,
        log=lambda *_: None,
    )

    for seed in (0, 1):
        d = tmp_path / "runs" / "C4_base" / f"seed_{seed:02d}"
        for name in (
            "run_config.json",
            "events.jsonl",
            "metrics.json",
            "pareto_archive.json",
        ):
            assert (d / name).is_file(), name
        m = json.loads((d / "metrics.json").read_text())
        assert m["condition"] == "C4_base"
        assert m["c4"] is not None
        assert m["c4"]["steps"] >= 1
        # agentic steps feed the funnel
        assert m["validity_funnel"]["n_prop"] == m["c4"]["steps"]

    notes = result["notes"]
    assert any("C4_base TOOL-LOOP" in n for n in notes)

    hyp = list(
        __import__("csv").DictReader(
            (tmp_path / "results" / "hypothesis_tests.csv").open()
        )
    )
    # no C5 present -> H2 pending
    h2 = next(r for r in hyp if r["hypothesis"] == "H2")
    assert h2["decision"] == "PENDING_C4"
