"""
Cross-cutting invariants for the A12 harness -- properties that span
identity + ledger + apply_proposal + events + drivers + metrics + the
runner, asserted against the artifact files a real sweep writes.
Per-module behaviour is covered in the other test_experiment_* files.
"""

import csv
import json
from pathlib import Path

import pytest

from src.data.registry import DataRegistry
from src.experiment.drivers import Nsga2Driver
from src.experiment.events import EventLog, read_events
from src.experiment.identity import build_run_config
from src.experiment.metrics import load_run
from src.optimization.search_space import load_verified_real_search_space
from src.llm.generator import ProposalGenerator
from scripts.run_experiment import run_one, run_sweep
from scripts.run_c5_real_pilot import main as c5_shim_main


PILOT_BOM_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"
SEARCH_SPACE_PATH = "data/benchmark/real_search_space.json"


# ----------------------------------------------------------------------
# fakes
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
            "proposal_id": f"P_{part_id}_{material}",
            "part_id": part_id,
            "change_type": "material",
            "target_field": "material_id",
            "old_value": "AL_6061_T6",
            "new_value": material,
        }
    )


RICH_RESPONSES = [
    _proposal("PILOT_001", "AL_7075_T6"),
    _proposal("PILOT_002", "AL_6061_T6"),
    _proposal("PILOT_003", "STEEL_S235"),
    _proposal("PILOT_004", "STEEL_S235"),
    _proposal("PILOT_005", "AL_6061_T6"),
]
POOR_RESPONSES = [_proposal("PILOT_001", "AL_7075_T6")]


def _material_cost(material):
    table = {
        "AL_7075_T6": (55.0, 0.20),
        "STEEL_S235": (48.0, 0.25),
        "AL_6061_T6": (60.0, 0.22),
    }
    return table.get(material, (62.0, 0.23))


def fake_evaluator(bom):
    cost, mass = _material_cost(bom["parts"][0]["material_id"])
    return {
        "objective_vector": [cost, mass],
        "objectives": {"cost_eur": cost, "mass_kg": mass},
        "constraints": {"status": "NOT_EVALUATED", "feasible": None},
    }


def gen_factory_for(responses):
    def factory(condition):
        return ProposalGenerator(
            CyclingBackend(responses), registry=DataRegistry()
        )

    return factory


def run_small_sweep(tmp_path, *, responses, budget, seeds=(0, 1)):
    run_sweep(
        conditions=["C1", "C5"],
        seeds=list(seeds),
        budget=budget,
        parts_spec="all",
        out_root=tmp_path,
        overwrite=False,
        dry_run=False,
        skip_c3_probe=True,
        generator_factory=gen_factory_for(responses),
        retriever_factory=lambda c: None,
        generative_evaluator=fake_evaluator,
        log=lambda *_: None,
    )


def all_run_dirs(tmp_path):
    root = tmp_path / "runs"
    return sorted(
        p.parent
        for p in root.glob("*/seed_*/run_config.json")
    )


# ----------------------------------------------------------------------
# invariant 1: equal-budget accounting is consistent across files
# ----------------------------------------------------------------------


def test_budget_invariant_events_match_metrics_for_every_run(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    for run_dir in all_run_dirs(tmp_path):
        loaded = load_run(run_dir)
        events = loaded["events"]
        metrics = loaded["metrics"]

        consumed = sum(
            1
            for e in events
            if (e.get("evaluation") or {}).get("consumed_objective_budget")
        )
        assert consumed == metrics["budget"]["n_eval_consumed"]
        cache_hits = sum(
            1
            for e in events
            if (e.get("evaluation") or {}).get("objective_eval_cache_hit")
        )
        assert cache_hits == metrics["budget"]["objective_eval_cache_hits"]

        if metrics["terminal_status"] == "COMPLETE":
            assert consumed == metrics["budget"]["n_eval_target"]
        else:
            assert consumed <= metrics["budget"]["n_eval_target"]


# ----------------------------------------------------------------------
# invariant 2: the validity funnel is monotone non-increasing
# ----------------------------------------------------------------------


def test_funnel_counts_monotone_for_generative_runs(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    for run_dir in all_run_dirs(tmp_path):
        metrics = load_run(run_dir)["metrics"]
        if metrics["condition"] != "C1":
            continue
        c = metrics["validity_funnel"]["counts"]
        chain = [
            c["parse"],
            c["schema"],
            c["identifier"],
            c["applicability"],
            c["objective_evaluated"],
        ]
        assert chain == sorted(chain, reverse=True)
        assert c["parse"] <= metrics["validity_funnel"]["n_prop"]


# ----------------------------------------------------------------------
# invariant 3: run_id agrees across every file and is globally unique
# ----------------------------------------------------------------------


def test_run_id_consistent_across_files_and_unique(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    index = json.loads((tmp_path / "run_index.json").read_text())
    seen = set()
    for run_dir in all_run_dirs(tmp_path):
        loaded = load_run(run_dir)
        rid = loaded["run_config"]["run_id"]
        assert loaded["metrics"]["run_id"] == rid
        assert all(e["run_id"] == rid for e in loaded["events"])
        if loaded["pareto_archive"]:
            assert loaded["pareto_archive"]["run_id"] == rid
        assert rid in index["runs"]
        assert rid not in seen
        seen.add(rid)
    assert len(seen) == len(index["runs"])


# ----------------------------------------------------------------------
# invariant 4: event_index is contiguous 0..N-1
# ----------------------------------------------------------------------


def test_event_index_is_contiguous(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    for run_dir in all_run_dirs(tmp_path):
        events = read_events(run_dir / "events.jsonl")
        assert [e["event_index"] for e in events] == list(
            range(len(events))
        )


# ----------------------------------------------------------------------
# invariant 5: pareto_archive.json <-> metrics.json agree
# ----------------------------------------------------------------------


def test_pareto_archive_matches_metrics(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    for run_dir in all_run_dirs(tmp_path):
        loaded = load_run(run_dir)
        pa = loaded["pareto_archive"]
        mo = loaded["metrics"]["multiobjective"]
        assert pa["hypervolume"] == pytest.approx(mo["hypervolume"])
        assert pa["archive_size"] == mo["pareto_archive_size"]
        assert len(pa["entries"]) == pa["archive_size"]
        assert mo["delta_hv"] == pytest.approx(mo["hypervolume"])


# ----------------------------------------------------------------------
# invariant 6: event_type / validity shape by condition
# ----------------------------------------------------------------------


def test_event_type_and_validity_shape_by_condition(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    for run_dir in all_run_dirs(tmp_path):
        loaded = load_run(run_dir)
        condition = loaded["run_config"]["condition"]
        events = loaded["events"]
        if condition == "C1":
            assert all(e["event_type"] == "proposal" for e in events)
            assert all(
                isinstance(e["validity"], dict) for e in events
            )
        else:  # C5
            assert all(
                e["event_type"] == "nsga2_evaluation" for e in events
            )
            assert all(e["validity"] is None for e in events)
            assert all(e["hallucination"] is None for e in events)


# ----------------------------------------------------------------------
# invariant 7: software stamp is uniform within a run
# ----------------------------------------------------------------------


def test_software_stamp_uniform_within_run(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4)
    for run_dir in all_run_dirs(tmp_path):
        events = read_events(run_dir / "events.jsonl")
        stamps = {json.dumps(e["software"], sort_keys=True) for e in events}
        assert len(stamps) == 1
        soft = events[0]["software"]
        assert soft["harness_version"]
        assert soft["evaluator_version"]


# ----------------------------------------------------------------------
# invariant 8: COMPLETE_SPACE_EXHAUSTED path end-to-end
# ----------------------------------------------------------------------


def test_space_exhaustion_end_to_end(tmp_path):
    run_small_sweep(tmp_path, responses=POOR_RESPONSES, budget=8, seeds=(0,))
    c1_dir = tmp_path / "runs" / "C1" / "seed_00"
    metrics = load_run(c1_dir)["metrics"]
    assert metrics["terminal_status"] == "COMPLETE_SPACE_EXHAUSTED"
    assert metrics["budget"]["n_eval_consumed"] == 1
    assert metrics["budget"]["n_eval_consumed"] < metrics["budget"][
        "n_eval_target"
    ]

    notes = (tmp_path / "results" / "RUN_NOTES.md").read_text()
    assert "CANDIDATE-DIVERSITY CEILING" in notes
    assert "ZERO-HALLUCINATION BASELINE" in notes


# ----------------------------------------------------------------------
# invariant 9: rollup CSV shapes
# ----------------------------------------------------------------------


def test_rollup_csv_shapes(tmp_path):
    run_small_sweep(tmp_path, responses=RICH_RESPONSES, budget=4, seeds=(0, 1, 2))
    results = tmp_path / "results"

    with (results / "seed_summary.csv").open() as h:
        seed_rows = list(csv.DictReader(h))
    assert len(seed_rows) == 6  # (C1 + C5) x 3 seeds

    with (results / "condition_summary.csv").open() as h:
        cond_rows = list(csv.DictReader(h))
    assert {r["condition"] for r in cond_rows} == {"C1", "C5"}
    assert all(int(r["n_seeds"]) == 3 for r in cond_rows)

    with (results / "hypothesis_tests.csv").open() as h:
        hyp_rows = list(csv.DictReader(h))
    pending = [r for r in hyp_rows if r["decision"] == "PENDING_C4"]
    assert {r["hypothesis"] for r in pending} == {"H2", "H3", "H4"}


# ----------------------------------------------------------------------
# invariant 10: C5 is identical across all four entry points
# ----------------------------------------------------------------------


def test_c5_parity_across_entry_points(tmp_path):
    seed, budget = 5, 14
    baseline = json.loads(Path(PILOT_BOM_PATH).read_text())
    ss = load_verified_real_search_space(SEARCH_SPACE_PATH, PILOT_BOM_PATH)

    # 1. direct Nsga2Driver
    cfg = build_run_config("C5", seed=seed, n_eval=budget, target_parts=[])
    log = EventLog(
        tmp_path / "d.jsonl", run_id=cfg["run_id"], condition="C5", seed=seed
    )
    direct = Nsga2Driver(cfg, baseline_bom=baseline, search_space=ss).run(
        log, pareto_archive_path=tmp_path / "d_pareto.json"
    )
    log.close()

    # 2. run_one
    m_one = run_one(
        "C5",
        seed,
        budget=budget,
        parts=[],
        runs_root=tmp_path / "one" / "runs",
        baseline_bom=baseline,
        search_space=ss,
        overwrite=False,
        dry_run=False,
    )

    # 3. run_c5_real_pilot shim
    c5_shim_main(
        ["--seed", str(seed), "--budget", str(budget),
         "--output-dir", str(tmp_path / "shim")]
    )
    m_shim = json.loads(
        (
            tmp_path / "shim" / "runs" / "C5" / f"seed_{seed:02d}" / "metrics.json"
        ).read_text()
    )

    # 4. run_sweep
    run_sweep(
        conditions=["C5"],
        seeds=[seed],
        budget=budget,
        parts_spec="all",
        out_root=tmp_path / "sweep",
        overwrite=False,
        dry_run=False,
        log=lambda *_: None,
    )
    m_sweep = json.loads(
        (
            tmp_path / "sweep" / "runs" / "C5" / f"seed_{seed:02d}" / "metrics.json"
        ).read_text()
    )

    hv = direct.hypervolume
    for m in (m_one, m_shim, m_sweep):
        assert m["multiobjective"]["hypervolume"] == pytest.approx(hv)
        assert m["multiobjective"]["pareto_archive_size"] == direct.archive_size
        assert m["budget"]["n_eval_consumed"] == budget
