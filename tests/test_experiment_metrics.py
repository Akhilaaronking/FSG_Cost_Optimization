import csv
import json
from pathlib import Path

import pytest

from src.data.registry import DataRegistry
from src.experiment.drivers import GenerativeDriver, Nsga2Driver
from src.experiment.events import EventLog, read_events
from src.experiment.identity import build_run_config
from src.experiment.metrics import (
    CONDITION_SUMMARY_COLUMNS,
    HYPOTHESIS_TEST_COLUMNS,
    SEED_SUMMARY_COLUMNS,
    _average_ranks,
    _constraints,
    _describe,
    _hallucination,
    _multiobjective,
    _validity_funnel,
    build_condition_summary,
    build_seed_summary,
    compute_metrics,
    hypothesis_tests,
    seed_summary_row,
    wilcoxon_paired,
    write_csv,
)
from src.llm.generator import ProposalGenerator
from src.optimization.search_space import load_verified_real_search_space


PILOT_BOM_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"
SEARCH_SPACE_PATH = "data/benchmark/real_search_space.json"


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------


def proposal_event(
    *,
    parse=True,
    schema=True,
    authority=True,
    applicable=None,
    evaluated=False,
    cache_hit=False,
    hallucinated=False,
    categories=(),
    constraints=None,
    archive_status=None,
):
    event = {
        "event_type": "proposal",
        "validity": {
            "parse_valid": parse,
            "schema_valid": schema,
            "authority_valid": authority,
            "applicability_valid": (
                applicable if applicable is not None else authority
            ),
        },
        "hallucination": {
            "hallucinated": hallucinated,
            "categories": list(categories),
        },
    }
    if evaluated or cache_hit:
        event["evaluation"] = {
            "consumed_objective_budget": evaluated and not cache_hit,
            "objective_eval_cache_hit": cache_hit,
            "constraints": constraints
            or {"status": "NOT_EVALUATED", "evaluated": False},
        }
    if archive_status:
        event["archive"] = {"status": archive_status, "archive_size_after": 1}
    return event


def test_validity_funnel_counts_and_monotone_rates():
    events = [
        proposal_event(evaluated=True),  # full funnel
        proposal_event(authority=False),  # dies at identifier
        proposal_event(schema=False),  # dies at schema
        proposal_event(parse=False),  # dies at parse
        proposal_event(applicable=False),  # dies at applicability
    ]
    funnel = _validity_funnel(events)
    assert funnel["n_prop"] == 5
    counts = funnel["counts"]
    assert counts == {
        "parse": 4,
        "schema": 3,
        "identifier": 2,
        "applicability": 1,
        "objective_evaluated": 1,
    }
    rates = [funnel["rates"][k] for k in (
        "parse",
        "schema",
        "identifier",
        "applicability",
        "objective_evaluated",
    )]
    assert rates == sorted(rates, reverse=True)  # non-increasing


def test_hallucination_rates_and_category_counter():
    events = [
        proposal_event(hallucinated=True, categories=["UNKNOWN_MATERIAL_ID"]),
        proposal_event(hallucinated=True, categories=["UNKNOWN_MATERIAL_ID"]),
        proposal_event(hallucinated=False),
        proposal_event(schema=False, hallucinated=True, categories=["PARSE_ERROR"]),
    ]
    result = _hallucination(events)
    assert result["hr_all_proposals"] == 3 / 4
    # schema-valid-only denominator is the 3 parse+schema-valid events
    assert result["hr_schema_valid_only"] == 2 / 3
    assert result["categories"] == {
        "UNKNOWN_MATERIAL_ID": 2,
        "PARSE_ERROR": 1,
    }


def test_constraints_not_evaluated_gives_none_cvr():
    events = [proposal_event(evaluated=True) for _ in range(3)]
    result = _constraints(events)
    assert result["n_deterministically_checked"] == 0
    assert result["cvr_proposal_level"] is None
    assert result["cvr_rule_level"] is None


def test_constraints_computed_when_evaluated():
    events = [
        proposal_event(
            evaluated=True,
            constraints={
                "evaluated": True,
                "proposal_level_violation": True,
                "rule_level_violations": 2,
                "rule_level_checks": 10,
                "missing_essential_fields": ["x"],
            },
        ),
        proposal_event(
            evaluated=True,
            constraints={
                "evaluated": True,
                "proposal_level_violation": False,
                "rule_level_violations": 0,
                "rule_level_checks": 10,
                "missing_essential_fields": [],
            },
        ),
    ]
    result = _constraints(events)
    assert result["n_deterministically_checked"] == 2
    assert result["cvr_proposal_level"] == 0.5
    assert result["cvr_rule_level"] == 2 / 20
    assert result["missing_essential_fields_count"] == 1


def test_multiobjective_from_archive():
    archive = {
        "reference_point": [120.0, 2.4],
        "hypervolume": 30.0,
        "normalized_hypervolume": 0.104,
        "archive_size": 2,
        "entries": [
            {"cost_eur": 60.0, "mass_kg": 1.0},
            {"cost_eur": 90.0, "mass_kg": 0.5},
        ],
    }
    mo = _multiobjective(archive)
    assert mo["delta_hv"] == 30.0  # HV_baseline = 0
    assert mo["ideal_point"] == [60.0, 0.5]
    assert mo["min_ideal_point_distance"] == pytest.approx(
        min(
            ((60 - 60) / 60) ** 2 + ((1.0 - 0.5) / 1.9) ** 2,
            ((90 - 60) / 60) ** 2 + ((0.5 - 0.5) / 1.9) ** 2,
        )
        ** 0.5
    )


def test_multiobjective_empty_archive():
    mo = _multiobjective({"reference_point": [1, 1], "entries": []})
    assert mo["ideal_point"] is None
    assert mo["min_ideal_point_distance"] is None


# -- statistics helpers -----------------------------------------


def test_average_ranks_handles_ties():
    assert _average_ranks([0.1, 0.1, 0.05]) == [2.5, 2.5, 1.0]


def test_wilcoxon_paired_all_positive_n3_is_underpowered():
    stats = wilcoxon_paired([0.10, 0.20, 0.15])
    assert stats["n_nonzero_pairs"] == 3
    assert stats["W_statistic"] == 6.0
    assert stats["p_one_sided"] == pytest.approx(0.125)
    assert stats["effect_size_rrb"] == 1.0  # all reductions
    assert stats["effect_size_dz"] is not None
    assert stats["min_achievable_p_one_sided"] == pytest.approx(1 / 8)
    assert stats["underpowered"] is True


def test_wilcoxon_paired_drops_zeros():
    stats = wilcoxon_paired([0.0, 0.0, 0.0])
    assert stats["n_nonzero_pairs"] == 0
    assert stats["p_one_sided"] is None
    assert stats["effect_size_rrb"] is None


def test_wilcoxon_paired_mixed_signs():
    stats = wilcoxon_paired([0.1, 0.1, -0.05])
    # ranks 2.5, 2.5, 1 ; w+ = 5, w- = 1 ; total 6
    assert stats["effect_size_rrb"] == pytest.approx((5 - 1) / 6)


def test_describe_basic_and_edges():
    assert _describe([1, 2, 3]) == {
        "median": 2,
        "iqr": pytest.approx(1.0),
        "mean": pytest.approx(2.0),
        "sd": pytest.approx((2 / 3) ** 0.5),
    }
    assert _describe([]) == {
        "median": None,
        "iqr": None,
        "mean": None,
        "sd": None,
    }
    single = _describe([5])
    assert single["median"] == 5 and single["sd"] == 0.0
    mixed = _describe([1, None, 3])
    assert mixed["mean"] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# compute_metrics -- integration through the drivers
# ----------------------------------------------------------------------


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


def two_part_baseline():
    bom = json.loads(Path(PILOT_BOM_PATH).read_text())
    bom["parts"] = bom["parts"][:2]
    return bom


def material_proposal(part_id, new_value):
    return json.dumps(
        {
            "proposal_id": f"P_{new_value}",
            "part_id": part_id,
            "change_type": "material",
            "target_field": "material_id",
            "old_value": "AL_6061_T6",
            "new_value": new_value,
        }
    )


def fake_evaluator():
    weights = {
        "AL_7075_T6": (55.0, 0.20),
        "STEEL_S235": (48.0, 0.25),
        "TI_GRADE5": (68.0, 0.13),
    }

    def evaluate(bom):
        cost, mass = weights.get(
            bom["parts"][0]["material_id"], (63.0, 0.22)
        )
        return {
            "objective_vector": [cost, mass],
            "objectives": {"cost_eur": cost, "mass_kg": mass},
            "constraints": {"status": "NOT_EVALUATED", "feasible": None},
        }

    return evaluate


def test_compute_metrics_for_generative_run(tmp_path):
    cfg = build_run_config(
        "C1", seed=0, n_eval=3, target_parts=["PILOT_001", "PILOT_002"]
    )
    driver = GenerativeDriver(
        cfg,
        generator=ProposalGenerator(
            CyclingBackend(
                [
                    material_proposal("PILOT_001", "AL_7075_T6"),
                    material_proposal("PILOT_001", "STEEL_S235"),
                    material_proposal("PILOT_001", "TI_GRADE5"),
                ]
            ),
            registry=DataRegistry(),
        ),
        baseline_bom=two_part_baseline(),
        evaluator=fake_evaluator(),
    )
    log = EventLog(tmp_path / "events.jsonl", run_id=cfg["run_id"], condition="C1", seed=0)
    outcome = driver.run(
        log, pareto_archive_path=tmp_path / "pareto_archive.json"
    )
    log.close()

    events = read_events(tmp_path / "events.jsonl")
    pareto = json.loads((tmp_path / "pareto_archive.json").read_text())
    metrics = compute_metrics(
        cfg,
        events,
        terminal_status=outcome.terminal_status,
        wall_clock_sec=outcome.wall_clock_sec,
        pareto_archive=pareto,
    )

    assert metrics["condition"] == "C1"
    assert metrics["terminal_status"] == "COMPLETE"
    assert metrics["budget"]["n_eval_consumed"] == 3
    assert metrics["budget"]["proposal_attempts"] == metrics[
        "validity_funnel"
    ]["n_prop"]
    r = metrics["validity_funnel"]["rates"]
    assert r["parse"] == 1.0 and r["objective_evaluated"] == 1.0
    assert metrics["multiobjective"]["hypervolume"] > 0
    assert metrics["multiobjective"]["delta_hv"] == metrics[
        "multiobjective"
    ]["hypervolume"]
    assert metrics["constraints"]["cvr_proposal_level"] is None  # NOT_EVALUATED
    assert metrics["efficiency"]["t100_sec_per_100_proposals"] is not None


def test_compute_metrics_for_nsga2_run(tmp_path):
    cfg = build_run_config("C5", seed=0, n_eval=12, target_parts=[])
    baseline = json.loads(Path(PILOT_BOM_PATH).read_text())
    search_space = load_verified_real_search_space(
        SEARCH_SPACE_PATH, PILOT_BOM_PATH
    )
    driver = Nsga2Driver(
        cfg, baseline_bom=baseline, search_space=search_space
    )
    log = EventLog(tmp_path / "events.jsonl", run_id=cfg["run_id"], condition="C5", seed=0)
    outcome = driver.run(
        log, pareto_archive_path=tmp_path / "pareto_archive.json"
    )
    log.close()

    metrics = compute_metrics(
        cfg,
        read_events(tmp_path / "events.jsonl"),
        terminal_status=outcome.terminal_status,
        wall_clock_sec=outcome.wall_clock_sec,
        pareto_archive=json.loads(
            (tmp_path / "pareto_archive.json").read_text()
        ),
    )
    assert metrics["condition"] == "C5"
    assert metrics["budget"]["n_eval_consumed"] == 12
    assert metrics["validity_funnel"]["n_prop"] == 0  # no proposals in C5
    assert metrics["multiobjective"]["hypervolume"] > 0
    assert metrics["efficiency"]["t100_sec_per_100_proposals"] is not None


# ----------------------------------------------------------------------
# rollups + hypothesis tests
# ----------------------------------------------------------------------


def fake_metrics(condition, seed, *, hr, cvr=None, hv, terminal="COMPLETE"):
    return {
        "run_id": f"sha256:{condition}{seed}",
        "condition": condition,
        "seed": seed,
        "terminal_status": terminal,
        "budget": {
            "n_eval_target": 50,
            "n_eval_consumed": 10,
            "proposal_attempts": 40,
            "objective_eval_cache_hits": 3,
        },
        "validity_funnel": {
            "n_prop": 40,
            "counts": {},
            "rates": {
                "parse": 1.0,
                "schema": 0.95,
                "identifier": 0.9,
                "applicability": 0.85,
                "objective_evaluated": 0.25,
            },
        },
        "hallucination": {
            "hr_all_proposals": hr,
            "hr_schema_valid_only": hr,
            "categories": {},
        },
        "constraints": {
            "n_deterministically_checked": 0 if cvr is None else 10,
            "cvr_proposal_level": cvr,
            "cvr_rule_level": None,
            "missing_essential_fields_count": 0,
        },
        "objectives": {
            "baseline": {"cost_eur": 63.0, "mass_kg": 0.22},
            "best_cost_eur": 48.0,
            "mass_of_best_cost_kg": 0.25,
            "lowest_mass_kg": 0.13,
            "cost_of_lowest_mass_eur": 68.0,
            "n_pareto_improving": 2,
        },
        "multiobjective": {
            "reference_point": [75.6, 0.264],
            "hypervolume": hv,
            "normalized_hypervolume": hv / (75.6 * 0.264),
            "delta_hv": hv,
            "pareto_archive_size": 3,
            "ideal_point": [48.0, 0.13],
            "min_ideal_point_distance": 0.3,
        },
        "efficiency": {
            "wall_clock_sec": 100.0 + seed,
            "t100_sec_per_100_proposals": 250.0,
            "eta_hv_per_sec": hv / 100.0,
            "total_tokens": None,
        },
        "software": {},
    }


def test_seed_summary_row_and_columns():
    m = fake_metrics("C2", 0, hr=0.2, hv=10.0)
    row = seed_summary_row(m)
    assert set(row) == set(SEED_SUMMARY_COLUMNS)
    assert row["condition"] == "C2"
    assert row["hr_all"] == 0.2
    assert row["hypervolume"] == 10.0


def test_build_condition_summary_aggregates_by_condition():
    metrics_list = [
        fake_metrics("C2", 0, hr=0.30, hv=8.0),
        fake_metrics("C2", 1, hr=0.20, hv=10.0),
        fake_metrics("C2", 2, hr=0.10, hv=12.0),
        fake_metrics("C3", 0, hr=0.10, hv=9.0),
        fake_metrics("C3", 1, hr=0.05, hv=11.0),
        fake_metrics("C3", 2, hr=0.00, hv=13.0),
    ]
    rows = build_condition_summary(metrics_list)
    assert {r["condition"] for r in rows} == {"C2", "C3"}
    c2 = next(r for r in rows if r["condition"] == "C2")
    assert c2["n_seeds"] == 3
    assert c2["n_complete"] == 3
    assert c2["hv_median"] == 10.0
    assert c2["hr_all_median"] == pytest.approx(0.20)
    assert set(rows[0]) == set(CONDITION_SUMMARY_COLUMNS)


def test_hypothesis_tests_h1_underpowered_and_pending_rows():
    metrics_by_condition = {
        "C2": [
            fake_metrics("C2", 0, hr=0.30, hv=8.0),
            fake_metrics("C2", 1, hr=0.20, hv=10.0),
            fake_metrics("C2", 2, hr=0.25, hv=12.0),
        ],
        "C3": [
            fake_metrics("C3", 0, hr=0.10, hv=9.0),
            fake_metrics("C3", 1, hr=0.08, hv=11.0),
            fake_metrics("C3", 2, hr=0.05, hv=13.0),
        ],
    }
    rows = hypothesis_tests(metrics_by_condition)
    h1_hr = next(
        r
        for r in rows
        if r["hypothesis"] == "H1" and r["metric"] == "hallucination_rate"
    )
    assert h1_hr["c_ref"] == "C2" and h1_hr["c_test"] == "C3"
    assert h1_hr["relative_reduction_pct"] > 30.0
    assert h1_hr["threshold_met"] is True
    assert h1_hr["p_one_sided"] == pytest.approx(0.125)
    assert h1_hr["significant"] is False  # 0.125 > 0.025
    assert h1_hr["decision"] == "NOT_SUPPORTED"
    assert "underpowered" in h1_hr["notes"]

    h1_cvr = next(
        r
        for r in rows
        if r["hypothesis"] == "H1"
        and r["metric"] == "constraint_violation_rate"
    )
    assert h1_cvr["decision"] == "NOT_COMPUTABLE"  # cvr is None on every seed

    pending = {r["hypothesis"] for r in rows if r["decision"] == "PENDING_C4"}
    assert pending == {"H2", "H3", "H4"}


def test_hypothesis_tests_without_c3():
    rows = hypothesis_tests({"C2": [fake_metrics("C2", 0, hr=0.2, hv=8.0)]})
    h1 = next(r for r in rows if r["hypothesis"] == "H1")
    assert h1["decision"] == "NOT_COMPUTABLE"


def test_write_csv_round_trips(tmp_path):
    rows = build_seed_summary(
        [
            fake_metrics("C2", 0, hr=0.2, hv=8.0),
            fake_metrics("C3", 0, hr=0.1, hv=9.0),
        ]
    )
    path = write_csv(tmp_path / "seed_summary.csv", rows, SEED_SUMMARY_COLUMNS)
    with path.open() as handle:
        loaded = list(csv.DictReader(handle))
    assert [r["condition"] for r in loaded] == ["C2", "C3"]
    assert loaded[0]["hr_all"] == "0.2"


def test_hypothesis_test_columns_are_stable():
    rows = hypothesis_tests({})
    path_cols = set(HYPOTHESIS_TEST_COLUMNS)
    for row in rows:
        assert set(row).issubset(path_cols)
