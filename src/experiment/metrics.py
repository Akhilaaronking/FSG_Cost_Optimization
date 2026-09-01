"""
Metrics for the A12 harness (thesis 11.7-11.11, 11.15, 11.19).

Per-run:  compute_metrics(...) -> metrics.json, a function of the run's
          run_config.json + events.jsonl + pareto_archive.json plus the
          terminal status and wall-clock the driver reports.

Rollups:  build_seed_summary / build_condition_summary -> the two CSVs
          in results/ (11.19), with descriptive statistics per 11.15
          (median, IQR, mean, sd).

Stats:    hypothesis_tests(...) -> results/hypothesis_tests.csv. Only
          H1 (C3 vs C2) is computable with {C1,C2,C3,C5}; H2/H3/H4 rows
          are emitted as PENDING_C4. Paired one-sided Wilcoxon
          signed-rank, Bonferroni alpha* = 0.05 / 2, effect sizes
          d_z (eq 11.27) and rank-biserial (eq 11.28). At the reduced
          seed count the test is pre-declared underpowered (docs/A12
          section 8).
"""

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

try:  # scipy is a locked dependency (see ENV_DEVIATIONS.txt)
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:  # pragma: no cover
    _scipy_wilcoxon = None


H1_ALPHA = 0.05
H1_FAMILY_SIZE = 2  # hallucination rate + constraint-violation rate
H1_ALPHA_CORRECTED = H1_ALPHA / H1_FAMILY_SIZE
H1_HR_THRESHOLD_PCT = 30.0
H1_CVR_THRESHOLD_PCT = 20.0

H2_ALPHA = 0.05           # single primary comparison, no family correction
H3_HR_TARGET = 0.05       # eq 10.39
C4_CONDITION_NAMES = ("C4", "C4_base")
C4_ABLATION_SUFFIXES = ("no_rag", "no_schema", "no_validator")


# ----------------------------------------------------------------------
# per-run metrics
# ----------------------------------------------------------------------


_PROPOSAL_EVENT_TYPES = ("proposal", "agentic_step")

# fields apply_proposal can write; a candidate whose diff touches only
# these is a purely categorical move (eq 3.53 categorical subset).
_CATEGORICAL_FIELDS = ("material_id", "process_id")


def _proposal_events(events: list[dict]) -> list[dict]:
    # C4 "agentic_step" events carry the same validity / hallucination /
    # evaluation shape as C1/C2/C3 "proposal" events, so the funnel,
    # hallucination and constraint metrics apply unchanged.
    return [
        e
        for e in events
        if e.get("event_type") in _PROPOSAL_EVENT_TYPES
    ]


def _rate(numerator: int, denominator: int):
    return (numerator / denominator) if denominator else None


def _validity_funnel(proposal_events: list[dict]) -> dict:
    n_prop = len(proposal_events)
    parse = schema = identifier = applicability = evaluated = 0
    for event in proposal_events:
        v = event.get("validity") or {}
        if not v.get("parse_valid"):
            continue
        parse += 1
        if not v.get("schema_valid"):
            continue
        schema += 1
        if not v.get("authority_valid"):
            continue
        identifier += 1
        if not v.get("applicability_valid"):
            continue
        applicability += 1
        ev = event.get("evaluation")
        if ev is not None and (
            ev.get("consumed_objective_budget")
            or ev.get("objective_eval_cache_hit")
        ):
            evaluated += 1

    counts = {
        "parse": parse,
        "schema": schema,
        "identifier": identifier,
        "applicability": applicability,
        "objective_evaluated": evaluated,
    }
    rates = {
        key: _rate(value, n_prop) for key, value in counts.items()
    }
    return {"n_prop": n_prop, "counts": counts, "rates": rates}


def _hallucination(proposal_events: list[dict]) -> dict:
    n_prop = len(proposal_events)
    hallucinated_all = 0
    schema_valid = 0
    hallucinated_schema_valid = 0
    categories: Counter = Counter()

    for event in proposal_events:
        halluc = event.get("hallucination") or {}
        is_hallucinated = bool(halluc.get("hallucinated"))
        if is_hallucinated:
            hallucinated_all += 1
        categories.update(halluc.get("categories") or [])

        v = event.get("validity") or {}
        if v.get("parse_valid") and v.get("schema_valid"):
            schema_valid += 1
            if is_hallucinated:
                hallucinated_schema_valid += 1

    return {
        "hr_all_proposals": _rate(hallucinated_all, n_prop),
        "hr_schema_valid_only": _rate(
            hallucinated_schema_valid, schema_valid
        ),
        "categories": dict(categories),
    }


def _constraints(proposal_events: list[dict]) -> dict:
    checked = 0
    proposal_violations = 0
    rule_violations = 0
    rule_checks = 0
    missing_fields = 0

    for event in proposal_events:
        ev = event.get("evaluation")
        if not ev:
            continue
        c = ev.get("constraints") or {}
        if not c.get("evaluated"):
            continue
        checked += 1
        if c.get("proposal_level_violation"):
            proposal_violations += 1
        if isinstance(c.get("rule_level_violations"), (int, float)):
            rule_violations += c["rule_level_violations"]
        if isinstance(c.get("rule_level_checks"), (int, float)):
            rule_checks += c["rule_level_checks"]
        missing_fields += len(c.get("missing_essential_fields") or [])

    return {
        "n_deterministically_checked": checked,
        "cvr_proposal_level": _rate(proposal_violations, checked),
        "cvr_rule_level": _rate(rule_violations, rule_checks),
        "missing_essential_fields_count": missing_fields,
    }


def _archive_objectives(pareto_archive: dict | None) -> dict:
    entries = (pareto_archive or {}).get("entries") or []
    if not entries:
        return {
            "best_cost_eur": None,
            "mass_of_best_cost_kg": None,
            "lowest_mass_kg": None,
            "cost_of_lowest_mass_eur": None,
        }
    by_cost = min(entries, key=lambda e: (e["cost_eur"], e["mass_kg"]))
    by_mass = min(entries, key=lambda e: (e["mass_kg"], e["cost_eur"]))
    return {
        "best_cost_eur": by_cost["cost_eur"],
        "mass_of_best_cost_kg": by_cost["mass_kg"],
        "lowest_mass_kg": by_mass["mass_kg"],
        "cost_of_lowest_mass_eur": by_mass["cost_eur"],
    }


def _multiobjective(pareto_archive: dict | None) -> dict:
    archive = pareto_archive or {}
    entries = archive.get("entries") or []
    ref = archive.get("reference_point")
    hv = archive.get("hypervolume")

    ideal_point = None
    min_distance = None
    if entries:
        ideal_point = [
            min(e["cost_eur"] for e in entries),
            min(e["mass_kg"] for e in entries),
        ]
        if ref:
            spans = [
                ref[0] - ideal_point[0],
                ref[1] - ideal_point[1],
            ]
            distances = []
            for e in entries:
                norm = [
                    (e["cost_eur"] - ideal_point[0]) / spans[0]
                    if spans[0]
                    else 0.0,
                    (e["mass_kg"] - ideal_point[1]) / spans[1]
                    if spans[1]
                    else 0.0,
                ]
                distances.append(math.hypot(norm[0], norm[1]))
            min_distance = min(distances)

    # eq 3.53 -- HV over the archive restricted to purely categorical
    # (material_id / process_id) candidates. While those are the only
    # decision variables it equals `hypervolume`; it will diverge once
    # geometry/continuous variables enter.
    from src.optimization.hypervolume import hypervolume_2d

    categorical_hv = None
    if entries and ref:
        cat_points = [
            {
                "candidate_id": e.get("candidate_id", ""),
                "objective_vector": e.get("objective_vector")
                or [e["cost_eur"], e["mass_kg"]],
            }
            for e in entries
            if all(
                m.get("field") in _CATEGORICAL_FIELDS
                for m in e.get("modifications", [])
            )
        ]
        categorical_hv = hypervolume_2d(cat_points, ref)

    return {
        "reference_point": ref,
        "hypervolume": hv,
        "normalized_hypervolume": archive.get("normalized_hypervolume"),
        # HV_baseline = 0 (empty start archive) for every condition
        "delta_hv": hv,
        "categorical_subset_hypervolume": categorical_hv,
        "pareto_archive_size": archive.get("archive_size"),
        "ideal_point": ideal_point,
        "min_ideal_point_distance": min_distance,
    }


def _n_pareto_improving(proposal_events: list[dict]) -> int:
    return sum(
        1
        for e in proposal_events
        if (e.get("archive") or {}).get("status") == "pareto_improving"
    )


def _efficiency(
    *,
    wall_clock_sec: float,
    n_prop: int,
    fallback_denominator: int,
    delta_hv,
    events: list[dict],
) -> dict:
    denom = n_prop or fallback_denominator
    t100 = (
        100.0 * wall_clock_sec / denom if denom and wall_clock_sec else None
    )
    eta_hv = (
        delta_hv / wall_clock_sec
        if (delta_hv is not None and wall_clock_sec)
        else None
    )
    prompt_tokens = completion_tokens = 0
    have_tokens = False
    for e in events:
        tc = (e.get("efficiency") or {}).get("token_counts") or {}
        if tc.get("prompt") is not None:
            prompt_tokens += tc["prompt"]
            have_tokens = True
        if tc.get("completion") is not None:
            completion_tokens += tc["completion"]
            have_tokens = True
    return {
        "wall_clock_sec": wall_clock_sec,
        "t100_sec_per_100_proposals": t100,
        "eta_hv_per_sec": eta_hv,
        "total_tokens": (prompt_tokens + completion_tokens)
        if have_tokens
        else None,
    }


_C4_STOP_RULE = {
    "COMPLETE": "budget",
    "COMPLETE_CONVERGED": "convergence",
    "ABORTED_BUDGET_UNREACHED": "attempt_cap",
    "ABORTED_PROVIDER": "provider_error",
}


def _c4_block(
    events: list[dict], run_config: dict, terminal_status: str
) -> dict | None:
    """The C4 loop summary (docs/A13 section 10), recomputed from the
    agentic_step events. None for non-C4 runs."""
    steps = [
        e for e in events if e.get("event_type") == "agentic_step"
    ]
    if not steps:
        return None

    accepted = [
        e for e in steps if (e.get("agentic") or {}).get("accepted")
    ]
    intents = Counter(
        e["agentic"]["selection"]["intent"] for e in steps
    )
    hv_trajectory = [
        e["agentic"]["hv_after"]
        for e in steps
        if (e.get("evaluation") or {}).get("consumed_objective_budget")
        and e["agentic"].get("hv_after") is not None
    ]

    # a step with retry_of_selection == 0 starts a new selection episode;
    # the episode's retry count is its last step's retry_of_selection.
    episode_retries: list[int] = []
    current = None
    for e in steps:
        r = e["agentic"].get("retry_of_selection", 0)
        if r == 0:
            if current is not None:
                episode_retries.append(current)
            current = 0
        else:
            current = r
    if current is not None:
        episode_retries.append(current)

    loop = (run_config.get("condition_spec") or {}).get("c4_loop") or {}
    conv = loop.get("convergence") or {}

    # the driver breaks on convergence for exactly one of two reasons and
    # checks hv_plateau first (src/experiment/c4_driver.py _converged);
    # reproduce that decision from the recomputed HV trajectory.
    convergence_reason = None
    if terminal_status == "COMPLETE_CONVERGED":
        L = int(conv.get("look_back_L") or 0)
        eps = float(conv.get("epsilon_hv") or 0.0)
        window = hv_trajectory[-L:] if L else []
        if len(window) == L and L and (max(window) - min(window)) < eps:
            convergence_reason = "hv_plateau"
        else:
            convergence_reason = "archive_unchanged"

    return {
        "steps": len(steps),
        "accepted_steps": len(accepted),
        "acceptance_rate": len(accepted) / len(steps),
        "selection_intent_counts": dict(intents),
        "mean_retries_per_selection": (
            sum(episode_retries) / len(episode_retries)
            if episode_retries
            else 0.0
        ),
        "hv_trajectory": hv_trajectory,
        "converged": terminal_status == "COMPLETE_CONVERGED",
        "convergence_reason": convergence_reason,
        "stop_rule": _C4_STOP_RULE.get(terminal_status, terminal_status),
        "ablation": loop.get("ablation"),
    }


def compute_metrics(
    run_config: dict,
    events: list[dict],
    *,
    terminal_status: str,
    wall_clock_sec: float,
    pareto_archive: dict | None = None,
) -> dict:
    """Assemble metrics.json for one run (docs/A12 section 5)."""
    condition = run_config["condition"]
    seed = run_config["seed"]
    budget = run_config["identity"]["budget"]

    proposal_events = _proposal_events(events)
    funnel = _validity_funnel(proposal_events)
    n_prop = funnel["n_prop"]

    fresh = sum(
        1
        for e in events
        if (e.get("evaluation") or {}).get("consumed_objective_budget")
    )
    cache_hits = sum(
        1
        for e in events
        if (e.get("evaluation") or {}).get("objective_eval_cache_hit")
    )
    nsga2_evals = sum(
        1 for e in events if e.get("event_type") == "nsga2_evaluation"
    )

    multiobjective = _multiobjective(pareto_archive)
    baseline_vector = (pareto_archive or {}).get("baseline_vector")

    return {
        "run_id": run_config["run_id"],
        "condition": condition,
        "seed": seed,
        "terminal_status": terminal_status,
        "budget": {
            "n_eval_target": budget["n_eval"],
            "n_eval_consumed": fresh,
            "proposal_attempts": n_prop,
            "objective_eval_cache_hits": cache_hits,
        },
        "validity_funnel": funnel,
        "hallucination": _hallucination(proposal_events),
        "constraints": _constraints(proposal_events),
        "objectives": {
            "baseline": {
                "cost_eur": baseline_vector[0]
                if baseline_vector
                else None,
                "mass_kg": baseline_vector[1]
                if baseline_vector
                else None,
            },
            **_archive_objectives(pareto_archive),
            "n_pareto_improving": _n_pareto_improving(proposal_events),
        },
        "multiobjective": multiobjective,
        "efficiency": _efficiency(
            wall_clock_sec=wall_clock_sec,
            n_prop=n_prop,
            fallback_denominator=nsga2_evals,
            delta_hv=multiobjective["delta_hv"],
            events=events,
        ),
        "c4": _c4_block(events, run_config, terminal_status),
        "software": (events[0]["software"] if events else None),
    }


# ----------------------------------------------------------------------
# disk helpers
# ----------------------------------------------------------------------


def write_metrics(run_dir, metrics: dict) -> Path:
    path = Path(run_dir) / "metrics.json"
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_run(run_dir) -> dict:
    """Read a runs/<condition>/seed_NN/ directory back into memory."""
    from src.experiment.events import read_events

    run_dir = Path(run_dir)
    run_config = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    events = read_events(run_dir / "events.jsonl")
    pareto_path = run_dir / "pareto_archive.json"
    pareto_archive = (
        json.loads(pareto_path.read_text(encoding="utf-8"))
        if pareto_path.is_file()
        else None
    )
    metrics_path = run_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else None
    )
    return {
        "run_config": run_config,
        "events": events,
        "pareto_archive": pareto_archive,
        "metrics": metrics,
    }


# ----------------------------------------------------------------------
# rollups
# ----------------------------------------------------------------------

SEED_SUMMARY_COLUMNS = [
    "condition",
    "seed",
    "run_id",
    "terminal_status",
    "n_eval_target",
    "n_eval_consumed",
    "n_prop",
    "parse_rate",
    "schema_rate",
    "identifier_rate",
    "applicability_rate",
    "objective_evaluated_rate",
    "hr_all",
    "hr_schema_valid",
    "cvr_proposal",
    "cvr_rule",
    "hypervolume",
    "normalized_hv",
    "delta_hv",
    "pareto_size",
    "best_cost_eur",
    "lowest_mass_kg",
    "min_ideal_distance",
    "wall_clock_sec",
    "t100_sec",
    "eta_hv",
    "total_tokens",
]

CONDITION_SUMMARY_COLUMNS = [
    "condition",
    "n_seeds",
    "n_complete",
    "hv_median",
    "hv_iqr",
    "hv_mean",
    "hv_sd",
    "norm_hv_median",
    "norm_hv_iqr",
    "hr_all_median",
    "hr_all_iqr",
    "hr_all_mean",
    "hr_all_sd",
    "cvr_proposal_median",
    "cvr_proposal_iqr",
    "parse_rate_mean",
    "schema_rate_mean",
    "identifier_rate_mean",
    "applicability_rate_mean",
    "best_cost_eur_median",
    "lowest_mass_kg_median",
    "wall_clock_sec_median",
    "t100_sec_median",
]


def seed_summary_row(metrics: dict) -> dict:
    funnel = metrics["validity_funnel"]
    rates = funnel["rates"]
    return {
        "condition": metrics["condition"],
        "seed": metrics["seed"],
        "run_id": metrics["run_id"],
        "terminal_status": metrics["terminal_status"],
        "n_eval_target": metrics["budget"]["n_eval_target"],
        "n_eval_consumed": metrics["budget"]["n_eval_consumed"],
        "n_prop": funnel["n_prop"],
        "parse_rate": rates["parse"],
        "schema_rate": rates["schema"],
        "identifier_rate": rates["identifier"],
        "applicability_rate": rates["applicability"],
        "objective_evaluated_rate": rates["objective_evaluated"],
        "hr_all": metrics["hallucination"]["hr_all_proposals"],
        "hr_schema_valid": metrics["hallucination"][
            "hr_schema_valid_only"
        ],
        "cvr_proposal": metrics["constraints"]["cvr_proposal_level"],
        "cvr_rule": metrics["constraints"]["cvr_rule_level"],
        "hypervolume": metrics["multiobjective"]["hypervolume"],
        "normalized_hv": metrics["multiobjective"][
            "normalized_hypervolume"
        ],
        "delta_hv": metrics["multiobjective"]["delta_hv"],
        "pareto_size": metrics["multiobjective"]["pareto_archive_size"],
        "best_cost_eur": metrics["objectives"]["best_cost_eur"],
        "lowest_mass_kg": metrics["objectives"]["lowest_mass_kg"],
        "min_ideal_distance": metrics["multiobjective"][
            "min_ideal_point_distance"
        ],
        "wall_clock_sec": metrics["efficiency"]["wall_clock_sec"],
        "t100_sec": metrics["efficiency"]["t100_sec_per_100_proposals"],
        "eta_hv": metrics["efficiency"]["eta_hv_per_sec"],
        "total_tokens": metrics["efficiency"]["total_tokens"],
    }


def build_seed_summary(metrics_list: list[dict]) -> list[dict]:
    rows = [seed_summary_row(m) for m in metrics_list]
    return sorted(rows, key=lambda r: (r["condition"], r["seed"]))


def _describe(values: list) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"median": None, "iqr": None, "mean": None, "sd": None}
    median = statistics.median(clean)
    mean = statistics.fmean(clean)
    sd = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    if len(clean) >= 2:
        q1, _, q3 = statistics.quantiles(clean, n=4, method="inclusive")
        iqr = q3 - q1
    else:
        iqr = 0.0
    return {"median": median, "iqr": iqr, "mean": mean, "sd": sd}


def build_condition_summary(metrics_list: list[dict]) -> list[dict]:
    by_condition: dict[str, list[dict]] = {}
    for m in metrics_list:
        by_condition.setdefault(m["condition"], []).append(m)

    rows = []
    for condition, group in sorted(by_condition.items()):
        hv = _describe([g["multiobjective"]["hypervolume"] for g in group])
        nhv = _describe(
            [g["multiobjective"]["normalized_hypervolume"] for g in group]
        )
        hr = _describe(
            [g["hallucination"]["hr_all_proposals"] for g in group]
        )
        cvr = _describe(
            [g["constraints"]["cvr_proposal_level"] for g in group]
        )
        rates = {
            key: _describe(
                [g["validity_funnel"]["rates"][key] for g in group]
            )["mean"]
            for key in (
                "parse",
                "schema",
                "identifier",
                "applicability",
            )
        }
        rows.append(
            {
                "condition": condition,
                "n_seeds": len(group),
                "n_complete": sum(
                    1
                    for g in group
                    if g["terminal_status"]
                    in (
                        "COMPLETE",
                        "COMPLETE_SPACE_EXHAUSTED",
                        "COMPLETE_CONVERGED",
                    )
                ),
                "hv_median": hv["median"],
                "hv_iqr": hv["iqr"],
                "hv_mean": hv["mean"],
                "hv_sd": hv["sd"],
                "norm_hv_median": nhv["median"],
                "norm_hv_iqr": nhv["iqr"],
                "hr_all_median": hr["median"],
                "hr_all_iqr": hr["iqr"],
                "hr_all_mean": hr["mean"],
                "hr_all_sd": hr["sd"],
                "cvr_proposal_median": cvr["median"],
                "cvr_proposal_iqr": cvr["iqr"],
                "parse_rate_mean": rates["parse"],
                "schema_rate_mean": rates["schema"],
                "identifier_rate_mean": rates["identifier"],
                "applicability_rate_mean": rates["applicability"],
                "best_cost_eur_median": _describe(
                    [g["objectives"]["best_cost_eur"] for g in group]
                )["median"],
                "lowest_mass_kg_median": _describe(
                    [g["objectives"]["lowest_mass_kg"] for g in group]
                )["median"],
                "wall_clock_sec_median": _describe(
                    [g["efficiency"]["wall_clock_sec"] for g in group]
                )["median"],
                "t100_sec_median": _describe(
                    [
                        g["efficiency"]["t100_sec_per_100_proposals"]
                        for g in group
                    ]
                )["median"],
            }
        )
    return rows


def write_csv(path, rows: list[dict], columns: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    return path


# ----------------------------------------------------------------------
# hypothesis tests (thesis 11.15, docs/A12 section 8)
# ----------------------------------------------------------------------


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while (
            j + 1 < len(order)
            and values[order[j + 1]] == values[order[i]]
        ):
            j += 1
        average = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def wilcoxon_paired(diffs: list[float]) -> dict:
    """
    One-sided (``greater``) paired Wilcoxon signed-rank on ``diffs``
    (ref - test, so positive = a reduction). Zeros dropped. Returns the
    statistic, exact one-sided p (scipy), d_z (eq 11.27), rank-biserial
    (eq 11.28), and whether the sample can even reach the corrected
    alpha.
    """
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    result = {
        "n_nonzero_pairs": n,
        "W_statistic": None,
        "p_one_sided": None,
        "effect_size_dz": None,
        "effect_size_rrb": None,
        "min_achievable_p_one_sided": (1.0 / (2**n)) if n else None,
        "underpowered": None,
    }
    if n == 0:
        return result

    ranks = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    result["effect_size_rrb"] = (w_plus - w_minus) / (n * (n + 1) / 2)

    if n >= 2:
        sd = statistics.pstdev(nonzero)
        if sd:
            result["effect_size_dz"] = statistics.fmean(nonzero) / sd

    if _scipy_wilcoxon is not None:
        try:
            wr = _scipy_wilcoxon(
                nonzero, alternative="greater", method="exact"
            )
            result["W_statistic"] = float(wr.statistic)
            result["p_one_sided"] = float(wr.pvalue)
        except Exception:  # pragma: no cover - degenerate samples
            pass

    result["underpowered"] = (
        result["min_achievable_p_one_sided"] > H1_ALPHA_CORRECTED
    )
    return result


HYPOTHESIS_TEST_COLUMNS = [
    "hypothesis",
    "comparison",
    "metric",
    "c_ref",
    "c_test",
    "ref_mean",
    "test_mean",
    "relative_reduction_pct",
    "absolute_reduction",
    "W_statistic",
    "p_one_sided",
    "alpha_corrected",
    "effect_size_dz",
    "effect_size_rrb",
    "n_nonzero_pairs",
    "threshold_pct",
    "threshold_met",
    "significant",
    "decision",
    "notes",
]


def _paired_by_seed(ref_metrics, test_metrics, extractor):
    ref = {m["seed"]: extractor(m) for m in ref_metrics}
    test = {m["seed"]: extractor(m) for m in test_metrics}
    seeds = sorted(set(ref) & set(test))
    pairs = [
        (ref[s], test[s])
        for s in seeds
        if ref[s] is not None and test[s] is not None
    ]
    return pairs


def _h1_row(metric_name, ref_metrics, test_metrics, extractor, threshold):
    pairs = _paired_by_seed(ref_metrics, test_metrics, extractor)
    row = {
        "hypothesis": "H1",
        "comparison": "C3_vs_C2",
        "metric": metric_name,
        "c_ref": "C2",
        "c_test": "C3",
        "alpha_corrected": H1_ALPHA_CORRECTED,
        "threshold_pct": threshold,
        "notes": "",
    }
    if len(pairs) < 2:
        row["decision"] = "NOT_COMPUTABLE"
        row["notes"] = (
            "fewer than 2 seeds with a value in both C2 and C3 "
            f"(had {len(pairs)}); metric may be unavailable on the "
            "frozen benchmark (11.10)"
        )
        return row

    ref_vals = [p[0] for p in pairs]
    test_vals = [p[1] for p in pairs]
    ref_mean = statistics.fmean(ref_vals)
    test_mean = statistics.fmean(test_vals)
    diffs = [r - t for r, t in pairs]

    row["ref_mean"] = ref_mean
    row["test_mean"] = test_mean
    row["absolute_reduction"] = ref_mean - test_mean
    row["relative_reduction_pct"] = (
        100.0 * (ref_mean - test_mean) / ref_mean if ref_mean else None
    )

    stats = wilcoxon_paired(diffs)
    row["W_statistic"] = stats["W_statistic"]
    row["p_one_sided"] = stats["p_one_sided"]
    row["effect_size_dz"] = stats["effect_size_dz"]
    row["effect_size_rrb"] = stats["effect_size_rrb"]
    row["n_nonzero_pairs"] = stats["n_nonzero_pairs"]

    if row["relative_reduction_pct"] is None:
        row["threshold_met"] = row["absolute_reduction"] > 0
    else:
        row["threshold_met"] = (
            row["relative_reduction_pct"] >= threshold
        )
    row["significant"] = (
        stats["p_one_sided"] is not None
        and stats["p_one_sided"] < H1_ALPHA_CORRECTED
    )
    row["decision"] = (
        "SUPPORTED"
        if (row["threshold_met"] and row["significant"])
        else "NOT_SUPPORTED"
    )
    notes = []
    if stats["underpowered"]:
        notes.append(
            "underpowered: with n="
            f"{stats['n_nonzero_pairs']} nonzero pairs the minimum "
            f"achievable one-sided p is "
            f"{stats['min_achievable_p_one_sided']:.3f} > "
            f"{H1_ALPHA_CORRECTED} -- descriptive/effect-size only"
        )
    if row["relative_reduction_pct"] is None:
        notes.append("C2 mean is 0; reporting absolute reduction")
    row["notes"] = "; ".join(notes)
    return row


def _pick_c4(metrics_by_condition: dict) -> tuple[str | None, list]:
    for name in C4_CONDITION_NAMES:
        group = metrics_by_condition.get(name)
        if group:
            return name, group
    return None, []


def _mean_over_seeds(metrics: list, extractor):
    vals = [
        v for v in (extractor(m) for m in metrics) if v is not None
    ]
    return statistics.fmean(vals) if vals else None


def _h2_row(metric_name, c4_name, c4, c5, extractor):
    row = {
        "hypothesis": "H2",
        "comparison": f"{c4_name}_vs_C5",
        "metric": metric_name,
        "c_ref": "C5",
        "c_test": c4_name,
        "alpha_corrected": H2_ALPHA,
        "notes": "positive absolute_reduction = C4 hypervolume advantage",
    }
    pairs = _paired_by_seed(c5, c4, extractor)  # (c5_val, c4_val)
    if len(pairs) < 2:
        row["decision"] = "NOT_COMPUTABLE"
        row["notes"] = f"fewer than 2 shared seeds (had {len(pairs)})"
        return row

    c5_mean = statistics.fmean(p[0] for p in pairs)
    c4_mean = statistics.fmean(p[1] for p in pairs)
    diffs = [c4v - c5v for c5v, c4v in pairs]  # C4 - C5
    stats = wilcoxon_paired(diffs)  # one-sided 'greater': C4 > C5

    row["ref_mean"] = c5_mean
    row["test_mean"] = c4_mean
    row["absolute_reduction"] = c4_mean - c5_mean
    row["W_statistic"] = stats["W_statistic"]
    row["p_one_sided"] = stats["p_one_sided"]
    row["effect_size_dz"] = stats["effect_size_dz"]
    row["effect_size_rrb"] = stats["effect_size_rrb"]
    row["n_nonzero_pairs"] = stats["n_nonzero_pairs"]
    row["threshold_met"] = c4_mean >= c5_mean  # eq 3.52 / 3.53
    row["significant"] = (
        stats["p_one_sided"] is not None
        and stats["p_one_sided"] < H2_ALPHA
    )
    row["decision"] = (
        "SUPPORTED"
        if (row["threshold_met"] and row["significant"])
        else "NOT_SUPPORTED"
    )
    extra = []
    if stats["underpowered"]:
        extra.append(
            f"underpowered: min achievable one-sided p "
            f"{stats['min_achievable_p_one_sided']:.3f} > {H2_ALPHA}"
        )
    extra.append("a non-significant H2 is a valid result (11.13)")
    row["notes"] = "; ".join(extra)
    return row


def _h3_rows(metrics_by_condition: dict, c4_name: str, full: list) -> list[dict]:
    hr = lambda m: m["hallucination"]["hr_all_proposals"]
    hr_full = _mean_over_seeds(full, hr)

    ablations = {
        suffix: metrics_by_condition.get(f"{c4_name}_{suffix}", [])
        for suffix in C4_ABLATION_SUFFIXES
    }
    hr_abl = {
        s: _mean_over_seeds(g, hr) for s, g in ablations.items() if g
    }

    base = {
        "hypothesis": "H3",
        "comparison": f"{c4_name}_vs_ablations",
        "c_ref": c4_name,
        "alpha_corrected": H2_ALPHA,
    }
    below5 = {
        **base,
        "metric": "hr_full_below_0.05",
        "c_test": "-",
        "ref_mean": hr_full,
        "threshold_pct": H3_HR_TARGET,
        "threshold_met": (hr_full is not None and hr_full < H3_HR_TARGET),
        "decision": (
            "SUPPORTED"
            if (hr_full is not None and hr_full < H3_HR_TARGET)
            else "NOT_SUPPORTED"
        ),
        "notes": f"HR_full={hr_full}",
    }

    if not hr_abl:
        ordering = {
            **base,
            "metric": "hr_full_below_min_ablation",
            "c_test": "-",
            "ref_mean": hr_full,
            "decision": "PENDING_ABLATIONS",
            "notes": (
                "no ablation runs present "
                f"({c4_name}_no_rag / _no_schema / _no_validator)"
            ),
        }
    else:
        min_abl = min(hr_abl.values())
        ordering = {
            **base,
            "metric": "hr_full_below_min_ablation",
            "c_test": "/".join(sorted(hr_abl)),
            "ref_mean": hr_full,
            "test_mean": min_abl,
            "absolute_reduction": (
                (min_abl - hr_full)
                if (hr_full is not None and min_abl is not None)
                else None
            ),
            "threshold_met": (
                hr_full is not None and hr_full < min_abl
            ),
            "decision": (
                "SUPPORTED"
                if (hr_full is not None and hr_full < min_abl)
                else "NOT_SUPPORTED"
            ),
            "notes": (
                "; ".join(
                    f"HR_{s}={v}" for s, v in sorted(hr_abl.items())
                )
                + " (3-seed pilot; descriptive, no family correction)"
            ),
        }
    return [below5, ordering]


def hypothesis_tests(metrics_by_condition: dict[str, list[dict]]) -> list[dict]:
    """
    Rows for results/hypothesis_tests.csv.
      H1  C3 vs C2   -- paired one-sided Wilcoxon (docs/A12 section 8)
      H2  C4 vs C5   -- final HV + categorical-subset HV (eq 11.21 / 3.52 / 3.53)
      H3  C4 ablations -- HR_full < 0.05 and < min(ablations) (eq 10.38-10.39)
      H4  transfer   -- PENDING_C4
    Rows for a hypothesis whose inputs are absent are emitted PENDING/
    NOT_COMPUTABLE rather than omitted.
    """
    rows = []
    c2 = metrics_by_condition.get("C2", [])
    c3 = metrics_by_condition.get("C3", [])

    if c2 and c3:
        rows.append(
            _h1_row(
                "hallucination_rate",
                c2,
                c3,
                lambda m: m["hallucination"]["hr_all_proposals"],
                H1_HR_THRESHOLD_PCT,
            )
        )
        rows.append(
            _h1_row(
                "constraint_violation_rate",
                c2,
                c3,
                lambda m: m["constraints"]["cvr_proposal_level"],
                H1_CVR_THRESHOLD_PCT,
            )
        )
    else:
        rows.append(
            {
                "hypothesis": "H1",
                "comparison": "C3_vs_C2",
                "metric": "hallucination_rate",
                "decision": "NOT_COMPUTABLE",
                "notes": "C2 and/or C3 metrics not present",
            }
        )

    # -- H2: C4 (or C4_base) vs C5 --
    c4_name, c4 = _pick_c4(metrics_by_condition)
    c5 = metrics_by_condition.get("C5", [])
    if c4 and c5:
        rows.append(
            _h2_row(
                "final_hypervolume",
                c4_name,
                c4,
                c5,
                lambda m: m["multiobjective"]["hypervolume"],
            )
        )
        rows.append(
            _h2_row(
                "categorical_subset_hypervolume",
                c4_name,
                c4,
                c5,
                lambda m: m["multiobjective"].get(
                    "categorical_subset_hypervolume"
                ),
            )
        )
    else:
        rows.append(
            {
                "hypothesis": "H2",
                "comparison": "C4_vs_C5",
                "metric": "final_hypervolume",
                "decision": "PENDING_C4",
                "notes": "C4/C4_base and/or C5 metrics not present",
            }
        )

    # -- H3: C4 full vs its ablations --
    if c4:
        rows.extend(_h3_rows(metrics_by_condition, c4_name, c4))
    else:
        rows.append(
            {
                "hypothesis": "H3",
                "comparison": "C4_vs_ablations",
                "metric": "",
                "decision": "PENDING_C4",
                "notes": "C4/C4_base metrics not present",
            }
        )

    # -- H4: transfer case --
    rows.append(
        {
            "hypothesis": "H4",
            "comparison": "transfer_case",
            "metric": "",
            "decision": "PENDING_C4",
            "notes": "transfer case not in scope",
        }
    )
    return rows
