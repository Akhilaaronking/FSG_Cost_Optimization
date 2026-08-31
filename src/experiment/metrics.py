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


# ----------------------------------------------------------------------
# per-run metrics
# ----------------------------------------------------------------------


def _proposal_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("event_type") == "proposal"]


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

    return {
        "reference_point": ref,
        "hypervolume": hv,
        "normalized_hypervolume": archive.get("normalized_hypervolume"),
        # HV_baseline = 0 (empty start archive) for every condition
        "delta_hv": hv,
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
                    in ("COMPLETE", "COMPLETE_SPACE_EXHAUSTED")
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


def hypothesis_tests(metrics_by_condition: dict[str, list[dict]]) -> list[dict]:
    """
    Rows for results/hypothesis_tests.csv. H1 (C3 vs C2) is evaluated
    from paired seed-level metrics; H2/H3/H4 need C4 and are emitted as
    PENDING_C4 (docs/A12 section 8).
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

    for hypothesis, comparison in (
        ("H2", "C4_vs_C5"),
        ("H3", "C4_vs_ablations"),
        ("H4", "transfer_case"),
    ):
        rows.append(
            {
                "hypothesis": hypothesis,
                "comparison": comparison,
                "metric": "",
                "decision": "PENDING_C4",
                "notes": "C4 not in A12 scope",
            }
        )
    return rows
