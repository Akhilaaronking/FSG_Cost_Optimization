"""
A12 unified experiment runner (docs/A12 section 7).

Runs any of C1 / C2 / C3 / C5 under one run-identity model, one event
log, one equal-budget definition. Conditions execute in the fixed order
C1 -> C2 -> C3 -> C5 (never interleaved) so the Ollama model, the MLX
LoRA model and the MiniLM embedder are never all resident at once.

    python scripts/run_experiment.py                       # full sweep, seeds 0-9
    python scripts/run_experiment.py --condition C2 --seeds 0-9
    python scripts/run_experiment.py --condition C1,C2,C3,C5 --budget 50
    python scripts/run_experiment.py --dry-run             # write run_config.json only

Outputs, under --out (default "."):
    runs/<condition>/seed_NN/{run_config.json, events.jsonl,
                              pareto_archive.json, metrics.json}
    results/{seed_summary.csv, condition_summary.csv,
             hypothesis_tests.csv, RUN_NOTES.md, figures/}
    run_index.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.evaluator import evaluate_bom
from src.experiment.drivers import GenerativeDriver, Nsga2Driver
from src.experiment.events import EventLog, read_events
from src.experiment.identity import (
    BENCHMARK_PATH,
    CONDITIONS,
    DEFAULT_N_EVAL,
    DEFAULT_SEEDS,
    GENERATIVE_CONDITIONS,
    MLX_BASE_MODEL,
    OLLAMA_MODEL_ID,
    PROJECT_ROOT,
    SEARCH_SPACE_PATH,
    build_run_config,
    git_identity,
    seed_dir_name,
    write_run_config,
)
from src.experiment.metrics import (
    CONDITION_SUMMARY_COLUMNS,
    HYPOTHESIS_TEST_COLUMNS,
    SEED_SUMMARY_COLUMNS,
    build_condition_summary,
    build_seed_summary,
    compute_metrics,
    hypothesis_tests,
    write_csv,
    write_metrics,
)
from src.experiment.probe import probe_c3

# fixed non-interleaved order
CONDITION_ORDER = ("C1", "C2", "C3", "C5")


# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------


def parse_seeds(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(part) for part in text.split(",") if part.strip() != ""]


def parse_conditions(text: str) -> list[str]:
    if text.strip().lower() == "all":
        chosen = set(CONDITION_ORDER)
    else:
        chosen = {c.strip().upper() for c in text.split(",") if c.strip()}
    unknown = chosen - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown condition(s): {sorted(unknown)}")
    return [c for c in CONDITION_ORDER if c in chosen]


def parse_parts(text: str, baseline_bom: dict) -> list[str]:
    all_ids = [p["part_id"] for p in baseline_bom["parts"]]
    if text.strip().lower() == "all":
        return all_ids
    wanted = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in all_ids]
    if unknown:
        raise ValueError(f"unknown part id(s): {unknown}")
    return wanted


# ----------------------------------------------------------------------
# real backend / retriever factories (injected in tests)
# ----------------------------------------------------------------------


def build_generator(condition: str):
    from src.data.registry import DataRegistry
    from src.llm.generator import ProposalGenerator

    if condition in ("C1", "C2"):
        from src.llm.backend import OllamaBackend

        backend = OllamaBackend(OLLAMA_MODEL_ID)
    elif condition == "C3":
        from src.llm.backend import MLXLoRABackend

        backend = MLXLoRABackend(model_name=MLX_BASE_MODEL)
    else:
        raise ValueError(condition)
    return ProposalGenerator(backend, registry=DataRegistry())


def build_retriever(condition: str):
    if condition not in ("C2", "C3"):
        return None
    from src.rag.embeddings import SentenceTransformerEmbedder
    from src.rag.retriever import RagRetriever

    return RagRetriever(
        str(PROJECT_ROOT / "data" / "rag" / "corpus.jsonl"),
        SentenceTransformerEmbedder(),
    )


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------


def run_one(
    condition: str,
    seed: int,
    *,
    budget: int,
    parts: list[str],
    runs_root: Path,
    baseline_bom: dict,
    search_space: dict | None,
    generator=None,
    retriever=None,
    generative_evaluator=None,
    overwrite: bool,
    dry_run: bool,
    deviations: list[dict] | None = None,
) -> dict | None:
    seed_dir = runs_root / condition / seed_dir_name(seed)
    cfg = build_run_config(
        condition,
        seed=seed,
        n_eval=budget,
        target_parts=parts,
        deviations=deviations,
    )
    write_run_config(cfg, seed_dir)
    if dry_run:
        return None

    log = EventLog(
        seed_dir / "events.jsonl",
        run_id=cfg["run_id"],
        condition=condition,
        seed=seed,
        if_exists="overwrite" if overwrite else "error",
    )
    try:
        if condition in GENERATIVE_CONDITIONS:
            driver = GenerativeDriver(
                cfg,
                generator=generator,
                baseline_bom=baseline_bom,
                retriever=retriever,
                evaluator=generative_evaluator,
            )
        else:
            driver = Nsga2Driver(
                cfg,
                baseline_bom=baseline_bom,
                search_space=search_space,
            )
        outcome = driver.run(
            log, pareto_archive_path=seed_dir / "pareto_archive.json"
        )
    finally:
        log.close()

    pareto_path = seed_dir / "pareto_archive.json"
    pareto = (
        json.loads(pareto_path.read_text(encoding="utf-8"))
        if pareto_path.is_file()
        else None
    )
    metrics = compute_metrics(
        cfg,
        read_events(seed_dir / "events.jsonl"),
        terminal_status=outcome.terminal_status,
        wall_clock_sec=outcome.wall_clock_sec,
        pareto_archive=pareto,
    )
    write_metrics(seed_dir, metrics)
    return metrics


# ----------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------


def run_sweep(
    *,
    conditions: list[str],
    seeds: list[int],
    budget: int,
    parts_spec: str,
    out_root: Path,
    overwrite: bool,
    dry_run: bool,
    skip_c3_probe: bool = False,
    generator_factory=build_generator,
    retriever_factory=build_retriever,
    generative_evaluator=None,
    log=print,
) -> dict:
    out_root = Path(out_root)
    runs_root = out_root / "runs"
    results_root = out_root / "results"

    baseline_bom = json.loads(
        (PROJECT_ROOT / BENCHMARK_PATH).read_text(encoding="utf-8")
    )
    parts = parse_parts(parts_spec, baseline_bom)

    search_space = None
    if "C5" in conditions:
        from src.optimization.search_space import (
            load_verified_real_search_space,
        )

        search_space = load_verified_real_search_space(
            str(PROJECT_ROOT / SEARCH_SPACE_PATH),
            str(PROJECT_ROOT / BENCHMARK_PATH),
        )

    all_metrics: list[dict] = []
    blocked: list[dict] = []

    for condition in [c for c in CONDITION_ORDER if c in conditions]:
        deviations = None

        if condition == "C3" and not dry_run and not skip_c3_probe:
            probe = probe_c3(deep=True)
            log(probe.summary_line())
            if not probe.ok:
                blocked.append(
                    {"condition": "C3", "probe": probe.as_dict()}
                )
                log(
                    "C3: SKIPPED for the sweep (blocked, environment); "
                    "C1/C2/C5 unaffected."
                )
                continue
            deviations = None

        generator = None
        retriever = None
        if condition in GENERATIVE_CONDITIONS and not dry_run:
            log(f"[{condition}] loading backend + retriever ...")
            generator = generator_factory(condition)
            retriever = retriever_factory(condition)

        for seed in seeds:
            started = time.perf_counter()
            metrics = run_one(
                condition,
                seed,
                budget=budget,
                parts=parts,
                runs_root=runs_root,
                baseline_bom=baseline_bom,
                search_space=search_space,
                generator=generator,
                retriever=retriever,
                generative_evaluator=generative_evaluator,
                overwrite=overwrite,
                dry_run=dry_run,
                deviations=deviations,
            )
            if dry_run:
                log(f"[{condition} seed {seed}] run_config.json written")
                continue
            elapsed = time.perf_counter() - started
            all_metrics.append(metrics)
            b = metrics["budget"]
            log(
                f"[{condition} seed {seed}] {metrics['terminal_status']:24s} "
                f"n_eval={b['n_eval_consumed']}/{b['n_eval_target']} "
                f"n_prop={b['proposal_attempts']} "
                f"hr={metrics['hallucination']['hr_all_proposals']} "
                f"HV={metrics['multiobjective']['hypervolume']:.3f} "
                f"({elapsed:.0f}s)"
            )

        # free the models before the next condition
        del generator, retriever

    if dry_run:
        _write_run_index(out_root, runs_root, conditions, seeds)
        return {"metrics": [], "blocked": blocked, "notes": []}

    by_condition: dict[str, list[dict]] = {}
    for m in all_metrics:
        by_condition.setdefault(m["condition"], []).append(m)

    write_csv(
        results_root / "seed_summary.csv",
        build_seed_summary(all_metrics),
        SEED_SUMMARY_COLUMNS,
    )
    write_csv(
        results_root / "condition_summary.csv",
        build_condition_summary(all_metrics),
        CONDITION_SUMMARY_COLUMNS,
    )
    write_csv(
        results_root / "hypothesis_tests.csv",
        hypothesis_tests(by_condition),
        HYPOTHESIS_TEST_COLUMNS,
    )
    (results_root / "figures").mkdir(parents=True, exist_ok=True)
    _write_run_index(out_root, runs_root, conditions, seeds)

    notes = findings(all_metrics, blocked)
    write_run_notes(results_root, all_metrics, by_condition, blocked, notes)
    for line in _summary_block(all_metrics, by_condition, blocked, notes):
        log(line)

    return {"metrics": all_metrics, "blocked": blocked, "notes": notes}


def _write_run_index(out_root, runs_root, conditions, seeds):
    index = {}
    for condition in conditions:
        for seed in seeds:
            cfg_path = (
                runs_root
                / condition
                / seed_dir_name(seed)
                / "run_config.json"
            )
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                index[cfg["run_id"]] = str(
                    cfg_path.parent.relative_to(out_root)
                )
    payload = {
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "git": git_identity(),
        "conditions": conditions,
        "seeds": seeds,
        "deviations": [],
        "runs": index,
    }
    (out_root / "run_index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# data-driven findings -- emitted only when the data shows them
# ----------------------------------------------------------------------


def findings(all_metrics: list[dict], blocked: list[dict]) -> list[str]:
    notes: list[str] = []
    gen = [
        m for m in all_metrics if m["condition"] in GENERATIVE_CONDITIONS
    ]

    # 1. candidate-diversity ceiling
    exhausted = [
        m
        for m in gen
        if m["terminal_status"] == "COMPLETE_SPACE_EXHAUSTED"
    ]
    if exhausted:
        consumed = sorted(
            m["budget"]["n_eval_consumed"] for m in exhausted
        )
        target = exhausted[0]["budget"]["n_eval_target"]
        med = consumed[len(consumed) // 2]
        notes.append(
            "CANDIDATE-DIVERSITY CEILING: "
            f"{len(exhausted)}/{len(gen)} generative runs terminated "
            f"COMPLETE_SPACE_EXHAUSTED at a median of {med} distinct "
            f"candidates (range {consumed[0]}-{consumed[-1]}) against the "
            f"N={target} budget. The atomic material_id/process_id space "
            "over this benchmark is that small. The equal-budget ceiling "
            "is shared with C5, but only C5 (NSGA-II) reaches it -- read "
            "C1/C2/C3 vs C5 hypervolume as a search-reach difference, not "
            "a like-for-like budget comparison."
        )

    # 2. zero-hallucination baseline
    c1 = [m for m in all_metrics if m["condition"] == "C1"]
    if c1 and all(
        (m["hallucination"]["hr_all_proposals"] or 0.0) == 0.0
        and m["validity_funnel"]["n_prop"] > 0
        for m in c1
    ):
        total_prop = sum(m["validity_funnel"]["n_prop"] for m in c1)
        notes.append(
            "ZERO-HALLUCINATION BASELINE: C1 (ungrounded base LLM) "
            f"produced 0 hallucinations across all {total_prop} proposals "
            f"in {len(c1)} seeds on this benchmark. RAG (C2) and "
            "fine-tuning (C3) cannot reduce a rate already at zero, so "
            "H1's hallucination-reduction axis is reported as an absolute "
            "difference near 0 / NOT_COMPUTABLE (thesis 11.12). The "
            "measurable C1->C2->C3 signal here is candidate selection and "
            "the HV / validity-funnel trend (docs/A12 section 8)."
        )

    if blocked:
        for entry in blocked:
            notes.append(
                f"{entry['condition']} BLOCKED (environment): "
                f"{entry['probe']['detail']}"
            )

    aborted = [
        m
        for m in all_metrics
        if m["terminal_status"].startswith("ABORTED")
    ]
    if aborted:
        detail = ", ".join(
            f"{m['condition']}/seed{m['seed']}={m['terminal_status']}"
            for m in aborted
        )
        notes.append(
            f"ABORTED RUNS ({len(aborted)}): {detail} -- excluded from "
            "inferential analysis until rerun under the same frozen "
            "config (thesis 11.18)."
        )
    return notes


def _summary_block(all_metrics, by_condition, blocked, notes) -> list[str]:
    lines = ["", "=" * 72, "A12 SWEEP SUMMARY", "=" * 72]
    for condition in CONDITION_ORDER:
        group = by_condition.get(condition)
        if not group:
            continue
        n = len(group)
        hv = sorted(m["multiobjective"]["hypervolume"] for m in group)
        consumed = sorted(
            m["budget"]["n_eval_consumed"] for m in group
        )
        hr = [
            m["hallucination"]["hr_all_proposals"] or 0.0 for m in group
        ]
        statuses = {m["terminal_status"] for m in group}
        lines.append(
            f"{condition}: {n} seeds | status {sorted(statuses)} | "
            f"n_eval {consumed[0]}-{consumed[-1]} | "
            f"HV median {hv[len(hv) // 2]:.3f} | "
            f"hr_all max {max(hr):.3f}"
        )
    lines.append("-" * 72)
    if notes:
        lines.append("FINDINGS (from this run's data):")
        for note in notes:
            lines.append("")
            for wrapped in _wrap(note, 70):
                lines.append("  " + wrapped)
    else:
        lines.append("No diversity-ceiling / zero-baseline findings triggered.")
    lines.append("=" * 72)
    return lines


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    out, line = [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def write_run_notes(
    results_root: Path, all_metrics, by_condition, blocked, notes
) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A12 Run Notes",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Git commit: {git_identity().get('commit')}",
        "",
        "This file records the interpretive context for this run's",
        "results/ CSVs so it travels with the artifacts.",
        "",
        "## Per-condition",
        "",
        "| condition | seeds | terminal statuses | n_eval range | HV median | hr_all max |",
        "|---|---|---|---|---|---|",
    ]
    for condition in CONDITION_ORDER:
        group = by_condition.get(condition)
        if not group:
            continue
        hv = sorted(m["multiobjective"]["hypervolume"] for m in group)
        consumed = sorted(m["budget"]["n_eval_consumed"] for m in group)
        hr = [m["hallucination"]["hr_all_proposals"] or 0.0 for m in group]
        statuses = ", ".join(
            sorted({m["terminal_status"] for m in group})
        )
        lines.append(
            f"| {condition} | {len(group)} | {statuses} | "
            f"{consumed[0]}-{consumed[-1]} | {hv[len(hv) // 2]:.3f} | "
            f"{max(hr):.3f} |"
        )
    lines += ["", "## Findings (data-driven)", ""]
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none triggered")
    lines.append("")
    path = results_root / "RUN_NOTES.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="A12 unified experiment runner (C1/C2/C3/C5)."
    )
    parser.add_argument("--condition", default="all")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="e.g. '0-9' or '0,1,2' (default: 0-9, the thesis protocol)",
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument("--parts", default="all")
    parser.add_argument("--out", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-c3-probe", action="store_true")
    args = parser.parse_args(argv)

    conditions = parse_conditions(args.condition)
    seeds = parse_seeds(args.seeds)

    print(
        f"A12 runner | conditions {conditions} | seeds {seeds} | "
        f"budget {args.budget} | parts {args.parts} | out {args.out}"
        + (" | DRY RUN" if args.dry_run else "")
    )

    result = run_sweep(
        conditions=conditions,
        seeds=seeds,
        budget=args.budget,
        parts_spec=args.parts,
        out_root=Path(args.out),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        skip_c3_probe=args.skip_c3_probe,
    )
    if any(
        m["terminal_status"].startswith("ABORTED")
        for m in result["metrics"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
