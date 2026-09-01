"""
Regenerate metrics.json for existing committed runs from their source-of-truth
artifacts (run_config.json + events.jsonl + pareto_archive.json), using the
CURRENT src/experiment/metrics.py.

Why: metrics.json is declared regenerable from those three files (docs/A12
section 5). The committed C1/C2/C3/C5 metrics.json were written before
categorical_subset_hypervolume (eq 3.53) existed in _multiobjective, so H2's
categorical-subset row can't pair C5 against C4_base. This regenerates them in
place with no re-run (no LLM/evaluator calls) -- terminal_status and
wall_clock_sec are carried over unchanged from the existing file since neither
is derivable from events alone.

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/refresh_metrics.py [--write]
Without --write it reports what would change and writes nothing.
"""

import json
import sys
from pathlib import Path

from src.experiment.events import read_events
from src.experiment.metrics import compute_metrics

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

CONDITIONS = ["C1", "C2", "C3", "C5"]


def _seed_dirs():
    for cond in CONDITIONS:
        yield from sorted((RUNS / cond).glob("seed_*"))


def main() -> int:
    do_write = "--write" in sys.argv[1:]
    changed = 0
    for seed_dir in _seed_dirs():
        cfg_path = seed_dir / "run_config.json"
        events_path = seed_dir / "events.jsonl"
        metrics_path = seed_dir / "metrics.json"
        pareto_path = seed_dir / "pareto_archive.json"
        if not (cfg_path.is_file() and metrics_path.is_file()):
            continue

        run_config = json.loads(cfg_path.read_text(encoding="utf-8"))
        old = json.loads(metrics_path.read_text(encoding="utf-8"))
        events = read_events(events_path)
        pareto = (
            json.loads(pareto_path.read_text(encoding="utf-8"))
            if pareto_path.is_file()
            else None
        )

        new = compute_metrics(
            run_config,
            events,
            terminal_status=old["terminal_status"],
            wall_clock_sec=old["efficiency"]["wall_clock_sec"],
            pareto_archive=pareto,
        )

        old_hv = old["multiobjective"]["hypervolume"]
        new_hv = new["multiobjective"]["hypervolume"]
        if abs(old_hv - new_hv) > 1e-9:
            print(
                f"!! HV DRIFT {seed_dir}: old={old_hv} new={new_hv} "
                "-- not writing, investigate"
            )
            continue

        cat_hv = new["multiobjective"].get("categorical_subset_hypervolume")
        print(
            f"{seed_dir}: HV unchanged ({new_hv:.4f}), "
            f"categorical_subset_hypervolume={cat_hv}"
        )
        changed += 1
        if do_write:
            metrics_path.write_text(
                json.dumps(new, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    print(f"\n{changed} run(s) {'refreshed' if do_write else 'would refresh'}")
    if not do_write:
        print("(dry run -- pass --write to update metrics.json files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
