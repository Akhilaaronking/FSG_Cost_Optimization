"""
Run one C5 (NSGA-II) seed.

Thin shim over the A12 harness: this is exactly the code path that
`scripts/run_experiment.py --condition C5` takes for a single seed
(build_run_config -> Nsga2Driver -> compute_metrics), writing the
unified run layout under ``runs/C5/seed_NN/``. All the NSGA-II math and
the search-space-guarded evaluator now live in
``src/experiment/drivers.py``; nothing is duplicated here.

    python scripts/run_c5_real_pilot.py --seed 0 --budget 50

Note: --budget defaults to 50 (the A12 protocol). The earlier standalone
pilot defaulted to 100 and wrote its own results/c5_pilot/ format; those
files are retained as a historical record (EXPERIMENT_DEVIATIONS.txt) but
are superseded for A12.
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.run_experiment import run_one
from src.experiment.identity import (
    BENCHMARK_PATH,
    DEFAULT_N_EVAL,
    PROJECT_ROOT,
    SEARCH_SPACE_PATH,
    seed_dir_name,
)
from src.optimization.search_space import load_verified_real_search_space


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one C5 (NSGA-II) seed via the A12 Nsga2Driver. Same "
            "code path as `run_experiment.py --condition C5`."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument(
        "--output-dir",
        default=".",
        help="root containing runs/ (default: current directory)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.budget <= 0:
        print("--budget must be positive", file=sys.stderr)
        return 2

    baseline_bom = json.loads(
        (PROJECT_ROOT / BENCHMARK_PATH).read_text(encoding="utf-8")
    )
    parts = [p["part_id"] for p in baseline_bom["parts"]]
    search_space = load_verified_real_search_space(
        str(PROJECT_ROOT / SEARCH_SPACE_PATH),
        str(PROJECT_ROOT / BENCHMARK_PATH),
    )

    runs_root = Path(args.output_dir) / "runs"
    metrics = run_one(
        "C5",
        args.seed,
        budget=args.budget,
        parts=parts,
        runs_root=runs_root,
        baseline_bom=baseline_bom,
        search_space=search_space,
        overwrite=args.overwrite,
        dry_run=False,
    )

    budget = metrics["budget"]
    mo = metrics["multiobjective"]
    obj = metrics["objectives"]
    out_dir = runs_root / "C5" / seed_dir_name(args.seed)

    print("=" * 72)
    print(f"C5 seed {args.seed} via Nsga2Driver (A12 unified layout)")
    print("=" * 72)
    print(f"terminal_status : {metrics['terminal_status']}")
    print(
        f"evaluations     : {budget['n_eval_consumed']}/"
        f"{budget['n_eval_target']}"
    )
    print(f"pareto size     : {mo['pareto_archive_size']}")
    print(
        f"hypervolume     : {mo['hypervolume']:.6f}  "
        f"(normalized {mo['normalized_hypervolume']:.6f})"
    )
    print(f"best cost       : {obj['best_cost_eur']} EUR")
    print(f"lowest mass     : {obj['lowest_mass_kg']} kg")
    print(f"outputs         : {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
