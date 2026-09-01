"""
Fold the committed C4-base 10-seed run (runs/C4_base/seed_NN/) into the A12
results tree so the H2 (C4-base vs C5) hypothesis-test row computes, without
re-running C1/C2/C3/C5.

  by_condition =
    C1/C2/C3/C5        -> committed runs/<c>/seed_NN/metrics.json (A12 sweep,
                          refreshed by scripts/refresh_metrics.py so
                          categorical_subset_hypervolume is present)
    C4_base            -> committed runs/C4_base/seed_NN/metrics.json
    C4_base_no_*       -> committed runs/C4_base/C4_base_no_*/seed_NN/metrics.json
                          (3-seed H3 ablation pilot)

Writes results/{seed_summary,condition_summary,hypothesis_tests,RUN_NOTES.md}.
H1 rows are unchanged (same C2/C3 metrics); H2 moves from PENDING to computed;
H3 rows compute against the 3-seed ablation pilot (descriptive).

Usage:
    PYTHONPATH="$PWD" .venv/bin/python scripts/aggregate_h2.py [--write]
Without --write it prints the H2/H3 rows and writes nothing.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import findings, write_run_notes  # noqa: E402

from src.experiment.metrics import (
    CONDITION_SUMMARY_COLUMNS,
    HYPOTHESIS_TEST_COLUMNS,
    SEED_SUMMARY_COLUMNS,
    build_condition_summary,
    build_seed_summary,
    hypothesis_tests,
    write_csv,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def _load(paths: list[Path]) -> list[dict]:
    out = []
    for p in sorted(paths):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def main() -> int:
    do_write = "--write" in sys.argv[1:]

    by_condition: dict[str, list[dict]] = {}

    for cond in ("C1", "C2", "C3", "C4_base", "C5"):
        by_condition[cond] = _load(
            list((RUNS / cond).glob("seed_*/metrics.json"))
        )
    if not by_condition["C4_base"]:
        print(f"no C4_base metrics under {RUNS}/C4_base/seed_*/")
        return 1

    for suffix in ("no_rag", "no_schema", "no_validator"):
        label = f"C4_base_{suffix}"
        paths = list(
            (RUNS / "C4_base" / label).glob("seed_*/metrics.json")
        )
        if paths:
            by_condition[label] = _load(paths)

    for k, v in by_condition.items():
        print(f"  {k:22s} {len(v)} seeds")

    ht = hypothesis_tests(by_condition)
    print("\n--- H2 / H3 rows ---")
    for row in ht:
        if row["hypothesis"] in ("H2", "H3"):
            keys = (
                "hypothesis comparison metric c_ref c_test ref_mean "
                "test_mean absolute_reduction p_one_sided effect_size_dz "
                "n_nonzero_pairs threshold_met significant decision notes"
            ).split()
            print(
                " | ".join(
                    f"{k}={row.get(k)}" for k in keys if row.get(k) is not None
                )
            )

    if not do_write:
        print("\n(dry run -- pass --write to update results/*.csv)")
        return 0

    results = REPO / "results"
    write_csv(
        results / "seed_summary.csv",
        build_seed_summary(
            [m for group in by_condition.values() for m in group]
        ),
        SEED_SUMMARY_COLUMNS,
    )
    write_csv(
        results / "condition_summary.csv",
        build_condition_summary(
            [m for group in by_condition.values() for m in group]
        ),
        CONDITION_SUMMARY_COLUMNS,
    )
    write_csv(
        results / "hypothesis_tests.csv", ht, HYPOTHESIS_TEST_COLUMNS
    )

    all_metrics = [m for group in by_condition.values() for m in group]
    notes = findings(all_metrics, blocked=[])
    write_run_notes(results, all_metrics, by_condition, [], notes)

    print(f"\nwrote {results}/seed_summary.csv, condition_summary.csv, "
          "hypothesis_tests.csv, RUN_NOTES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
