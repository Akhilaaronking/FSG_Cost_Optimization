import argparse
import csv
import json
import time
from pathlib import Path

from src.evaluator import evaluate_bom
from src.optimization.hypervolume import (
    hypervolume_2d,
)
from src.optimization.nsga2 import (
    nsga2_optimize,
)
from src.optimization.search_space import (
    load_verified_real_search_space,
    validate_candidate_within_search_space,
)
from src.llm.prompt_builder import (
    sha256_json,
)


BOM_PATH = Path(
    "data/benchmark/pilot_10_parts_ground_truth.json"
)
REAL_SEARCH_SPACE_PATH = Path(
    "data/benchmark/real_search_space.json"
)


def _load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _parts_by_id(bom: dict) -> dict:
    return {
        part["part_id"]: part
        for part in bom.get("parts", [])
    }


def _modifications(
    baseline_bom: dict,
    candidate_bom: dict,
) -> list[dict]:
    baseline_parts = _parts_by_id(
        baseline_bom
    )
    modifications = []

    for part in candidate_bom.get("parts", []):
        part_id = part["part_id"]
        baseline = baseline_parts[part_id]

        for field in [
            "material_id",
            "process_id",
        ]:
            if part.get(field) != baseline.get(field):
                modifications.append({
                    "part_id": part_id,
                    "field": field,
                    "baseline": baseline.get(field),
                    "candidate": part.get(field),
                })

    return modifications


def _safe_real_evaluator(
    search_space: dict,
    reference_point: list[float],
):
    penalty = [
        reference_point[0] * 10.0,
        reference_point[1] * 10.0,
    ]

    def evaluator(bom: dict) -> dict:
        admissibility = validate_candidate_within_search_space(
            bom,
            search_space,
        )

        if not admissibility["valid"]:
            return {
                "objective_vector": penalty,
                "objectives": {
                    "cost_eur": None,
                    "mass_kg": None,
                },
                "constraints": {
                    "status": "SEARCH_SPACE_REJECTED",
                    "feasible": False,
                    "violation_count": len(
                        admissibility["errors"]
                    ),
                    "errors": admissibility["errors"],
                },
            }

        try:
            result = evaluate_bom(
                bom,
                evaluate_constraints=False,
            )
        except Exception as exc:
            return {
                "objective_vector": penalty,
                "objectives": {
                    "cost_eur": None,
                    "mass_kg": None,
                },
                "constraints": {
                    "status": "DETERMINISTIC_EVALUATION_FAILED",
                    "feasible": False,
                    "violation_count": 1,
                    "errors": [str(exc)],
                },
            }

        result = dict(result)
        result["constraints"] = {
            **result.get("constraints", {}),
            "status": "ENGINEERING_ADMISSIBLE_EVALUATED",
            "feasible": True,
            "violation_count": 0,
            "meaning": (
                "Feasible means material/process choices "
                "are within the approved real search space "
                "and deterministic cost/mass evaluation "
                "succeeded. This is not a full FSG "
                "compliance claim."
            ),
        }
        return result

    return evaluator


def _archive_rows(
    archive: list[dict],
    baseline_bom: dict,
    baseline_objective: list[float],
) -> list[dict]:
    rows = []

    for item in archive:
        candidate_bom = item["candidate"].bom
        cost, mass = item["objective_vector"]

        rows.append({
            "candidate_id": item["candidate_id"],
            "cost_eur": cost,
            "mass_kg": mass,
            "cost_delta_eur": cost
            - baseline_objective[0],
            "mass_delta_kg": mass
            - baseline_objective[1],
            "cost_improvement_pct": (
                baseline_objective[0] - cost
            )
            / baseline_objective[0]
            * 100.0,
            "mass_improvement_pct": (
                baseline_objective[1] - mass
            )
            / baseline_objective[1]
            * 100.0,
            "modifications": _modifications(
                baseline_bom,
                candidate_bom,
            ),
        })

    return sorted(
        rows,
        key=lambda row: (
            row["cost_eur"],
            row["mass_kg"],
            row["candidate_id"],
        ),
    )


def _write_outputs(
    *,
    output_dir: Path,
    seed: int,
    run_record: dict,
    archive_rows: list[dict],
    summary: dict,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_path = output_dir / f"run_seed_{seed}.json"
    csv_path = output_dir / f"pareto_seed_{seed}.csv"
    summary_path = output_dir / f"summary_seed_{seed}.json"

    with run_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_record,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "candidate_id",
            "cost_eur",
            "mass_kg",
            "cost_delta_eur",
            "mass_delta_kg",
            "cost_improvement_pct",
            "mass_improvement_pct",
            "modifications",
        ]
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in archive_rows:
            csv_row = dict(row)
            csv_row["modifications"] = json.dumps(
                row["modifications"],
                sort_keys=True,
            )
            writer.writerow(csv_row)

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    return {
        "run": str(run_path),
        "pareto_csv": str(csv_path),
        "summary": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the first real Formula Student C5 "
            "material/process optimisation pilot."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-dir",
        default="results/c5_pilot",
    )
    args = parser.parse_args()

    if args.budget <= 0:
        raise ValueError(
            "--budget must be positive"
        )

    baseline_bom = _load_json(BOM_PATH)
    real_source = _load_json(
        REAL_SEARCH_SPACE_PATH
    )
    search_space = load_verified_real_search_space(
        REAL_SEARCH_SPACE_PATH,
        BOM_PATH,
    )
    baseline_eval = evaluate_bom(
        baseline_bom,
        evaluate_constraints=False,
    )
    baseline_objective = baseline_eval[
        "objective_vector"
    ]
    reference_point = [
        baseline_objective[0] * 1.2,
        baseline_objective[1] * 1.2,
    ]

    start = time.perf_counter()
    result = nsga2_optimize(
        baseline_bom,
        search_space,
        _safe_real_evaluator(
            search_space,
            reference_point,
        ),
        population_size=min(20, args.budget),
        generations=args.budget,
        seed=args.seed,
        evaluation_budget=args.budget,
        mutation_rate=0.35,
        reference_point=reference_point,
    )
    runtime_sec = time.perf_counter() - start

    archive_rows = _archive_rows(
        result["pareto_archive"],
        baseline_bom,
        baseline_objective,
    )
    hypervolume = hypervolume_2d(
        result["pareto_archive"],
        reference_point,
    )
    normalized_hypervolume = (
        hypervolume
        / (reference_point[0] * reference_point[1])
    )

    best_cost = min(
        archive_rows,
        key=lambda row: (
            row["cost_eur"],
            row["mass_kg"],
        ),
    )
    lowest_mass = min(
        archive_rows,
        key=lambda row: (
            row["mass_kg"],
            row["cost_eur"],
        ),
    )

    run_record = {
        "label": "REAL C5 PILOT — NOT FINAL THESIS EXPERIMENT",
        "seed": args.seed,
        "requested_evaluation_budget": args.budget,
        "actual_evaluations": result[
            "evaluation_count"
        ],
        "cache_hits": result["cache_hits"],
        "runtime_sec": runtime_sec,
        "termination_reason": result[
            "termination_reason"
        ],
        "baseline": {
            "cost_eur": baseline_eval[
                "objectives"
            ]["cost_eur"],
            "mass_kg": baseline_eval[
                "objectives"
            ]["mass_kg"],
            "objective_vector": baseline_objective,
        },
        "reference_point": reference_point,
        "hypervolume": hypervolume,
        "normalized_hypervolume": normalized_hypervolume,
        "archive_size": len(archive_rows),
        "pareto_archive": archive_rows,
        "search_space_hash": sha256_json(
            search_space
        ),
        "benchmark_hash": sha256_json(
            baseline_bom
        ),
        "source_search_space_hash": sha256_json(
            real_source
        ),
        "code_config": {
            "optimizer": "src.optimization.nsga2.nsga2_optimize",
            "evaluator": "src.evaluator.evaluate_bom",
            "decision_variables": [
                "material",
                "process",
            ],
            "population_size": min(
                20,
                args.budget,
            ),
            "mutation_rate": 0.35,
        },
        "history": result["history"],
    }

    summary = {
        "label": run_record["label"],
        "seed": args.seed,
        "baseline_cost_eur": baseline_objective[0],
        "baseline_mass_kg": baseline_objective[1],
        "evaluations": result["evaluation_count"],
        "cache_hits": result["cache_hits"],
        "distinct_candidates": result[
            "evaluation_count"
        ],
        "pareto_archive_size": len(
            archive_rows
        ),
        "best_cost_eur": best_cost["cost_eur"],
        "mass_of_best_cost_design_kg": best_cost[
            "mass_kg"
        ],
        "lowest_mass_kg": lowest_mass["mass_kg"],
        "cost_of_lowest_mass_design_eur": lowest_mass[
            "cost_eur"
        ],
        "hypervolume": hypervolume,
        "normalized_hypervolume": normalized_hypervolume,
        "runtime_sec": runtime_sec,
        "output_note": (
            "Pilot only; no statistical C4/C5 comparison "
            "or thesis hypothesis conclusion."
        ),
    }

    paths = _write_outputs(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        run_record=run_record,
        archive_rows=archive_rows,
        summary=summary,
    )

    print("=" * 80)
    print("REAL C5 PILOT — NOT FINAL THESIS EXPERIMENT")
    print("=" * 80)
    print(
        f"Baseline cost: {baseline_objective[0]:.2f} EUR"
    )
    print(
        f"Baseline mass: {baseline_objective[1]:.9f} kg"
    )
    print(
        "Evaluations:",
        result["evaluation_count"],
    )
    print(
        "Pareto archive size:",
        len(archive_rows),
    )
    print(
        f"Best cost found: {best_cost['cost_eur']:.2f} EUR"
    )
    print(
        "Mass of best-cost design: "
        f"{best_cost['mass_kg']:.9f} kg"
    )
    print(
        f"Lowest mass found: {lowest_mass['mass_kg']:.9f} kg"
    )
    print(
        "Cost of lowest-mass design: "
        f"{lowest_mass['cost_eur']:.2f} EUR"
    )
    print(f"Hypervolume: {hypervolume}")
    print(
        "Normalized hypervolume:",
        normalized_hypervolume,
    )
    print(f"Runtime: {runtime_sec:.6f} sec")
    print(
        "Number of distinct candidates:",
        result["evaluation_count"],
    )
    print("Outputs:", paths)


if __name__ == "__main__":
    main()
