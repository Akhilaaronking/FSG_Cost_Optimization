"""
Condition drivers for the A12 harness (docs/A12 sections 6-7).

  GenerativeDriver -- C1 / C2 / C3. Atomic proposals against the frozen
    baseline x0 (never stacked). One EqualBudgetLedger; the loop runs
    until n_eval fresh objective evaluations, the proposal-attempt cap,
    or the distinct candidate space is exhausted.

  Nsga2Driver -- C5. Reuses the existing NSGA-II math unchanged; only the
    evaluator is wrapped so every fresh objective evaluation flows
    through the same ledger and is logged as one nsga2_evaluation event.

Both write events through an already-open EventLog and (optionally) a
pareto_archive.json. metrics.json is computed separately (step 7) by
replaying those files.
"""

import json
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

from src.evaluator import evaluate_bom
from src.optimization.hypervolume import hypervolume_2d
from src.optimization.nsga2 import nsga2_optimize
from src.optimization.pareto import dominates
from src.optimization.search_space import (
    validate_candidate_within_search_space,
)

from src.experiment.apply_proposal import apply_proposal
from src.experiment.events import (
    attach_applicability,
    attach_archive,
    attach_evaluation,
    event_from,
    nsga2_event,
)
from src.experiment.ledger import EqualBudgetLedger, canonical_bom_hash
from src.llm.conditions import generate_c1, generate_c2, generate_c3


REFERENCE_POINT_FACTOR = 1.2  # eq 11.10
MODIFICATION_FIELDS = ("material_id", "process_id")

_GENERATORS = {
    "C1": generate_c1,
    "C2": generate_c2,
    "C3": generate_c3,
}


# -- shared helpers ------------------------------------------------


def reference_point(baseline_vector) -> list:
    return [value * REFERENCE_POINT_FACTOR for value in baseline_vector]


def bom_modifications(baseline_bom: dict, candidate_bom: dict) -> list:
    """material_id / process_id deltas, part by part (C5 pilot shape)."""
    baseline = {
        part["part_id"]: part
        for part in baseline_bom.get("parts", [])
    }
    changes = []
    for part in candidate_bom.get("parts", []):
        base = baseline.get(part["part_id"], {})
        for field_name in MODIFICATION_FIELDS:
            if part.get(field_name) != base.get(field_name):
                changes.append(
                    {
                        "part_id": part["part_id"],
                        "field": field_name,
                        "baseline": base.get(field_name),
                        "candidate": part.get(field_name),
                    }
                )
    return changes


def archive_entry(
    candidate_id: str,
    objective_vector,
    baseline_vector,
    modifications,
) -> dict:
    cost, mass = objective_vector[0], objective_vector[1]
    base_cost, base_mass = baseline_vector[0], baseline_vector[1]
    return {
        "candidate_id": candidate_id,
        "objective_vector": [cost, mass],
        "cost_eur": cost,
        "mass_kg": mass,
        "baseline_delta": {
            "cost_eur": base_cost - cost
            if cost is not None
            else None,
            "mass_kg": base_mass - mass if mass is not None else None,
            "cost_improvement_pct": (base_cost - cost) / base_cost * 100.0
            if (cost is not None and base_cost)
            else None,
            "mass_improvement_pct": (base_mass - mass) / base_mass * 100.0
            if (mass is not None and base_mass)
            else None,
        },
        "modifications": list(modifications),
    }


class ParetoArchive:
    """
    Minimisation, 2 objectives. Classifies each offered candidate as
    duplicate / dominated / non_dominated / pareto_improving (11.7
    stage 8) and keeps the running non-dominated set.
    """

    def __init__(self):
        self._entries: list[dict] = []
        self._hashes: set[str] = set()

    def offer(
        self,
        *,
        candidate_id: str,
        objective_vector,
        bom_hash: str,
        modifications,
    ) -> tuple[str, int]:
        if bom_hash in self._hashes:
            return "duplicate", len(self._entries)
        self._hashes.add(bom_hash)

        vector = [objective_vector[0], objective_vector[1]]
        entry = {
            "candidate_id": candidate_id,
            "objective_vector": vector,
            "bom_hash": bom_hash,
            "modifications": list(modifications),
        }

        if not self._entries:
            self._entries.append(entry)
            return "pareto_improving", 1

        if any(
            dominates(e["objective_vector"], vector)
            for e in self._entries
        ):
            return "dominated", len(self._entries)

        improves = any(
            dominates(vector, e["objective_vector"])
            for e in self._entries
        )
        self._entries = [
            e
            for e in self._entries
            if not dominates(vector, e["objective_vector"])
        ]
        self._entries.append(entry)
        return (
            "pareto_improving" if improves else "non_dominated"
        ), len(self._entries)

    def entries(self) -> list[dict]:
        return sorted(
            self._entries,
            key=lambda e: (
                e["objective_vector"][0],
                e["objective_vector"][1],
                e["candidate_id"],
            ),
        )


def _safe_real_evaluator(search_space: dict, penalty_reference: list):
    """
    Port of scripts/run_c5_real_pilot.py::_safe_real_evaluator. Rejects
    candidates outside the engineering-reviewed search space with a
    large penalty; otherwise returns the deterministic cost/mass with a
    'this is not full FSG compliance' constraints note.
    """
    penalty = [penalty_reference[0] * 10.0, penalty_reference[1] * 10.0]

    def evaluator(bom: dict) -> dict:
        admissibility = validate_candidate_within_search_space(
            bom, search_space
        )
        if not admissibility["valid"]:
            return {
                "objective_vector": penalty,
                "objectives": {"cost_eur": None, "mass_kg": None},
                "constraints": {
                    "status": "SEARCH_SPACE_REJECTED",
                    "feasible": False,
                    "violation_count": len(admissibility["errors"]),
                    "errors": admissibility["errors"],
                },
            }

        try:
            result = dict(
                evaluate_bom(bom, evaluate_constraints=False)
            )
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "objective_vector": penalty,
                "objectives": {"cost_eur": None, "mass_kg": None},
                "constraints": {
                    "status": "DETERMINISTIC_EVALUATION_FAILED",
                    "feasible": False,
                    "violation_count": 1,
                    "errors": [str(exc)],
                },
            }

        result["constraints"] = {
            **result.get("constraints", {}),
            "status": "ENGINEERING_ADMISSIBLE_EVALUATED",
            "feasible": True,
            "violation_count": 0,
        }
        return result

    return evaluator


def _generative_evaluator(evaluate_constraints: bool = False):
    def evaluator(bom: dict) -> dict:
        return evaluate_bom(
            bom, evaluate_constraints=evaluate_constraints
        )

    return evaluator


# -- outcome -----------------------------------------------------


@dataclass
class RunOutcome:
    condition: str
    seed: int
    run_id: str
    terminal_status: str
    events_written: int
    wall_clock_sec: float
    ledger: dict
    baseline_vector: list
    reference_point: list
    archive_size: int
    hypervolume: float
    normalized_hypervolume: float
    pareto_archive_path: str | None = None
    detail: str | None = None
    extra: dict = field(default_factory=dict)


def _write_pareto_archive(
    path,
    *,
    condition,
    seed,
    run_id,
    baseline_vector,
    ref_point,
    entries,
) -> tuple[str, float, float]:
    hv = hypervolume_2d(entries, ref_point)
    denom = ref_point[0] * ref_point[1]
    nhv = hv / denom if denom else 0.0
    payload = {
        "condition": condition,
        "seed": seed,
        "run_id": run_id,
        "reference_point": ref_point,
        "baseline_vector": list(baseline_vector),
        "hypervolume": hv,
        "normalized_hypervolume": nhv,
        "archive_size": len(entries),
        "entries": entries,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path), hv, nhv


# -- generative driver (C1/C2/C3) ------------------------------


class GenerativeDriver:
    def __init__(
        self,
        run_config: dict,
        *,
        generator,
        baseline_bom: dict,
        retriever=None,
        evaluator=None,
        stall_limit: int | None = None,
    ):
        self.run_config = run_config
        self.condition = run_config["condition"]
        if self.condition not in _GENERATORS:
            raise ValueError(
                f"GenerativeDriver does not handle {self.condition!r}"
            )
        self.seed = int(run_config["seed"])
        self.run_id = run_config["run_id"]

        budget = run_config["identity"]["budget"]
        self.n_eval = int(budget["n_eval"])
        self.attempt_cap = budget.get("proposal_attempt_cap")

        self.target_part_ids = list(
            run_config["condition_spec"]["target_parts"]
        )
        self.generator = generator
        self.retriever = retriever
        self.baseline_bom = baseline_bom
        self._parts_by_id = {
            part["part_id"]: part
            for part in baseline_bom.get("parts", [])
        }
        self.ledger = EqualBudgetLedger(
            evaluator or _generative_evaluator(),
            n_eval=self.n_eval,
            proposal_attempt_cap=self.attempt_cap,
        )
        self.stall_limit = (
            stall_limit
            if stall_limit is not None
            else 2 * max(1, len(self.target_part_ids))
        )

    def _generate(self, target_part: dict) -> dict:
        fn = _GENERATORS[self.condition]
        if self.condition == "C1":
            return fn(
                self.generator,
                bom=self.baseline_bom,
                target_part=target_part,
                seed=self.seed,
            )
        return fn(
            self.generator,
            bom=self.baseline_bom,
            target_part=target_part,
            retriever=self.retriever,
            seed=self.seed,
        )

    def _part_cycle(self):
        parts = self.target_part_ids
        start = self.seed % len(parts)
        return cycle(parts[start:] + parts[:start])

    def run(self, event_log, *, pareto_archive_path=None) -> RunOutcome:
        started = time.perf_counter()

        baseline_eval = evaluate_bom(
            self.baseline_bom, evaluate_constraints=False
        )
        baseline_vector = list(baseline_eval["objective_vector"])
        ref_point = reference_point(baseline_vector)

        archive = ParetoArchive()
        parts = self._part_cycle()
        terminal = None
        detail = None
        stall = 0

        while not self.ledger.budget_exhausted:
            if self.ledger.attempts_exhausted:
                terminal = "ABORTED_BUDGET_UNREACHED"
                detail = (
                    f"hit proposal_attempt_cap={self.attempt_cap} "
                    f"with {self.ledger.objective_evaluations}/"
                    f"{self.n_eval} objective evaluations"
                )
                break

            part_id = next(parts)
            self.ledger.record_proposal_attempt()

            try:
                rec = self._generate(self._parts_by_id[part_id])
            except Exception as exc:
                terminal = "ABORTED_PROVIDER"
                detail = f"{type(exc).__name__}: {exc}"
                break

            event = event_from(rec, target_part_id=part_id)
            progressed = False

            if (
                rec.get("parse_valid")
                and rec.get("schema_valid")
                and rec.get("authority_valid")
            ):
                application = apply_proposal(
                    self.baseline_bom, rec.get("proposal") or {}
                )
                attach_applicability(event, application)

                if application.applicability_valid:
                    progressed = self._evaluate_and_archive(
                        event,
                        application,
                        archive,
                        baseline_vector,
                    )

            event_log.write(event)
            stall = 0 if progressed else stall + 1

            if stall >= self.stall_limit:
                terminal = "COMPLETE_SPACE_EXHAUSTED"
                detail = (
                    f"{self.stall_limit} attempts with no new distinct "
                    f"candidate; {self.ledger.objective_evaluations}/"
                    f"{self.n_eval} objective evaluations consumed"
                )
                break

        if terminal is None:
            terminal = "COMPLETE"

        entries = archive.entries()
        path, hv, nhv = (None, 0.0, 0.0)
        if pareto_archive_path is not None:
            path, hv, nhv = _write_pareto_archive(
                pareto_archive_path,
                condition=self.condition,
                seed=self.seed,
                run_id=self.run_id,
                baseline_vector=baseline_vector,
                ref_point=ref_point,
                entries=[
                    archive_entry(
                        e["candidate_id"],
                        e["objective_vector"],
                        baseline_vector,
                        e["modifications"],
                    )
                    for e in entries
                ],
            )
        else:
            hv = hypervolume_2d(entries, ref_point)
            denom = ref_point[0] * ref_point[1]
            nhv = hv / denom if denom else 0.0

        return RunOutcome(
            condition=self.condition,
            seed=self.seed,
            run_id=self.run_id,
            terminal_status=terminal,
            events_written=event_log.count,
            wall_clock_sec=time.perf_counter() - started,
            ledger=self.ledger.snapshot(),
            baseline_vector=baseline_vector,
            reference_point=ref_point,
            archive_size=len(entries),
            hypervolume=hv,
            normalized_hypervolume=nhv,
            pareto_archive_path=path,
            detail=detail,
        )

    def _evaluate_and_archive(
        self, event, application, archive, baseline_vector
    ) -> bool:
        candidate_id = (
            f"{self.condition}_s{self.seed}_"
            f"p{self.ledger.proposal_attempts:04d}"
        )
        try:
            t0 = time.perf_counter()
            outcome = self.ledger.consume(application.bom)
            eval_dt = time.perf_counter() - t0
        except Exception as exc:  # pragma: no cover - defensive
            event["evaluation"] = {
                "consumed_objective_budget": False,
                "objective_eval_cache_hit": False,
                "bom_hash": canonical_bom_hash(application.bom),
                "objectives": {"cost_eur": None, "mass_kg": None},
                "objective_vector": None,
                "baseline_delta": {},
                "constraints": {
                    "status": "DETERMINISTIC_EVALUATION_FAILED",
                    "evaluated": False,
                    "errors": [str(exc)],
                },
            }
            return False

        attach_evaluation(
            event, outcome, baseline_vector, eval_runtime_sec=eval_dt
        )
        status, size = archive.offer(
            candidate_id=candidate_id,
            objective_vector=outcome.result["objective_vector"],
            bom_hash=outcome.bom_hash,
            modifications=application.modifications,
        )
        attach_archive(event, status, size)
        return outcome.consumed_budget


# -- NSGA-II driver (C5) ------------------------------------


class Nsga2Driver:
    def __init__(
        self,
        run_config: dict,
        *,
        baseline_bom: dict,
        search_space: dict,
        evaluator=None,
    ):
        if run_config["condition"] != "C5":
            raise ValueError("Nsga2Driver only handles C5")
        self.run_config = run_config
        self.condition = "C5"
        self.seed = int(run_config["seed"])
        self.run_id = run_config["run_id"]
        self.n_eval = int(
            run_config["identity"]["budget"]["n_eval"]
        )
        nsga2_spec = run_config["condition_spec"].get("nsga2") or {}
        self.population_size = min(
            int(nsga2_spec.get("population_size", 20)), self.n_eval
        )
        self.mutation_rate = float(
            nsga2_spec.get("mutation_rate", 0.35)
        )
        self.baseline_bom = baseline_bom
        self.search_space = search_space
        self._injected_evaluator = evaluator

    def run(self, event_log, *, pareto_archive_path=None) -> RunOutcome:
        started = time.perf_counter()

        baseline_eval = evaluate_bom(
            self.baseline_bom, evaluate_constraints=False
        )
        baseline_vector = list(baseline_eval["objective_vector"])
        ref_point = reference_point(baseline_vector)

        real_evaluator = self._injected_evaluator or _safe_real_evaluator(
            self.search_space, ref_point
        )
        ledger = EqualBudgetLedger(
            real_evaluator,
            n_eval=self.n_eval,
            proposal_attempt_cap=None,
        )

        def instrumented(bom: dict) -> dict:
            t0 = time.perf_counter()
            outcome = ledger.consume(bom)
            eval_dt = time.perf_counter() - t0
            index = ledger.objective_evaluations
            event = nsga2_event(
                candidate_id=f"nsga2_s{self.seed}_e{index:05d}",
                modifications=bom_modifications(self.baseline_bom, bom),
            )
            attach_evaluation(
                event,
                outcome,
                baseline_vector,
                eval_runtime_sec=eval_dt,
            )
            event_log.write(event)
            return outcome.result

        result = nsga2_optimize(
            self.baseline_bom,
            self.search_space,
            instrumented,
            population_size=self.population_size,
            generations=self.n_eval,
            seed=self.seed,
            evaluation_budget=self.n_eval,
            mutation_rate=self.mutation_rate,
            reference_point=ref_point,
        )

        extra = {
            "nsga2_evaluation_count": result["evaluation_count"],
            "nsga2_cache_hits": result["cache_hits"],
            "nsga2_termination_reason": result["termination_reason"],
            "generations_completed": result["generations_completed"],
        }
        terminal = "COMPLETE"
        detail = None
        if ledger.objective_evaluations != result["evaluation_count"]:
            extra["ledger_drift"] = {
                "ledger": ledger.objective_evaluations,
                "nsga2": result["evaluation_count"],
            }
            detail = (
                "ledger vs NSGA-II evaluation-count drift "
                f"({ledger.objective_evaluations} vs "
                f"{result['evaluation_count']})"
            )

        entries = [
            archive_entry(
                item["candidate_id"],
                item["objective_vector"],
                baseline_vector,
                bom_modifications(
                    self.baseline_bom, item["candidate"].bom
                ),
            )
            for item in result["pareto_archive"]
        ]

        path, hv, nhv = (None, 0.0, 0.0)
        if pareto_archive_path is not None:
            path, hv, nhv = _write_pareto_archive(
                pareto_archive_path,
                condition="C5",
                seed=self.seed,
                run_id=self.run_id,
                baseline_vector=baseline_vector,
                ref_point=ref_point,
                entries=entries,
            )
        else:
            hv = hypervolume_2d(entries, ref_point)
            denom = ref_point[0] * ref_point[1]
            nhv = hv / denom if denom else 0.0

        return RunOutcome(
            condition="C5",
            seed=self.seed,
            run_id=self.run_id,
            terminal_status=terminal,
            events_written=event_log.count,
            wall_clock_sec=time.perf_counter() - started,
            ledger=ledger.snapshot(),
            baseline_vector=baseline_vector,
            reference_point=ref_point,
            archive_size=len(entries),
            hypervolume=hv,
            normalized_hypervolume=nhv,
            pareto_archive_path=path,
            detail=detail,
            extra=extra,
        )
