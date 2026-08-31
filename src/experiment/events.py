"""
Event log for the A12 harness (thesis 11.16, and the 11.7 validity funnel).

One JSON object per line in ``runs/<condition>/seed_NN/events.jsonl``.
One record per event that is either a generation attempt (C1/C2/C3) or
an objective-evaluation attempt (C5). The record carries the full
11.7 funnel, the 11.9 hallucination flags, the deterministic result and
its constraints, archive outcome, timings and software versions -- so
every Chapter 12 table is regenerable from result files (11.16.1).

This module owns event *shape*: building a record from a generator
return value (``event_from``) or an NSGA-II candidate (``nsga2_event``),
filling in the pieces the driver learns later (``attach_*``), deriving
the funnel stage, and writing / reading the file. The drivers
(step 6) own the loop; metrics.py (step 7) replays what is written here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluator.unified_evaluator import EVALUATOR_VERSION
from src.experiment.identity import HARNESS_VERSION, git_identity
from src.llm.prompt_builder import sha256_text


EVENT_TYPES = ("proposal", "nsga2_evaluation")

FUNNEL_STAGES = (
    "parse",
    "schema",
    "identifier",
    "applicability",
    "feasibility",
    "objective_evaluation",
    "archive",
)

ARCHIVE_STATUSES = (
    "dominated",
    "non_dominated",
    "duplicate",
    "pareto_improving",
)

_EVAL_FAILURE_STATUSES = (
    "DETERMINISTIC_EVALUATION_FAILED",
    "SEARCH_SPACE_REJECTED",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_software_block() -> dict:
    """
    Version stamp for every event. Computed once by the EventLog rather
    than per event (git is a subprocess call).
    """
    git = git_identity()
    commit = git.get("commit")
    return {
        "harness_version": HARNESS_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "git_commit": commit[:12] if commit else None,
        "git_tracked_dirty": git.get("tracked_dirty"),
    }


# -- building a record ------------------------------------------------


def event_from(
    rec: dict,
    *,
    target_part_id: str | None = None,
    parent_candidate_id: str | None = None,
) -> dict:
    """
    Build the generation-side of an event from a ``generate_c1/c2/c3``
    return value. The driver fills ``validity.applicability_valid``,
    ``generation.modifications``, ``evaluation`` and ``archive`` later
    via the ``attach_*`` helpers; ``run_id / condition / seed /
    event_index / ts_utc / software`` are stamped by ``EventLog.write``.
    """
    retrieval = rec.get("retrieval") or {}
    rag_enabled = bool(retrieval.get("rag_enabled"))

    proposal = rec.get("proposal") or {}
    resolved_part_id = target_part_id or proposal.get("part_id")

    event = {
        "event_type": "proposal",
        "generation": {
            "target_part_id": resolved_part_id,
            "parent_candidate_id": parent_candidate_id,
            "raw_output_sha256": sha256_text(rec.get("raw_output") or ""),
            "parsed_proposal": rec.get("proposal"),
            "prompt_hash": rec.get("prompt_hash"),
            "modifications": [],
        },
        "validity": {
            "parse_valid": bool(rec.get("parse_valid")),
            "schema_valid": bool(rec.get("schema_valid")),
            "authority_valid": bool(rec.get("authority_valid")),
            "applicability_valid": False,
            "unknown_identifiers": list(
                rec.get("unknown_identifiers") or []
            ),
            "protected_field_writes": [],
            "funnel_stage_reached": None,
        },
        "hallucination": {
            "hallucinated": bool(rec.get("hallucinated")),
            "categories": list(rec.get("hallucination_categories") or []),
        },
        "efficiency": {
            "gen_runtime_sec": rec.get("runtime_sec"),
            "eval_runtime_sec": None,
            "token_counts": {"prompt": None, "completion": None},
        },
    }

    if rag_enabled:
        event["retrieval"] = {
            "rag_enabled": True,
            "query_text": retrieval.get("query_text"),
            "retrieved_chunk_ids": list(
                retrieval.get("retrieved_chunk_ids") or []
            ),
            "retrieved_source_ids": list(
                retrieval.get("retrieved_source_ids") or []
            ),
            "similarity_scores": list(
                retrieval.get("similarity_scores") or []
            ),
        }
    else:
        event["retrieval"] = {"rag_enabled": False}

    return event


def nsga2_event(
    *,
    candidate_id: str,
    parent_candidate_id=None,
    modifications: list | None = None,
    gen_runtime_sec: float | None = None,
) -> dict:
    """
    Build the generation-side of an NSGA-II objective-evaluation event.
    An evolutionary candidate has no proposal funnel, so ``validity``
    and ``hallucination`` are null and ``retrieval`` is omitted
    (docs/A12 section 4).
    """
    return {
        "event_type": "nsga2_evaluation",
        "generation": {
            "target_part_id": None,
            "parent_candidate_id": parent_candidate_id,
            "candidate_id": candidate_id,
            "raw_output_sha256": None,
            "parsed_proposal": None,
            "prompt_hash": None,
            "modifications": list(modifications or []),
        },
        "validity": None,
        "hallucination": None,
        "efficiency": {
            "gen_runtime_sec": gen_runtime_sec,
            "eval_runtime_sec": None,
            "token_counts": {"prompt": None, "completion": None},
        },
    }


# -- filling in what the driver learns later -----------------------


def attach_applicability(event: dict, application) -> dict:
    """
    Fold a :class:`~src.experiment.apply_proposal.ProposalApplication`
    into the event's validity + generation blocks.
    """
    validity = event["validity"]
    validity["applicability_valid"] = bool(application.applicability_valid)
    validity["protected_field_writes"] = list(
        application.protected_field_writes
    )
    event["generation"]["modifications"] = list(application.modifications)
    if getattr(application, "is_noop", False):
        event["generation"]["is_noop"] = True
    return event


def build_evaluation_block(
    outcome,
    baseline_vector,
    *,
    eval_runtime_sec: float | None = None,
) -> dict:
    """
    Normalise an evaluator result (via a ledger ``ConsumeOutcome``) plus
    the frozen baseline objective vector into the event ``evaluation``
    block (docs/A12 section 4). Shared by both drivers.
    """
    result = outcome.result or {}
    objectives = result.get("objectives") or {}
    vector = result.get("objective_vector")

    cost = objectives.get("cost_eur")
    mass = objectives.get("mass_kg")
    base_cost, base_mass = (baseline_vector or [None, None])[:2]

    def _pct(base, value):
        if base in (None, 0) or value is None:
            return None
        return (base - value) / base * 100.0

    baseline_delta = {
        "cost_eur": (base_cost - cost)
        if (base_cost is not None and cost is not None)
        else None,
        "mass_kg": (base_mass - mass)
        if (base_mass is not None and mass is not None)
        else None,
        "cost_improvement_pct": _pct(base_cost, cost),
        "mass_improvement_pct": _pct(base_mass, mass),
    }

    raw_constraints = result.get("constraints") or {}
    status = raw_constraints.get("status")
    violation_count = raw_constraints.get("violation_count")
    evaluated = status not in (None, "NOT_EVALUATED")

    constraints = {
        "status": status,
        "evaluated": evaluated,
        "feasible": raw_constraints.get("feasible"),
        "violation_count": violation_count,
        "proposal_level_violation": (violation_count > 0)
        if isinstance(violation_count, (int, float))
        else None,
        "rule_level_violations": violation_count,
        "rule_level_checks": raw_constraints.get(
            "available_optimizer_rules"
        ),
        "missing_essential_fields": list(
            raw_constraints.get("missing_essential_fields") or []
        ),
    }

    return {
        "consumed_objective_budget": bool(outcome.consumed_budget),
        "objective_eval_cache_hit": bool(outcome.cache_hit),
        "bom_hash": outcome.bom_hash,
        "objectives": {"cost_eur": cost, "mass_kg": mass},
        "objective_vector": list(vector) if vector is not None else None,
        "baseline_delta": baseline_delta,
        "constraints": constraints,
        "_eval_runtime_sec": eval_runtime_sec,
    }


def attach_evaluation(
    event: dict,
    outcome,
    baseline_vector,
    *,
    eval_runtime_sec: float | None = None,
) -> dict:
    block = build_evaluation_block(
        outcome, baseline_vector, eval_runtime_sec=eval_runtime_sec
    )
    runtime = block.pop("_eval_runtime_sec", None)
    event["evaluation"] = block
    if runtime is not None:
        event["efficiency"]["eval_runtime_sec"] = runtime
    return event


def attach_archive(
    event: dict, status: str, archive_size_after: int
) -> dict:
    if status not in ARCHIVE_STATUSES:
        raise ValueError(
            f"archive status {status!r} not in {ARCHIVE_STATUSES}"
        )
    event["archive"] = {
        "status": status,
        "archive_size_after": int(archive_size_after),
    }
    return event


# -- funnel stage --------------------------------------------------


def derive_funnel_stage(event: dict) -> str:
    """
    The deepest 11.7 funnel stage this event reached, from its validity
    flags and whether evaluation / archive were attached.
    """
    if event.get("event_type") == "nsga2_evaluation":
        return "archive" if "archive" in event else "objective_evaluation"

    validity = event.get("validity") or {}
    if not validity.get("parse_valid"):
        return "parse"
    if not validity.get("schema_valid"):
        return "schema"
    if not validity.get("authority_valid"):
        return "identifier"
    if not validity.get("applicability_valid"):
        return "applicability"

    evaluation = event.get("evaluation")
    if evaluation is None:
        return "feasibility"
    if (evaluation.get("constraints") or {}).get(
        "status"
    ) in _EVAL_FAILURE_STATUSES:
        return "feasibility"
    if "archive" not in event:
        return "objective_evaluation"
    return "archive"


# -- the log -----------------------------------------------------


def _validate_event(event: dict) -> None:
    if event.get("event_type") not in EVENT_TYPES:
        raise ValueError(
            f"event_type {event.get('event_type')!r} not in {EVENT_TYPES}"
        )
    if "generation" not in event:
        raise ValueError("event missing 'generation' block")
    evaluation = event.get("evaluation")
    if evaluation is not None:
        for key in ("consumed_objective_budget", "bom_hash"):
            if key not in evaluation:
                raise ValueError(
                    f"evaluation block missing {key!r}"
                )


class EventLog:
    """
    Append-only writer for one run's events.jsonl. Stamps the run
    identity, a sequential ``event_index``, a timestamp and the software
    block onto every record, derives the funnel stage if the caller did
    not set it, and flushes each line.
    """

    def __init__(
        self,
        path,
        *,
        run_id: str,
        condition: str,
        seed: int,
        software: dict | None = None,
        if_exists: str = "error",
    ):
        self.path = Path(path)
        self.run_id = run_id
        self.condition = condition
        self.seed = int(seed)
        self.software = software or default_software_block()
        self._count = 0

        if if_exists not in ("error", "overwrite"):
            raise ValueError("if_exists must be 'error' or 'overwrite'")
        if (
            if_exists == "error"
            and self.path.exists()
            and self.path.stat().st_size > 0
        ):
            raise FileExistsError(
                f"{self.path} already exists and is non-empty; pass "
                f"if_exists='overwrite' to replace it (thesis 11.18: "
                f"a rerun reuses the same seed and frozen config)"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    # -- context manager --

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    @property
    def count(self) -> int:
        return self._count

    # -- write --

    def write(self, event: dict) -> dict:
        record = dict(event)
        record["run_id"] = self.run_id
        record["condition"] = self.condition
        record["seed"] = self.seed
        record["event_index"] = self._count
        record.setdefault("ts_utc", _utcnow())
        record["software"] = dict(self.software)

        validity = record.get("validity")
        if isinstance(validity, dict) and validity.get(
            "funnel_stage_reached"
        ) is None:
            validity["funnel_stage_reached"] = derive_funnel_stage(record)

        _validate_event(record)
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        self._count += 1
        return record


def read_events(path) -> list[dict]:
    """Load a run's events.jsonl back into a list of dicts (metrics/tests)."""
    events = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: corrupt events.jsonl line"
                ) from exc
    return events
