"""
C4 agentic tool-loop driver (docs/A13).

One continuous search per seed. The loop body is thesis eq 10.30-10.35:

    s_t  = Select(A_t, history)          # ArchiveGuidedSelector (A13 A)
    r_t  = Retrieve(s_t, k)              # RAG top-k        (skipped: no_rag)
    pi_t = LLM(s_t, r_t, feedback, schema)   #               (relaxed: no_schema, via the backend)
    x'_t = Apply(x_t, pi_t)              # apply_proposal -- to x_t, NOT x0
    y_t  = E(x'_t)                       # ledger.consume  -- one budget unit iff fresh
    A_{t+1} = ND(A_t u {x'_t})           # ParetoArchive.offer
    accept  = offer status in {pareto_improving, non_dominated}   # (skipped: no_validator)

Stops at N fresh objective evaluations (eq 11.4), on convergence
dHV_recent < eps (eq 4.2) -> COMPLETE_CONVERGED, or on the proposal
attempt cap -> ABORTED_BUDGET_UNREACHED.

Reuses the A12 machinery: EqualBudgetLedger, apply_proposal,
ParetoArchive, events, RunOutcome. The archive is NOT seeded with the
baseline as a scoring member -- consistent with C1/C2/C3/C5 in the
committed A12 sweep (deviates from thesis 4.9's literal wording, but
deviates uniformly, which is what the paired C4-vs-C5 comparison needs).
"""

import time
from itertools import count as _count

from src.evaluator import evaluate_bom
from src.optimization.hypervolume import hypervolume_2d

from src.experiment.apply_proposal import apply_proposal
from src.experiment.c4_feedback import build_archive_text, build_feedback_text
from src.experiment.c4_select import ArchiveGuidedSelector
from src.experiment.drivers import (
    ATTEMPT_SEED_STRIDE,
    ParetoArchive,
    RunOutcome,
    _generative_evaluator,
    _write_pareto_archive,
    archive_entry,
    bom_modifications,
    reference_point,
)
from src.experiment.events import (
    attach_applicability,
    attach_archive,
    attach_evaluation,
    event_from,
)
from src.experiment.ledger import EqualBudgetLedger, canonical_bom_hash
from src.llm.prompt_builder import build_c4_prompt
from src.rag.context_formatter import format_retrieval_context
from src.rag.query_builder import build_engineering_query


ABLATIONS = (None, "no_rag", "no_schema", "no_validator")
ACCEPT_STATUSES = ("pareto_improving", "non_dominated")


class C4Driver:
    def __init__(
        self,
        run_config: dict,
        *,
        generator,
        baseline_bom: dict,
        retriever=None,
        evaluator=None,
    ):
        self.condition = run_config["condition"]
        self.seed = int(run_config["seed"])
        self.run_id = run_config["run_id"]

        spec = run_config["condition_spec"]
        loop = spec["c4_loop"]
        self.n_eval = int(loop["n_eval"])
        self.look_back_L = int(loop["convergence"]["look_back_L"])
        self.epsilon_hv = float(loop["convergence"]["epsilon_hv"])
        self.retry_cap_K = int(loop["retry_cap_K"])
        self.attempt_cap = loop.get("proposal_attempt_cap")
        self.ablation = loop.get("ablation")
        if self.ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {self.ablation!r}")

        self.target_part_ids = list(spec["target_parts"])
        self.generator = generator
        self.retriever = retriever
        self.baseline_bom = baseline_bom
        self._raw_evaluator = evaluator or _generative_evaluator()
        # ledger is built in run() once the 1.2x reference point (for the
        # eval-failure penalty) is known.
        self.ledger = None

        self.selector = ArchiveGuidedSelector(
            self.target_part_ids,
            seed=self.seed,
            explore_after=self.look_back_L,
        )

    # -- retrieval (skipped for the no_rag ablation) --------------------

    def _retrieve(self, selection, target_part):
        if self.ablation == "no_rag" or self.retriever is None:
            return None, {"rag_enabled": False}
        query = build_engineering_query(
            target_part, user_intent=selection.intent
        )
        results = self.retriever.retrieve(query, top_k=5)
        return format_retrieval_context(results), {
            "rag_enabled": True,
            "top_k": 5,
            "query_text": query,
            "retrieved_chunk_ids": [
                r.chunk.chunk_id for r in results
            ],
            "retrieved_source_ids": [
                r.chunk.source_id for r in results
            ],
            "similarity_scores": [r.score for r in results],
        }

    # -- the loop ----------------------------------------------------

    def run(self, event_log, *, pareto_archive_path=None) -> RunOutcome:
        started = time.perf_counter()

        baseline_eval = evaluate_bom(
            self.baseline_bom, evaluate_constraints=False
        )
        baseline_vector = list(baseline_eval["objective_vector"])
        ref_point = reference_point(baseline_vector)

        # a candidate whose deterministic evaluation raises (e.g. a
        # process swap the frozen benchmark cannot cost) is scored with a
        # large penalty -> dominated -> rejected, and the loop continues.
        # Mirrors scripts/run_c5_real_pilot.py's _safe_real_evaluator.
        self.ledger = EqualBudgetLedger(
            _c4_safe_evaluator(self._raw_evaluator, ref_point),
            n_eval=self.n_eval,
            proposal_attempt_cap=self.attempt_cap,
        )

        archive = ParetoArchive()
        x_t = self.baseline_bom
        parts_by_id = {
            p["part_id"]: p for p in x_t["parts"]
        }

        hv_trajectory: list[float] = []
        # archive membership snapshot taken at each fresh evaluation, so
        # _converged can detect a stalled search even while HV is pinned
        # at 0 (every archive member outside the 1.2x reference point).
        archive_id_history: list[frozenset] = []
        convergence_reason = None
        intent_counts: dict[str, int] = {}
        accepted_steps = 0
        retries_seen: list[int] = []
        step_counter = _count()

        prev = {
            "selection": None,
            "evaluation": None,
            "accepted": None,
            "reason": None,
            "archive_status": None,
        }
        pending_selection = None
        pending_retries = 0
        terminal = None
        detail = None

        while not self.ledger.budget_exhausted:
            if self.ledger.attempts_exhausted:
                terminal = "ABORTED_BUDGET_UNREACHED"
                detail = (
                    f"proposal_attempt_cap={self.attempt_cap} hit at "
                    f"{self.ledger.objective_evaluations}/{self.n_eval}"
                )
                break

            converged, convergence_reason = self._converged(
                hv_trajectory, archive_id_history
            )
            if converged:
                terminal = "COMPLETE_CONVERGED"
                detail = (
                    f"{convergence_reason} over last {self.look_back_L} "
                    f"evaluations (epsilon_hv={self.epsilon_hv}); "
                    f"{self.ledger.objective_evaluations}/{self.n_eval} "
                    f"evaluations consumed"
                )
                break

            if pending_selection is not None:
                selection = pending_selection
            else:
                selection = self.selector.select(
                    archive_entries=archive.entries(),
                    last_evaluation=_selector_view(prev),
                )
                pending_retries = 0
            intent_counts[selection.intent] = (
                intent_counts.get(selection.intent, 0) + 1
            )

            target_part = parts_by_id[selection.part_id]
            retrieved_context, retrieval_meta = self._retrieve(
                selection, target_part
            )
            feedback_text = build_feedback_text(
                previous_evaluation=prev["evaluation"],
                previous_selection=prev["selection"],
                previous_accepted=prev["accepted"],
                previous_rejection_reason=(
                    prev["reason"] if pending_selection is not None else None
                ),
                baseline_vector=baseline_vector,
            )
            archive_text = build_archive_text(
                archive_entries=archive.entries(),
                baseline_vector=baseline_vector,
                last_archive_status=prev["archive_status"],
            )
            bundle = build_c4_prompt(
                x_t,
                target_part,
                self.generator.registry,
                selection=selection,
                feedback_text=feedback_text,
                archive_text=archive_text,
                retrieved_context=retrieved_context,
            )

            self.ledger.record_proposal_attempt()
            attempt_seed = (
                self.seed * ATTEMPT_SEED_STRIDE
                + self.ledger.proposal_attempts
            )
            try:
                rec = self.generator.generate(
                    bom=x_t,
                    target_part=target_part,
                    condition=self.condition,
                    seed=attempt_seed,
                    retrieved_context=retrieved_context,
                    retrieval_metadata=retrieval_meta,
                    prompt_bundle_override=bundle,
                )
            except Exception as exc:
                terminal = "ABORTED_PROVIDER"
                detail = f"{type(exc).__name__}: {exc}"
                break

            step_index = next(step_counter)
            event = event_from(rec, target_part_id=selection.part_id)
            event["event_type"] = "agentic_step"
            hash_before = canonical_bom_hash(x_t)

            accepted = False
            archive_status = None
            reason = None
            application = None
            outcome = None

            funnel_ok = (
                rec.get("parse_valid")
                and (
                    rec.get("schema_valid")
                    or self.ablation == "no_schema"
                )
                and rec.get("authority_valid")
            )
            if funnel_ok and rec.get("proposal"):
                application = apply_proposal(x_t, rec["proposal"])
                attach_applicability(event, application)
                if application.applicability_valid and not application.is_noop:
                    t0 = time.perf_counter()
                    outcome = self.ledger.consume(application.bom)
                    eval_dt = time.perf_counter() - t0
                    attach_evaluation(
                        event,
                        outcome,
                        baseline_vector,
                        eval_runtime_sec=eval_dt,
                    )
                    archive_status, size = archive.offer(
                        candidate_id=(
                            f"{self.condition}_s{self.seed}_t{step_index:03d}"
                        ),
                        objective_vector=outcome.result["objective_vector"],
                        bom_hash=outcome.bom_hash,
                        # cumulative diff vs the frozen baseline, not the
                        # single atomic move of this step
                        modifications=bom_modifications(
                            self.baseline_bom, application.bom
                        ),
                    )
                    attach_archive(event, archive_status, size)

                    feasible = (
                        (outcome.result.get("constraints") or {}).get(
                            "feasible"
                        )
                        is not False
                    )
                    if self.ablation == "no_validator":
                        accepted = True
                    else:
                        accepted = (
                            feasible
                            and archive_status in ACCEPT_STATUSES
                        )
                    if not accepted:
                        reason = _rejection_reason(
                            archive_status,
                            feasible,
                            (outcome.result.get("constraints") or {}).get(
                                "status"
                            ),
                        )

                    if outcome.consumed_budget:
                        hv_trajectory.append(
                            hypervolume_2d(archive.entries(), ref_point)
                        )
                        archive_id_history.append(
                            frozenset(
                                e["candidate_id"]
                                for e in archive.entries()
                            )
                        )
                elif application.is_noop:
                    reason = "no-op (no change vs the working BOM)"
                else:
                    reason = "not applicable: " + (
                        application.errors[0]
                        if application.errors
                        else "unknown"
                    )
            else:
                reason = "failed validity funnel (parse/schema/authority)"

            hash_after = (
                canonical_bom_hash(application.bom)
                if (accepted and application is not None)
                else hash_before
            )
            event["agentic"] = {
                "step_index": step_index,
                "selection": {
                    "part_id": selection.part_id,
                    "intent": selection.intent,
                    "policy_reason": selection.policy_reason,
                },
                "working_state_hash_before": hash_before,
                "working_state_hash_after": hash_after,
                "feedback_given": {
                    "had_previous": prev["evaluation"] is not None,
                    "previous_accepted": prev["accepted"],
                    "retry_of_selection": pending_retries,
                },
                "accepted": accepted,
                "retry_of_selection": pending_retries,
                "archive_status": archive_status,
                "hv_after": (
                    hv_trajectory[-1] if hv_trajectory else None
                ),
                "delta_hv_recent": self._delta_hv_recent(hv_trajectory),
            }
            event_log.write(event)

            if accepted:
                x_t = application.bom
                accepted_steps += 1
                self.selector.note_step(selection, archive_status, True)
                retries_seen.append(pending_retries)
                pending_selection = None
                pending_retries = 0
                prev.update(
                    selection=selection,
                    evaluation=(outcome.result if outcome else None),
                    accepted=True,
                    reason=None,
                    archive_status=archive_status,
                )
            else:
                pending_selection = selection
                pending_retries += 1
                prev.update(
                    selection=selection,
                    evaluation=(outcome.result if outcome else None),
                    accepted=False,
                    reason=reason,
                    archive_status=archive_status,
                )
                if pending_retries >= self.retry_cap_K:
                    self.selector.note_step(
                        selection, archive_status, False
                    )
                    retries_seen.append(pending_retries)
                    pending_selection = None
                    pending_retries = 0

        if terminal is None:
            terminal = "COMPLETE"

        entries = archive.entries()
        path, hv, nhv = _finalise_archive(
            pareto_archive_path,
            condition=self.condition,
            seed=self.seed,
            run_id=self.run_id,
            baseline_vector=baseline_vector,
            ref_point=ref_point,
            entries=entries,
        )

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
            extra={
                "c4": {
                    "steps": event_log.count,
                    "accepted_steps": accepted_steps,
                    "acceptance_rate": (
                        accepted_steps / max(1, event_log.count)
                    ),
                    "selection_intent_counts": intent_counts,
                    "mean_retries_per_selection": (
                        sum(retries_seen) / len(retries_seen)
                        if retries_seen
                        else 0.0
                    ),
                    "hv_trajectory": hv_trajectory,
                    "converged": terminal == "COMPLETE_CONVERGED",
                    "convergence_reason": convergence_reason,
                    "stop_rule": _stop_rule(terminal),
                    "ablation": self.ablation,
                }
            },
        )

    # -- helpers ---------------------------------------------------

    def _delta_hv_recent(self, hv_trajectory: list) -> float | None:
        if len(hv_trajectory) <= self.look_back_L:
            return None
        return hv_trajectory[-1] - hv_trajectory[-1 - self.look_back_L]

    def _converged(
        self, hv_trajectory: list, archive_id_history: list
    ) -> tuple[bool, str | None]:
        """Convergence check (eq 4.2).

        Previously gated on ``hv > 0.0``, which meant a working state with
        HV pinned at 0 (every accepted candidate outside the 1.2x
        baseline reference point) could never converge -- the loop ground
        on until the proposal-attempt cap even though the search had long
        since stalled. Now:

          A. HV has plateaued over the last L evaluations, regardless of
             its absolute level (``delta_HV_recent < epsilon_hv``); or
          B. the archive membership has not changed across the last L
             evaluations (covers the degenerate case where the HV number
             itself stays 0 but the search is genuinely stuck).
        """
        L = self.look_back_L
        if len(hv_trajectory) <= L:
            return False, None

        window = hv_trajectory[-L:]
        if max(window) - min(window) < self.epsilon_hv:
            return True, "hv_plateau"

        recent_ids = archive_id_history[-L:]
        if len(recent_ids) == L and len(set(recent_ids)) == 1:
            return True, "archive_unchanged"

        return False, None


def _selector_view(prev: dict) -> dict | None:
    ev = prev.get("evaluation")
    sel = prev.get("selection")
    if ev is None or sel is None:
        return None
    return {
        "constraints": ev.get("constraints"),
        "modified_part_id": sel.part_id,
    }


def _c4_safe_evaluator(inner, penalty_reference: list):
    penalty = [penalty_reference[0] * 10.0, penalty_reference[1] * 10.0]

    def evaluator(bom: dict) -> dict:
        try:
            return inner(bom)
        except Exception as exc:
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

    return evaluator


def _rejection_reason(archive_status, feasible, status=None) -> str:
    if status == "DETERMINISTIC_EVALUATION_FAILED":
        return "deterministic evaluation failed for this candidate"
    if not feasible:
        return "infeasible (deterministic constraint violation)"
    if archive_status == "dominated":
        return "dominated by the current non-dominated front"
    if archive_status == "duplicate":
        return "duplicate of a candidate already in the archive"
    return f"not accepted (archive status {archive_status})"


def _stop_rule(terminal: str) -> str:
    return {
        "COMPLETE": "budget",
        "COMPLETE_CONVERGED": "convergence",
        "ABORTED_BUDGET_UNREACHED": "attempt_cap",
        "ABORTED_PROVIDER": "provider_error",
    }.get(terminal, terminal)


def _finalise_archive(
    path,
    *,
    condition,
    seed,
    run_id,
    baseline_vector,
    ref_point,
    entries,
):
    rows = [
        archive_entry(
            e["candidate_id"],
            e["objective_vector"],
            baseline_vector,
            e["modifications"],
        )
        for e in entries
    ]
    if path is not None:
        return _write_pareto_archive(
            path,
            condition=condition,
            seed=seed,
            run_id=run_id,
            baseline_vector=baseline_vector,
            ref_point=ref_point,
            entries=rows,
        )
    hv = hypervolume_2d(entries, ref_point)
    denom = ref_point[0] * ref_point[1]
    return None, hv, (hv / denom if denom else 0.0)
