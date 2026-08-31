"""
C4 feedback assembly (docs/A13 section 4).

Turns the previous step's evaluator result + archive state into the two
text blocks the C4 prompt carries: EVALUATOR FEEDBACK and ARCHIVE STATE.
Pure formatters — no model calls, no side effects.
"""


def _pct(base, value):
    if not base or value is None:
        return None
    return (base - value) / base * 100.0


def build_feedback_text(
    *,
    previous_evaluation: dict | None,
    previous_selection=None,
    previous_accepted: bool | None = None,
    previous_rejection_reason: str | None = None,
    baseline_vector: list | None = None,
) -> str:
    """EVALUATOR FEEDBACK block for the C4 prompt."""
    if previous_evaluation is None:
        return "No previous step -- this is the first modification."

    objectives = previous_evaluation.get("objectives") or {}
    cost = objectives.get("cost_eur")
    mass = objectives.get("mass_kg")
    base_cost, base_mass = (baseline_vector or [None, None])[:2]

    lines = []
    if previous_selection is not None:
        lines.append(
            f"Previous target: {previous_selection.part_id} "
            f"(goal: {previous_selection.intent})."
        )
    if cost is not None and mass is not None:
        dc = f"{base_cost - cost:+.2f}" if base_cost is not None else "n/a"
        dm = f"{base_mass - mass:+.5f}" if base_mass is not None else "n/a"
        cp = _pct(base_cost, cost)
        mp = _pct(base_mass, mass)
        lines.append(
            f"Result: cost {cost:.2f} EUR (vs baseline {dc}"
            + (f", {cp:+.1f}%" if cp is not None else "")
            + f"), mass {mass:.5f} kg (vs baseline {dm}"
            + (f", {mp:+.1f}%" if mp is not None else "")
            + ")."
        )
    else:
        lines.append(
            "Result: not scored (proposal rejected before deterministic "
            "evaluation)."
        )

    constraints = previous_evaluation.get("constraints") or {}
    if constraints.get("evaluated"):
        vc = constraints.get("violation_count")
        lines.append(
            f"Deterministic constraints: "
            + ("no violations." if not vc else f"{vc} violation(s).")
        )
    else:
        lines.append(
            "Deterministic constraints: not evaluated on this benchmark."
        )

    if previous_accepted is False:
        reason = previous_rejection_reason or "did not improve the archive"
        lines.append(
            f"The previous proposal was REJECTED ({reason}); the working "
            f"BOM was not advanced. Try a materially different change."
        )
    elif previous_accepted is True:
        lines.append(
            "The previous proposal was ACCEPTED and is part of the "
            "working BOM below."
        )

    return "\n".join(lines)


def build_archive_text(
    *,
    archive_entries: list,
    baseline_vector: list | None = None,
    last_archive_status: str | None = None,
) -> str:
    """ARCHIVE STATE block for the C4 prompt."""
    if not archive_entries:
        base = baseline_vector or [None, None]
        return (
            "Archive holds only the baseline "
            f"(cost {base[0]}, mass {base[1]}). No improvement yet."
        )

    best_cost = min(archive_entries, key=lambda e: e["objective_vector"][0])
    best_mass = min(archive_entries, key=lambda e: e["objective_vector"][1])
    lines = [
        f"Non-dominated front: {len(archive_entries)} point(s).",
        f"Best cost so far: {best_cost['objective_vector'][0]:.2f} EUR "
        f"(mass {best_cost['objective_vector'][1]:.5f} kg).",
        f"Lowest mass so far: {best_mass['objective_vector'][1]:.5f} kg "
        f"(cost {best_mass['objective_vector'][0]:.2f} EUR).",
    ]
    if last_archive_status:
        lines.append(
            f"Last accepted move classified as: {last_archive_status}."
        )
    return "\n".join(lines)
