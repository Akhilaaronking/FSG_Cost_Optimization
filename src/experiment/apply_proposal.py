"""
Atomic proposal application for the A12 harness (thesis 11.7, stage 4).

The generative conditions C1/C2/C3 produce one atomic modification per
proposal (proposal.schema.json). Each accepted proposal is applied to
the *frozen baseline* BOM independently -- never stacked onto a running
candidate (docs/A12 section 6, decision Q3). This keeps C1-C3 as
proposal-quality conditions; cumulative search is C4's job.

``apply_proposal`` is a pure structural helper:

  * it deep-clones the baseline and sets exactly one field,
  * it enforces stage-4 applicability -- the change must touch an
    allowed optimisation variable of a real part and must not mutate a
    protected ground-truth field,
  * it does NOT look at the registry or the engineering-reviewed search
    space. Identifier existence (stage 3) is already checked by the
    generator; search-space admissibility is a downstream quality
    question for C1-C3, not a funnel gate.
"""

from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field


# The A12 decision variables (docs/A12 section 3, condition_spec).
OPTIMISATION_FIELDS = ("material_id", "process_id")

# change_type -> the part field it writes, for the schema's constrained
# change types. Other change types resolve their field from target_field.
CHANGE_TYPE_FIELD = {
    "material": "material_id",
    "process": "process_id",
}

# Independently-prepared ground truth and part identity. A proposal that
# targets any of these is a stage-4 applicability failure
# (protected_field_writes), not merely out of scope.
PROTECTED_FIELDS = frozenset(
    {
        "part_id",
        "name",
        "geometry_assumption",
        "dimensions_mm",
        "volume_m3",
        "mass_kg",
        "manual_calculation",
    }
)


@dataclass(frozen=True)
class ProposalApplication:
    """Outcome of applying one atomic proposal to the frozen baseline."""

    applicability_valid: bool
    bom: dict | None
    target_part_id: str | None
    target_field: str | None
    change_type: str | None
    baseline_value: object
    new_value: object
    modifications: list = dataclass_field(default_factory=list)
    protected_field_writes: list = dataclass_field(default_factory=list)
    is_noop: bool = False
    errors: list = dataclass_field(default_factory=list)


def _not_applicable(
    *,
    part_id,
    target_field,
    change_type,
    baseline_value=None,
    new_value=None,
    protected_field_writes=None,
    errors,
) -> ProposalApplication:
    return ProposalApplication(
        applicability_valid=False,
        bom=None,
        target_part_id=part_id,
        target_field=target_field,
        change_type=change_type,
        baseline_value=baseline_value,
        new_value=new_value,
        modifications=[],
        protected_field_writes=list(protected_field_writes or []),
        is_noop=False,
        errors=list(errors),
    )


def apply_proposal(bom: dict, proposal: dict) -> ProposalApplication:
    """
    Apply one atomic proposal to ``bom`` (the frozen baseline) and
    return a :class:`ProposalApplication`.

    ``bom`` is never mutated. On success ``result.bom`` is a deep copy
    with exactly one field changed and ``result.modifications`` holds
    the canonical ``[{part_id, field, baseline, candidate}]`` record
    used by both events.jsonl and the C5 pareto rows.
    """
    part_id = proposal.get("part_id")
    change_type = proposal.get("change_type")
    target_field = proposal.get("target_field")
    new_value = proposal.get("new_value")

    parts = bom.get("parts", [])
    index_by_id = {
        part.get("part_id"): position
        for position, part in enumerate(parts)
    }

    if part_id not in index_by_id:
        return _not_applicable(
            part_id=part_id,
            target_field=target_field,
            change_type=change_type,
            new_value=new_value,
            errors=[f"unknown part_id {part_id!r}"],
        )

    part = parts[index_by_id[part_id]]

    # Resolve which field the proposal writes.
    resolved_field = CHANGE_TYPE_FIELD.get(change_type, target_field)

    # For the schema-constrained change types, target_field must agree.
    expected = CHANGE_TYPE_FIELD.get(change_type)
    if expected is not None and target_field != expected:
        return _not_applicable(
            part_id=part_id,
            target_field=target_field,
            change_type=change_type,
            new_value=new_value,
            errors=[
                f"change_type {change_type!r} requires target_field "
                f"{expected!r}, got {target_field!r}"
            ],
        )

    if resolved_field in PROTECTED_FIELDS:
        return _not_applicable(
            part_id=part_id,
            target_field=resolved_field,
            change_type=change_type,
            baseline_value=part.get(resolved_field),
            new_value=new_value,
            protected_field_writes=[resolved_field],
            errors=[
                f"target_field {resolved_field!r} is a protected "
                f"ground-truth field and must not be mutated"
            ],
        )

    if resolved_field not in OPTIMISATION_FIELDS:
        return _not_applicable(
            part_id=part_id,
            target_field=resolved_field,
            change_type=change_type,
            baseline_value=part.get(resolved_field),
            new_value=new_value,
            errors=[
                f"target_field {resolved_field!r} is not an A12 "
                f"optimisation variable (expected one of "
                f"{list(OPTIMISATION_FIELDS)})"
            ],
        )

    if not isinstance(new_value, str) or not new_value:
        return _not_applicable(
            part_id=part_id,
            target_field=resolved_field,
            change_type=change_type,
            baseline_value=part.get(resolved_field),
            new_value=new_value,
            errors=[
                f"new_value for {resolved_field!r} must be a non-empty "
                f"string, got {type(new_value).__name__}"
            ],
        )

    baseline_value = part.get(resolved_field)
    is_noop = baseline_value == new_value

    candidate = deepcopy(bom)
    candidate["parts"][index_by_id[part_id]][resolved_field] = new_value

    modifications = (
        []
        if is_noop
        else [
            {
                "part_id": part_id,
                "field": resolved_field,
                "baseline": baseline_value,
                "candidate": new_value,
            }
        ]
    )

    return ProposalApplication(
        applicability_valid=True,
        bom=candidate,
        target_part_id=part_id,
        target_field=resolved_field,
        change_type=change_type,
        baseline_value=baseline_value,
        new_value=new_value,
        modifications=modifications,
        protected_field_writes=[],
        is_noop=is_noop,
        errors=[],
    )
