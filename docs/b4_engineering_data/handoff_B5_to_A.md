# B5 Handoff — Deterministic Constraint Dataset

**From:** engineering data & validation
**To:** pipeline / optimization
**Date:** 2026-08-22

## B4 correction — confirmed matched

Your reported €312.02 (vs frozen €312.03) is correct — PILOT_004 was already fixed to v2
on my end (material €2.79, total €9.78) before I saw your message. Sums now match exactly.
See `docs/CHANGELOG.md` for the v2 correction log.

## Deterministic constraint dataset

Two files, same content, pick whichever is easier to ingest:
- `data/processed/deterministic_constraints_B5.json`
- `data/processed/deterministic_constraints_B5.csv`

**27 entries total: 26 from FSG Rules 2026 v1.1 Section S3, plus 1 derived internal
quality gate (see below).**

| Field | Meaning |
|---|---|
| `rule_id` | e.g. `S_3.5.12` — maps to the FSG rule number |
| `rule_category` | `hard` / `engineering` / `interpretive` |
| `affected_part_category` | What the rule applies to (e.g. "CCBOM - all parts") |
| `parameter_field` | The specific field/attribute being constrained |
| `operator` | e.g. `==`, `in`, `not_in`, `<=`, or `subjective_review` for non-deterministic rules |
| `limit_value` | The threshold/set/value to check against |
| `units` | Unit type, where applicable |
| `source_id` | Always `FSG_RULES_2026` — see `docs/source_register.csv` |
| `fsg_reference` | Exact rule number + page in the source PDF |
| `deterministic` | `true`/`false` — see below |
| `notes` | Rationale for classification and any caveats |

## Important: 18 deterministic, 9 non-deterministic — please route accordingly

**18 rules (`deterministic: true`)** have a real `operator` + `limit_value` pair your
pipeline can check in code — e.g. `S_3.4.9`: unit_system must be in the metric set;
`S_3.5.12`: currency must equal EUR.

**9 rules (`deterministic: false`)** are `engineering` or `interpretive` category —
these have `operator: "subjective_review"` and no real threshold, because none
exists in the rule text. Examples: `S_3.5.11` ("costs as realistic as possible" —
no fixed number to check against) and `S_3.4.8` (part naming clarity — no ontology
exists to machine-verify this). **Please don't build automated pass/fail checks for
these 9** — they need to stay flagged for human/judge review, or your pipeline will
produce false confidence on things that genuinely require engineering judgement.

## New: DERIVED_QG_001 — internal provenance quality gate (NOT an FSG rule)

Per your request, I've added a check: **every material/process/fastener cost value
used by the evaluator must have a non-null `source_id` that exists in
`docs/source_register.csv`.**

This is `rule_category: "derived_quality_gate"` in the dataset — deliberately
**separate** from `S_3.5.11`. Important distinction:

- `S_3.5.11` ("realistic costs") **stays interpretive/non-deterministic** — no
  fixed threshold can prove "realism," so it's not being reclassified.
- `DERIVED_QG_001` only checks **traceability** (does a source exist?), not
  correctness or realism of the value itself.
- **Passing DERIVED_QG_001 is necessary but not sufficient** for `S_3.5.11`
  compliance — a cost could have a valid source and still not be realistic (e.g.
  a wrong material grade), and this gate won't catch that.

Please implement this as internal tooling/CI-style validation, not as something
that appears in FSG compliance reporting — it's a data-quality check we're adding
for our own pipeline's benefit, not a competition requirement.

Let me know if you want the schema adjusted or additional fields.
