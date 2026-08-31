# Provenance Notes

## Purpose
This document defines the conventions used in `source_register.csv` and
`data/raw/` so that every value in the processed tables can be traced back
to a specific, dated source.

## source_id naming convention
- Official FSG documents: `FSG_<TOPIC>_<YEAR>` e.g. `FSG_RULES_2026`
- Material datasheets: `MAT_<NNN>` e.g. `MAT_001`
- Supplier snapshots: `SUP_<SUPPLIERNAME>_<YYYYMMDD>` e.g. `SUP_MISUMI_20260821`
- Public team reports: `TEAM_<TEAMNAME_OR_ID>_<YEAR>`
- Synthetic data batches: `SYN_<TOPIC>_<NNN>`

IDs are permanent once assigned. Never reuse or reassign an ID, even if the
source is later found to be wrong or unusable — mark it deprecated instead.

## Snapshot rule
If a web source (supplier catalogue, online datasheet) changes over time,
create a NEW source_id with a new date suffix. Never overwrite or edit an
existing raw file. Old snapshots stay in `data/raw/` even if unused.

## Authority hierarchy (for resolving disagreements between sources)
1. Official FSG rules/cost documents (highest authority)
2. Manufacturer/material datasheets
3. Supplier catalogue/API data
4. Public team reports (treat as illustrative, not authoritative)
5. Generic catalogue / synthetic data (lowest — always flagged synthetic=true)

When two sources disagree, record both values, cite this hierarchy to choose
the one used downstream, and keep a note of the disagreement — do not
silently discard the lower-authority value.

## Redistribution status
Every source_id gets a `redistributable` value of one of:
- `yes` — confirmed safe to include in public benchmark release
- `no` — confirmed proprietary/restricted, research use only
- `check` — not yet verified; default for all new entries until confirmed

Nothing is marked `yes` by default. Public team reports default to `no`
unless the team has given explicit permission.

## What "raw" means
`data/raw/` holds original files exactly as obtained (PDF, screenshot,
CSV export, HTML save). Never edit these in place. All cleaning,
normalising, and unit conversion happens only in `data/processed/`,
where every derived value still carries its source_id.

## Validation
Before treating a processed table as final, spot-check that:
- every source_id used in `data/processed/*` exists as a row in
  `source_register.csv`
- every source_id in the register has a corresponding file (or dated
  subfolder) in `data/raw/`
