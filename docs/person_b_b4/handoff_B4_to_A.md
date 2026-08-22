# B4 Handoff — Pilot 10-Part BOM

**From:** Aaron (Person B — data & validation)
**To:** Person A (pipeline/optimization)
**Date:** 2026-08-22
**Status:** FROZEN, ready for unit testing

## What's included

`data/benchmark/pilot_10_parts.json` — 10 parts spanning metals, plastics, and
composite, across 6 manufacturing processes. Every part has:
- Material (`material_id` — matches a row in `data/processed/materials.csv`)
- Process (`process_id` — matches a row in `data/processed/processes.csv`)
- Fasteners (`fastener_id` — matches rows in `data/processed/fasteners.csv`)
- Geometry assumptions and calculated volume/mass
- A full **manual ground-truth cost calculation** (material + process + fastener = total)

## What I need from you (the PASS gate)

Run your pipeline's cost/mass calculator against these same 10 parts and compare
against the `manual_calculation.total_cost_eur` value in each part record.

| Part | My manual total (EUR) |
|---|---|
| PILOT_001 — Pedal box bracket | 25.32 |
| PILOT_002 — Suspension pickup plate | 37.82 |
| PILOT_003 — Chassis mounting tab | 8.25 |
| PILOT_004 — Roll hoop gusset | 9.79 |
| PILOT_005 — Steering column spacer | 7.80 |
| PILOT_006 — Bearing carrier bushing | 5.76 |
| PILOT_007 — Aero mounting standoff | 13.90 |
| PILOT_008 — Battery box side panel | 194.79 |
| PILOT_009 — Sensor mounting bracket (3D print) | 5.49 |
| PILOT_010 — Wiring harness clip | 3.11 |
| **Sum** | **312.03** |

If your pipeline's numbers don't match mine within a small rounding tolerance,
that's a bug to chase down together **before** this scales to the full 30-40
part development BOM (next stage) — much cheaper to catch now.

## Known limitations / things to be aware of

1. **All process/print times are my engineering-judgement estimates**, not
   sourced from anywhere external — see each part's `calc_notes` for the
   assumed time. If your pipeline calculates time differently (e.g. from
   actual geometry/toolpath), expect some legitimate divergence here — that's
   not a bug, just a different estimation method. Worth discussing which
   approach we standardize on going forward.
2. **PILOT_008's cost is dominated by material** — the composite plate came
   from small-batch retail pricing (~EUR500/kg), not bulk aerospace stock.
   Real, but worth flagging so it doesn't look like an error.
3. **PILOT_010's fastener cost exceeds its material+process cost** — small
   parts can be fastener-dominated. Also not a bug, just how the numbers
   shake out for tiny items.
4. **Process rates in `processes.csv` mix incompatible bases** — some are
   full shop rates (machine + labor + overhead), others are raw employee
   wages, and units vary (per-minute, per-hour, per-meter, per-gram, per-m²).
   Check the `notes` column per process before assuming they're
   directly comparable.

## Where everything lives

- `docs/source_register.csv` — every source, dated and URL'd
- `docs/provenance_notes.md` — ID conventions and rules
- `data/raw/` — screenshots/evidence for every sourced number
- `scripts/validate_provenance.py` — run this anytime; it checks register
  consistency and catches CSV structural bugs (comma-in-field issues, mainly)
- `docs/CHANGELOG.md` — freeze history for `pilot_10_parts.json`

Let me know once you've run the comparison — happy to jump on a call if
anything doesn't line up.
