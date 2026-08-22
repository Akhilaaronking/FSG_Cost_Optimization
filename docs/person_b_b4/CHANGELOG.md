# Changelog — Frozen Data Files

Every time a file is frozen (pilot BOM, development BOM, final benchmark,
human baseline, etc.), add an entry here. Never overwrite a frozen file —
create a new version (`_v2`, `_v3`, ...) and log why.

Format:

## <file path> — v<N> — <date>
- Frozen by:
- Reason for freeze / what stage this supports:
- Known limitations at freeze time:
- Hash (optional, e.g. `sha256sum <file>`):

---

## data/benchmark/pilot_10_parts.json — v1 (FROZEN) — 2026-08-22
- Frozen by: Aaron
- Reason for freeze / what stage this supports: B4 pilot BOM for Person A's unit tests - all 10 parts complete
- Known limitations at freeze time:
  - All process/print times are engineering-judgement estimates, not sourced from any external reference
  - PILOT_008 (battery box panel) cost is dominated by small-batch composite pricing (EUR499.75/kg) - worth a methodology note on small-quantity vs bulk sourcing cost inflation
  - PILOT_010 (wiring clip) fastener cost exceeds part material+process cost - illustrates how small parts can be fastener-cost-dominated, worth noting in BOM analysis
  - Sum of all 10 part totals: EUR312.03
- Hash: 96b012d5e7b1ac7e02f212aa3d61e680d35aebf67f0116e45e8ca05c9a9bacdb (sha256)

## data/benchmark/pilot_10_parts.json — v2 (FROZEN, supersedes v1) — 2026-08-22
- Frozen by: Aaron
- Reason for freeze: correction to PILOT_004 flagged by Person A during A4 validation cross-check
- Change: PILOT_004 material_cost_eur corrected 2.80 -> 2.79; total_cost_eur corrected 9.79 -> 9.78
  (v1 had a rounding error: exact calc is 0.05652 kg x EUR49.44/kg = EUR2.7943488 -> EUR2.79,
  not EUR2.80 as originally recorded)
- All other 9 parts (001, 002, 003, 005, 006, 007, 008, 009, 010) re-verified by hand against
  their stored values during this correction pass - no further errors found
- New sum of all 10 part totals: EUR312.02 (was EUR312.03 in v1)
- Hash: 80a14a6eb7bb240e643674249eb24c11936e929b25377876be2de4f358196d92 (sha256)
- Downstream files updated to match: data/benchmark/development_30_parts.json (B6),
  docs/handoff_B4_to_A.md
