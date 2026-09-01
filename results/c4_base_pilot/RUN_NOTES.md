# A12 Run Notes

Generated: 2026-09-01T03:39:19Z
Git commit: 28e1b99ad3192bc568bec56d6b53c0a47df58a6c

This file records the interpretive context for this run's
results/ CSVs so it travels with the artifacts.

## Per-condition

| condition | seeds | terminal statuses | n_eval range | HV median | hr_all max |
|---|---|---|---|---|---|

## Findings (data-driven)

- C4_base_no_rag TOOL-LOOP: 3 seeds | stop rules {'convergence': 3} | n_eval 11-18 of 50 | acceptance rate mean 0.18 | HV median 8.337. Unlike C1/C2/C3, C4 compounds changes on a working state, so it can consume the full budget -- read C4-vs-C5 hypervolume as like-for-like (eq 11.4 / 11.21).
- C4_base_no_schema PRODUCED NO EVALUATED CANDIDATES: all 3 seeds ran the loop but no proposal reached deterministic evaluation -- treat its rows as a condition-level defect, not a result (cf. C3).
- C4_base_no_validator TOOL-LOOP: 3 seeds | stop rules {'convergence': 3} | n_eval 11-11 of 50 | acceptance rate mean 0.34 | HV median 8.441. Unlike C1/C2/C3, C4 compounds changes on a working state, so it can consume the full budget -- read C4-vs-C5 hypervolume as like-for-like (eq 11.4 / 11.21).
- ABORTED RUNS (3): C4_base_no_schema/seed0=ABORTED_BUDGET_UNREACHED, C4_base_no_schema/seed1=ABORTED_BUDGET_UNREACHED, C4_base_no_schema/seed2=ABORTED_BUDGET_UNREACHED -- excluded from inferential analysis until rerun under the same frozen config (thesis 11.18).

