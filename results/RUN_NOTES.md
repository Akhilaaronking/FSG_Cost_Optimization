# A12 Run Notes

Generated: 2026-09-01T16:15:43Z
Git commit: df8eae0777533b158747c58dfbb8fb1bcfaac97e

This file records the interpretive context for this run's
results/ CSVs so it travels with the artifacts.

## Per-condition

| condition | seeds | terminal statuses | n_eval range | HV median | hr_all max |
|---|---|---|---|---|---|
| C1 | 10 | COMPLETE_SPACE_EXHAUSTED | 11-15 | 9.180 | 0.019 |
| C2 | 10 | COMPLETE_SPACE_EXHAUSTED | 13-16 | 8.867 | 0.000 |
| C3 | 10 | COMPLETE_SPACE_EXHAUSTED | 0-0 | 0.000 | 1.000 |
| C4_base | 10 | COMPLETE_CONVERGED | 11-22 | 9.437 | 0.000 |
| C4_base_no_rag | 3 | COMPLETE_CONVERGED | 11-18 | 8.337 | 0.000 |
| C4_base_no_schema | 3 | ABORTED_BUDGET_UNREACHED | 0-0 | 0.000 | 1.000 |
| C4_base_no_validator | 3 | COMPLETE_CONVERGED | 11-11 | 8.441 | 0.061 |
| C5 | 10 | COMPLETE | 50-50 | 17.120 | 0.000 |

## Findings (data-driven)

- C3 PRODUCED NO VALID PROPOSALS: all 10 seeds generated proposals (200 total) but none passed the validity funnel -- every proposal failed at the 'schema' stage (hr_all = 1.0). C3 contributes no candidates and no hypervolume; treat its rows in the CSVs as a condition-level defect, not a result.
- CANDIDATE-DIVERSITY CEILING: 20/20 C1/C2 runs terminated COMPLETE_SPACE_EXHAUSTED at a median of 13 distinct candidates (range 11-16) against the N=50 budget (C3 excluded -- see above). The atomic material_id/process_id space over this benchmark is that small. The equal-budget ceiling is shared with C5, but only C5 (NSGA-II) reaches it -- read generative vs C5 hypervolume as a search-reach difference, not a like-for-like budget comparison.
- C4_base TOOL-LOOP: 10 seeds | stop rules {'convergence': 10} | n_eval 11-22 of 50 | acceptance rate mean 0.22 | HV median 9.437. Unlike C1/C2/C3, C4 compounds changes on a working state, so it can consume the full budget -- read C4-vs-C5 hypervolume as like-for-like (eq 11.4 / 11.21).
- C4_base_no_rag TOOL-LOOP: 3 seeds | stop rules {'convergence': 3} | n_eval 11-18 of 50 | acceptance rate mean 0.18 | HV median 8.337. Unlike C1/C2/C3, C4 compounds changes on a working state, so it can consume the full budget -- read C4-vs-C5 hypervolume as like-for-like (eq 11.4 / 11.21).
- C4_base_no_schema PRODUCED NO EVALUATED CANDIDATES: all 3 seeds ran the loop but no proposal reached deterministic evaluation -- treat its rows as a condition-level defect, not a result (cf. C3).
- C4_base_no_validator TOOL-LOOP: 3 seeds | stop rules {'convergence': 3} | n_eval 11-11 of 50 | acceptance rate mean 0.34 | HV median 8.441. Unlike C1/C2/C3, C4 compounds changes on a working state, so it can consume the full budget -- read C4-vs-C5 hypervolume as like-for-like (eq 11.4 / 11.21).
- ABORTED RUNS (3): C4_base_no_schema/seed0=ABORTED_BUDGET_UNREACHED, C4_base_no_schema/seed1=ABORTED_BUDGET_UNREACHED, C4_base_no_schema/seed2=ABORTED_BUDGET_UNREACHED -- excluded from inferential analysis until rerun under the same frozen config (thesis 11.18).

