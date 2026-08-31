# A12 Run Notes

Generated: 2026-08-31T15:35:37Z
Git commit: d015be2e21af6f9714d19d6f456e664d56f00478

This file records the interpretive context for this run's
results/ CSVs so it travels with the artifacts.

## Per-condition

| condition | seeds | terminal statuses | n_eval range | HV median | hr_all max |
|---|---|---|---|---|---|
| C1 | 10 | COMPLETE_SPACE_EXHAUSTED | 11-15 | 9.180 | 0.019 |
| C2 | 10 | COMPLETE_SPACE_EXHAUSTED | 13-16 | 8.867 | 0.000 |
| C3 | 10 | COMPLETE_SPACE_EXHAUSTED | 0-0 | 0.000 | 1.000 |
| C5 | 10 | COMPLETE | 50-50 | 17.120 | 0.000 |

## Findings (data-driven)

- C3 PRODUCED NO VALID PROPOSALS: all 10 seeds generated proposals (200 total) but none passed the validity funnel -- every proposal failed at the 'schema' stage (hr_all = 1.0). C3 contributes no candidates and no hypervolume; treat its rows in the CSVs as a condition-level defect, not a result.
- CANDIDATE-DIVERSITY CEILING: 20/20 C1/C2 runs terminated COMPLETE_SPACE_EXHAUSTED at a median of 13 distinct candidates (range 11-16) against the N=50 budget (C3 excluded -- see above). The atomic material_id/process_id space over this benchmark is that small. The equal-budget ceiling is shared with C5, but only C5 (NSGA-II) reaches it -- read generative vs C5 hypervolume as a search-reach difference, not a like-for-like budget comparison.


---

## C3 — DEFERRED (do not interpret C3 rows as a result)

**Sweep executed 2026-08-31; C3 was found non-functional and is deferred
alongside C4, consistent with docs/A11_C3_TRAINING_STATUS.md's own scope
limitation.**

What happened: every C3 proposal failed at the schema stage
(`schema_rate_mean = 0.0`, `hr_all = 1.0`, `n_eval_consumed = 0` on all 10
seeds). Under the shared `build_proposal_prompt` the MLX LoRA adapter
emits a near-empty object (`{"change_type": "material_id"}`).

Root cause (train/inference mismatch, not a harness bug):
- `OllamaBackend` (C1/C2) enforces structured output via `format=<schema>`;
  `MLXLoRABackend` (C3) does free-form generation with no schema
  constraint.
- The C3 adapter (A11) was trained on 1,212 short `system` + terse `user`
  chat pairs — 3 fixed system prompts, ~1/3 of them *rejection/validation*
  tasks ("check this output, return a rejection JSON"), not free proposal
  generation. At inference it receives the 3,725-char A10 proposal prompt
  as a single `user` message with no system prompt — out of distribution.

Time-boxed fix attempt (2026-08-31): feeding C3 a training-style
`system` + compact `user` prompt raised valid output from 0/N to ~2/10
(schema+authority). The adapter still omits the required `proposal_id`
field on ~80% of generations and occasionally loops. Not usable as an
experimental condition. Not escalated to dataset/retraining work.

Consequences for the CSVs:
- `condition_summary.csv` / `seed_summary.csv`: the C3 rows are the
  honest raw output of a defective condition. Read them as a **negative
  finding about the A11 adapter's fitness under the A12 proposal prompt**,
  not as a fine-tuning result.
- `hypothesis_tests.csv` H1 (C3 vs C2): the `NOT_SUPPORTED` /
  `NOT_COMPUTABLE` rows are an artefact of C3 producing no valid
  proposals. **H1 is not evaluated in this run.** It becomes evaluable
  when a working C3 adapter / prompt exists — the harness supports it
  with no code change.

C1, C2 and C5 results in this run are unaffected and stand.
