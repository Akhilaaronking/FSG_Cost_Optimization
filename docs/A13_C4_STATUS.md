# A13 C4 Agentic Tool-Loop Status

## Status

The **C4 agentic tool-loop** (`C4Driver`) is implemented, tested (≈65 C4 tests;
full repository suite 361 passing) and **pilot-run against the base backend
(`C4_base` — Ollama `llama3.1:8b` + RAG + deterministic tool-loop)**.

- **C4-base**, **10 seeds**, `N = 50` (`runs/C4_base/seed_00..09/`): runs
  end-to-end, zero hallucinations, all proposals
  parse/schema/identifier/applicability-valid. Every seed converges early
  (`COMPLETE_CONVERGED`, `hv_plateau`) at 11–22 evaluations — the atomic
  candidate space is still shallow, and compounding on a working state buys
  only a handful of accepted moves before the front stops improving. HV
  median 9.44 (range 0.0–11.56; seed 0's 0.0 is the mass-drift case, §"Seed 0
  drifts").
- **Canonical C4** (fine-tuned C3 backend swapped in for Ollama, zero loop
  changes) was smoke-tested (1 seed, `N=50`, 2026-09-01) after the C3 backend
  fixes (commit `05a605c`: repetition penalty + system/user chat roles) were
  applied and verified. Result: **diagnosed non-functional**, same root cause
  as C3 -- see "Canonical C4 -- Diagnosed Non-Functional" below. The loop,
  identity, metrics and CSV rows are all in place for it.
- **H3 ablation pilot** (3 seeds each): `no_rag` and `no_validator` behave as
  expected; `no_schema` is **non-functional** on this backend (same class as
  C3) and is recorded as a degenerate/aborted ablation, excluded from
  inferential H3 (11.18).
- **H2 (C4 vs C5)** is **computed as C4-base vs C5** (10 shared seeds), per
  the contingency above, since canonical C4 is diagnosed non-functional.
  Result: **NOT_SUPPORTED** on both `final_hypervolume` and
  `categorical_subset_hypervolume` (eq 3.52 / 3.53) — C4-base HV mean 8.73 vs
  C5 15.91, `d_z = −1.90`, one-sided `p = 1.0`. A non-significant H2 is a
  valid result (11.13). See "H2 — C4-base vs C5" below.
- **H3** is computed at the pilot scope (10-seed C4-base HR vs 3-seed
  ablations, descriptive): eq 10.39 SUPPORTED, eq 10.38 NOT_SUPPORTED. A full
  10-seed H3 is not run (see "On going to full 10-seed H3").

## Canonical C4 -- Diagnosed Non-Functional (same mechanism as C3)

A 1-seed smoke test of canonical C4 (`--condition C4 --seeds 0 --budget 50`)
was run after the C3 backend fixes (repetition penalty, system/user chat
roles -- commit `05a605c`) were applied and verified via isolated diagnostic.
Canonical C4 reuses the C3 fine-tuned adapter, wrapped in
`build_c4_prompt()`, which adds SELECTION / FEEDBACK / ARCHIVE blocks on top
of `build_proposal_prompt()` -- an even longer prompt than C3's.

Result: identical failure signature to the diagnosed C3 pilot --
`raw_output_sha256` constant across steps and target parts, output truncates
to `{"change_type":"material_id"}`, 100% `SCHEMA_ERROR` / `UNKNOWN_PART_ID`,
zero evaluations, across 39+ steps and 5 distinct parts before the run was
stopped. This confirms canonical C4 is non-functional for the same root
cause already diagnosed for C3: the LoRA adapter (trained on short-form,
1-3 sentence templated examples) does not generalize to production-length
structured prompts -- and C4's prompt is longer still than C3's.

**Disposition:** H2 is evaluated as C4-base vs C5 rather than canonical-C4
vs C5, per the contingency already documented above -- not a new scope
decision. A weak or non-significant H2 result under C4-base is itself a
valid, citable result (per SS11.13, already invoked for the C4-base seed-0
drift finding), reflecting the base backend's known limitations (no
backtracking, weaker generation) rather than a null result from a broken
measurement.

Canonical C4 and C3 are reported together as a single diagnosed negative
finding: schema-constrained fine-tuning on short templated examples does not
transfer to the long, structured, RAG-integrated prompts used in production,
across both the single-shot (C3) and agentic-loop (canonical C4) conditions.

Design: `docs/A13_C4_AGENTIC_LOOP_DESIGN.md` (decisions LOCKED 2026-08-31).
Deviation record: `EXPERIMENT_DEVIATIONS.txt` (A13 section).

## The loop (thesis eq 10.30–10.35)

One continuous search per seed on an **evolving working state `x_t`**, unlike
C1/C2/C3 which apply every proposal to the frozen baseline `x0` independently:

```
s_t  = Select(A_t, history)        ArchiveGuidedSelector — deterministic, seeded
r_t  = Retrieve(s_t, k=5)          RAG top-k            (skipped: no_rag)
π_t  = LLM(s_t, r_t, feedback, schema)                  (relaxed: no_schema)
x'_t = Apply(x_t, π_t)             apply_proposal — to x_t, NOT x0 (compounding)
y_t  = E(x'_t)                     EqualBudgetLedger.consume — 1 budget unit iff fresh
A_{t+1} = ND(A_t ∪ {x'_t})         ParetoArchive.offer
accept  = feasible ∧ offer_status ∈ {pareto_improving, non_dominated}   (skipped: no_validator)
```

On `accept`, `x_t ← x'_t` and the selection episode closes; on reject the same
selection is retried up to `K = 3`, then the selector moves to a different
part/intent (`x_t` never resets — bounded backtracking is a v1-future
extension, §13 decision 7).

## Stopping (§6 of the design, eq 4.2)

| Rule | Terminal status |
|---|---|
| `N = 50` fresh objective evaluations reached | `COMPLETE` |
| `hv_plateau` — `max−min` of HV over the last `L = 10` evals `< ε = 0.1`, **at any level incl. 0** | `COMPLETE_CONVERGED` |
| `archive_unchanged` — archive membership identical across the last `L` evals | `COMPLETE_CONVERGED` |
| proposal-attempt cap `max(150, 6·N) = 300` hit before `N` | `ABORTED_BUDGET_UNREACHED` |
| provider exception | `ABORTED_PROVIDER` |

`_converged` originally gated on `HV > 0`, so a working state whose whole
archive sat outside the 1.2× reference point (HV pinned at 0) could never
converge and ground on to the attempt cap — pilot seed 0 was 591 LLM calls /
61 min. Fixed in `28e1b99`; `convergence_reason` is now recorded in
`metrics.json`'s `c4` block and `RunOutcome.extra`.

## C4-base run (2026-09-01, `N = 50`, attempt cap 300, 10 seeds)

Baseline 312.02 EUR / 0.6507 kg; 1.2× reference point 374.4 EUR / 0.7809 kg.
Every seed terminated `COMPLETE_CONVERGED` via `hv_plateau`.

| seed | n_eval | HV | best cost | lowest mass | wall |
|---|---|---|---|---|---|
| 0 | 11 | **0.000** | 178.33 | 0.7850 | 401 s |
| 1 | 11 | 9.099 | 179.01 | 0.6458 | 501 s |
| 2 | 14 | 9.070 | 179.23 | 0.6507 | 593 s |
| 3 | 22 | 10.042 | 177.29 | 0.6397 | 1130 s |
| 4 | 11 | 9.418 | 179.76 | 0.6298 | 544 s |
| 5 | 12 | 8.375 | 178.33 | 0.6507 | 283 s |
| 6 | 16 | 10.250 | 178.85 | 0.6340 | 624 s |
| 7 | 11 | 10.078 | 179.75 | 0.6236 | 236 s |
| 8 | 11 | 9.437 | 180.01 | 0.6298 | 172 s |
| 9 | 14 | 11.555 | 179.23 | 0.6149 | 352 s |

HV mean **8.73**, median 9.43, sd 3.02 (the mean is dragged by seed 0's 0.0).
Acceptance rate mean 0.22. Funnel parse / schema / identifier / applicability
all 1.00; `hr_all` 0.00 on every seed. ~4 evaluations per 9 proposals — the
rest are no-ops or duplicates from the base model repeating itself, absorbed
by the retry cap.

**Reading:**

- **The loop is mechanically sound.** Selection is deterministic from the seed,
  the working state compounds across accepted steps (archive entries carry
  multi-part cumulative diffs), the equal-budget ledger and convergence stop
  behave.
- **9 of 10 seeds converge at HV 8.4–11.6** — around and a little above the
  atomic C1/C2 results from the A12 sweep (HV 8.87–9.18). Compounding buys a
  modest amount over the atomic conditions on this benchmark, well short of
  C5's NSGA-II reach (median 17.1).
- **Seed 0 drifts.** Compounding cost-reducing material swaps pushed aggregate
  mass to 0.785 kg, just past the 1.2× reference bound (0.781 kg), so its whole
  non-dominated set scores zero hypervolume. This is a genuine C4-base
  characteristic — weak base model, no backtracking — not a harness defect. It
  is exactly the per-seed variance the paired C4-vs-C5 comparison (eq 11.21) is
  built to capture. Bounded backtracking (§13 decision 7, deferred) is the
  obvious mitigation.

## H2 — C4-base vs C5 (eq 11.21, `results/hypothesis_tests.csv`)

Paired over 10 shared seeds, `dᵢ = HV_{C4base,i} − HV_{C5,i}`, one-sided
Wilcoxon signed-rank ("greater"). `results/` was assembled by
`scripts/aggregate_h2.py` from the committed C1/C2/C3/C5 metrics + this
C4-base run + the ablation pilot; C5 is **not** re-run.
`scripts/refresh_metrics.py` first regenerated the committed
C1/C2/C3/C5 `metrics.json` (HV unchanged) so
`categorical_subset_hypervolume` (added after the A12 sweep) is present for
the eq 3.53 row.

| metric | C5 mean | C4-base mean | dᵢ mean | d_z | p (1-sided) | threshold_met | decision |
|---|---|---|---|---|---|---|---|
| `final_hypervolume` (eq 3.52) | 15.91 | 8.73 | −7.17 | −1.90 | 1.00 | False | **NOT_SUPPORTED** |
| `categorical_subset_hypervolume` (eq 3.53) | 15.91 | 8.73 | −7.17 | −1.90 | 1.00 | False | **NOT_SUPPORTED** |

The two rows are identical because `material_id` / `process_id` are the only
decision variables, so the categorical-subset archive equals the full archive
(the metric will diverge once geometry/continuous variables enter). C4-base
does not reach C5's hypervolume on this benchmark; per 11.13 a
non-significant H2 is itself a valid, citable result, and here it reflects
the base backend's known limits (no backtracking, weaker generation, the
seed-0 drift) against a mature NSGA-II — not a broken measurement.

## H3 ablation pilot (2026-09-01, 3 seeds each)

| ablation | terminal ×3 | n_eval | `hr_all` | HV (per seed) | note |
|---|---|---|---|---|---|
| **`no_rag`** — skip Retrieve, empty context | `COMPLETE_CONVERGED` | 11–18 | 0.00, 0.00, 0.00 | 0 / 9.36 / 8.34 | converges; RAG removal did **not** raise hallucination |
| **`no_schema`** — drop `format=<schema>`, best-effort parse | `ABORTED_BUDGET_UNREACHED` ×3 | **0** | **1.00 ×3** | 0 / 0 / 0 | **non-functional** — see below |
| **`no_validator`** — `Accept()` ignores feasibility | `COMPLETE_CONVERGED` | 11 | 0.025, 0.061, 0.00 | 0 / 9.10 / 8.44 | id/applicability rate drops to 0.94–0.98; accepts dominated (accept rate ↑ to 0.28–0.42) |

`hypothesis_tests.csv` (10-seed C4-base HR vs 3-seed ablation HR, descriptive
— no family-wise correction):

- `HR_full < 0.05` (eq 10.39) — **SUPPORTED** (`HR_full = 0.0` over all 10
  seeds).
- `HR_full < min(HR_ablations)` (eq 10.38) — **NOT_SUPPORTED**: `HR_no_rag`
  is also 0.0, so the strict inequality fails on a tie at the floor. The base
  backend is already hallucination-free on this benchmark (the A12
  zero-hallucination-baseline finding), so the fine-tuning/RAG axis of H3 has
  no headroom here, just as H1's hallucination axis was `NOT_COMPUTABLE`.
  `no_schema` (`HR = 1.0`, degenerate) is correctly excluded from the `min()`.

### `no_schema` — non-functional (deferred, cf. C3)

With `OllamaBackend(enforce_schema=False)` the `format=<json schema>`
constraint is removed and `llama3.1:8b` free-form output under the C4 prompt
**does not parse** as a proposal at all — `parse_rate` 0.0, `hr_all` 1.0, zero
evaluations across 300 attempts × 3 seeds (~26.5 min/seed under the tightened
cap; it was ~2.4 h/seed at the old 1500 cap). Same failure class as C3: without
grammar-constrained decoding the base model cannot emit a usable proposal.
Recorded as a degenerate ablation, excluded from inferential H3 (11.18). The
"schema enforcement is load-bearing for the base backend" claim has a clean
3/3 citation. A working `no_schema` ablation needs either a canonical C4 run
(fine-tuned adapter) or the `docs/A12_C3_CONSTRAINED_DECODING_PLAN.md` port.

### On taking the ablations to 10 seeds

C4-base itself is now 10-seed; the three ablation conditions remain at 3.
§13 decision 5: run them to 10 only if the pilot shows the expected ordering.
It partly does — `no_schema` collapses the funnel and `no_validator` lets
invalid identifiers through, both as predicted. `no_rag` does **not** raise
hallucination, consistently with A12's zero-hallucination baseline. Because
eq 10.38 fails on a tie at `HR = 0` and the base backend has no headroom below
zero, taking the ablations to 10 seeds would mostly re-confirm "cannot beat
zero". A full H3 is better spent on canonical C4 once C3 is unblocked.

## Modules (new in A13)

| Module | Role | Design § |
|---|---|---|
| `src/experiment/c4_select.py` | `ArchiveGuidedSelector` — `Select(A_t, history) → {part_id, intent}`, deterministic/seeded: repair ▸ explore (`diversify`) ▸ exploit (cost-min / mass-min corner) | 5 |
| `src/experiment/c4_feedback.py` | `build_feedback_text` / `build_archive_text` — the EVALUATOR FEEDBACK and ARCHIVE STATE prompt blocks | 3, 10 |
| `src/llm/prompt_builder.py::build_c4_prompt` | wraps `build_proposal_prompt` with the SELECTION / FEEDBACK / ARCHIVE blocks; template tag `A13.C4.v1` | 10, 12 |
| `src/experiment/c4_driver.py` | the loop; `_c4_safe_evaluator` penalty wrap; `_converged` (`hv_plateau` / `archive_unchanged`); `RunOutcome.extra["c4"]` | 3, 6 |
| `src/experiment/metrics.py::_c4_block` | `c4` block recomputed from `agentic_step` events — acceptance, intent counts, retry episodes, HV trajectory, `convergence_reason`; `categorical_subset_hypervolume`; H2/H3 rows | 10, 11 |
| `src/experiment/identity.py` | C4 labels, `c4_loop_spec`, `c4_attempt_cap(N) = max(150, 6N)`, `method` identity block (so `L`/`ε`/`K`/ablation changes → new `run_id`) | 12 |
| `scripts/run_experiment.py` | `C1→C2→C3→C4→C5` ordering; `C4Driver` branch; C4 labels parse case-sensitively; `findings()` C4 block; per-condition report tables iterate `_ordered(by_condition)` so C4 rows appear | 12 |
| `scripts/refresh_metrics.py` | regenerate committed C1/C2/C3/C5 `metrics.json` from source artifacts (HV unchanged) so `categorical_subset_hypervolume` is present for eq 3.53 | — |
| `scripts/aggregate_h2.py` | fold the committed C4-base 10-seed run + ablation pilot into `results/*.csv` + `RUN_NOTES.md` so H2/H3 compute without re-running C1–C5 | — |

## Reused unchanged from A12

`EqualBudgetLedger`, `apply_proposal`, `ParetoArchive`, `events.py`
(`event_from` / `attach_*`), `RunOutcome`, `reference_point`,
`hypervolume_2d`, `GenerativeDriver`'s `_safe_real_evaluator` pattern, the
11.19 directory contract, `RagRetriever`.

## How to run

```bash
# C4-base, 10 seeds (Ollama backend)
PYTHONPATH="$PWD" .venv/bin/python scripts/run_experiment.py --condition C4_base --seeds 0-9 --budget 50

# H3 ablations (3-seed pilot)
PYTHONPATH="$PWD" .venv/bin/python scripts/run_experiment.py \
  --condition C4_base_no_rag,C4_base_no_schema,C4_base_no_validator --seeds 0-2 --budget 50

# fold H2/H3 into results/ against the committed C1/C2/C3/C5 metrics (no re-run)
PYTHONPATH="$PWD" .venv/bin/python scripts/refresh_metrics.py --write
PYTHONPATH="$PWD" .venv/bin/python scripts/aggregate_h2.py --write
```

Canonical `C4` (fine-tuned backend) is wired but **diagnosed non-functional**
(see above), same root cause as C3.

## Scope limitation

Consistent with `docs/A11_C3_TRAINING_STATUS.md` and
`docs/A12_EXPERIMENT_HARNESS_STATUS.md`: this delivers the C4 loop, a 10-seed
C4-base run and a 3-seed ablation pilot on the frozen B4 pilot benchmark, and
computes **H2 as C4-base vs C5** (NOT_SUPPORTED) and H3 at pilot scope.
Canonical C4 is diagnosed non-functional and **not** run to completion; a
full 10-seed H3 and C6 are out of scope. No result here establishes
C4-vs-C5 hypervolume superiority or final H2–H4 acceptance — and the H2
result stands as a valid non-significant finding (11.13), not such an
establishment.
