# A12 Experiment Harness Status

## Status

The unified experiment harness for conditions **C1, C2, C3, C5** is implemented
and tested (150 A12 tests; full repository suite 304 passing).

**The full 10-seed sweep was executed on 2026-08-31.** C1, C2 and C5 completed
cleanly. **C3 was found non-functional and is deferred alongside C4** — the A11
LoRA adapter does not produce schema-valid proposals under the shared proposal
prompt (see "Sweep result" below). H1 (C3 vs C2) is therefore not evaluated in
this run; it becomes evaluable, with no harness change, when a working C3
adapter/prompt exists.

Design: `docs/A12_EXPERIMENT_HARNESS_DESIGN.md`.
Deviation record: `EXPERIMENT_DEVIATIONS.txt` (A12 section).
Run artifacts: `runs/<condition>/seed_NN/`, `results/*.csv`, `results/RUN_NOTES.md`.

## Sweep result (2026-08-31)

Total wall-clock ≈ 64 min (matches the ~60 min projection). Baseline
312.02 EUR / 0.6507 kg.

| condition | terminal ×10 | n_eval | HV (mean ± sd) | best cost | lowest mass | funnel |
|---|---|---|---|---|---|---|
| **C1** base LLM | `COMPLETE_SPACE_EXHAUSTED` | 11–15 | 9.18 ± 0.00 | 187.15 | 0.6446 | parse/schema 1.0, id/applic 0.998 |
| **C2** + RAG | `COMPLETE_SPACE_EXHAUSTED` | 13–16 | 8.93 ± 0.12 | 187.15 | 0.6446 | all 1.0 |
| **C3** fine-tuned + RAG | `COMPLETE_SPACE_EXHAUSTED` | **0** | **0.00** | — | — | parse 1.0, **schema 0.0** |
| **C5** NSGA-II | `COMPLETE` | 50 | **15.91 ± 2.92** | 150.99 | 0.5765 | n/a |

Read `results/RUN_NOTES.md` for the interpretive context (candidate-diversity
ceiling; C3 deferred; C1/C2/C5 stand). Observations:

- **Candidate-diversity ceiling.** All 30 generative runs exhausted the atomic
  `material_id`/`process_id` space (~11–16 distinct candidates) well below N=50.
  Only C5 (NSGA-II) reaches the budget. C1/C2/C3 vs C5 hypervolume is a
  search-reach comparison, not like-for-like.
- **RAG (C2 vs C1)** removed C1's single hallucination (seed 1 invented
  `NYLON_PA66`) — funnel 0.998 → 1.0 — but did not move the objective front
  (identical best-cost / lowest-mass endpoints; marginally lower HV; +75 %
  wall time). On this benchmark RAG's benefit is validity, not optimisation.
- **C5** is the only condition with real seed variance and is well ahead on
  hypervolume, on 50 evaluations vs ~13.

### C3 defect (deferred)

Every C3 proposal fails at schema. Under `build_proposal_prompt` the MLX LoRA
adapter emits `{"change_type": "material_id"}`. This is a train/inference
mismatch, not a harness bug:

- `MLXLoRABackend` does free-form generation with no structured-output
  constraint (`OllamaBackend` uses `format=<schema>`).
- The A11 C3 dataset is 1,212 short `system` + terse `user` chat pairs (3 fixed
  system prompts, ~1/3 rejection/validation tasks), not free proposal
  generation from the A10 prompt.

A time-boxed fix (training-style short system + compact user prompt) raised
valid output from 0 to ~2/10 — the adapter still omits the required
`proposal_id` on ~80 % of generations. Not pursued further; not escalated to
retraining. C3 is deferred with C4, exactly as `docs/A11_C3_TRAINING_STATUS.md`
already scopes it.

It does **not** implement C4 (agentic optimisation loop) or C6 (human baseline).
`results/hypothesis_tests.csv` emits H2 / H3 / H4 rows as `PENDING_C4`.

## Modules

| Module | Role | Thesis |
|---|---|---|
| `src/experiment/identity.py` | run identity, `run_config.json`, `RunID = H(B,R,S,M,A,P,Q,s,N,Git)`; refuses to co-file runs whose identity differs | 11.4 |
| `src/experiment/ledger.py` | `EqualBudgetLedger` — budget is *fresh* deterministic objective evaluations; cache hits and pre-eval funnel failures do not consume it | 11.6 |
| `src/experiment/apply_proposal.py` | atomic proposal application vs the frozen baseline `x0`; stage-4 applicability (protected ground-truth fields; decision variables `material_id` / `process_id`) | 11.7 |
| `src/experiment/events.py` | `events.jsonl` writer; the 11.7 validity funnel; `event_from` / `nsga2_event` / `attach_*` / `derive_funnel_stage` | 11.16, 11.7 |
| `src/experiment/probe.py` | C3 environment probe — MLX import + `models/c3_adapter` load + one generation; `ready` / `blocked, environment` | docs/A12 Q2 |
| `src/experiment/drivers.py` | `GenerativeDriver` (C1/C2/C3), `Nsga2Driver` (C5, wraps the ported `_safe_real_evaluator`); `ParetoArchive`; `pareto_archive.json` | 11.7, 11.8 |
| `src/experiment/metrics.py` | `compute_metrics` → `metrics.json`; `seed_summary` / `condition_summary` rollups; `hypothesis_tests` (H1 paired Wilcoxon, effect sizes, `underpowered` flag) | 11.7–11.11, 11.15, 11.19 |
| `scripts/run_experiment.py` | the CLI — sweep orchestration, C1→C2→C3→C5 ordering, data-driven findings, `results/RUN_NOTES.md` | 11.19 |
| `scripts/run_c5_real_pilot.py` | thin shim over `Nsga2Driver` (one C5 code path) | — |

## Protocol

- **Conditions:** C1 (base LLM, no RAG), C2 (base LLM + RAG top-k 5), C3 (MLX LoRA fine-tune + RAG), C5 (NSGA-II).
- **Seeds:** 10 independent, `{0…9}`, shared across all four conditions (11.5). Full thesis protocol, **no reduction** — calibration put the sweep at ~60 min.
- **Budget:** `N = 50` deterministic objective evaluations, uniform across conditions (11.6). C5 re-run at 50; the old `results/c5_pilot/` (N=100) is superseded, retained as history.
- **Parts:** all 10 frozen B4 pilot parts eligible for generative proposals.
- **Order:** C1 → C2 → C3 → C5, never interleaved — the Ollama model, the MLX LoRA model and the MiniLM embedder are never co-resident.
- **Decode seed:** per proposal attempt, `run_seed * 10_000 + attempt_index` (disjoint sub-sequence per run seed; reproducible; paired across C1/C2/C3).

## Directory contract (11.19)

```
runs/<condition>/seed_NN/
    run_config.json      identity (11.4); deviations: []
    events.jsonl         one record per proposal (C1/C2/C3) or evaluation (C5)
    pareto_archive.json  final non-dominated set + HV against the 1.2x reference point
    metrics.json         run-level summary (regenerable from the above)
results/
    seed_summary.csv         one row per condition x seed
    condition_summary.csv    per-condition median / IQR / mean / sd (11.15)
    hypothesis_tests.csv     H1 (C3 vs C2); H2/H3/H4 = PENDING_C4
    RUN_NOTES.md             data-driven interpretive context (see Findings)
    figures/                 (populated in a later pass)
run_index.json               run_id -> path, git, deviations
```

Naming deviations from the thesis text (both permitted by 11.19's "names may
differ"): `manifest.json` → `run_config.json`; `<seed>` → `seed_NN`. Terminal-status
vocabulary adds `COMPLETE_SPACE_EXHAUSTED` to the 11.18 set.

## Statistical analysis (11.15)

- **H1 — fine-tuning benefit (C3 vs C2), 11.12.** Paired one-sided Wilcoxon
  signed-rank (`scipy` exact, zeros dropped) on per-seed hallucination rate and
  constraint-violation rate; Bonferroni `α* = 0.05 / 2 = 0.025`; effect sizes
  `d_z` (eq 11.27) and rank-biserial (eq 11.28). Each row carries
  `min_achievable_p_one_sided = 1/2ⁿ` and an `underpowered` flag computed from the
  *effective* nonzero-pair count.
- **H2 / H3 / H4** need C4 → `PENDING_C4`.

## C3 environment

`probe_c3(deep=True)` **passes on this host**: `mlx` 0.32.2, `mlx-lm` 0.31.3, 4-bit
base model + `models/c3_adapter` load + generation. If the probe fails at run time,
C3 is skipped as `blocked, environment`, a note is recorded, and C1/C2/C5 still run;
a written C3 run is marked `terminal_status = INVALID_CONFIGURATION` (11.18).

Note: the probe only verifies that generation *runs* — it does not check that the
output is a schema-valid proposal. In the 2026-08-31 sweep the probe passed but the
C3 adapter still produced no usable proposals (see "Sweep result"). A future probe
enhancement could generate one real proposal and check it parses + passes schema.

## Calibration (real, 2026-08-31)

N=50 C1 run, seed 0, all 10 parts, Ollama `llama3.1:8b`:

| | |
|---|---|
| wall-clock | 103 s |
| proposal attempts (K) | 37 |
| distinct candidates (`n_eval_consumed`) | 11 |
| per-attempt wall | 2.79 s |
| validity funnel | parse / schema / identifier / applicability / eval all 1.00 |
| hallucination rate | 0.0 |
| terminal_status | `COMPLETE_SPACE_EXHAUSTED` |

Projected full sweep (10 seeds × C1/C2/C3 + C5, serial): **≈ 60 min**.

## Findings carried into reporting

The runner inspects each sweep's own metrics and writes these to
`results/RUN_NOTES.md` (and stdout) — only when the data shows them:

1. **Candidate-diversity ceiling.** The atomic `material_id`/`process_id` space
   over this benchmark is ~11 wide. C1/C2/C3 runs terminate
   `COMPLETE_SPACE_EXHAUSTED` at ~11; only C5 reaches N=50. Read C1/C2/C3 vs C5
   hypervolume as a search-reach difference, not a like-for-like budget comparison.
2. **Zero-hallucination baseline.** The base LLM is already hallucination-free on
   this benchmark, so RAG and fine-tuning cannot reduce that rate; H1's
   hallucination axis is reported as absolute difference ≈ 0 / `NOT_COMPUTABLE`
   (11.12). The measurable C1→C2→C3 signal is candidate selection and the HV /
   validity-funnel trend, reported descriptively.

## How to run

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/run_experiment.py
# or a subset:
PYTHONPATH="$PWD" .venv/bin/python scripts/run_experiment.py --condition C1,C2 --seeds 0-9
```

## Scope limitation

Consistent with `docs/A11_C3_TRAINING_STATUS.md`: the C4 autonomous optimisation
loop and the large-scale final C1–C6 comparative experiment are not implemented
here. Results from this harness support the H1 (C3 vs C2) comparison and the
descriptive C1→C2→C3→C5 trend on the frozen B4 pilot benchmark; they do not by
themselves establish C4-vs-C5 hypervolume superiority or final H2–H4 acceptance.
