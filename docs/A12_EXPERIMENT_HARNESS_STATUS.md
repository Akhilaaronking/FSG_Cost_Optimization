# A12 Experiment Harness Status

## Status

The unified experiment harness for conditions **C1, C2, C3, C5** is implemented
and tested (150 A12 tests; full repository suite 304 passing). The real 10-seed
sweep has **not yet been executed** — this document describes the built harness
and the protocol it will run.

Design: `docs/A12_EXPERIMENT_HARNESS_DESIGN.md`.
Deviation record: `EXPERIMENT_DEVIATIONS.txt` (A12 section).

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
