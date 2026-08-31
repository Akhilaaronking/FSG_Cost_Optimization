# A12 — Unified Experiment Harness: Design for Review

**Status:** design approved 2026-08-31 (§10). Ready to build (§11). No implementation code written yet.
**Date:** 2026-08-31
**Author:** Aaron
**Sign-off:** schema §3–§5 and scope §1/§9 approved; six open questions resolved in §10.

This document specifies a single harness that runs conditions **C1, C2, C3, C5**
under one run-identity model, one event log, one metrics contract, and one
equal-budget definition, following the thesis methodology in
`Akhileswar Dullam_3122042_Final Draft.docx`, Chapter 11:

| Thesis section | What it governs | Where it lands here |
|---|---|---|
| 11.4 Configuration Freezing and Experimental Identity | `RunID = H(B,R,S,M,A,P,Q,s,N,Git)` | §3 `run_config.json` |
| 11.5 Random Seeds, Replication and Pairing | 10 shared seeds, paired differences | §6, §9 (reduced to 3, documented) |
| 11.6 Equal-Budget Principle | budget = deterministic objective evaluations, not wall-clock | §6 |
| 11.7 Proposal Validity Funnel | parse → schema → identifier → applicability → feasibility → objective → archive | §4 `events.jsonl` |
| 11.16 Experiment Logging and Result Schema | one auditable record per proposal/evaluation | §4 |
| 11.19 Reproducibility Package and Directory Contract | `runs/<condition>/<seed>/…`, `results/*.csv` | §2 |
| 11.15 Statistical Analysis Procedure | paired Wilcoxon, Bonferroni, effect size | §8 |

---

## 1. Scope and non-goals

**In scope (this pass):**

- **C1** — base LLM, no retrieval. *Working today* (`generate_c1`, `OllamaBackend` / `llama3.1:8b`).
- **C2** — base LLM + RAG top-k=5. *Working today* (`generate_c2` + `SentenceTransformerEmbedder`).
- **C3** — fine-tuned LLM (MLX LoRA adapter) + RAG. *Backend and adapter exist* (`MLXLoRABackend`, `models/c3_adapter`, A11); **no runner path** and no equal-budget wiring. A host `mlx_lm` import + adapter-load probe runs before any C3 sweep; on failure C3 is reported `blocked, environment` and C1/C2/C5 still run (§10 Q2).
- **C5** — NSGA-II over the deterministic evaluator. *Working pilot* (`scripts/run_c5_real_pilot.py`); the pilot is **reduced to a thin shim over `Nsga2Driver`** so there is one C5 code path (§10 Q6). Re-run at N=50.

**Explicitly NOT in scope (deferred to a later pass, per user):**

- **C4** — full agentic search loop (cumulative state, tool feedback, evolving archive). Not designed here. The event/metrics schema leaves room for it (`parent_candidate_id`, `archive.*`) but no C4 driver is specified.
- **C6** — human expert baseline. Not designed here.
- **H2 / H3 / H4** hypothesis tests — all require C4. Only the **H1 (C3 vs C2)** comparison is statistically actionable this pass (§8).
- Figure generation (`results/figures/`). Directory is created; plots are a later pass.

**Non-goals:** changing any existing evaluator, generator, prompt, retriever, or NSGA-II math. The harness *orchestrates and logs*; it does not alter scored behaviour. `EVALUATOR_VERSION = "A6.1"` and the frozen B4 benchmark are inputs, not things this pass touches.

---

## 2. Directory contract (thesis 11.19)

11.19 prescribes the layout below; it also states "exact names may differ, but the same information must be present." Two documented naming choices are noted inline.

```
runs/
  C1/
    seed_00/
      run_config.json        # 11.4 run identity  (fills the 11.19 "manifest.json" role — name per Requirement 1)
      events.jsonl           # 11.16 — one record per proposal/evaluation event
      metrics.json           # 11.19 — run-level summary
      pareto_archive.json    # final non-dominated accepted candidates (kept for every condition, see §5)
    seed_01/ …
    seed_02/ …
  C2/ seed_00/ …             # same 4 files
  C3/ seed_00/ …
  C5/
    seed_00/
      run_config.json
      events.jsonl           # one record per NSGA-II objective evaluation
      metrics.json
      pareto_archive.json
results/
  seed_summary.csv           # 11.19 — one row per condition × seed
  condition_summary.csv      # 11.19 — aggregated descriptive statistics (Requirement 3)
  hypothesis_tests.csv       # 11.19 — H1 rows only this pass; H2/H3/H4 rows present but marked PENDING_C4
  figures/                   # created empty this pass
run_index.json               # run_id → path map + the deviations block from §9 (harness-level, not in 11.19; additive)
```

Naming deviations from 11.19 (both permitted by its own "names may differ" clause, both recorded in §9):

1. `run_config.json` instead of `manifest.json` — matches Requirement 1 and the fact that the previous run attempt already referenced `run_config.json` (see `EXPERIMENT_DEVIATIONS.txt`).
2. `seed_00`/`seed_01`/`seed_02` instead of bare `<seed>` — zero-padded for stable lexical sort.

---

## 3. `run_config.json` schema (thesis 11.4 — run identity)

11.4: *"The identity of a run is defined by the combination of the benchmark version, rule-set version, source snapshot, model identifier, adapter or fine-tune identifier, prompt version, retriever setup, seed, evaluation budget, and software commit. … Any change to one element produces a new experimental identity and must not be silently merged into an earlier result set."*

One file per `runs/<condition>/seed_NN/`. Self-contained: `run_id` is recomputable from `identity` alone.

```jsonc
{
  "run_id": "sha256:9f3c1a...e2",        // 16-hex of canonical-JSON(identity); the H(...) of eq 11.1
  "condition": "C2",                      // C1 | C2 | C3 | C5
  "seed": 0,
  "created_utc": "2026-08-31T14:22:07Z",
  "harness_version": "A12.1",

  "identity": {                           // every field below feeds run_id — the 10 elements of eq 11.1

    "benchmark": {                        // B
      "name": "B4_pilot_10_parts",
      "path": "data/benchmark/pilot_10_parts_ground_truth.json",
      "version": "v2",                    // from the file's "version": 2 / "frozen_date"
      "frozen_date": "2026-08-22",
      "sha256": "sha256:80a14a6eb7bb..."  // raw file bytes (not the C5 pilot's sha256_json of the parsed object)
    },

    "ruleset_snapshot": {                 // R
      "deterministic_constraints_sha256": "…",   // data/processed/deterministic_constraints_B5.json
      "rule_classification_sha256": "…",         // data/processed/rule_classification_B5.json
      "evaluator_version": "A6.1",               // src/evaluator EVALUATOR_VERSION
      "routed_rule_counts": { "deterministic": 17, "review": 9, "quality_gate": 1 }
    },

    "source_snapshot": {                  // S
      "registry_files": {
        "materials.csv":  "sha256:…",
        "processes.csv":  "sha256:…",
        "fasteners.csv":  "sha256:…",
        "suppliers.csv":  "sha256:…"
      },
      "search_space": {
        "path": "data/benchmark/real_search_space.json",
        "schema_version": "OptionA_v1_verified",
        "sha256": "…"
      }
    },

    "model": {                           // M
      "role": "base",                    // base | fine_tuned | none
      "backend_name": "ollama",          // ollama | mlx_lora | none
      "model_id": "llama3.1:8b",
      "model_digest": "46e0c10c039e...", // ollama digest  (C3: mlx repo + revision)  (C5: null)
      "decode": { "temperature": 0.2, "max_tokens": 512, "seed_supported": true }
    },

    "adapter": {                         // A
      "adapter_id": null                 // C1/C2/C5: null
      // C3:
      //   "adapter_id": "models/c3_adapter",
      //   "adapter_sha256": "8195bde354264b95...",   // A11 canonical adapter
      //   "base_model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
      //   "lora": { "rank": 16, "scale": 20.0, "dropout": 0.05, "layers": 16, "iters": 300, "checkpoint": 300 },
      //   "training_data_sha256": "e7870a5729ca1dd6..."
    },

    "prompt": {                         // P
      "prompt_version": "A10.1",
      "prompt_template_sha256": "…",     // sha256(PROMPT_TEMPLATE_STRUCTURE) — already emitted as prompt_template_hash
      "builder_module": "src.llm.prompt_builder"
      // C5: null
    },

    "retrieval": {                      // Q
      "rag_enabled": true,              // C1: false ; C5: null
      "embedder": "sentence-transformers/all-MiniLM-L6-v2",
      "vector_backend": "numpy_cosine",
      "top_k": 5,
      "corpus_path": "data/rag/corpus.jsonl",
      "corpus_version": "A9.1",         // data/rag/corpus_manifest.json
      "corpus_sha256": "…"
    },

    "budget": {                        // N — locked at 50 for all four conditions (§10 Q1)
      "definition": "deterministic_objective_evaluations",   // NOT wall-clock (11.6)
      "n_eval": 50,
      "duplicate_policy": "cache_by_canonical_bom_hash; a cache hit does NOT consume budget",
      "proposal_attempt_cap": 1500      // generative conditions only; null for C5
    },

    "git": {                          // Git
      "commit": "a02aa5237ae8767add2bea157f8c35891b3628d3",
      "tracked_dirty": false,          // uncommitted tracked changes present?
      "untracked_files": 1             // informational; does not change run_id
    }
  },

  "condition_spec": {                  // NOT part of run_id — describes how the driver is wired
    "driver": "GenerativeDriver",      // GenerativeDriver (C1/C2/C3) | Nsga2Driver (C5)
    "generator_fn": "src.llm.conditions.generate_c2",
    "decision_variables": ["material_id", "process_id"],
    "target_parts": ["PILOT_001", "...", "PILOT_010"],   // generative: parts eligible for proposals
    "proposal_application": "atomic_vs_frozen_baseline",  // each accepted proposal applied to x0 independently (§6, Q3)
    "nsga2": null
    // C5: { "population_size": 20, "mutation_rate": 0.35, "generations": <=n_eval,
    //       "reference_point_rule": "1.2x baseline (eq 11.10)" }
  },

  "deviations": [                      // §9 — copied verbatim into every run_config.json
    {
      "id": "SEED_COUNT",
      "thesis_ref": "11.5",
      "thesis_value": "10 independent seeds, shared across paired conditions",
      "actual_value": "3 seeds {0,1,2}, shared across C1/C2/C3/C5",
      "reason": "time constraint; C3 MLX LoRA inference cost on M4/16GB",
      "consequence": "paired Wilcoxon underpowered — min two-sided p = 0.25 at n=3 (see §8)",
      "pattern": "EXPERIMENT_DEVIATIONS.txt / ENV_DEVIATIONS.txt",
      "approved_by": null
    }
  ]
}
```

### Per-condition identity summary

| element | C1 | C2 | C3 | C5 |
|---|---|---|---|---|
| model.backend | ollama | ollama | mlx_lora | none |
| model.model_id | llama3.1:8b | llama3.1:8b | Meta-Llama-3.1-8B-Instruct-4bit | — |
| adapter.adapter_id | null | null | `models/c3_adapter` (+ sha) | null |
| prompt | A10.1 | A10.1 | A10.1 | null |
| retrieval.rag_enabled | false | true | true | null |
| condition_spec.driver | GenerativeDriver | GenerativeDriver | GenerativeDriver | Nsga2Driver |
| benchmark / ruleset / source / budget / git | identical across all four |

If C3 is run at a different `top_k`, prompt version, or adapter checkpoint than C2, that is a *different Q/P/A* and therefore a different `run_id` — it must not be filed alongside a C2 run as if paired. The harness enforces this by refusing to write into an existing `seed_NN/` whose `run_config.json` has a mismatching `run_id`.

---

## 4. `events.jsonl` schema (thesis 11.16 + 11.7 funnel + 11.9 / 11.10)

One JSON object per line. **One record per event that is either (a) a generation attempt or (b) an objective-evaluation attempt.** For C1/C2/C3 that is one record per proposal; for C5 one record per NSGA-II candidate handed to the evaluator.

11.16 minimum fields (all present below): run ID, condition, seed, proposal index, parent candidate, raw model output hash, parsed proposal, retrieved context IDs, schema status, hallucination flags, constraint results, cost, mass, archive status, runtime, software versions.

```jsonc
{
  "run_id": "sha256:9f3c1a...e2",
  "condition": "C2",
  "seed": 0,
  "event_index": 12,                     // 0-based, sequential within the run
  "event_type": "proposal",             // "proposal" (C1/C2/C3) | "nsga2_evaluation" (C5)
  "ts_utc": "2026-08-31T14:23:19Z",

  "generation": {                       // populated for generative conditions
    "target_part_id": "PILOT_002",
    "parent_candidate_id": null,        // always null for C1/C2/C3 (atomic vs baseline); reserved for C4/C5 lineage
    "raw_output_sha256": "…",           // hash of the raw model string (raw string itself NOT stored here)
    "parsed_proposal": { "proposal_id": "...", "part_id": "PILOT_002",
                         "change_type": "material", "target_field": "material_id",
                         "old_value": "AL_7075_T6", "new_value": "AL_6061_T6" },   // null if unparseable
    "prompt_hash": "…",                 // sha256(final prompt) — from generator
    "modifications": [                  // canonical (part_id, field, baseline, candidate) list actually applied
      { "part_id": "PILOT_002", "field": "material_id", "baseline": "AL_7075_T6", "candidate": "AL_6061_T6" }
    ]
  },

  "retrieval": {                        // C2/C3 only; { "rag_enabled": false } for C1; omitted for C5
    "rag_enabled": true,
    "query_text": "…",
    "retrieved_chunk_ids": ["chunk_0031", "…"],
    "retrieved_source_ids": ["SR_014", "…"],
    "similarity_scores": [0.71, 0.66, 0.51, 0.44, 0.40]
  },

  "validity": {                         // 11.7 stages 1–4
    "parse_valid": true,                // stage 1
    "schema_valid": true,               // stage 2
    "authority_valid": true,            // stage 3 — closed-world identifier validity
    "applicability_valid": true,        // stage 4 — allowed field of an allowed part, no protected-field mutation
    "unknown_identifiers": [],
    "protected_field_writes": [],
    "funnel_stage_reached": "objective_evaluation"
    // one of: parse | schema | identifier | applicability | feasibility | objective_evaluation | archive
  },

  "hallucination": {                    // 11.9
    "hallucinated": false,
    "categories": []                    // UNKNOWN_MATERIAL_ID | UNKNOWN_PROCESS_ID | UNKNOWN_FASTENER_ID
                                        // | UNSUPPORTED_FACTUAL_ATTRIBUTE | UNSUPPORTED_RULE_ASSERTION
                                        // | FABRICATED_NUMERIC_JUSTIFICATION   (11.9 taxonomy)
  },

  "evaluation": {                       // present iff the funnel reached feasibility/objective stage OR event_type == nsga2_evaluation
    "consumed_objective_budget": true,  // true iff a FRESH evaluate_bom() call was made (not a cache hit)
    "objective_eval_cache_hit": false,
    "bom_hash": "…",                    // canonical hash of the applied candidate BOM
    "objectives": { "cost_eur": 305.11, "mass_kg": 0.641200 },
    "objective_vector": [305.11, 0.641200],
    "baseline_delta": { "cost_eur": -6.91, "mass_kg": -0.009511,
                        "cost_improvement_pct": 2.21, "mass_improvement_pct": 1.46 },
    "constraints": {                    // 11.10 — mirrors _safe_real_evaluator / unified_evaluator output
      "status": "ENGINEERING_ADMISSIBLE_EVALUATED",
      // ENGINEERING_ADMISSIBLE_EVALUATED | NOT_EVALUATED | SEARCH_SPACE_REJECTED | DETERMINISTIC_EVALUATION_FAILED
      "evaluated": true,
      "feasible": true,
      "violation_count": 0,
      "proposal_level_violation": false,
      "rule_level_violations": 0,
      "rule_level_checks": 17,
      "missing_essential_fields": []
    }
  },

  "archive": {                          // 11.7 stage 8
    "status": "pareto_improving",       // dominated | non_dominated | duplicate | pareto_improving
    "archive_size_after": 6
  },

  "efficiency": {                       // 11.11
    "gen_runtime_sec": 5.19,            // model call; 0 for C5
    "eval_runtime_sec": 0.011,
    "token_counts": { "prompt": null, "completion": null }   // wire from ollama /api/generate eval_count if present
  },

  "software": { "harness_version": "A12.1", "evaluator_version": "A6.1",
                "git_commit": "a02aa523", "git_tracked_dirty": false }
}
```

**C5 rows** (`event_type: "nsga2_evaluation"`): `generation.modifications`, `generation.parent_candidate_id`, `evaluation.*`, `archive.*`, `efficiency.eval_runtime_sec` are populated. `retrieval` omitted; `validity`/`hallucination` set to `null` (no proposal funnel for an evolutionary candidate — it is generated by operators, not parsed from text). This asymmetry is intentional and is exactly the 11.7-vs-11.8 split in the thesis.

**Budget invariant:** for a run with `terminal_status == COMPLETE`, `sum(1 for e in events if e.evaluation.consumed_objective_budget) == identity.budget.n_eval`.

---

## 5. `metrics.json` (per run) and the CSV rollups (Requirement 3)

`metrics.json` is a pure function of that run's `events.jsonl` + `run_config.json` — regenerable, never hand-edited (11.16.1: *"Tables and figures in Chapter 12 can be regenerated directly from result files."*).

```jsonc
{
  "run_id": "sha256:9f3c1a...e2",
  "condition": "C2", "seed": 0,
  "terminal_status": "COMPLETE",        // 11.18 vocabulary — see §6

  "budget": { "n_eval_target": 50, "n_eval_consumed": 50,
              "proposal_attempts": 74, "objective_eval_cache_hits": 5 },

  "validity_funnel": {                  // 11.7  VR_k = N_k / N_prop
    "n_prop": 74,
    "counts":  { "parse": 74, "schema": 72, "identifier": 68, "applicability": 66, "objective_evaluated": 50 },
    "rates":   { "parse": 1.000, "schema": 0.973, "identifier": 0.919, "applicability": 0.892 }
  },

  "hallucination": {                    // 11.9  HR = N_hallucinated / N_proposals  (primary = all proposals)
    "hr_all_proposals": 0.081,
    "hr_schema_valid_only": 0.056,      // secondary rate (11.9)
    "categories": { "UNKNOWN_MATERIAL_ID": 4, "UNKNOWN_PROCESS_ID": 2 }
  },

  "constraints": {                     // 11.10  CVR = N_violation / N_deterministically_checked
    "cvr_proposal_level": 0.000,
    "cvr_rule_level": 0.000,
    "n_deterministically_checked": 50,
    "missing_essential_fields_count": 0
  },

  "objectives": {
    "baseline": { "cost_eur": 312.02, "mass_kg": 0.6507108 },
    "best_cost_eur": 288.40, "mass_of_best_cost_kg": 0.7100,
    "lowest_mass_kg": 0.5800, "cost_of_lowest_mass_eur": 309.00,
    "n_pareto_improving": 3
  },

  "multiobjective": {                  // 11.8
    "reference_point": [374.424, 0.78085296],   // 1.2x baseline  (eq 11.10)
    "hypervolume": 12.41,
    "normalized_hypervolume": 0.0425,
    "delta_hv": 12.41,                 // eq 11.12  (HV_baseline = 0 for an empty start archive)
    "pareto_archive_size": 4,
    "ideal_point": [288.40, 0.5800],   // eq 11.13
    "min_ideal_point_distance": 0.108  // eq 11.14, normalised
  },

  "efficiency": {                     // 11.11
    "wall_clock_sec": 402.7,
    "t100_sec_per_100_proposals": 544.2,   // eq 11.17 = 100 * T_wall / N_prop
    "eta_hv_per_sec": 0.0308,               // eq 11.18 = delta_HV / T_wall
    "total_tokens": null
  },

  "software": { "harness_version": "A12.1", "evaluator_version": "A6.1",
                "git_commit": "a02aa523", "git_tracked_dirty": false }
}
```

Notes:
- `pareto_archive.json` is kept for **every** condition (not just C5) so cross-condition HV in `condition_summary.csv` is computed the same way everywhere — same reference-point rule, same `hypervolume_2d`.
- For **C1** the archive is simply whichever accepted proposals were non-dominated; HV is still meaningful and comparable.
- `delta_hv` uses `HV_baseline = 0` (empty archive at start) for generative conditions; for C5 it equals the pilot's reported HV. Both are the Lebesgue measure under the same `r` — comparable.

### `results/seed_summary.csv` — one row per condition × seed (11.19)

```
condition,seed,run_id,terminal_status,n_eval_target,n_eval_consumed,n_prop,
parse_rate,schema_rate,identifier_rate,applicability_rate,
hr_all,hr_schema_valid,cvr_proposal,cvr_rule,
hypervolume,normalized_hv,delta_hv,pareto_size,
best_cost_eur,lowest_mass_kg,min_ideal_distance,
wall_clock_sec,t100_sec,eta_hv,total_tokens
```

### `results/condition_summary.csv` — aggregate over seeds (11.19, Requirement 3)

Descriptive statistics per 11.15 (*"median, interquartile range, mean, and standard deviation … when appropriate"*):

```
condition,n_seeds,n_complete,
hv_median,hv_iqr,hv_mean,hv_sd,
norm_hv_median,norm_hv_iqr,
hr_all_median,hr_all_iqr,hr_all_mean,hr_all_sd,
cvr_proposal_median,cvr_proposal_iqr,
parse_rate_mean,schema_rate_mean,identifier_rate_mean,applicability_rate_mean,
best_cost_eur_median,lowest_mass_kg_median,
wall_clock_sec_median,t100_sec_median
```

### `results/hypothesis_tests.csv` (11.19)

```
hypothesis,comparison,metric,c_ref,c_test,relative_reduction_pct,W_statistic,p_value_exact,p_one_sided,alpha_corrected,effect_size_dz,effect_size_rrb,decision,notes
H1,C3_vs_C2,hallucination_rate,...,...,...,...,...,...,0.025,...,...,...,
H1,C3_vs_C2,constraint_violation_rate,...,...,...,...,...,...,0.025,...,...,...,
H2,C4_vs_C5,final_hypervolume,,,,,,,,,,PENDING_C4,"C4 not in A12 scope"
H3,C4_vs_ablations,...,,,,,,,,,,PENDING_C4,
H4,transfer,...,,,,,,,,,,PENDING_C4,
```

---

## 6. Equal-budget definition (thesis 11.5 pairing + 11.6)

11.6: *"the main budget is the number of candidate evaluations done by the fixed deterministic evaluator … If a candidate fails before deterministic evaluation … it still counts in the proposal-generation statistics [but] does not use an objective-evaluator call."*

### The shared ledger

A single `EqualBudgetLedger` object is used by both drivers:

| counter | incremented when | caps the run? |
|---|---|---|
| `objective_evaluations` | a **fresh** `evaluate_bom()` call completes (cache miss) | **yes — run ends at `n_eval`** |
| `proposal_attempts` | any generation attempt (C1/C2/C3), valid or not | soft cap `proposal_attempt_cap` → terminal status |
| `objective_eval_cache_hits` | candidate BOM hash already evaluated this run | no |

`ledger.consume(candidate_bom) -> (result, was_cache_hit)`:
hashes the canonical BOM; on hit returns the cached objective result and increments `cache_hits`; on miss calls `evaluate_bom(..., evaluate_constraints=False)` exactly as the C5 pilot does, caches, and increments `objective_evaluations`.

This makes the budget **identical in kind** across C1/C2/C3/C5: *N distinct deterministic objective evaluations of the frozen evaluator*. Wall-clock, tokens, and proposal attempts are recorded (§4, §5) but never gate the run.

### Generative driver loop (C1 / C2 / C3)

```
while ledger.objective_evaluations < N:
    if ledger.proposal_attempts >= attempt_cap:          -> terminal = ABORTED_BUDGET_UNREACHED ; break
    part = next_target_part(seed)                        # deterministic round-robin seeded by `seed`
    rec  = generate_cX(generator, bom=x0, target_part=part, seed=seed, retriever=…)   # EXISTING fn, unchanged
    ledger.proposal_attempts += 1
    ev = event_from(rec)                                 # parse/schema/authority/hallucination already in rec
    if rec.parse_valid and rec.schema_valid and rec.authority_valid:
        cand = apply_proposal(x0, rec.proposal)          # NEW helper (design only) — clone x0, set one field
        if cand.applicability_ok:                        # allowed field, allowed part, no protected write
            obj, hit = ledger.consume(cand)
            ev.evaluation = obj
            ev.archive    = archive.offer(cand, obj.objective_vector)
    events.write(ev)
terminal = terminal or COMPLETE
```

Key properties:
- **Atomic proposals vs the frozen baseline** (`x0`). Each accepted proposal is applied to `x0` independently — not stacked. This keeps C1/C2/C3 as *proposal-quality* conditions (their thesis role: 11.3 *"C1 to C3 focus mainly on the quality of proposals"*). Cumulative search is C4's job. **See Q3 for sign-off.**
- The **seed** governs only: (a) backend decode seed (already plumbed through `generate_cX`), (b) target-part iteration order. Same 3 seeds across C1/C2/C3 ⇒ paired comparisons (11.5).
- **Distinct-space exhaustion:** with atomic proposals over ~2–3 materials × 1–2 processes per part, the distinct candidate space across 10 parts is on the order of tens. If the ledger runs out of *new* candidates before reaching `N`, the run ends `COMPLETE_SPACE_EXHAUSTED` (new status, §"terminal statuses") at `n_eval_consumed < N`, reported honestly. The equal-budget *ceiling* is still identical; that a generative condition bottoms out earlier than C5 is itself a result (smaller effective reach per proposal).

### NSGA-II driver (C5)

Reuse `run_c5_real_pilot.py`'s math. Wrap `_safe_real_evaluator` in an instrumenting decorator that, per call, routes through the same `EqualBudgetLedger` and writes one `nsga2_evaluation` event. Call `nsga2_optimize(..., evaluation_budget=N, reference_point=1.2*baseline)`. Emit `pareto_archive.json` from `result["pareto_archive"]`; emit `metrics.json` from the same numbers the pilot already prints. Net change to C5 behaviour: **none** — only output routing.

### Terminal run statuses (thesis 11.18)

| status | meaning | inferential use |
|---|---|---|
| `COMPLETE` | `n_eval_consumed == n_eval` | full |
| `COMPLETE_SPACE_EXHAUSTED` | generative: distinct admissible candidate space exhausted before `n_eval` | full, with `n_eval_consumed` noted (additive to 11.18 vocabulary; documented) |
| `ABORTED_BUDGET_UNREACHED` | `proposal_attempt_cap` hit before `n_eval` (e.g. C1 funnel losses too high) | excluded until rerun; failed log preserved |
| `ABORTED_IMPLEMENTATION` | harness/backend bug | excluded until fixed + rerun same seed |
| `ABORTED_PROVIDER` | Ollama/MLX unavailable mid-run | rerun same seed |
| `INVALID_CONFIGURATION` | `run_id` mismatch on resume, or frozen-input hash drift | invalidate before inspection |
| `CORRUPT_LOG` | `events.jsonl` unparseable / invariant broken | exclude unless reconstructable |

Per 11.18: a required rerun **reuses the same seed and frozen config** — a new seed is never substituted to get a nicer number.

---

## 7. The runner

Single entry point, `scripts/run_experiment.py` (to be built next pass):

```
python scripts/run_experiment.py \
    --condition C2 \
    --seeds 0,1,2 \
    --budget 50 \
    --parts PILOT_001..PILOT_010 \
    --out runs/ \
    [--dry-run]        # writes run_config.json only, computes run_id, exits
```

Dispatch:

```
build_identity(condition, budget)          # hashes frozen inputs + git → identity block
for seed in seeds:
    cfg = build_run_config(condition, seed, identity)      # → runs/<c>/seed_NN/run_config.json  (11.4)
    guard: refuse if seed_NN/run_config.json exists with a different run_id   (→ INVALID_CONFIGURATION)
    log = EventLog(runs/<c>/seed_NN/events.jsonl)
    driver = GenerativeDriver(condition, cfg) if condition in {C1,C2,C3} else Nsga2Driver(cfg)
    status = driver.run(budget=cfg.identity.budget.n_eval, ledger=EqualBudgetLedger(), events=log)
    metrics = compute_metrics(events_path, cfg)            # → metrics.json  (§5)
    append_row(results/seed_summary.csv, metrics)
rollup(results/seed_summary.csv) → results/condition_summary.csv
```

### Build vs adapt

| Component | Today | This design needs |
|---|---|---|
| `generate_c1` / `generate_c2` / `generate_c3` | exist (`src/llm/conditions.py`) | **called as-is**, no change |
| `OllamaBackend` (C1/C2) | working, verified | none |
| `MLXLoRABackend` + `models/c3_adapter` | exist; A11 "direct backend inference test" passed | **wire into runner** behind the host probe below |
| `mlx_lm` host probe | **missing** | new: import `mlx_lm` + load `models/c3_adapter` once at C3 startup; on failure emit `terminal_status = INVALID_CONFIGURATION` / condition report `blocked, environment`, continue C1/C2/C5 (§10 Q2) |
| `nsga2_optimize` + `_safe_real_evaluator` | working (C5 pilot) | **wrap** evaluator for event logging; redirect outputs into `runs/C5/seed_NN/` |
| `apply_proposal(bom, proposal) -> bom` | **missing** | new pure helper: `change_type`→field map, deep-clone `x0`, set one value; flag protected-field writes = applicability check (11.7 stage 4) |
| `EqualBudgetLedger` | **missing** | new (~40 lines) |
| `EventLog` writer / `build_run_config` / `compute_metrics` | **missing** | new — the bulk of the work |
| identity hashing | partial (`sha256_json` in `prompt_builder`) | extend: file SHAs, git commit/dirty, canonical run_id |
| `seed_summary` / `condition_summary` rollup | **missing** | new (stdlib `csv` + `statistics`) |
| `run_c5_real_pilot.py` | standalone | **reduce to a shim** calling `Nsga2Driver` — one C5 code path (§10 Q6) |

No existing test, generator, prompt, evaluator, or optimiser file is modified.

---

## 8. Statistical analysis this pass can support (thesis 11.15)

11.15 mandates: Wilcoxon signed-rank on paired seed differences (zero-diffs dropped, exact small-sample p where the library allows), Bonferroni `α* = α/m` within a hypothesis family, effect sizes `d_z = mean(d)/sd(d)` (eq 11.27) and rank-biserial `r_rb` (eq 11.28), always reported next to absolute engineering deltas.

### What is computable with {C1, C2, C3, C5} present

**H1 — Fine-Tuning Benefit (11.12): C3 vs C2.** *In scope.*
- Paired differences per seed for `hallucination_rate` and `constraint_violation_rate`.
- `R_H = 100·(HR_C2 − HR_C3)/HR_C2` (eq 11.19); `R_V = 100·(CVR_C2 − CVR_C3)/CVR_C2` (eq 11.20). If `HR_C2 == 0`, report the absolute difference (11.12).
- Pre-registered thresholds: ≥ 30 % relative HR reduction **and** ≥ 20 % relative CVR reduction.
- Family size `m = 2` → `α* = 0.025`.
- Wilcoxon signed-rank, one-sided (direction pre-specified), plus `d_z` and `r_rb`.

**Descriptive only (no hypothesis claim):**
- C1 → C2 → C3 validity-funnel progression and HR/CVR trend (median/IQR/mean/sd).
- C5 hypervolume distribution; C2/C3 vs C5 hypervolume shown side-by-side as *context*, not as H2 (H2 requires C4).

### Out of scope until C4 exists

H2 (C4 vs C5), H3 (C4 ablations), H4 (transfer). Rows are pre-created in `hypothesis_tests.csv` with `decision = PENDING_C4`.

### The seed-reduction consequence — stated plainly

With **n = 3** nonzero paired differences, the Wilcoxon signed-rank null distribution has 2³ = 8 equally likely sign patterns. The most extreme outcome (all differences same sign) gives:
- two-sided exact p = 2/8 = **0.25**
- one-sided exact p = 1/8 = **0.125**

Both exceed α = .05 and the Bonferroni α* = .025. **It is therefore impossible for this pass to produce a statistically significant H1 result**, regardless of how large the effect is. This pass yields:
- the **effect direction and magnitude** (`R_H`, `R_V`, `d_z`, `r_rb`),
- the **absolute engineering deltas** (percentage-point HR/CVR change, EUR, grams, HV),
- the Wilcoxon statistic and exact p **reported but pre-declared underpowered**.

Full inferential H1 requires re-running C2 and C3 at the thesis's 10 seeds. This is recorded in `run_config.deviations[]`, in `EXPERIMENT_DEVIATIONS.txt`, and must be carried into any Chapter 12 wording ("descriptive evidence consistent with H1; not a significance test").

---

## 9. Seed-count reduction proposal (Requirement 5)

Same pattern as `ENV_DEVIATIONS.txt` / `EXPERIMENT_DEVIATIONS.txt`: **reduced, documented, not silently substituted.**

### Proposal

| Parameter | Thesis (11.5 / 11.6) | Locked for A12 (§10) | Rationale |
|---|---|---|---|
| Seeds | 10 independent, shared across paired conditions | **3 seeds: {0, 1, 2}**, shared across C1/C2/C3/C5 | Pairing preserved (11.5). C3 = MLX LoRA 8B-4bit inference on M4/16 GB; a 3-seed × 4-condition sweep is a working-session cost, a 10-seed sweep is a multi-day cost. |
| Budget `N` | Not fixed by the thesis (the C5 pilot's own default is 100) | **N = 50** deterministic objective evaluations, all four conditions; **C5 re-run at 50** | One budget number for all four. C5 re-run costs ~0.1 s/seed, so aligning C5 down is free; the existing `results/c5_pilot/` (N=100, 10 seeds) is superseded for A12 purposes. |

### How it is recorded (not silent)

1. `run_config.json → deviations[]` — verbatim block, in **every** run's config (schema shown in §3).
2. `EXPERIMENT_DEVIATIONS.txt` — appended entry in the existing CONTEXT / DECISION / WHAT WAS RUN / RESULT / REPORTING BOUNDARY format.
3. `metrics.json → terminal_status` and `condition_summary.csv → n_seeds` — the reduced count is visible in every rollup.
4. §8 consequence text travels with the H1 result wherever it is quoted.

### Full deviations table for this harness

| Thesis spec | A12 harness | Reason | Recorded in |
|---|---|---|---|
| 11.5 — 10 seeds | 3 seeds {0,1,2} | time; C3 MLX inference cost | `run_config.deviations[]`, `EXPERIMENT_DEVIATIONS.txt` |
| 11.5 — inferential Wilcoxon at α=.05 | descriptive + effect size; test reported, pre-declared underpowered (min p = 0.25) | consequence of the seed cut | §8, `EXPERIMENT_DEVIATIONS.txt` |
| 11.19 — `manifest.json` | `run_config.json` (identical field set) | Requirement 1; 11.19 permits renaming | §2 |
| 11.19 — `runs/<condition>/<seed>/` | `runs/<condition>/seed_NN/` | zero-pad for sort order | §2 |
| 11.18 — fixed status vocabulary | + `COMPLETE_SPACE_EXHAUSTED` | atomic generative proposals can exhaust a small distinct space | §6 |
| `N` unfixed in thesis | `N = 50`, uniform across conditions; C5 re-run to match; `results/c5_pilot/` (N=100 ×10) superseded for A12 | one budget number for all four | `run_config.budget`, `EXPERIMENT_DEVIATIONS.txt` |
| `run_c1_c2_pilot.py` — 3 hard-coded parts, seeds {0,1} | all 10 pilot parts eligible, seeds {0,1,2} | broader funnel statistics; that script was a pipeline check | §10 Q4 |
| C1–C6 full matrix | C1, C2, C3, C5 only | staged delivery; C4/C6 next pass | this doc, §1 |

---

## 10. Resolved decisions (2026-08-31)

| # | Question | Decision |
|---|---|---|
| 1 | Budget `N` | **N = 50 for all four conditions; C5 re-run at 50.** Existing `results/c5_pilot/` (N=100 ×10 seeds) is superseded for A12. |
| 2 | C3 host readiness | **Add an `mlx_lm` import + `models/c3_adapter` load probe** at C3 startup. On failure: C3 reported `blocked, environment` (like the old A10 note), C1/C2/C5 still run. |
| 3 | Atomic vs cumulative proposals | **Atomic.** Each accepted proposal applied to the frozen baseline `x0` independently, no stacking. C1–C3 stay proposal-quality conditions (11.3); cumulative search is C4. |
| 4 | Part scope (generative) | **All 10 pilot parts** eligible for proposals. Supersedes the current 3-part hard-coded subset in `run_c1_c2_pilot.py`. |
| 5 | `pareto_archive.json` for C1/C2/C3 | **Keep for every condition** so cross-condition HV uses one code path and one reference-point rule. |
| 6 | `run_c5_real_pilot.py` | **Reduce to a thin shim over `Nsga2Driver`.** One C5 code path. |

---

## 11. Build order

1. `src/experiment/identity.py` — file SHAs, git info, `build_run_config`, `run_id`.
2. `src/experiment/ledger.py` — `EqualBudgetLedger`.
3. `src/experiment/apply_proposal.py` — atomic application vs frozen `x0` + applicability check (11.7 stage 4).
4. `src/experiment/events.py` — `EventLog`, `event_from(rec)`.
5. `src/experiment/probe.py` — `mlx_lm` import + `models/c3_adapter` load check; returns a pass/blocked verdict the runner acts on for C3.
6. `src/experiment/drivers.py` — `GenerativeDriver` (C1/C2/C3, all 10 parts, atomic), `Nsga2Driver` (C5).
7. `src/experiment/metrics.py` — `compute_metrics`, `seed_summary` / `condition_summary` rollups.
8. `scripts/run_experiment.py` — CLI.
9. `scripts/run_c5_real_pilot.py` — reduce to a shim over `Nsga2Driver` (keep the CLI flags, drop the duplicated math).
10. Tests: budget invariant (`Σ consumed_objective_budget == n_eval`), funnel counting, `run_id` stability, `COMPLETE_SPACE_EXHAUSTED` path, C5 parity vs the pre-shim pilot on a fixed seed.
11. `EXPERIMENT_DEVIATIONS.txt` append (seed count, N=50, C5 supersession, underpowered-H1 note) + a short `docs/A12_EXPERIMENT_HARNESS_STATUS.md` with the real runs.

*Scope and schema are signed off (§10). Build proceeds on your go.*
