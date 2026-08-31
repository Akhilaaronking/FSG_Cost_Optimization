# A13 — C4 Agentic Tool-Loop: Design for Review

**Status:** design approved 2026-08-31 (§13). Building per §14.
**Date:** 2026-08-31
**Author:** Aaron
**Depends on:** A12 harness (`docs/A12_EXPERIMENT_HARNESS_DESIGN.md`) — reuses its
ledger, events, `apply_proposal`, archive and metrics rather than rebuilding them.

C4 is the **full proposed framework** in thesis Table 4.5 (§4.15): fine-tuned
LLM **+** RAG **+** a deterministic tool-loop that iterates on evaluator feedback,
rather than the one-shot-per-attempt generation of C1/C2/C3. It is the condition
that H2 (C4 vs C5) and H3 (C4 ablations) are built on.

| Thesis element | What it fixes | Where it lands here |
|---|---|---|
| §4.15 / Table 4.5 | C4 = FT-LLM + RAG + deterministic iteration ("Full proposed framework; H2/H3") | §1, §4 |
| eq 10.1–10.3 | LLM: context→proposal; Tools: proposal→verified state; Archive: verified state→search memory | §3 |
| eq 10.26–10.29 | the four tools: `T_mass`, `T_cost`, `T_constraint`, `T_supplier` — already the unified evaluator `E(x)` | §3 |
| **eq 10.30–10.35** | `Select → Retrieve → LLM → Apply → Evaluate → Archive` — the loop body | §3 |
| §4.9 eq 4.2 / eq 9.34–9.35 | `ΔHV_recent(t) = HV_t − HV_{t−L}`; stop at budget `N` or convergence `< ε` | §6 |
| eq 11.4 | `B_eval(C4) = B_eval(C5) = N` — equal budget, deterministic objective evaluations | §7 |
| eq 11.21 / §11.13 | `dᵢ^HV = HVᵢ,C4 − HVᵢ,C5` — H2 paired difference | §11 |
| §11.14 / eq 10.38–10.39 | C4−RAG / C4−Schema / C4−Validator; `HR_full < min(...)`, `HR_full < 0.05` | §10, §11 |

---

## 1. Scope and non-goals

**In scope (design):** the C4 agentic loop as a new driver in the A12 harness —
loop structure, selection policy, feedback assembly, per-state stopping,
run-level stopping, `run_config` / `events.jsonl` / `metrics.json` additions, the
H3 ablation variants, and how H2/H3 slot into the existing `hypothesis_tests`.

**Not in scope:** C6 (human baseline); any change to C1/C2/C3/C5; implementation
code; the C3 constrained-decoding port (`docs/A12_C3_CONSTRAINED_DECODING_PLAN.md`).

**Explicit non-goal:** re-deriving the evaluator or the archive. `E(x)` (eqs
10.26–10.29) is `src.evaluator.evaluate_bom`; the non-dominated archive
(eq 10.35 / 3.32) is A12's `ParetoArchive`. C4 orchestrates and logs.

---

## 2. The one-line difference

C1/C2/C3 answer *"how good is a single proposal against the frozen baseline?"*.
**C4 answers *"how far can an LLM-driven search get in N evaluations, with the
evaluator in the loop?"*** — the same question C5 (NSGA-II) answers, which is
why H2 is C4 vs C5 and why both are held to the same `N` (eq 11.4).

---

## 3. The loop (thesis eq 10.30–10.35)

Per seed, one continuous search. `x_0` = frozen baseline BOM; `A_0 = {x_0}`
(archive seeded with the baseline, per §4.9).

```
t = 0
x_t = x0
A_t = ParetoArchive(seeded with x0)
history = []                               # (selection, proposal, y, accepted, reason) per step
retries = {}                               # per-selection retry counter
while not stop(t, A_t, ledger):
    s_t   = Select(A_t, history)                       # eq 10.30  -- which part / what intent
    r_t   = Retrieve(s_t, k=5)                         # eq 10.31  -- RAG top-k  (skipped for C4-RAG)
    ctx   = Feedback(x_t, y_{t-1}, A_t, last_rejection)  # evaluator feedback into the prompt
    pi_t  = LLM(s_t, r_t, ctx, schema)                 # eq 10.32  -- schema-constrained (relaxed for C4-Schema)
    ledger.record_proposal_attempt()
    ev    = event_from(pi_t)                           # A12 funnel: parse/schema/authority/hallucination
    if funnel_ok(pi_t):
        appl  = apply_proposal(x_t, pi_t)              # eq 10.33  -- APPLIED TO x_t, NOT x0
        attach_applicability(ev, appl)
        if appl.applicability_valid:
            y_t, hit = ledger.consume(appl.bom)        # eq 10.34  -- E(x'_t); one budget unit iff fresh
            attach_evaluation(ev, y_t)
            accept = Accept(appl.bom, y_t, x_t, A_t)   # deterministic gate (skipped for C4-Validator)
            status, size = A_t.offer(appl.bom, y_t)    # eq 10.35  -- ND(A_t ∪ {x'_t})
            attach_archive(ev, status, size)
            if accept:
                x_{t+1} = appl.bom ; retries[s_t] = 0
            else:
                x_{t+1} = x_t ; retries[s_t] += 1      # stay; Select() moves on after K
    events.write(ev) ; history.append(...) ; t += 1
terminal = classify_terminal(...)                      # 11.18 vocabulary + C4 additions
```

`stop(...)` and `Select(...)` are §6 and §5. `Feedback(...)` is §4.

---

## 4. Q1 — what makes C4 mechanically different from C1/C2/C3

| | C1 / C2 / C3 (A12 `GenerativeDriver`) | **C4 (`C4Driver`)** |
|---|---|---|
| **Working state** | every accepted proposal applied to the **frozen `x0`**, independently, never stacked (A12 decision Q3) | a **single evolving `x_t`**: `x'_t = Apply(x_t, π_t)` (eq 10.33). Changes compound across steps. |
| **Selection** | fixed round-robin over parts, `start = seed % |parts|` | `s_t = Select(A_t, history_t)` (eq 10.30) — an explicit policy over the archive + history (§5) |
| **LLM inputs** | part + registry + (RAG for C2/C3); **no result of any prior attempt** | + the previous evaluation `y_{t-1}` (cost/mass vs baseline, violations), the archive summary, and any rejection reason — the LLM **iterates on feedback** |
| **Accept/reject** | none — every funnel-passing proposal is scored and offered | a deterministic `Accept()` gate decides whether `x_t` advances; rejects cause a retry then a `Select()` move |
| **Reachable space** | atomic single-swap vs `x0` ⇒ ~11–16 distinct candidates (measured) ⇒ every generative run `COMPLETE_SPACE_EXHAUSTED` well below `N` | compounded swaps ⇒ ~`∏ᵢ |choicesᵢ|` ≈ 10³–10⁵ reachable states ⇒ **C4 can actually consume the full `N`** (§7) |
| **Thesis role** | "proposal quality" (§11.3) | "deterministic iteration / search" — the H2/H3 condition |

The compounding state is the whole point: it is exactly what A12 deliberately
kept out of C1/C2/C3 ("cumulative search is C4's job", A12 §6).

---

## 5. `Select(A_t, history_t)` — selection policy

`Select` returns a **selection** `s_t = {part_id, intent}` where `intent ∈
{reduce_cost, reduce_mass, fix_violation, diversify}`. Two candidate policies:

- **(A) Deterministic archive-guided (recommended for the first C4 build).**
  No extra LLM call, fully seed-reproducible.
  - *exploit:* pick the archive member on the current cost-min or mass-min
    corner (alternating by step parity), pick a part on it not yet modified
    this "pass", `intent` = the objective that corner optimises;
  - *explore:* if the last `L_explore` steps produced no `pareto_improving`
    archive event, round-robin to an unmodified part with `intent=diversify`;
  - *repair:* if `y_{t-1}` had a deterministic violation, `intent=fix_violation`
    on the offending part (overrides the above).
  - Randomness: a single `random.Random(seed)` breaks ties and orders the
    round-robin, so the whole trajectory is a function of `seed`.

- **(B) LLM-chosen selection.** The LLM picks the next part + intent given the
  archive summary. More "agentic", but: (i) a second generation call per step
  (budget/time), (ii) reproducibility depends on the decode seed only, (iii)
  harder to ablate. **Deferred** — revisit if (A) plateaus.

**LOCKED (2026-08-31): policy (A), `archive_guided_v1`.** (B) is a future extension.

---

## 6. Q3 — stopping rules

Three nested limits, all frozen in `run_config` before the run (per §4.9):

1. **Run budget `N` (hard, equal-budget, eq 11.4).** The loop ends when the
   ledger has counted `N = 50` **fresh** `E(x)` calls — identical definition to
   C5. Cache hits and pre-eval funnel failures do not consume it.
2. **Convergence early-stop (soft, eq 4.2 / 9.35).** After `t ≥ L`, stop if
   `ΔHV_recent(t) = HV_t − HV_{t−L} < ε` (or the range variant
   `max(HV_{t−L+1..t}) − min(...) < ε_HV`). `HV_t` is computed after every
   `y_t` — cheap, no budget cost. Proposed `L = 10`, `ε` = a small absolute HV
   (≈ 0.1, i.e. ~1 % of a single-swap improvement's HV contribution on this
   benchmark). **Frozen before the run**, recorded in `run_config` and
   `EXPERIMENT_DEVIATIONS.txt` if it differs from any thesis-stated value.
3. **Per-selection retry cap `K` (bounds wasted work).** If `Accept()` rejects
   the proposal for `s_t`, `x_t` stays and `retries[s_t] += 1`; the rejection
   reason is fed into the next prompt. After `K` rejections on the same
   selection, `Select()` is forced to a different part/intent. Proposed
   `K = 3`. Not a convergence calc — just a local circuit-breaker.
4. **Global attempt cap** (backstop, as in C1–C3): if
   `proposal_attempts ≥ attempt_cap` before `N` fresh evals →
   `ABORTED_BUDGET_UNREACHED`.

**`Accept(x', y, A_t)` — via existing `ParetoArchive.offer()` semantics (LOCKED
2026-08-31).** After `E(x')` and `status, _ = A_t.offer(x', y)`:
- pre-checks (before `offer`): **reject** on parse/schema/authority/applicability
  failure, no-op, or deterministic infeasibility (`y` has a violation);
- then **accept** iff `status ∈ {pareto_improving, non_dominated}` — the
  `non_dominated` class already covers a lateral move that extends the frontier
  (e.g. cheaper-but-heavier);
- **reject** iff `status ∈ {dominated, duplicate}`.

No separate "improves ≥ 1 objective vs `x_t`" rule — the archive classification
is the single source of truth. For **C4−Validator**, `Accept` degrades to
"parseable ⇒ accept" (feasibility and archive status are logged but not used to
gate `x_{t+1}`).

Terminal statuses: A12's 11.18 set plus **`COMPLETE_CONVERGED`** (stopped on
rule 2 with `n_eval_consumed < N`). `COMPLETE` = hit `N`.

---

## 7. Equal budget — why C4 is the first generative condition that can use it

C1/C2/C3 exhausted a ~11–16-wide atomic space and terminated
`COMPLETE_SPACE_EXHAUSTED` at `n_eval_consumed ≈ 13 << 50` (A12 sweep). C4's
`x'_t = Apply(x_t, π_t)` compounds changes, so its reachable-state space is the
product over parts (~10³–10⁵), not the sum. C4 will therefore normally run to
`N = 50` (or `COMPLETE_CONVERGED` near it) — making **C4-vs-C5 a genuine
like-for-like `N`-evaluation comparison**, which C1/C2/C3-vs-C5 never was. This
is the one place the shared ceiling is also a shared *reach*.

Ledger usage is unchanged: `ledger.consume(x'_t)` — fresh eval ⇒ +1 budget,
cache hit ⇒ 0. `assert_consistent_with` still applies as a sanity cross-check
against the step count.

---

## 8. Q2 — backend: canonical C4 vs C4-base

Table 4.5 makes canonical C4 a **fine-tuned** LLM (the C3 adapter) + RAG +
tools. But C3 is deferred (`docs/A12_EXPERIMENT_HARNESS_STATUS.md`): the adapter
produces no schema-valid proposals under the deployment prompt.

**C4's loop is backend-agnostic** — `C4Driver` takes a `ProposalGenerator` just
like `GenerativeDriver`. So:

- **`C4-base` (build + validate now):** the loop against the working Ollama
  `llama3.1:8b` backend + RAG + tools. Not the canonical fine-tuned C4, but it
  exercises every mechanism, produces the full artifact set, and makes H2
  (C4-base vs C5) and H3 (ablations of C4-base) runnable **tonight's-scope**.
  Labeled `C4_base` in `run_config.condition_spec` and `RUN_NOTES.md`.
- **`C4` (canonical):** swap the backend to the fixed C3 adapter once
  `docs/A12_C3_CONSTRAINED_DECODING_PLAN.md` lands or A11 retrains. Zero loop
  changes. The thesis H2/H3 numbers come from this run; `C4-base` is a
  development/validation precursor, same status as the A10 C1/C2 pilot vs the
  real sweep.

**C4 does NOT hard-depend on a working C3. LOCKED (2026-08-31): build `C4-base`
now against Ollama `llama3.1:8b`; swap to canonical when C3 is unblocked.**

---

## 9. What is reused from A12 vs new

| Component | Reuse / new |
|---|---|
| `EqualBudgetLedger` | **reuse unchanged** — `consume`, `record_proposal_attempt`, `budget_exhausted`, `attempts_exhausted` |
| `apply_proposal(bom, proposal)` | **reuse** — called with `x_t` instead of `x0`; protected-field / applicability / no-op logic already correct against an arbitrary base |
| `ParetoArchive` (`drivers.py`) | **reuse** — `offer()` is eq 10.35; seeded with `x0` per §4.9 |
| `events.py` — `event_from`, `attach_*`, `EventLog`, `derive_funnel_stage` | **reuse**, + a new `agentic` block and `event_type = "agentic_step"` (§10) |
| `metrics.py` — `compute_metrics`, rollups, `wilcoxon_paired`, `hypothesis_tests` | **reuse**, + convergence/acceptance metrics and H2/H3 rows (§11) |
| `run_experiment.py` | **edit** — add `C4` (and `C4_base`, ablations) to `CONDITION_ORDER` → `C1→C2→C3→C4→C5`; a `C4Driver` branch; `findings()` note for `COMPLETE_CONVERGED` vs budget |
| `identity.py` | **edit** — `condition_spec.c4_loop` block (N, L, ε, K, select_policy, feedback_mode, ablation) |
| `probe.py` | **reuse** — run the C3 probe only when C4's backend is the fine-tuned adapter |
| `src/experiment/c4_driver.py` | **new** (~180–260 lines) — the loop |
| `src/experiment/c4_select.py` | **new** (~60–100 lines) — policy (A) |
| `src/llm/prompt_builder.py` — `build_c4_prompt()` | **new** (~50 lines) — selection + feedback + archive summary + schema; a distinct template from C1/C2/C3 |
| `src/experiment/c4_feedback.py` | **new** (~40 lines) — assemble `y_{t-1}` + archive state + rejection reason into prompt text |

Net: **~4 new files, ~4 edited.** No new external dependency (C4-base uses
Ollama; canonical C4 uses the existing MLX path).

---

## 10. `events.jsonl` / `metrics.json` additions for C4

Per-step event (`event_type = "agentic_step"`) keeps the A12 shape and adds:

```jsonc
"agentic": {
  "step_index": 12,
  "selection": { "part_id": "PILOT_004", "intent": "reduce_mass",
                 "policy_reason": "exploit: mass-min corner, part unmodified this pass" },
  "working_state_hash_before": "sha256:...",     // x_t
  "working_state_hash_after":  "sha256:...",     // x_{t+1} (== before if rejected)
  "feedback_given": { "prev_cost_delta": -6.9, "prev_mass_delta": 0.01,
                      "prev_violations": 0, "rejection_reason": null },
  "accepted": true,
  "retry_of_selection": 0,
  "hv_after": 12.83,
  "delta_hv_recent": 0.41                        // ΔHV over the last L steps
}
```

`metrics.json` gains a `c4` block: `steps`, `accepted_steps`,
`acceptance_rate`, `mean_retries_per_selection`, `selection_intent_counts`,
`hv_trajectory` (list of `hv_after`), `converged` (bool), `stop_rule`
(`budget` / `convergence` / `attempt_cap`), plus the standard funnel /
hallucination / multiobjective blocks computed over the steps.

---

## 11. H3 ablations and statistical analysis

### H3 ablations (§11.14) — flags on `C4Driver`

| Variant | Change | Expected (thesis) |
|---|---|---|
| `C4-full` | — | reference |
| `C4−RAG` | skip eq 10.31; empty retrieved context | ↑ unsupported rule/data claims |
| `C4−Schema` | drop schema constraint from the LLM call; best-effort parse | ↑ parse/schema failure |
| `C4−Validator` | `Accept()` → "parseable ⇒ accept"; feasibility not used to reject | ↑ accepted violations, ↓ archive trust |

Each variant is a separate set of runs. **Scope decision:** full 10 seeds ×
3 variants × `N=50`, or a 3-seed H3 pilot. Recommendation: 3-seed pilot first
(H3 is a within-C4 comparison; direction matters more than power initially),
full 10 seeds if the pilot shows the expected ordering.

### H2 — C4 vs C5 (§11.13, eq 11.21)

Slots straight into `metrics.hypothesis_tests`:
- `dᵢ^HV = HVᵢ,C4 − HVᵢ,C5` per shared seed; paired Wilcoxon (two-sided —
  a negative H2 is a meaningful result), `d_z`, rank-biserial.
- Overall criterion `HV_C4 ≥ HV_C5` (eq 3.52); stronger criterion on the
  categorical-variable subset `HV_C4,cat > HV_C5,cat` (eq 3.53) — needs a
  categorical-subset HV computed from the archive (material/process-only moves),
  a small metrics addition.
- The `hypothesis_tests.csv` `H2` row moves from `PENDING_C4` to computed.

### H3 in the CSV

`HR_full < min(HR_{C4−RAG}, HR_{C4−Schema}, HR_{C4−Validator})` (eq 10.38) and
`HR_full < 0.05` (eq 10.39), family-wise corrected. `H3` row computed when the
ablation metrics are present.

---

## 12. `run_config` additions (identity 11.4)

```jsonc
"condition_spec": {
  "driver": "C4Driver",
  "backend_role": "base",            // "base" (Ollama, C4-base) | "fine_tuned" (canonical C4)
  "generator_fn": "src.experiment.c4_driver.C4Driver",
  "decision_variables": ["material_id", "process_id"],
  "target_parts": ["PILOT_001", "...", "PILOT_010"],
  "c4_loop": {
    "budget_definition": "deterministic_objective_evaluations",
    "n_eval": 50,
    "convergence": { "look_back_L": 10, "epsilon_hv": 0.1, "variant": "delta" },
    "retry_cap_K": 3,
    "proposal_attempt_cap": 1500,
    "select_policy": "archive_guided_v1",   // §5 (A)
    "feedback_mode": "prev_eval+archive+rejection",
    "ablation": null                        // null | "no_rag" | "no_schema" | "no_validator"
  }
}
```

`select_policy`, `feedback_mode`, `n_eval`, `L`, `ε`, `K` are all part of the
run identity — a change to any of them is a new `run_id` (11.4).

---

## 13. Decisions — LOCKED 2026-08-31

| # | Question | Decision |
|---|---|---|
| 1 | `Select` policy | **(A) `archive_guided_v1`** — deterministic, seeded, no extra LLM call. (B) LLM-chosen is a future extension. |
| 2 | Backend for the first C4 run | **`C4-base`** against Ollama `llama3.1:8b` + RAG + tools; canonical (fine-tuned C3) swapped in when C3 is unblocked, zero loop changes. |
| 3 | Convergence `L`, `ε` | **`L = 10`, `ε = 0.1`** absolute HV; frozen in `run_config` before any run; recorded in `EXPERIMENT_DEVIATIONS.txt`. |
| 4 | Per-selection retry cap `K` | **`K = 3`**. |
| 5 | H3 ablation scope | **3-seed pilot** per variant (C4−RAG / C4−Schema / C4−Validator); full 10 seeds only if the pilot shows the expected ordering. |
| 6 | `Accept()` | **Via `ParetoArchive.offer()` classification** — accept iff `status ∈ {pareto_improving, non_dominated}` (the `non_dominated` class is the lateral move). No separate single-objective rule. |
| 7 | Backtracking on stagnation | **Skipped for v1.** `Select()` moves to a different part/intent after `K` retries; `x_t` never resets. Noted as a future extension. |
| 8 | Categorical-subset HV (eq 3.53) | **Add to `metrics.py`** alongside the C4 metrics — HV over the archive restricted to material/process-only moves. |

---

## 14. Build order (after sign-off — not now)

1. `src/experiment/c4_select.py` — policy (A), pure + seeded.
2. `src/llm/prompt_builder.py::build_c4_prompt` + `src/experiment/c4_feedback.py`.
3. `src/experiment/c4_driver.py` — the loop, reusing ledger / `apply_proposal` /
   `ParetoArchive` / events; terminal-status classification incl.
   `COMPLETE_CONVERGED`.
4. `events.py` — `agentic` block + `event_type`; `metrics.py` — `c4` block,
   categorical-subset HV, H2/H3 rows.
5. `run_experiment.py` — `C1→C2→C3→C4→C5`; `C4Driver` branch; `--ablation` flag;
   `findings()` for convergence vs budget.
6. Tests: loop budget invariant (`Σ fresh E(x) == n_eval` or `COMPLETE_CONVERGED`
   with `< N`), state-compounding (`x_t` diverges from `x0` across accepted
   steps), `Select` reproducibility from seed, `Accept` gate truth table,
   convergence early-stop fires, ablation flags change the funnel as expected,
   H2 row computes.
7. `C4-base` pilot: 3 seeds, inspect `hv_trajectory` / acceptance rate / stop
   rule; then full 10 seeds.
8. H3 ablation pilot.
9. Docs: `docs/A13_C4_STATUS.md`; `EXPERIMENT_DEVIATIONS.txt` (frozen `N/L/ε/K`,
   `C4-base` vs canonical, ablation scope).

*§5 / §6 / §13 signed off 2026-08-31. Build proceeds.*
