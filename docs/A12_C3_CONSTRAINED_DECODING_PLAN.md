# Scoping plan — LFE `TokenEnforcer` → MLX `logits_processor` for `MLXLoRABackend`

**Status: plan only, not scheduled.** Preserves the design for a later
decision. No implementation. See `docs/A12_EXPERIMENT_HARNESS_STATUS.md`
("C3 defect") and `EXPERIMENT_DEVIATIONS.txt` (SWEEP RESULT) for why C3
is currently deferred.

## Goal

Force C3 (MLX LoRA) generation to emit structurally schema-valid JSON,
the way Ollama's `format=<schema>` already does for C1/C2 — so C3
proposals stop failing at the `schema` funnel stage.

## Why it's tractable

- **No grammar authoring.** Reuse `schemas/ollama_proposal_output.schema.json`
  — the flat, enum-constrained schema already used for C1/C2.
  `lmformatenforcer.JsonSchemaParser` consumes it directly.
- **LFE core is framework-agnostic.**
  `TokenEnforcer(tokenizer_data, parser).get_allowed_tokens(seq).allowed_tokens
  -> list[int]`. Only the transformers / vllm / llamacpp *adapters* are
  missing MLX — the engine isn't.
- **mlx-lm plumbing already works.** `mlx_lm.generate(..., logits_processors=[fn])`
  passes straight through to the generation loop (verified during
  investigation, 2026-08-31). A processor is
  `(tokens: mx.array, logits: mx.array) -> mx.array`.

## What this does NOT fix

Constrained decoding enforces *shape* — required keys,
`change_type` / `target_field` enums, non-empty string `new_value`. It
does **not** guarantee `new_value` is an admissible registry ID (that
would need a dynamic per-part enum schema — the "significant custom" path
we're avoiding). Expected effect: `schema` pass-rate -> ~100%;
`identifier` pass-rate then depends on prompt quality. So this pairs with
a companion prompt sub-task (change #3 below).

## Changes

| # | File | New/edit | ~Size | What |
|---|---|---|---|---|
| 1 | `src/llm/mlx_format_enforcer.py` | **new** | ~80–120 lines | `build_mlx_tokenizer_data(tokenizer)` (one-time ~128k-token vocab scan, cached) + `make_schema_logits_processor(tokenizer_data, parser, prompt_len)` returning the `(tokens, logits) -> logits` callable that masks disallowed tokens to `-inf`. |
| 2 | `src/llm/backend.py` — `MLXLoRABackend` | edit | ~25 lines | Optional `json_schema` ctor arg; build & cache `TokenEnforcerTokenizerData` in `_load()`; in `generate()`, if schema set, construct/reuse the processor and pass `logits_processors=[…]` to `mlx_generate`. |
| 3 | `src/llm/prompt_builder.py` + `src/llm/conditions.py` | edit | ~40 lines | **Companion:** `build_c3_prompt()` — short system + compact user (target part, admissible materials/processes from the frozen search space, current values). `generate_c3` uses it instead of the 3.7k-char `build_proposal_prompt`. Needed regardless — constrained shape on a bad prompt still yields bad picks. |
| 4 | `scripts/run_experiment.py` — `build_generator("C3")` | edit | ~5 lines | Pass `json_schema=<ollama schema>` to `MLXLoRABackend`. |
| 5 | `src/experiment/probe.py` | edit | ~15 lines | Deep probe: check the generated proposal *parses + passes `validate_proposal_schema`*, not just that a string came back (gap noted in the status doc). |
| 6 | `requirements.txt` / `requirements-lock.txt` | edit | — | Add `lm-format-enforcer` (pulls `pydantic` v2 + `interegular`). **New dependency surface — needs sign-off.** |
| 7 | `tests/test_mlx_format_enforcer.py` | **new** | ~120 lines | Unit: tokenizer-data build on a tiny fake vocab; mask correctness (`allowed=[5,9]` ⇒ others `-inf`); parser reaches terminal state ⇒ EOS allowed. One slow integration test (real MLX generate ⇒ output passes schema), kept out of the default suite like the probe. |
| 8 | `docs/A12_EXPERIMENT_HARNESS_STATUS.md`, `EXPERIMENT_DEVIATIONS.txt` | edit | — | C3 status update if it works; record the constrained-decoding approach + the new dep. |

Net: **2 new files, ~6 edited**, ~1 external dependency.

## Effort & decision gate

| Phase | Est. |
|---|---|
| Adapter module + backend wiring | ~0.5 day |
| C3 prompt path (#3) | ~0.25 day |
| Unit tests | ~0.25 day |
| Quick check: C3 seeds 0–2, iterate on prompt / decode params | ~0.25 day |
| Docs + dependency lock regen | ~0.1 day |
| **Total** | **~1.5 days** |

**Gate after the quick check:** if `schema` pass-rate ≥ ~90 % *and*
`identifier` pass-rate is workable (≥ ~60 %) → full 10-seed C3 re-run
(~20 min) and H1 becomes evaluable. Otherwise → the real fix is A11
(retrain the adapter on generate-only examples with the deployment
prompt), and this port gets shelved.

## Risks / unknowns to resolve early

1. **Prompt-token offset semantics.** `get_allowed_tokens` must be fed the
   *completion* tokens only, not prompt+completion (else it tries to parse
   the prompt as JSON). The processor needs `tokens[prompt_len:]`. Verify
   LFE's expected input contract. *(Med — get this wrong and nothing
   generates.)*
2. **mlx-lm tokenizer surface.** `TokenizerWrapper` must expose `encode` /
   `decode` / `all_special_ids` / `eos_token_id` / `len()` for the vocab
   scan; may need to reach through to `._tokenizer`. *(Low.)*
3. **Per-step Python callback latency.** MLX's tight generation loop calling
   into `get_allowed_tokens` + rebuilding an `mx` mask each token. LFE is
   built for this, but measure — expect C3 per-proposal ~3 s → ~5–8 s.
   Evaluate `use_bitmask` mode. *(Low–Med.)*
4. **New dependency.** `pydantic` v2 enters the env (full suite stayed green
   with it installed during investigation, so low compat risk, but it's a
   deliberate lockfile change). *(Low.)*
5. **Outcome risk, not engineering risk.** Even with valid schema, C3 may
   pick weak candidates → H1 comes back "C3 valid but not better than C2."
   That's still a legitimate result, not a failure of the port.

## Investigation notes (2026-08-31)

- `mlx-lm` 0.31.3: no native schema/grammar constraint (only
  repetition / frequency / presence penalty `logits_processors`).
- `lm-format-enforcer` 0.11.3: no MLX integration module
  (`exllamav2`, `haystackv1/v2`, `llamacpp`, `transformers`, `trtllm`,
  `vllm` only). Core `TokenEnforcer` + `JsonSchemaParser` +
  `TokenEnforcerTokenizerData` are generic and usable.
- `outlines` 1.3.3: has nominal MLX support (`outlines.models.MLXLM`,
  `from_mlxlm`) but `Generator(model, json_schema)` hung > 2 min on index
  compilation over the full vocab — not viable as-is.
- Prompt experiments (no constraint): the full `build_proposal_prompt`
  yields 0 valid; a training-style short-system + compact-user prompt
  yields ~2/10 (adapter still drops `proposal_id` ~80 % of the time).
  This is why change #3 (prompt path) is bundled here.
