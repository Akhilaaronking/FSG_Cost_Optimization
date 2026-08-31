# A10 LLM Status

A10 implements the base LLM proposal-generation software path and a controlled C1/C2 pilot framework. It does not implement QLoRA, C3, C4, or the final thesis experiment matrix.

## Backend

Implemented backend abstraction:

- `StubLLMBackend` for software tests only
- `OllamaBackend` for local Ollama generation if Ollama and a model are already installed

As of 2026-08-31 a usable real LLM backend is available: `ollama` is installed and serving on `http://127.0.0.1:11434` with `llama3.1:8b` (digest `46e0c10c039e`, 4.9 GB) installed. `choose_ollama_model` selects `llama3.1:8b` (first entry in its preference list), so `OllamaBackend("llama3.1:8b")` is the model that drives real C1/C2 runs on this host.

Stub outputs must still not be reported as C1/C2 results; only `OllamaBackend` runs count.

## Prompt Architecture

C1 and C2 share one prompt structure:

- SYSTEM ROLE
- TASK
- CURRENT PART/BOM STATE
- ALLOWED OUTPUT SCHEMA
- AVAILABLE CANONICAL IDENTIFIERS
- CONSTRAINT/SAFETY INSTRUCTIONS
- OPTIONAL RETRIEVED CONTEXT
- OUTPUT INSTRUCTIONS

C1 passes no retrieved context. C2 passes top-k RAG evidence formatted with source IDs and references. The prompt explicitly forbids invented identifiers, FSG compliance claims, authoritative cost/mass calculations, and hidden chain-of-thought.

## Conditions

C1: base LLM, no retrieval context.

C2: same base LLM, same prompt template, same decoding configuration and seed where supported, plus RAG top-k = 5 retrieved context.

## Decoding Parameters

Default software configuration:

- `temperature = 0.2`
- `max_tokens = 512`
- `seed` supplied per run when supported by backend

## Hallucination Taxonomy

Implemented categories:

- `PARSE_ERROR`
- `SCHEMA_ERROR`
- `UNKNOWN_PART_ID`
- `UNKNOWN_MATERIAL_ID`
- `UNKNOWN_PROCESS_ID`
- `UNKNOWN_FASTENER_ID`
- `UNSUPPORTED_TARGET_FIELD`

Valid but weak engineering judgement is not automatically classified as factual hallucination.

## RAG

Default top-k: `5`.

Production embedding model configured: `sentence-transformers/all-MiniLM-L6-v2`.

Current vector backend: transparent NumPy cosine index.

As of 2026-08-31 `sentence_transformers` (6.0.0) is importable in `.venv`, so `SentenceTransformerEmbedder` with `all-MiniLM-L6-v2` now backs real C2 retrieval; the C1/C2 pilot below ran with it (mean 5.0 retrieved chunks per C2 proposal). The deterministic keyword-hash embedder remains software-validation only and must not be described as SentenceTransformer RAG.

## C1/C2 Pilot Run (2026-08-31)

`scripts/run_c1_c2_pilot.py` was run against the real backend above
(`PYTHONPATH=$PWD .venv/bin/python scripts/run_c1_c2_pilot.py`).

- Backend: `OllamaBackend` / `llama3.1:8b`; RAG: `SentenceTransformerEmbedder` (`all-MiniLM-L6-v2`), NumPy cosine index, top-k = 5
- Inputs: `data/benchmark/pilot_10_parts_ground_truth.json`, `data/rag/corpus.jsonl`
- Scope: 3 parts x 2 seeds (0, 1) x {C1, C2} = 12 proposals
- Outputs: `results/a10_c1_c2_pilot.jsonl`, `results/a10_c1_c2_pilot_summary.json`

Result: C1 and C2 both 6/6 parse-valid, 6/6 schema-valid, 6/6 authority-valid, 0/6 hallucinated; C2 averaged 5.0 retrieved chunks. Per-generation runtime 2.4-7.9 s.

This confirms the C1/C2 pipeline end-to-end on the real Llama 3.1 8B + MiniLM stack. It is a **pilot / pipeline check only** (the script self-labels its summary `PILOT / PIPELINE CHECK ONLY`): 12 proposals over 3 parts and 2 seeds is not statistically powered to show a C1-vs-C2 difference, and these numbers must not be reported as final thesis C1/C2 results. The final experiment matrix is still not built.

## Thesis Wording Checks

If the thesis states that Llama 3.1 8B was used: as of 2026-08-31 `llama3.1:8b` is installed and is the selected model, and the C1/C2 pilot ran on it. This is satisfied for the A10 pilot only — do not extend the claim to C3/C4 or final-thesis runs that have not been executed.

If the thesis states that FAISS or Chroma is the implemented backend, update it unless such a backend is subsequently added; A9/A10 currently use NumPy cosine retrieval.
