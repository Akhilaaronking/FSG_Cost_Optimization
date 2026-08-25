# A10 LLM Status

A10 implements the base LLM proposal-generation software path and a controlled C1/C2 pilot framework. It does not implement QLoRA, C3, C4, or the final thesis experiment matrix.

## Backend

Implemented backend abstraction:

- `StubLLMBackend` for software tests only
- `OllamaBackend` for local Ollama generation if Ollama and a model are already installed

Current environment check found no usable real LLM backend. Stub outputs must not be reported as C1/C2 thesis results.

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

The current environment does not have `sentence_transformers` importable, so production MiniLM retrieval validation and real C2 execution are blocked here. The deterministic keyword-hash embedder is for software validation only and must not be described as SentenceTransformer RAG.

## Thesis Wording Checks

If the thesis states that Llama 3.1 8B was used, update it unless that exact model is actually installed and selected for the run.

If the thesis states that FAISS or Chroma is the implemented backend, update it unless such a backend is subsequently added; A9/A10 currently use NumPy cosine retrieval.
