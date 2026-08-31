# A9 RAG Status

The A9 retrieval pipeline is operational. It is a grounding layer only; cost, mass and deterministic feasibility remain the responsibility of the deterministic evaluator and optimisation tools.

## Corpus Inputs

- `data/processed/deterministic_constraints_B5.json`
- `data/processed/rule_classification_B5.json`
- `docs/b4_engineering_data/source_register.csv`
- `docs/b4_engineering_data/handoff_B5_to_A.md`
- `docs/b4_engineering_data/provenance_notes.md`

No actual FSG Rules 2026 PDF or extracted full-text file was found in the repository during A9 inspection. The corpus therefore uses the local B5 rule extraction/classification records rather than downloading any external source.

## Source Types

- `fsg_rule`
- `derived_quality_gate`
- `source_register`
- `handoff_document`
- `provenance_document`

`S_3.5.11` remains an interpretive FSG requirement concerning realistic costs. `DERIVED_QG_001` remains an internal traceability quality gate and is not an FSG rule.

## Chunking

Default chunking uses `chunk_size_words = 180` and `overlap_words = 30`. Short individual rule documents normally remain atomic.

## Embeddings And Retrieval

Production embedder: `sentence-transformers/all-MiniLM-L6-v2`, loaded only when `SentenceTransformerEmbedder` is instantiated.

Software test embedder: `KeywordHashEmbedder`, a deterministic local hash-based encoder used only for unit tests and offline operational checks. Metrics produced with this embedder must not be reported as production SentenceTransformer RAG results.

Default retrieval depth: `top_k = 5`.

The current vector backend is a transparent NumPy cosine index.

If the thesis currently describes FAISS/Chroma as the implemented backend, that wording should be revised unless such a backend is subsequently added.

## Validation

Retrieval validation uses a small rule-ID-labelled query set in `data/rag/retrieval_validation_queries.json`. These labels are intended for operational pipeline checks, not scientific hypothesis testing.

The validation script reports Recall@5, Precision@5 and MRR. It must not be interpreted as C1-C6 experiment output or LLM performance evidence.

## Current Limitations

- No full FSG PDF/text is currently present locally.
- The validation query set is intentionally small.
- The NumPy index is simple and inspectable, but not a large-scale vector database.
- If the production sentence-transformer model is unavailable offline, the validation script falls back to a deterministic keyword-hash embedder for local operational checks. Real C2 runs are blocked until the production embedder is available.
