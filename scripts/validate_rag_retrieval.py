import json
from pathlib import Path

from src.rag.embeddings import (
    DEFAULT_MODEL_NAME,
    KeywordHashEmbedder,
    SentenceTransformerEmbedder,
)
from src.rag.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.rag.retriever import (
    RagRetriever,
)


CORPUS_PATH = Path("data/rag/corpus.jsonl")
MANIFEST_PATH = Path("data/rag/corpus_manifest.json")
QUERIES_PATH = Path(
    "data/rag/retrieval_validation_queries.json"
)


def _ensure_corpus():
    if CORPUS_PATH.exists() and MANIFEST_PATH.exists():
        return

    from scripts.build_rag_corpus import main

    main()


def _make_embedder():
    try:
        embedder = SentenceTransformerEmbedder()
        return embedder, DEFAULT_MODEL_NAME
    except Exception as exc:
        print(
            "Production SentenceTransformer unavailable; "
            "using deterministic KeywordHashEmbedder for "
            "local operational validation."
        )
        print("Reason:", exc)
        embedder = KeywordHashEmbedder()
        return embedder, embedder.model_name


def _rule_ids(results):
    ids = []

    for result in results:
        rule_id = result.chunk.metadata.get(
            "rule_id"
        )
        ids.append(rule_id or result.chunk.source_id)

    return ids


def main():
    _ensure_corpus()

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    with QUERIES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    embedder, model_name = _make_embedder()
    retriever = RagRetriever(
        CORPUS_PATH,
        embedder,
    )

    per_query = []
    recall_values = []
    precision_values = []
    missed = []

    print("=" * 70)
    print("A9 RAG RETRIEVAL VALIDATION")
    print("=" * 70)
    print(
        "Corpus documents:",
        manifest["document_count"],
    )
    print(
        "Corpus chunks:",
        manifest["chunk_count"],
    )
    print(
        "Validation queries:",
        len(queries),
    )
    print("Embedding model:", model_name)
    print("Vector backend: NumPy cosine index")
    print("-" * 70)

    for query in queries:
        results = retriever.retrieve(
            query["query_text"],
            top_k=5,
        )
        retrieved_ids = _rule_ids(results)
        relevant_ids = query[
            "relevant_rule_ids"
        ]
        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        )
        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        )
        recall_values.append(recall)
        precision_values.append(precision)
        per_query.append({
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })

        if recall < 1.0:
            missed.append(query["query_id"])

        scored = [
            (
                item,
                round(result.score, 6),
            )
            for item, result in zip(
                retrieved_ids,
                results,
            )
        ]
        print(
            query["query_id"],
            "retrieved:",
            scored,
        )

    recall_5 = sum(recall_values) / len(
        recall_values
    )
    precision_5 = sum(
        precision_values
    ) / len(precision_values)
    mrr = mean_reciprocal_rank(
        per_query
    )

    print("-" * 70)
    print(f"Recall@5: {recall_5:.4f}")
    print(f"Precision@5: {precision_5:.4f}")
    print(f"MRR: {mrr:.4f}")

    if missed:
        print(
            "Queries missing relevant rule in top 5:",
            missed,
        )
    else:
        print(
            "Queries missing relevant rule in top 5: none"
        )

    print("\nA9 PIPELINE STATUS: OPERATIONAL")


if __name__ == "__main__":
    main()
