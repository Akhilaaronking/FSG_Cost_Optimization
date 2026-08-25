import json
from pathlib import Path

from src.rag.models import (
    RagChunk,
)
from src.rag.vector_index import (
    VectorIndex,
)


def load_corpus_jsonl(
    path: Path | str,
) -> list[RagChunk]:
    chunks = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            payload = json.loads(line)
            chunks.append(
                RagChunk(**payload)
            )

    return chunks


class RagRetriever:
    def __init__(
        self,
        corpus_path: Path | str,
        embedder,
    ):
        self.corpus_path = Path(corpus_path)
        self.embedder = embedder
        self.chunks = load_corpus_jsonl(
            self.corpus_path
        )
        embeddings = self.embedder.encode(
            [
                chunk.text
                for chunk in self.chunks
            ]
        )
        self.index = VectorIndex().build(
            self.chunks,
            embeddings,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ):
        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty"
            )

        query_embedding = self.embedder.encode(
            [query]
        )

        return self.index.search(
            query_embedding,
            top_k=top_k,
            filters=filters,
        )
