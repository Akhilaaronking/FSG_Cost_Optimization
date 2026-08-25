from dataclasses import dataclass

import numpy as np

from src.rag.models import (
    RagChunk,
    RetrievalResult,
)


@dataclass
class VectorIndex:
    chunks: list[RagChunk] | None = None
    embeddings: np.ndarray | None = None

    def build(
        self,
        chunks: list[RagChunk],
        embeddings,
    ):
        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix"
            )

        if len(chunks) != matrix.shape[0]:
            raise ValueError(
                "Chunk count must match embedding rows"
            )

        self.chunks = list(chunks)
        self.embeddings = matrix
        return self

    def _matches_filter(
        self,
        chunk: RagChunk,
        filters: dict | None,
    ) -> bool:
        if not filters:
            return True

        if (
            "source_type" in filters
            and chunk.source_type
            != filters["source_type"]
        ):
            return False

        if (
            "source_id" in filters
            and chunk.source_id
            != filters["source_id"]
        ):
            return False

        if "rule_category" in filters:
            if chunk.metadata.get(
                "rule_category"
            ) != filters["rule_category"]:
                return False

        return True

    def search(
        self,
        query_embedding,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if self.chunks is None or self.embeddings is None:
            raise ValueError(
                "VectorIndex has not been built"
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 2:
            if query.shape[0] != 1:
                raise ValueError(
                    "Query embedding must contain one row"
                )
            query = query[0]

        if query.ndim != 1:
            raise ValueError(
                "Query embedding must be a vector"
            )

        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension mismatch"
            )

        scored = []

        for index, chunk in enumerate(self.chunks):
            if not self._matches_filter(
                chunk,
                filters,
            ):
                continue

            score = float(
                np.dot(
                    query,
                    self.embeddings[index],
                )
            )
            scored.append(
                (score, chunk.chunk_id, chunk)
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
            )
            for rank, (
                score,
                _chunk_id,
                chunk,
            ) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]
