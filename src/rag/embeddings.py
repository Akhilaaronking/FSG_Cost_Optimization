from typing import Protocol

import numpy as np


DEFAULT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class Embedder(Protocol):
    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...


def normalize_embeddings(
    embeddings,
) -> np.ndarray:
    array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix"
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )
    norms[norms == 0.0] = 1.0

    return (
        array / norms
    ).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.model_name = model_name

        from sentence_transformers import (
            SentenceTransformer,
        )

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return normalize_embeddings(
            embeddings
        )


class KeywordHashEmbedder:
    """
    Deterministic local embedder for offline validation.

    This is not the production thesis embedding model; it
    exists so tests and scripts can run without downloading
    external model files.
    """

    def __init__(
        self,
        dimensions: int = 256,
    ):
        self.dimensions = dimensions
        self.model_name = (
            f"keyword-hash-{dimensions}"
        )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        import hashlib
        import re

        rows = []

        for text in texts:
            vector = np.zeros(
                self.dimensions,
                dtype=np.float32,
            )

            for token in re.findall(
                r"[A-Za-z0-9_.]+",
                text.lower(),
            ):
                digest = hashlib.sha256(
                    token.encode("utf-8")
                ).digest()
                index = int.from_bytes(
                    digest[:4],
                    "big",
                ) % self.dimensions
                vector[index] += 1.0

            rows.append(vector)

        return normalize_embeddings(
            np.vstack(rows)
            if rows
            else np.zeros(
                (0, self.dimensions),
                dtype=np.float32,
            )
        )
