import pytest

import scripts.validate_rag_retrieval as validate_rag


class DummySentenceTransformerEmbedder:
    model_name = validate_rag.DEFAULT_MODEL_NAME


class DummyKeywordHashEmbedder:
    model_name = "keyword-hash-256"


def test_production_mode_uses_minilm(monkeypatch):
    monkeypatch.setattr(
        validate_rag,
        "SentenceTransformerEmbedder",
        DummySentenceTransformerEmbedder,
    )

    embedder, model_name = validate_rag._make_embedder()

    assert isinstance(
        embedder,
        DummySentenceTransformerEmbedder,
    )
    assert model_name == validate_rag.DEFAULT_MODEL_NAME


def test_production_mode_does_not_silently_fallback(monkeypatch):
    class MissingSentenceTransformer:
        def __init__(self):
            raise ImportError(
                "sentence_transformers unavailable"
            )

    monkeypatch.setattr(
        validate_rag,
        "SentenceTransformerEmbedder",
        MissingSentenceTransformer,
    )

    with pytest.raises(
        RuntimeError,
        match="no keyword fallback was used",
    ):
        validate_rag._make_embedder()


def test_development_fallback_must_be_explicit(monkeypatch):
    class MissingSentenceTransformer:
        def __init__(self):
            raise ImportError(
                "sentence_transformers unavailable"
            )

    monkeypatch.setattr(
        validate_rag,
        "SentenceTransformerEmbedder",
        MissingSentenceTransformer,
    )
    monkeypatch.setattr(
        validate_rag,
        "KeywordHashEmbedder",
        DummyKeywordHashEmbedder,
    )

    embedder, model_name = validate_rag._make_embedder(
        allow_development_fallback=True,
    )

    assert isinstance(
        embedder,
        DummyKeywordHashEmbedder,
    )
    assert model_name == "keyword-hash-256"
