import hashlib
import re

from src.rag.models import (
    RagChunk,
    RagDocument,
)


def deterministic_id(
    *parts,
    prefix: str,
) -> str:
    payload = "\n".join(
        str(part)
        for part in parts
    )
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _words_and_headings(
    text: str,
) -> tuple[list[str], list[str | None]]:
    words = []
    headings = []
    current_heading = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            current_heading = stripped

        for word in _words(line):
            words.append(word)
            headings.append(current_heading)

    return words, headings


def chunk_document(
    document: RagDocument,
    chunk_size_words: int = 180,
    overlap_words: int = 30,
) -> list[RagChunk]:
    if chunk_size_words <= 0:
        raise ValueError(
            "chunk_size_words must be positive"
        )

    if overlap_words < 0:
        raise ValueError(
            "overlap_words must be non-negative"
        )

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be smaller than chunk_size_words"
        )

    words, headings = _words_and_headings(
        document.text
    )

    if not words:
        return []

    if len(words) <= chunk_size_words:
        text = " ".join(words)
        return [
            RagChunk(
                chunk_id=deterministic_id(
                    document.document_id,
                    0,
                    text,
                    prefix="chunk",
                ),
                document_id=document.document_id,
                text=text,
                source_type=document.source_type,
                source_id=document.source_id,
                source_reference=document.source_reference,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                },
            )
        ]

    chunks = []
    start = 0
    index = 0

    while start < len(words):
        end = min(
            start + chunk_size_words,
            len(words),
        )
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        prefix = ""
        if start > 0 and headings[start]:
            heading = headings[start]
            if heading not in chunk_text:
                prefix = f"{heading} "

        text = f"{prefix}{chunk_text}".strip()

        if text:
            chunks.append(
                RagChunk(
                    chunk_id=deterministic_id(
                        document.document_id,
                        index,
                        text,
                        prefix="chunk",
                    ),
                    document_id=document.document_id,
                    text=text,
                    source_type=document.source_type,
                    source_id=document.source_id,
                    source_reference=document.source_reference,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                    },
                )
            )

        if end == len(words):
            break

        start = end - overlap_words
        index += 1

    return chunks


def chunk_documents(
    documents: list[RagDocument],
    chunk_size_words: int = 180,
    overlap_words: int = 30,
) -> list[RagChunk]:
    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )
        )

    return chunks
