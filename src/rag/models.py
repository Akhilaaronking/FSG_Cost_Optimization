from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagDocument:
    document_id: str
    text: str
    source_type: str
    source_id: str | None = None
    source_reference: str | None = None
    metadata: dict = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    document_id: str
    text: str
    source_type: str
    source_id: str | None = None
    source_reference: str | None = None
    metadata: dict = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RetrievalResult:
    chunk: RagChunk
    score: float
    rank: int
