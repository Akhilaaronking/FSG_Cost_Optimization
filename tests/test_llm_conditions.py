import json

from src.data.registry import DataRegistry
from src.llm.backend import (
    StubLLMBackend,
)
from src.llm.conditions import (
    generate_c1,
    generate_c2,
)
from src.llm.generator import (
    ProposalGenerator,
)
from src.rag.models import (
    RagChunk,
    RetrievalResult,
)


def bom():
    return {
        "parts": [
            {
                "part_id": "PILOT_001",
                "name": "Bracket",
                "material_id": "AL_6061_T6",
                "process_id": "CNC_MILLING",
            }
        ]
    }


def response():
    return json.dumps({
        "proposal_id": "PROP_001",
        "part_id": "PILOT_001",
        "change_type": "material",
        "target_field": "material_id",
        "old_value": "AL_6061_T6",
        "new_value": "AL_7075_T6",
        "reasoning_summary": "Concise summary.",
    })


class CountingRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        top_k=5,
        filters=None,
    ):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
        })
        chunk = RagChunk(
            chunk_id="chunk_TEST",
            document_id="doc_TEST",
            text="retrieved evidence",
            source_type="fsg_rule",
            source_id="FSG_RULES_2026",
            source_reference="S 3.5.12, p.115",
            metadata={"rule_id": "S_3.5.12"},
        )
        return [
            RetrievalResult(
                chunk=chunk,
                score=0.9,
                rank=1,
            )
        ]


def generator():
    return ProposalGenerator(
        StubLLMBackend(response()),
        registry=DataRegistry(),
    )


def test_c1_never_calls_retriever():
    trace = generate_c1(
        generator(),
        bom=bom(),
        target_part=bom()["parts"][0],
        seed=0,
    )

    assert trace["retrieval"]["rag_enabled"] is False


def test_c2_calls_retriever_once():
    retriever = CountingRetriever()

    generate_c2(
        generator(),
        bom=bom(),
        target_part=bom()["parts"][0],
        retriever=retriever,
        seed=0,
    )

    assert len(retriever.calls) == 1


def test_c2_uses_top_k_five():
    retriever = CountingRetriever()

    generate_c2(
        generator(),
        bom=bom(),
        target_part=bom()["parts"][0],
        retriever=retriever,
        seed=0,
    )

    assert retriever.calls[0]["top_k"] == 5


def test_c2_records_retrieval_trace():
    retriever = CountingRetriever()

    trace = generate_c2(
        generator(),
        bom=bom(),
        target_part=bom()["parts"][0],
        retriever=retriever,
        seed=0,
    )

    assert trace["retrieval"]["query_text"]
    assert trace["retrieval"]["retrieved_chunk_ids"] == [
        "chunk_TEST"
    ]
    assert trace["retrieval"]["retrieved_source_ids"] == [
        "FSG_RULES_2026"
    ]
    assert trace["retrieval"][
        "retrieved_source_references"
    ] == ["S 3.5.12, p.115"]
