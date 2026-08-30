from src.llm.generator import (
    ProposalGenerator,
)
from src.rag.context_formatter import (
    format_retrieval_context,
)
from src.rag.query_builder import (
    build_engineering_query,
)


def generate_c1(
    generator: ProposalGenerator,
    *,
    bom: dict,
    target_part: dict,
    seed: int | None = None,
    search_space: dict | None = None,
) -> dict:
    return generator.generate(
        bom=bom,
        target_part=target_part,
        condition="C1",
        seed=seed,
        retrieved_context=None,
        retrieval_metadata={
            "rag_enabled": False,
            "top_k": None,
        },
        search_space=search_space,
    )


def generate_c2(
    generator: ProposalGenerator,
    *,
    bom: dict,
    target_part: dict,
    retriever,
    seed: int | None = None,
    top_k: int = 5,
    change_type: str | None = None,
    target_field: str | None = None,
    user_intent: str | None = None,
    search_space: dict | None = None,
) -> dict:
    query_text = build_engineering_query(
        target_part,
        change_type=change_type,
        target_field=target_field,
        user_intent=user_intent,
    )
    results = retriever.retrieve(
        query_text,
        top_k=top_k,
    )
    retrieved_context = format_retrieval_context(
        results
    )
    retrieval_metadata = {
        "rag_enabled": True,
        "top_k": top_k,
        "query_text": query_text,
        "retrieved_chunk_ids": [
            result.chunk.chunk_id
            for result in results
        ],
        "retrieved_source_ids": [
            result.chunk.source_id
            for result in results
        ],
        "retrieved_source_references": [
            result.chunk.source_reference
            for result in results
        ],
        "similarity_scores": [
            result.score
            for result in results
        ],
    }

    return generator.generate(
        bom=bom,
        target_part=target_part,
        condition="C2",
        seed=seed,
        retrieved_context=retrieved_context,
        retrieval_metadata=retrieval_metadata,
        search_space=search_space,
    )


def generate_c3(
    generator: ProposalGenerator,
    *,
    bom: dict,
    target_part: dict,
    retriever,
    seed: int | None = None,
    top_k: int = 5,
    change_type: str | None = None,
    target_field: str | None = None,
    user_intent: str | None = None,
    search_space: dict | None = None,
) -> dict:
    if generator.backend.backend_name != "mlx_lora":
        raise ValueError(
            "C3 requires the fine-tuned MLX LoRA backend."
        )

    query_text = build_engineering_query(
        target_part,
        change_type=change_type,
        target_field=target_field,
        user_intent=user_intent,
    )

    results = retriever.retrieve(
        query_text,
        top_k=top_k,
    )

    retrieved_context = format_retrieval_context(
        results
    )

    retrieval_metadata = {
        "rag_enabled": True,
        "top_k": top_k,
        "query_text": query_text,
        "retrieved_chunk_ids": [
            result.chunk.chunk_id
            for result in results
        ],
        "retrieved_source_ids": [
            result.chunk.source_id
            for result in results
        ],
        "retrieved_source_references": [
            result.chunk.source_reference
            for result in results
        ],
        "similarity_scores": [
            result.score
            for result in results
        ],
    }

    return generator.generate(
        bom=bom,
        target_part=target_part,
        condition="C3",
        seed=seed,
        retrieved_context=retrieved_context,
        retrieval_metadata=retrieval_metadata,
        search_space=search_space,
    )
