def format_retrieval_context(
    results,
) -> str:
    blocks = []

    for result in results:
        chunk = result.chunk
        source_id = (
            chunk.source_id
            if chunk.source_id is not None
            else ""
        )
        source_reference = (
            chunk.source_reference
            if chunk.source_reference is not None
            else ""
        )

        blocks.append(
            "\n".join([
                f"[RANK] {result.rank}",
                f"[SOURCE_TYPE] {chunk.source_type}",
                f"[SOURCE_ID] {source_id}",
                f"[SOURCE_REFERENCE] {source_reference}",
                f"[TEXT] {chunk.text}",
            ])
        )

    return "\n\n".join(blocks)
