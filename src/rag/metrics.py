def _top_k(
    retrieved_ids,
    k: int,
) -> list:
    if k <= 0:
        raise ValueError(
            "k must be positive"
        )

    return list(retrieved_ids)[:k]


def _relevant_set(
    relevant_ids,
) -> set:
    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError(
            "relevant_ids must not be empty"
        )

    return relevant


def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k: int,
) -> float:
    relevant = _relevant_set(
        relevant_ids
    )
    retrieved = set(
        _top_k(retrieved_ids, k)
    )

    return len(
        retrieved & relevant
    ) / len(relevant)


def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k: int,
) -> float:
    relevant = _relevant_set(
        relevant_ids
    )
    retrieved = _top_k(
        retrieved_ids,
        k,
    )

    return len(
        set(retrieved) & relevant
    ) / k


def reciprocal_rank(
    retrieved_ids,
    relevant_ids,
) -> float:
    relevant = _relevant_set(
        relevant_ids
    )

    for index, retrieved_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if retrieved_id in relevant:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(
    query_results,
) -> float:
    if not query_results:
        raise ValueError(
            "query_results must not be empty"
        )

    total = 0.0

    for item in query_results:
        total += reciprocal_rank(
            item["retrieved_ids"],
            item["relevant_ids"],
        )

    return total / len(query_results)
