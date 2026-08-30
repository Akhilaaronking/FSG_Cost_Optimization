from __future__ import annotations


DEFAULT_SPLIT_SEED = 11


def grouped_part_split(
    part_ids: list[str],
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, str]:
    """
    Deterministic grouped split by part_id.

    The current B4 pilot has 10 parts, so an 80/10/10 split maps to
    8 train parts, 1 validation part, and 1 test part.
    """

    del seed

    sorted_ids = sorted(set(part_ids))

    if len(sorted_ids) < 3:
        raise ValueError(
            "At least three distinct part IDs are required"
        )

    train_count = max(
        1,
        round(len(sorted_ids) * 0.8),
    )
    validation_count = max(
        1,
        round(len(sorted_ids) * 0.1),
    )

    if train_count + validation_count >= len(sorted_ids):
        train_count = len(sorted_ids) - 2
        validation_count = 1

    split = {}

    for part_id in sorted_ids[:train_count]:
        split[part_id] = "train"

    for part_id in sorted_ids[
        train_count : train_count + validation_count
    ]:
        split[part_id] = "validation"

    for part_id in sorted_ids[
        train_count + validation_count :
    ]:
        split[part_id] = "test"

    return split


def part_ids_by_split(
    split_by_part_id: dict[str, str],
) -> dict[str, list[str]]:
    return {
        split_name: sorted(
            part_id
            for part_id, split in split_by_part_id.items()
            if split == split_name
        )
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    }

