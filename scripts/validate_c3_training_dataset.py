from __future__ import annotations

import argparse
from pathlib import Path

from src.training.validation import (
    DEFAULT_TRAINING_DIR,
    validate_training_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate A11 C3 training-data JSONL files."
    )
    parser.add_argument(
        "--training-dir",
        default=str(DEFAULT_TRAINING_DIR),
    )
    args = parser.parse_args()

    result = validate_training_dataset(
        Path(args.training_dir)
    )

    print("=" * 70)
    print("A11 C3 TRAINING DATA VALIDATION")
    print("=" * 70)
    print("Validation: PASS")
    print("Total examples:", result["record_count"])
    print("Counts:", result["counts"])
    print("Part IDs by split:", result["part_ids_by_split"])
    print("Dataset hash:", result["dataset_hash"])


if __name__ == "__main__":
    main()

