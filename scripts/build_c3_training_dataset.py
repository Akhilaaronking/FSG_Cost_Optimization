from __future__ import annotations

import argparse
from pathlib import Path

from src.training.dataset_builder import (
    DEFAULT_OUTPUT_DIR,
    build_and_write_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build A11 C3 training-data JSONL files."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()

    records, manifest = build_and_write_dataset(
        output_dir=Path(args.output_dir),
    )

    print("=" * 70)
    print("A11 C3 TRAINING DATA BUILD")
    print("=" * 70)
    print("Dataset type: programmatically generated instruction examples")
    print("Output dir:", Path(args.output_dir))
    print("Total examples:", len(records))
    print("Split counts:", manifest["split_counts"])
    print("Example type counts:", manifest["example_type_counts"])
    print("Change type counts:", manifest["change_type_counts"])
    print("Part IDs by split:", manifest["part_ids_by_split"])
    print("Dataset hash:", manifest["hashes"]["dataset_hash"])
    print("NOTE: data build only; no C3 model training was run.")


if __name__ == "__main__":
    main()

