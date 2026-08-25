import json
from dataclasses import asdict
from pathlib import Path

from src.rag.corpus_builder import (
    build_corpus,
    corpus_manifest,
)


OUTPUT_DIR = Path("data/rag")
CORPUS_PATH = OUTPUT_DIR / "corpus.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "corpus_manifest.json"


def main():
    chunk_size_words = 180
    overlap_words = 30

    documents, chunks = build_corpus(
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )
    manifest = corpus_manifest(
        documents,
        chunks,
        chunk_size_words,
        overlap_words,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CORPUS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    asdict(chunk),
                    sort_keys=True,
                )
                + "\n"
            )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    print("=" * 70)
    print("A9 — RAG CORPUS BUILD")
    print("=" * 70)
    print("Corpus path:", CORPUS_PATH)
    print("Manifest path:", MANIFEST_PATH)
    print(
        "Documents:",
        manifest["document_count"],
    )
    print(
        "Chunks:",
        manifest["chunk_count"],
    )
    print(
        "Source types:",
        manifest["source_type_counts"],
    )


if __name__ == "__main__":
    main()
