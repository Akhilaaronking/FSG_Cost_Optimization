import json
from pathlib import Path

from src.llm.backend import (
    OllamaBackend,
    choose_ollama_model,
    list_ollama_models,
    ollama_available,
)


BOM_PATH = Path(
    "data/benchmark/pilot_10_parts_ground_truth.json"
)
OUTPUT_PATH = Path(
    "results/a10_c1_c2_pilot.jsonl"
)
SUMMARY_PATH = Path(
    "results/a10_c1_c2_pilot_summary.json"
)


def _production_embeddings_available() -> bool:
    try:
        from src.rag.embeddings import (
            SentenceTransformerEmbedder,
        )

        SentenceTransformerEmbedder()
        return True
    except Exception:
        return False


def _load_bom():
    with BOM_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def _rates(records, condition):
    subset = [
        record
        for record in records
        if record["condition"] == condition
    ]

    if not subset:
        return {
            "proposal_count": 0,
            "parse_valid_rate": None,
            "schema_valid_rate": None,
            "authority_valid_rate": None,
            "hallucination_rate": None,
        }

    total = len(subset)

    return {
        "proposal_count": total,
        "parse_valid_rate": sum(
            record["parse_valid"]
            for record in subset
        )
        / total,
        "schema_valid_rate": sum(
            record["schema_valid"]
            for record in subset
        )
        / total,
        "authority_valid_rate": sum(
            record["authority_valid"]
            for record in subset
        )
        / total,
        "hallucination_rate": sum(
            record["hallucinated"]
            for record in subset
        )
        / total,
    }


def main():
    if not _production_embeddings_available():
        print("A10 SOFTWARE PIPELINE: PASS")
        print(
            "A10 REAL PILOT: BLOCKED — "
            "C2_REAL_RUN_BLOCKED_BY_EMBEDDING_BACKEND"
        )
        return

    if not ollama_available():
        print("A10 SOFTWARE PIPELINE: PASS")
        print(
            "A10 REAL PILOT: BLOCKED — "
            "NO REAL LLM BACKEND"
        )
        return

    models = list_ollama_models()
    chosen_model = choose_ollama_model(
        models
    )

    if not chosen_model:
        print("A10 SOFTWARE PIPELINE: PASS")
        print(
            "A10 REAL PILOT: BLOCKED — "
            "NO REAL LLM BACKEND"
        )
        return

    from src.data.registry import DataRegistry
    from src.llm.conditions import (
        generate_c1,
        generate_c2,
    )
    from src.llm.generator import (
        ProposalGenerator,
    )
    from src.rag.embeddings import (
        SentenceTransformerEmbedder,
    )
    from src.rag.retriever import (
        RagRetriever,
    )

    bom = _load_bom()
    parts = bom["parts"][:3]
    seeds = [0, 1]
    registry = DataRegistry()
    backend = OllamaBackend(
        chosen_model
    )
    generator = ProposalGenerator(
        backend,
        registry=registry,
    )
    retriever = RagRetriever(
        "data/rag/corpus.jsonl",
        SentenceTransformerEmbedder(),
    )

    records = []

    for part in parts:
        for seed in seeds:
            records.append(
                generate_c1(
                    generator,
                    bom=bom,
                    target_part=part,
                    seed=seed,
                )
            )
            records.append(
                generate_c2(
                    generator,
                    bom=bom,
                    target_part=part,
                    retriever=retriever,
                    seed=seed,
                )
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index, record in enumerate(records):
            run_record = {
                "run_id": f"A10_PILOT_{index:04d}",
                "condition": record["condition"],
                "seed": record["model"]["seed"],
                "model_name": record["model"]["model_name"],
                "backend_name": record["model"]["backend_name"],
                "part_id": record["proposal"].get("part_id")
                if record["proposal"]
                else "",
                "prompt_hash": record["prompt_hash"],
                "rag_enabled": record["retrieval"].get(
                    "rag_enabled",
                    False,
                ),
                "retrieval": record["retrieval"],
                "raw_response": record["raw_output"],
                "parsed_proposal": record["proposal"],
                "parse_valid": record["parse_valid"],
                "schema_valid": record["schema_valid"],
                "authority_valid": record["authority_valid"],
                "hallucinated": record["hallucinated"],
                "hallucination_categories": record[
                    "hallucination_categories"
                ],
                "runtime_sec": record["runtime_sec"],
            }
            file.write(
                json.dumps(
                    run_record,
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "label": "PILOT / PIPELINE CHECK ONLY",
        "total_proposals": len(records),
        "c1": _rates(records, "C1"),
        "c2": {
            **_rates(records, "C2"),
            "mean_retrieved_chunks": sum(
                len(
                    record["retrieval"].get(
                        "retrieved_chunk_ids",
                        [],
                    )
                )
                for record in records
                if record["condition"] == "C2"
            )
            / max(
                1,
                len([
                    record
                    for record in records
                    if record["condition"] == "C2"
                ]),
            ),
        },
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    print("A10 REAL PILOT: COMPLETE")
    print("Output:", OUTPUT_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
