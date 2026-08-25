from src.llm.backend import (
    choose_ollama_model,
    list_ollama_models,
    ollama_available,
)


def main():
    available = ollama_available()
    models = list_ollama_models()
    chosen = choose_ollama_model(
        models
    )

    print("=" * 70)
    print("A10 — LLM BACKEND CHECK")
    print("=" * 70)
    print("Backend available:", available)
    print(
        "Backend name:",
        "ollama" if available else None,
    )
    print("Installed models:", models)
    print("Chosen model:", chosen)
    print(
        "Generation can be run:",
        bool(available and chosen),
    )

    if not available or not chosen:
        print("REAL_LLM_BACKEND_NOT_AVAILABLE")


if __name__ == "__main__":
    main()
