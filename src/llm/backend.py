import json
import subprocess
import urllib.error
from pathlib import Path
import urllib.request
from typing import Protocol


class LLMBackend(Protocol):
    model_name: str
    backend_name: str

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        ...


class StubLLMBackend:
    backend_name = "stub"

    def __init__(
        self,
        response: str,
        model_name: str = "stub-test-model",
    ):
        self.response = response
        self.model_name = model_name
        self.calls = []

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        return self.response


def _load_ollama_output_schema() -> dict:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "ollama_proposal_output.schema.json"
    )

    with schema_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


class OllamaBackend:
    backend_name = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        if seed is not None:
            options["seed"] = seed

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": _load_ollama_output_schema(),
            "options": options,
        }).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                data = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama generation request failed"
            ) from exc

        return data.get("response", "")


def ollama_available() -> bool:
    return subprocess.run(
        ["which", "ollama"],
        capture_output=True,
        text=True,
    ).returncode == 0


def list_ollama_models() -> list[str]:
    if not ollama_available():
        return []

    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return []

    lines = result.stdout.splitlines()

    if len(lines) <= 1:
        return []

    models = []

    for line in lines[1:]:
        fields = line.split()
        if fields:
            models.append(fields[0])

    return models


def choose_ollama_model(
    models: list[str],
) -> str | None:
    preferred = [
        "llama3.1:8b",
        "llama3.1",
        "mistral:7b",
        "mistral",
    ]

    for name in preferred:
        if name in models:
            return name

    return models[0] if models else None
