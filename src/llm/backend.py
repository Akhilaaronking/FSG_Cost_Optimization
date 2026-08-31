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
        enforce_schema: bool = True,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        # enforce_schema=False is used only by the C4-Schema ablation
        # (docs/A13 section 11); everything else keeps structured output.
        self.enforce_schema = enforce_schema

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

        body = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self.enforce_schema:
            body["format"] = _load_ollama_output_schema()

        payload = json.dumps(body).encode("utf-8")

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


class MLXLoRABackend:
    backend_name = "mlx_lora"

    def __init__(
        self,
        model_name: str = (
            "mlx-community/"
            "Meta-Llama-3.1-8B-Instruct-4bit"
        ),
        adapter_path: str | Path | None = None,
    ):
        self.model_name = model_name

        if adapter_path is None:
            adapter_path = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "c3_adapter"
            )

        self.adapter_path = str(
            Path(adapter_path).expanduser().resolve()
        )
        self.quantization = "4-bit"
        self._model = None
        self._tokenizer = None

    def _validate_adapter(self) -> None:
        adapter_dir = Path(self.adapter_path)

        required = [
            adapter_dir / "adapter_config.json",
            adapter_dir / "adapters.safetensors",
        ]

        missing = [
            str(path)
            for path in required
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "C3 MLX LoRA adapter is incomplete. "
                "Missing: "
                + ", ".join(missing)
            )

    def _load(self):
        if (
            self._model is not None
            and self._tokenizer is not None
        ):
            return self._model, self._tokenizer

        self._validate_adapter()

        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "MLX-LM is required for the C3 backend."
            ) from exc

        self._model, self._tokenizer = load(
            self.model_name,
            adapter_path=self.adapter_path,
        )

        return self._model, self._tokenizer

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        try:
            import mlx.core as mx
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "MLX and MLX-LM are required "
                "for the C3 backend."
            ) from exc

        model, tokenizer = self._load()

        if seed is not None:
            mx.random.seed(seed)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        try:
            formatted_prompt = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except (AttributeError, TypeError):
            formatted_prompt = prompt

        sampler = make_sampler(
            temp=temperature,
        )

        return mlx_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )


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
