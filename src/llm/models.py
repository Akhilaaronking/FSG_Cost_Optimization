from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    backend_name: str
    temperature: float = 0.2
    max_tokens: int = 512
    seed: int | None = None


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    prompt_template_hash: str
    prompt_hash: str
    template_structure: str
    rag_enabled: bool
    metadata: dict = field(
        default_factory=dict
    )
