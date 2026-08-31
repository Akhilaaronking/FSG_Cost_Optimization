import time

from src.data.registry import DataRegistry
from src.llm.models import (
    LLMConfig,
)
from src.llm.prompt_builder import (
    build_proposal_prompt,
    sha256_json,
)
from src.llm.proposal_parser import (
    ProposalParseError,
    parse_proposal,
)
from src.llm.proposal_validator import (
    classify_hallucination,
    validate_proposal_authority,
    validate_proposal_schema,
)


class ProposalGenerator:
    def __init__(
        self,
        backend,
        registry: DataRegistry | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        self.backend = backend
        self.registry = registry or DataRegistry()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        *,
        bom: dict,
        target_part: dict,
        condition: str,
        seed: int | None = None,
        retrieved_context: str | None = None,
        retrieval_metadata: dict | None = None,
        search_space: dict | None = None,
        prompt_bundle_override=None,
    ) -> dict:
        # C4 passes a pre-built PromptBundle (build_c4_prompt); C1/C2/C3
        # leave it None and get the standard single-shot proposal prompt.
        prompt_bundle = prompt_bundle_override or build_proposal_prompt(
            bom,
            target_part,
            self.registry,
            retrieved_context=retrieved_context,
            search_space=search_space,
        )
        config = LLMConfig(
            model_name=self.backend.model_name,
            backend_name=self.backend.backend_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        model_config = {
            "model_name": config.model_name,
            "backend_name": config.backend_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "rag_enabled": prompt_bundle.rag_enabled,
        }

        # Record backend-specific provenance when available.
        # C1/C2 Ollama behavior is unchanged.
        adapter_path = getattr(
            self.backend,
            "adapter_path",
            None,
        )
        quantization = getattr(
            self.backend,
            "quantization",
            None,
        )

        if adapter_path is not None:
            model_config["adapter_path"] = adapter_path

        if quantization is not None:
            model_config["quantization"] = quantization
        errors = []
        proposal = None
        parse_valid = False
        schema_result = {
            "schema_valid": False,
            "errors": [],
        }
        authority_result = {
            "authority_valid": False,
            "unknown_identifiers": [],
            "errors": [],
        }

        start = time.perf_counter()
        raw_output = self.backend.generate(
            prompt_bundle.prompt,
            seed=seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        runtime_sec = time.perf_counter() - start

        try:
            proposal = parse_proposal(
                raw_output
            )
            parse_valid = True
        except ProposalParseError as exc:
            errors.append({
                "category": "PARSE_ERROR",
                "message": str(exc),
            })

        if proposal is not None:
            schema_result = validate_proposal_schema(
                proposal
            )
            authority_result = validate_proposal_authority(
                proposal,
                bom,
                registry=self.registry,
            )

        hallucination = classify_hallucination(
            parse_valid,
            schema_result,
            authority_result,
        )

        return {
            "condition": condition,
            "model": model_config,
            "model_config_hash": sha256_json(
                model_config
            ),
            "prompt_template_hash": prompt_bundle.prompt_template_hash,
            "prompt_hash": prompt_bundle.prompt_hash,
            "benchmark_input_hash": prompt_bundle.metadata[
                "benchmark_input_hash"
            ],
            "retrieval": retrieval_metadata
            or {
                "rag_enabled": False,
                "top_k": None,
            },
            "raw_output": raw_output,
            "proposal": proposal,
            "parse_valid": parse_valid,
            "schema_valid": schema_result[
                "schema_valid"
            ],
            "authority_valid": authority_result[
                "authority_valid"
            ],
            "hallucinated": hallucination[
                "hallucinated"
            ],
            "hallucination_categories": hallucination[
                "categories"
            ],
            "schema_errors": schema_result[
                "errors"
            ],
            "authority_errors": authority_result[
                "errors"
            ],
            "unknown_identifiers": authority_result[
                "unknown_identifiers"
            ],
            "errors": errors,
            "runtime_sec": runtime_sec,
        }
