import json

from src.data.registry import DataRegistry
from src.llm.backend import (
    StubLLMBackend,
)
from src.llm.generator import (
    ProposalGenerator,
)


def bom():
    return {
        "parts": [
            {
                "part_id": "PILOT_001",
                "name": "Bracket",
                "material_id": "AL_6061_T6",
                "process_id": "CNC_MILLING",
            }
        ]
    }


def response():
    return json.dumps({
        "proposal_id": "PROP_001",
        "part_id": "PILOT_001",
        "change_type": "material",
        "target_field": "material_id",
        "old_value": "AL_6061_T6",
        "new_value": "AL_7075_T6",
        "reasoning_summary": "Concise summary.",
    })


def test_generator_returns_complete_trace():
    generator = ProposalGenerator(
        StubLLMBackend(response()),
        registry=DataRegistry(),
    )

    trace = generator.generate(
        bom=bom(),
        target_part=bom()["parts"][0],
        condition="C1",
        seed=0,
    )

    assert trace["condition"] == "C1"
    assert trace["raw_output"] == response()
    assert trace["parse_valid"]
    assert trace["schema_valid"]
    assert trace["authority_valid"]
    assert not trace["hallucinated"]
    assert trace["prompt_hash"]
    assert trace["model_config_hash"]
    assert trace["runtime_sec"] >= 0


def test_raw_llm_output_retained():
    backend = StubLLMBackend(
        response()
    )
    generator = ProposalGenerator(
        backend,
        registry=DataRegistry(),
    )

    trace = generator.generate(
        bom=bom(),
        target_part=bom()["parts"][0],
        condition="C1",
    )

    assert trace["raw_output"] == backend.response


def test_no_chain_of_thought_field_introduced():
    generator = ProposalGenerator(
        StubLLMBackend(response()),
        registry=DataRegistry(),
    )

    trace = generator.generate(
        bom=bom(),
        target_part=bom()["parts"][0],
        condition="C1",
    )

    assert "chain_of_thought" not in trace
    assert "chain_of_thought" not in trace["proposal"]


def test_run_configuration_hash_reproducible():
    generator = ProposalGenerator(
        StubLLMBackend(response()),
        registry=DataRegistry(),
    )

    first = generator.generate(
        bom=bom(),
        target_part=bom()["parts"][0],
        condition="C1",
        seed=1,
    )
    second = generator.generate(
        bom=bom(),
        target_part=bom()["parts"][0],
        condition="C1",
        seed=1,
    )

    assert (
        first["model_config_hash"]
        == second["model_config_hash"]
    )
    assert first["prompt_hash"] == second["prompt_hash"]
