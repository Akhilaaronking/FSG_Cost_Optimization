from src.data.registry import DataRegistry
from src.llm.prompt_builder import (
    build_proposal_prompt,
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


def test_c1_prompt_contains_no_retrieved_context():
    bundle = build_proposal_prompt(
        bom(),
        bom()["parts"][0],
        DataRegistry(),
    )

    assert "NO RETRIEVED CONTEXT PROVIDED." in bundle.prompt


def test_c2_prompt_includes_retrieved_context():
    bundle = build_proposal_prompt(
        bom(),
        bom()["parts"][0],
        DataRegistry(),
        retrieved_context="[RANK] 1\n[TEXT] Evidence",
    )

    assert "[TEXT] Evidence" in bundle.prompt


def test_c1_c2_base_prompt_structure_equivalent():
    c1 = build_proposal_prompt(
        bom(),
        bom()["parts"][0],
        DataRegistry(),
    )
    c2 = build_proposal_prompt(
        bom(),
        bom()["parts"][0],
        DataRegistry(),
        retrieved_context="evidence",
    )

    assert c1.prompt_template_hash == c2.prompt_template_hash
    assert c1.template_structure == c2.template_structure


def test_prompt_deterministic_for_same_inputs():
    first = build_proposal_prompt(
        bom(),
        bom()["parts"][0],
        DataRegistry(),
    )
    second = build_proposal_prompt(
        bom(),
        bom()["parts"][0],
        DataRegistry(),
    )

    assert first.prompt_hash == second.prompt_hash
    assert first.prompt == second.prompt
