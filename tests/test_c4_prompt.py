import json

from src.data.registry import DataRegistry
from src.experiment.c4_feedback import build_archive_text, build_feedback_text
from src.experiment.c4_select import Selection
from src.llm.backend import StubLLMBackend
from src.llm.generator import ProposalGenerator
from src.llm.models import PromptBundle
from src.llm.prompt_builder import build_c4_prompt, build_proposal_prompt


BASELINE = [312.02, 0.6507108]


def bom():
    return {
        "parts": [
            {
                "part_id": "PILOT_001",
                "material_id": "AL_6061_T6",
                "process_id": "CNC_MILLING",
            },
            {
                "part_id": "PILOT_002",
                "material_id": "AL_7075_T6",
                "process_id": "CNC_MILLING",
            },
        ]
    }


def selection(intent="reduce_cost"):
    return Selection("PILOT_001", intent, "exploit: cost-min corner (c_x)")


def c4_bundle(**kw):
    defaults = dict(
        selection=selection(),
        feedback_text="No previous step -- this is the first modification.",
        archive_text="Archive holds only the baseline. No improvement yet.",
    )
    defaults.update(kw)
    return build_c4_prompt(
        bom(), bom()["parts"][0], DataRegistry(), **defaults
    )


# -- build_c4_prompt ---------------------------------------------


def test_c4_prompt_has_agentic_blocks_and_keeps_the_contract():
    b = c4_bundle()
    assert isinstance(b, PromptBundle)
    p = b.prompt
    assert "SELECTION (chosen by the search policy this step)" in p
    assert "EVALUATOR FEEDBACK (previous step)" in p
    assert "ARCHIVE STATE" in p
    assert "Target part: PILOT_001" in p
    assert "reduce the total BOM cost" in p
    # base proposal contract survives
    assert "proposal_id" in p
    assert "OUTPUT INSTRUCTIONS" in p
    assert "AVAILABLE CANONICAL IDENTIFIERS" in p


def test_c4_prompt_template_hash_differs_from_base():
    base = build_proposal_prompt(bom(), bom()["parts"][0], DataRegistry())
    b = c4_bundle()
    assert b.prompt_template_hash != base.prompt_template_hash
    assert "A13.C4.v1" in b.template_structure


def test_c4_prompt_metadata_carries_selection():
    b = c4_bundle(selection=selection("reduce_mass"))
    assert b.metadata["c4"] is True
    assert b.metadata["selection"] == {
        "part_id": "PILOT_001",
        "intent": "reduce_mass",
    }
    assert "reduce the total BOM mass" in b.prompt


def test_c4_prompt_rag_flag_follows_retrieved_context():
    assert c4_bundle().rag_enabled is False
    assert c4_bundle(retrieved_context="S 3.5.12 ...").rag_enabled is True


def test_intent_phrase_for_each_intent():
    for intent, phrase in [
        ("reduce_cost", "reduce the total BOM cost"),
        ("reduce_mass", "reduce the total BOM mass"),
        ("fix_violation", "remove a deterministic constraint violation"),
        ("diversify", "widen the search"),
    ]:
        assert phrase in c4_bundle(selection=selection(intent)).prompt


# -- feedback text ---------------------------------------------


def test_feedback_first_step():
    assert "first modification" in build_feedback_text(
        previous_evaluation=None
    )


def test_feedback_reports_deltas_and_acceptance():
    ev = {
        "objectives": {"cost_eur": 305.0, "mass_kg": 0.64},
        "constraints": {"status": "NOT_EVALUATED", "evaluated": False},
    }
    txt = build_feedback_text(
        previous_evaluation=ev,
        previous_selection=selection(),
        previous_accepted=True,
        baseline_vector=BASELINE,
    )
    assert "cost 305.00 EUR" in txt
    assert "+7.02" in txt  # 312.02 - 305.00
    assert "ACCEPTED" in txt
    assert "not evaluated on this benchmark" in txt


def test_feedback_reports_rejection_reason():
    ev = {"objectives": {"cost_eur": 320.0, "mass_kg": 0.70}}
    txt = build_feedback_text(
        previous_evaluation=ev,
        previous_accepted=False,
        previous_rejection_reason="dominated by the current front",
        baseline_vector=BASELINE,
    )
    assert "REJECTED (dominated by the current front)" in txt
    assert "materially different change" in txt


# -- archive text --------------------------------------------


def test_archive_text_empty():
    assert "only the baseline" in build_archive_text(
        archive_entries=[], baseline_vector=BASELINE
    )


def test_archive_text_with_entries():
    entries = [
        {"objective_vector": [150.0, 0.80]},
        {"objective_vector": [300.0, 0.55]},
    ]
    txt = build_archive_text(
        archive_entries=entries,
        baseline_vector=BASELINE,
        last_archive_status="pareto_improving",
    )
    assert "2 point(s)" in txt
    assert "Best cost so far: 150.00 EUR" in txt
    assert "Lowest mass so far: 0.55000 kg" in txt
    assert "pareto_improving" in txt


# -- generator override --------------------------------------


def test_generator_uses_prompt_bundle_override():
    backend = StubLLMBackend(
        json.dumps(
            {
                "proposal_id": "P1",
                "part_id": "PILOT_001",
                "change_type": "material",
                "target_field": "material_id",
                "new_value": "AL_7075_T6",
            }
        )
    )
    gen = ProposalGenerator(backend, registry=DataRegistry())
    bundle = c4_bundle()

    rec = gen.generate(
        bom=bom(),
        target_part=bom()["parts"][0],
        condition="C4",
        seed=0,
        prompt_bundle_override=bundle,
    )

    assert backend.calls[0]["prompt"] == bundle.prompt
    assert rec["prompt_hash"] == bundle.prompt_hash
    assert rec["parse_valid"] is True
