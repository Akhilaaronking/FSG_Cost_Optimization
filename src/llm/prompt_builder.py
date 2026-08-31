import hashlib
import json

from src.data.registry import DataRegistry
from src.llm.models import (
    PromptBundle,
)


PROMPT_TEMPLATE_STRUCTURE = """
SYSTEM ROLE
TASK
CURRENT PART/BOM STATE
ALLOWED OUTPUT SCHEMA
AVAILABLE CANONICAL IDENTIFIERS
CONSTRAINT/SAFETY INSTRUCTIONS
OPTIONAL RETRIEVED CONTEXT
OUTPUT INSTRUCTIONS
""".strip()


def sha256_text(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_json(
    value,
) -> str:
    return sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _part_ids(
    bom: dict,
) -> list[str]:
    return sorted(
        part["part_id"]
        for part in bom.get("parts", [])
        if part.get("part_id")
    )


def _allowed_ids_for_part(
    target_part: dict,
    registry: DataRegistry,
    search_space: dict | None,
) -> tuple[dict, str]:
    if search_space:
        for part_space in search_space.get(
            "parts",
            [],
        ):
            if (
                part_space.get("part_id")
                == target_part.get("part_id")
            ):
                return {
                    "material_ids": sorted(
                        part_space.get(
                            "material_choices",
                            [],
                        )
                    ),
                    "process_ids": sorted(
                        part_space.get(
                            "process_choices",
                            [],
                        )
                    ),
                    "fastener_ids": sorted({
                        choice["fastener_id"]
                        for choices in part_space.get(
                            "fastener_choices",
                            {},
                        ).values()
                        for choice in choices
                    }),
                }, "EXPLICIT_SEARCH_SPACE"

    return {
        "material_ids": sorted(
            registry.materials
        ),
        "process_ids": sorted(
            registry.processes
        ),
        "fastener_ids": sorted(
            registry.fasteners
        ),
    }, "SOFTWARE_IDENTIFIER_VALIDITY_ONLY"


def _proposal_schema_summary() -> dict:
    return {
        "proposal_id": "string",
        "part_id": "string",
        "change_type": [
            "material",
            "process",
            "raw_stock",
            "geometry",
            "fastener",
        ],
        "target_field": "string",
        "old_value": "optional",
        "new_value": "any JSON value",
        "reasoning_summary": (
            "optional concise summary, max 1000 chars"
        ),
    }


def build_proposal_prompt(
    bom: dict,
    target_part: dict,
    registry: DataRegistry | None = None,
    retrieved_context: str | None = None,
    search_space: dict | None = None,
) -> PromptBundle:
    registry = registry or DataRegistry()
    allowed_ids, allowed_scope = _allowed_ids_for_part(
        target_part,
        registry,
        search_space,
    )
    rag_enabled = bool(
        retrieved_context
    )

    context_text = (
        retrieved_context
        if retrieved_context
        else "NO RETRIEVED CONTEXT PROVIDED."
    )

    prompt = f"""
SYSTEM ROLE
You are a controlled proposal-generation component for a Master's thesis software pipeline.

TASK
Propose exactly ONE atomic BOM/design modification for the target part.

CURRENT PART/BOM STATE
BOM part IDs: {json.dumps(_part_ids(bom), sort_keys=True)}
Target part JSON: {json.dumps(target_part, sort_keys=True)}

ALLOWED OUTPUT SCHEMA
Return one JSON object matching this schema summary:
{json.dumps(_proposal_schema_summary(), sort_keys=True)}

AVAILABLE CANONICAL IDENTIFIERS
Identifier scope: {allowed_scope}
Registry/search-space material IDs: {json.dumps(allowed_ids["material_ids"], sort_keys=True)}
Registry/search-space process IDs: {json.dumps(allowed_ids["process_ids"], sort_keys=True)}
Registry/search-space fastener IDs: {json.dumps(allowed_ids["fastener_ids"], sort_keys=True)}
Registry membership validates identifier existence only. It does not prove engineering interchangeability.

CONSTRAINT/SAFETY INSTRUCTIONS
Use only canonical identifiers supplied in this prompt.
Do not invent materials, processes, fasteners, suppliers, Formula Student rules, or engineering facts.
Do not calculate final authoritative cost or mass.
Cost and mass will be recomputed by deterministic tools.
Do not claim FSG compliance.
If evidence is insufficient, prefer a conservative proposal.
reasoning_summary must be concise and must not contain hidden chain-of-thought.

OPTIONAL RETRIEVED CONTEXT
{context_text}

OUTPUT INSTRUCTIONS
Output exactly ONE valid JSON object and nothing else.
Do not include Markdown fences.
Do not include prose before or after the JSON.
Do not include a chain_of_thought field.

The output MUST contain:
- proposal_id
- part_id
- change_type
- target_field
- new_value
- reasoning_summary

Type rules:
- For change_type "material":
  target_field MUST be "material_id".
  new_value MUST be a canonical material ID STRING.
  Example: "new_value": "AL_7075_T6"
  NEVER use {{"material_id": "..."}} as new_value.

- For change_type "process":
  target_field MUST be "process_id".
  new_value MUST be a canonical process ID STRING.
  Example: "new_value": "CNC_MILLING"
  NEVER use {{"process_id": "..."}} as new_value.

- For change_type "fastener":
  target_field MUST be "fasteners".
  Use only canonical fastener IDs.

proposal_id and part_id MUST be strings.
part_id MUST equal the target part ID supplied above.
Do not omit target_field.
""".strip()

    return PromptBundle(
        prompt=prompt,
        prompt_template_hash=sha256_text(
            PROMPT_TEMPLATE_STRUCTURE
        ),
        prompt_hash=sha256_text(prompt),
        template_structure=PROMPT_TEMPLATE_STRUCTURE,
        rag_enabled=rag_enabled,
        metadata={
            "allowed_identifier_scope": allowed_scope,
            "benchmark_input_hash": sha256_json({
                "bom_part_ids": _part_ids(bom),
                "target_part": target_part,
            }),
        },
    )


# --- C4 agentic-loop prompt (docs/A13 section 3) ---------------------

C4_TEMPLATE_TAG = "A13.C4.v1"

_C4_ANCHOR = (
    "TASK\n"
    "Propose exactly ONE atomic BOM/design modification for the "
    "target part."
)

_C4_INTENT_PHRASE = {
    "reduce_cost": "reduce the total BOM cost",
    "reduce_mass": "reduce the total BOM mass",
    "fix_violation": (
        "change this part to remove a deterministic constraint violation"
    ),
    "diversify": (
        "make a different admissible change to this part to widen the "
        "search"
    ),
}


def build_c4_prompt(
    bom: dict,
    target_part: dict,
    registry: DataRegistry | None = None,
    *,
    selection,
    feedback_text: str,
    archive_text: str,
    retrieved_context: str | None = None,
    search_space: dict | None = None,
) -> PromptBundle:
    """
    C4 step prompt: the C1/C2/C3 proposal contract plus the agentic-loop
    context (which part the policy chose and why, the evaluator feedback
    from the previous step, and the current archive state). Applied to
    the *working state* ``bom`` (x_t), not the frozen baseline.

    Returns a PromptBundle so it drops into
    ``ProposalGenerator.generate(prompt_bundle_override=...)``.
    """
    base = build_proposal_prompt(
        bom,
        target_part,
        registry,
        retrieved_context=retrieved_context,
        search_space=search_space,
    )

    if _C4_ANCHOR not in base.prompt:
        raise RuntimeError(
            "build_proposal_prompt template changed; C4 anchor not found"
        )

    goal = _C4_INTENT_PHRASE.get(selection.intent, selection.intent)
    c4_block = (
        f"{_C4_ANCHOR}\n"
        f"This is one step of a deterministic tool-loop search. The "
        f"modification is applied to the CURRENT working BOM below "
        f"(which already contains earlier accepted changes), then "
        f"re-evaluated.\n\n"
        f"SELECTION (chosen by the search policy this step)\n"
        f"Target part: {selection.part_id}\n"
        f"Goal: {goal}\n"
        f"Policy note: {selection.policy_reason}\n\n"
        f"EVALUATOR FEEDBACK (previous step)\n"
        f"{feedback_text}\n\n"
        f"ARCHIVE STATE\n"
        f"{archive_text}"
    )

    prompt = base.prompt.replace(_C4_ANCHOR, c4_block, 1)

    return PromptBundle(
        prompt=prompt,
        prompt_template_hash=sha256_text(
            PROMPT_TEMPLATE_STRUCTURE + "\n" + C4_TEMPLATE_TAG
        ),
        prompt_hash=sha256_text(prompt),
        template_structure=(
            PROMPT_TEMPLATE_STRUCTURE + "\n" + C4_TEMPLATE_TAG
        ),
        rag_enabled=base.rag_enabled,
        metadata={
            **base.metadata,
            "c4": True,
            "selection": {
                "part_id": selection.part_id,
                "intent": selection.intent,
            },
        },
    )
