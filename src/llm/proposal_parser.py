import json
import re


class ProposalParseError(ValueError):
    pass


FENCED_JSON_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.DOTALL,
)


def parse_proposal(
    text: str,
) -> dict:
    if not text or not text.strip():
        raise ProposalParseError(
            "LLM output is empty"
        )

    stripped = text.strip()
    match = FENCED_JSON_RE.match(
        stripped
    )

    if match:
        stripped = match.group(1).strip()
    elif "```" in stripped:
        raise ProposalParseError(
            "Malformed fenced JSON block"
        )

    decoder = json.JSONDecoder()

    try:
        parsed, end = decoder.raw_decode(
            stripped
        )
    except json.JSONDecodeError as exc:
        raise ProposalParseError(
            f"Invalid JSON: {exc.msg}"
        ) from exc

    if stripped[end:].strip():
        raise ProposalParseError(
            "Multiple JSON objects or trailing text detected"
        )

    if not isinstance(parsed, dict):
        raise ProposalParseError(
            "Proposal must be a JSON object"
        )

    return parsed
