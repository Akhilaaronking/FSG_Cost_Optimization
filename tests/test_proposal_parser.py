import pytest

from src.llm.proposal_parser import (
    ProposalParseError,
    parse_proposal,
)


VALID_JSON = (
    '{"proposal_id":"P","part_id":"PILOT_001",'
    '"change_type":"material","target_field":"material_id",'
    '"new_value":"AL_7075_T6"}'
)


def test_valid_json_parses():
    assert parse_proposal(
        VALID_JSON
    )["proposal_id"] == "P"


def test_fenced_json_parses():
    assert parse_proposal(
        f"```json\n{VALID_JSON}\n```"
    )["part_id"] == "PILOT_001"


def test_malformed_json_rejected():
    with pytest.raises(
        ProposalParseError,
        match="Invalid JSON",
    ):
        parse_proposal("{not json")


def test_multiple_json_objects_rejected():
    with pytest.raises(
        ProposalParseError,
        match="Multiple JSON objects",
    ):
        parse_proposal("{} {}")


def test_prose_only_rejected():
    with pytest.raises(
        ProposalParseError,
    ):
        parse_proposal("I propose changing the part.")
