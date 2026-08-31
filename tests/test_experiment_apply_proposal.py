import copy

from src.experiment.apply_proposal import apply_proposal


def baseline():
    return {
        "stage": "B4",
        "parts": [
            {
                "part_id": "PILOT_001",
                "name": "Pedal box mounting bracket",
                "material_id": "AL_6061_T6",
                "process_id": "CNC_MILLING",
                "dimensions_mm": {"length": 80, "width": 60, "thickness": 5},
                "volume_m3": 2.4e-05,
                "mass_kg": 0.0648,
                "fasteners": [{"fastener_id": "BOLT_M6X20", "qty": 4}],
                "manual_calculation": {"total_cost_eur": 25.32},
            },
            {
                "part_id": "PILOT_002",
                "name": "Suspension pickup plate",
                "material_id": "AL_7075_T6",
                "process_id": "CNC_MILLING",
                "volume_m3": 5.8e-05,
                "mass_kg": 0.15736,
            },
        ],
    }


def proposal(**overrides):
    base = {
        "proposal_id": "PROP_001",
        "part_id": "PILOT_001",
        "change_type": "material",
        "target_field": "material_id",
        "old_value": "AL_6061_T6",
        "new_value": "AL_7075_T6",
    }
    base.update(overrides)
    return base


# -- happy path ----------------------------------------------------


def test_material_swap_is_applicable():
    result = apply_proposal(baseline(), proposal())

    assert result.applicability_valid is True
    assert result.is_noop is False
    assert result.bom["parts"][0]["material_id"] == "AL_7075_T6"
    assert result.target_field == "material_id"
    assert result.modifications == [
        {
            "part_id": "PILOT_001",
            "field": "material_id",
            "baseline": "AL_6061_T6",
            "candidate": "AL_7075_T6",
        }
    ]


def test_process_swap_is_applicable():
    result = apply_proposal(
        baseline(),
        proposal(
            change_type="process",
            target_field="process_id",
            old_value="CNC_MILLING",
            new_value="TIG_WELDING",
        ),
    )

    assert result.applicability_valid is True
    assert result.bom["parts"][0]["process_id"] == "TIG_WELDING"
    assert result.modifications[0]["field"] == "process_id"


def test_only_the_target_part_changes():
    result = apply_proposal(baseline(), proposal())

    assert result.bom["parts"][1] == baseline()["parts"][1]
    assert result.bom["parts"][0]["process_id"] == "CNC_MILLING"
    assert result.bom["stage"] == "B4"


# -- input is never mutated -------------------------------------


def test_baseline_is_not_mutated():
    original = baseline()
    snapshot = copy.deepcopy(original)

    apply_proposal(original, proposal())

    assert original == snapshot


def test_candidate_is_deep_independent_of_baseline():
    original = baseline()
    result = apply_proposal(original, proposal())

    result.bom["parts"][0]["material_id"] = "MUTATED"
    original["parts"][0]["dimensions_mm"]["length"] = 999

    assert original["parts"][0]["material_id"] == "AL_6061_T6"
    assert result.bom["parts"][0]["dimensions_mm"]["length"] == 80


# -- no-op --------------------------------------------------------


def test_noop_proposal_is_applicable_but_records_no_modification():
    result = apply_proposal(
        baseline(),
        proposal(new_value="AL_6061_T6"),  # same as current
    )

    assert result.applicability_valid is True
    assert result.is_noop is True
    assert result.modifications == []
    assert result.bom["parts"][0]["material_id"] == "AL_6061_T6"


# -- unknown part ---------------------------------------------


def test_unknown_part_id_is_not_applicable():
    result = apply_proposal(
        baseline(), proposal(part_id="PILOT_999")
    )

    assert result.applicability_valid is False
    assert result.bom is None
    assert "unknown part_id" in result.errors[0]


# -- protected ground-truth fields ---------------------------


def test_mass_kg_write_is_blocked_as_protected():
    result = apply_proposal(
        baseline(),
        proposal(
            change_type="geometry",
            target_field="mass_kg",
            new_value="0.01",
        ),
    )

    assert result.applicability_valid is False
    assert result.protected_field_writes == ["mass_kg"]
    assert result.bom is None


def test_volume_m3_write_is_blocked_as_protected():
    result = apply_proposal(
        baseline(),
        proposal(
            change_type="geometry",
            target_field="volume_m3",
            new_value="1e-6",
        ),
    )

    assert result.applicability_valid is False
    assert result.protected_field_writes == ["volume_m3"]


def test_part_id_rewrite_is_blocked_as_protected():
    result = apply_proposal(
        baseline(),
        proposal(
            change_type="geometry",
            target_field="part_id",
            new_value="PILOT_042",
        ),
    )

    assert result.applicability_valid is False
    assert result.protected_field_writes == ["part_id"]


# -- out of scope (valid target, but not an A12 decision variable) --


def test_fastener_change_is_out_of_scope_not_protected():
    result = apply_proposal(
        baseline(),
        proposal(
            change_type="fastener",
            target_field="fasteners",
            new_value=[{"fastener_id": "BOLT_M8X25", "qty": 2}],
        ),
    )

    assert result.applicability_valid is False
    assert result.protected_field_writes == []
    assert "not an A12 optimisation variable" in result.errors[0]


def test_raw_stock_change_is_out_of_scope():
    result = apply_proposal(
        baseline(),
        proposal(
            change_type="raw_stock",
            target_field="raw_stock_form",
            new_value="bar",
        ),
    )

    assert result.applicability_valid is False
    assert result.protected_field_writes == []


# -- inconsistent change_type / target_field ----------------


def test_change_type_target_field_mismatch_is_not_applicable():
    result = apply_proposal(
        baseline(),
        proposal(change_type="material", target_field="process_id"),
    )

    assert result.applicability_valid is False
    assert "requires target_field 'material_id'" in result.errors[0]


# -- malformed new_value -----------------------------------


def test_non_string_new_value_for_material_is_not_applicable():
    result = apply_proposal(
        baseline(),
        proposal(new_value={"material_id": "AL_7075_T6"}),
    )

    assert result.applicability_valid is False
    assert "must be a non-empty string" in result.errors[0]


def test_empty_new_value_is_not_applicable():
    result = apply_proposal(baseline(), proposal(new_value=""))

    assert result.applicability_valid is False
