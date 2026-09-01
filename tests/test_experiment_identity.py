import json

import pytest

from src.experiment.identity import (
    CONDITIONS,
    build_identity,
    build_run_config,
    c4_attempt_cap,
    compute_run_id,
    seed_dir_name,
    write_run_config,
)


TARGET_PARTS = [f"PILOT_{index:03d}" for index in range(1, 11)]


def run_config(condition, seed=0, n_eval=50):
    return build_run_config(
        condition,
        seed=seed,
        n_eval=n_eval,
        target_parts=TARGET_PARTS,
    )


def test_c4_attempt_cap_is_tight_with_a_floor():
    assert c4_attempt_cap(50) == 300          # 6 * N
    assert c4_attempt_cap(10) == 150          # floor wins
    assert c4_attempt_cap(1) == 150
    assert c4_attempt_cap(50) < 1500          # far below the generative cap


def test_c4_loop_and_budget_agree_on_the_attempt_cap():
    cfg = run_config("C4_base")
    assert cfg["condition_spec"]["c4_loop"]["proposal_attempt_cap"] == 300
    assert cfg["identity"]["budget"]["proposal_attempt_cap"] == 300


def test_all_conditions_build_a_run_config():
    for condition in CONDITIONS:
        config = run_config(condition)
        assert config["condition"] == condition
        assert config["run_id"].startswith("sha256:")
        assert config["identity"]["seed"] == 0
        assert config["identity"]["budget"]["n_eval"] == 50


def test_run_id_is_stable_for_identical_identity():
    first = compute_run_id(build_identity("C2", 0, 50))
    second = compute_run_id(build_identity("C2", 0, 50))
    assert first == second


def test_run_id_changes_with_seed():
    assert compute_run_id(build_identity("C2", 0, 50)) != compute_run_id(
        build_identity("C2", 1, 50)
    )


def test_run_id_changes_with_budget():
    assert compute_run_id(build_identity("C2", 0, 50)) != compute_run_id(
        build_identity("C2", 0, 100)
    )


def test_run_id_changes_between_conditions():
    ids = {
        condition: run_config(condition)["run_id"]
        for condition in CONDITIONS
    }
    assert len(set(ids.values())) == len(CONDITIONS)


def test_created_utc_is_not_part_of_identity():
    identity = build_identity("C1", 0, 50)
    assert "created_utc" not in identity
    assert "created_utc" not in json.dumps(identity)


def test_c3_carries_adapter_identity():
    adapter = run_config("C3")["identity"]["adapter"]
    assert adapter["adapter_id"] == "models/c3_adapter"
    assert adapter["adapter_sha256"].startswith("sha256:")
    assert adapter["base_model"].endswith("Instruct-4bit")


def test_generative_conditions_have_no_adapter():
    for condition in ("C1", "C2"):
        assert run_config(condition)["identity"]["adapter"] == {
            "adapter_id": None
        }


def test_c1_has_no_retrieval_config():
    assert run_config("C1")["identity"]["retrieval"] == {
        "rag_enabled": False
    }


def test_c2_and_c3_share_retrieval_config():
    c2 = run_config("C2")["identity"]["retrieval"]
    c3 = run_config("C3")["identity"]["retrieval"]
    assert c2["rag_enabled"] is True
    assert c2["top_k"] == 5
    assert c2["corpus_sha256"] == c3["corpus_sha256"]


def test_c5_has_no_model_prompt_or_retrieval():
    identity = run_config("C5")["identity"]
    assert identity["model"] is None
    assert identity["prompt"] is None
    assert identity["retrieval"] is None
    assert identity["budget"]["proposal_attempt_cap"] is None


def test_c5_uses_the_nsga2_driver():
    spec = run_config("C5")["condition_spec"]
    assert spec["driver"] == "Nsga2Driver"
    assert spec["nsga2"]["population_size"] == 20


def test_generative_conditions_use_the_generative_driver():
    for condition in ("C1", "C2", "C3"):
        spec = run_config(condition)["condition_spec"]
        assert spec["driver"] == "GenerativeDriver"
        assert spec["proposal_application"] == "atomic_vs_frozen_baseline"
        assert spec["generator_fn"].endswith(condition.lower())


def test_full_protocol_has_no_default_deviations():
    # Running the full 10-seed thesis protocol -> no standing deviation.
    assert run_config("C1")["deviations"] == []


def test_explicit_deviations_are_carried_through():
    note = {"id": "C3_ENV", "detail": "MLX probe failed"}
    cfg = build_run_config(
        "C3", seed=0, n_eval=50, target_parts=TARGET_PARTS, deviations=[note]
    )
    assert cfg["deviations"] == [note]


def test_default_seeds_is_ten():
    from src.experiment.identity import DEFAULT_SEEDS

    assert DEFAULT_SEEDS == tuple(range(10))


def test_seed_dir_name_is_zero_padded():
    assert seed_dir_name(0) == "seed_00"
    assert seed_dir_name(7) == "seed_07"


def test_write_run_config_round_trips(tmp_path):
    config = run_config("C1")
    seed_dir = tmp_path / "C1" / seed_dir_name(0)
    written = write_run_config(config, seed_dir)
    assert written.is_file()
    assert json.loads(written.read_text())["run_id"] == config["run_id"]


def test_write_run_config_refuses_identity_mismatch(tmp_path):
    seed_dir = tmp_path / "C2" / seed_dir_name(0)
    write_run_config(run_config("C2", n_eval=50), seed_dir)

    with pytest.raises(ValueError, match="11.4"):
        write_run_config(run_config("C2", n_eval=100), seed_dir)


def test_write_run_config_is_idempotent_for_same_identity(tmp_path):
    seed_dir = tmp_path / "C2" / seed_dir_name(0)
    config = run_config("C2")
    first = write_run_config(config, seed_dir)
    mtime = first.stat().st_mtime_ns
    second = write_run_config(config, seed_dir)
    assert second == first
    assert second.stat().st_mtime_ns == mtime
