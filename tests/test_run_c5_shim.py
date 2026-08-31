import json
from pathlib import Path

import pytest

from scripts.run_c5_real_pilot import main
from src.experiment.drivers import Nsga2Driver
from src.experiment.events import EventLog
from src.experiment.identity import build_run_config
from src.optimization.search_space import load_verified_real_search_space


PILOT_BOM_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"
SEARCH_SPACE_PATH = "data/benchmark/real_search_space.json"


def test_shim_writes_unified_layout(tmp_path):
    rc = main(["--seed", "0", "--budget", "12", "--output-dir", str(tmp_path)])
    assert rc == 0

    run_dir = tmp_path / "runs" / "C5" / "seed_00"
    for name in (
        "run_config.json",
        "events.jsonl",
        "pareto_archive.json",
        "metrics.json",
    ):
        assert (run_dir / name).is_file(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["condition"] == "C5"
    assert metrics["budget"]["n_eval_consumed"] == 12
    assert metrics["terminal_status"] == "COMPLETE"


def test_shim_matches_direct_driver_on_the_same_seed(tmp_path):
    # shim
    main(["--seed", "3", "--budget", "16", "--output-dir", str(tmp_path)])
    shim_metrics = json.loads(
        (tmp_path / "runs" / "C5" / "seed_03" / "metrics.json").read_text()
    )

    # direct Nsga2Driver, same seed + budget
    cfg = build_run_config("C5", seed=3, n_eval=16, target_parts=[])
    baseline = json.loads(Path(PILOT_BOM_PATH).read_text())
    ss = load_verified_real_search_space(SEARCH_SPACE_PATH, PILOT_BOM_PATH)
    log = EventLog(
        tmp_path / "direct.jsonl", run_id=cfg["run_id"], condition="C5", seed=3
    )
    outcome = Nsga2Driver(
        cfg, baseline_bom=baseline, search_space=ss
    ).run(log, pareto_archive_path=tmp_path / "direct_pareto.json")
    log.close()

    assert shim_metrics["multiobjective"]["hypervolume"] == pytest.approx(
        outcome.hypervolume
    )
    assert (
        shim_metrics["multiobjective"]["pareto_archive_size"]
        == outcome.archive_size
    )
    assert shim_metrics["budget"]["n_eval_consumed"] == 16


def test_shim_rejects_non_positive_budget(tmp_path):
    assert main(
        ["--seed", "0", "--budget", "0", "--output-dir", str(tmp_path)]
    ) == 2


def test_shim_overwrite_allows_rerun(tmp_path):
    args = ["--seed", "0", "--budget", "10", "--output-dir", str(tmp_path)]
    assert main(args) == 0
    assert main(args + ["--overwrite"]) == 0
