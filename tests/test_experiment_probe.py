from src.experiment.probe import (
    ADAPTER_REQUIRED_FILES,
    DEFAULT_ADAPTER_DIR,
    ProbeResult,
    check_adapter_files,
    check_imports,
    probe_c3,
)


# -- check_adapter_files -----------------------------------------


def _write_adapter(directory, files=ADAPTER_REQUIRED_FILES):
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        (directory / name).write_text("x", encoding="utf-8")


def test_adapter_files_ok_when_both_present(tmp_path):
    _write_adapter(tmp_path / "adapter")
    ok, missing = check_adapter_files(tmp_path / "adapter")
    assert ok is True
    assert missing == []


def test_adapter_files_reports_missing_weights(tmp_path):
    _write_adapter(tmp_path / "adapter", files=["adapter_config.json"])
    ok, missing = check_adapter_files(tmp_path / "adapter")
    assert ok is False
    assert missing == ["adapters.safetensors"]


def test_adapter_files_reports_all_missing_for_empty_dir(tmp_path):
    ok, missing = check_adapter_files(tmp_path / "nope")
    assert ok is False
    assert set(missing) == set(ADAPTER_REQUIRED_FILES)


def test_repo_adapter_is_present():
    """The committed models/c3_adapter/ must satisfy the file check."""
    ok, missing = check_adapter_files()
    assert ok is True, f"repo adapter missing {missing} at {DEFAULT_ADAPTER_DIR}"


# -- check_imports (reflects this interpreter) -----------------


def test_check_imports_returns_triple():
    ok, checks, error = check_imports()
    assert isinstance(ok, bool)
    assert isinstance(checks, dict)
    assert (error is None) == ok
    if ok:
        assert "mlx" in checks and "mlx_lm" in checks


# -- probe_c3 shallow -----------------------------------------


def test_probe_shallow_ok_when_env_ready(tmp_path):
    _write_adapter(tmp_path / "adapter")
    result = probe_c3(deep=False, adapter_dir=tmp_path / "adapter")

    imports_ok, _, _ = check_imports()
    if imports_ok:
        assert result.ok is True
        assert result.stage_reached == "adapter_files"
        assert result.verdict == "ready"
    else:
        assert result.ok is False
        assert result.stage_reached == "import"


def test_probe_shallow_blocked_on_missing_adapter(tmp_path):
    result = probe_c3(deep=False, adapter_dir=tmp_path / "absent")

    imports_ok, _, _ = check_imports()
    if imports_ok:
        assert result.ok is False
        assert result.stage_reached == "adapter_files"
        assert result.verdict == "blocked, environment"
        assert "adapters.safetensors" in result.detail
        assert result.remediation
    else:
        assert result.stage_reached == "import"


def test_probe_never_raises_on_bad_adapter_dir():
    # str path, nonexistent -- must return a result, not raise
    result = probe_c3(deep=False, adapter_dir="/definitely/not/here")
    assert isinstance(result, ProbeResult)
    assert result.ok is False


# -- ProbeResult shape -------------------------------------


def test_probe_result_as_dict_and_summary_for_ok():
    r = ProbeResult(
        ok=True,
        stage_reached="generate",
        detail="loaded and generated (12 chars).",
        checks={"mlx": "0.32.2", "mlx_lm": "0.31.3"},
    )
    assert r.verdict == "ready"
    assert "READY" in r.summary_line()
    d = r.as_dict()
    assert d["ok"] is True
    assert d["verdict"] == "ready"
    assert d["checks"]["mlx"] == "0.32.2"


def test_probe_result_summary_for_blocked_includes_fix():
    r = ProbeResult(
        ok=False,
        stage_reached="import",
        detail="MLX stack not importable",
        remediation="pip install mlx mlx-lm",
    )
    line = r.summary_line()
    assert "BLOCKED at 'import'" in line
    assert "fix: pip install mlx mlx-lm" in line
    assert r.as_dict()["verdict"] == "blocked, environment"
