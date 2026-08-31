"""
C3 environment probe (docs/A12 section 10, decision Q2).

The C3 condition needs the MLX LoRA stack and the trained adapter. A11
recorded a working load on a prior environment; this probe re-checks it
on *this* host before a sweep, so a missing dependency is a clean
``blocked, environment`` verdict up front rather than a failure
discovered mid-run.

The MLX imports live inside the functions: this module (and the whole
``src.experiment`` package) imports fine on a machine without MLX.

Verdict handling by the runner (step 8):
  ok        -> run C3 normally
  not ok    -> skip C3, report it "blocked, environment", still run
               C1/C2/C5; a written C3 run is marked
               terminal_status = INVALID_CONFIGURATION (11.18).
"""

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.experiment.identity import (
    C3_ADAPTER_DIR,
    MLX_BASE_MODEL,
    PROJECT_ROOT,
)


PROBE_STAGES = ("import", "adapter_files", "model_load", "generate")

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / C3_ADAPTER_DIR
ADAPTER_REQUIRED_FILES = ("adapter_config.json", "adapters.safetensors")

PROBE_PROMPT = "Reply with the single token: OK"


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    stage_reached: str
    detail: str
    checks: dict = field(default_factory=dict)
    remediation: str | None = None

    @property
    def verdict(self) -> str:
        return "ready" if self.ok else "blocked, environment"

    def summary_line(self) -> str:
        head = f"C3 environment probe: {'READY' if self.ok else 'BLOCKED'}"
        if self.ok:
            return f"{head} (reached '{self.stage_reached}') -- {self.detail}"
        tail = f"  fix: {self.remediation}" if self.remediation else ""
        return (
            f"{head} at '{self.stage_reached}' -- {self.detail}{tail}"
        )

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stage_reached": self.stage_reached,
            "verdict": self.verdict,
            "detail": self.detail,
            "checks": dict(self.checks),
            "remediation": self.remediation,
        }


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "ok"


def check_imports() -> tuple[bool, dict, str | None]:
    """(ok, checks, error). Reflects the current interpreter."""
    checks: dict = {}
    try:
        import mlx.core  # noqa: F401

        checks["mlx"] = _pkg_version("mlx")
    except Exception as exc:  # ImportError, and MLX's own load errors
        return False, checks, f"mlx: {type(exc).__name__}: {exc}"

    try:
        import mlx_lm  # noqa: F401

        checks["mlx_lm"] = _pkg_version("mlx-lm")
    except Exception as exc:
        return False, checks, f"mlx_lm: {type(exc).__name__}: {exc}"

    return True, checks, None


def check_adapter_files(
    adapter_dir: Path | str | None = None,
) -> tuple[bool, list[str]]:
    """(ok, missing_filenames)."""
    directory = Path(adapter_dir) if adapter_dir else DEFAULT_ADAPTER_DIR
    missing = [
        name
        for name in ADAPTER_REQUIRED_FILES
        if not (directory / name).is_file()
    ]
    return (not missing), missing


def probe_c3(
    *,
    deep: bool = True,
    adapter_dir: Path | str | None = None,
    base_model: str | None = None,
) -> ProbeResult:
    """
    Check whether C3 can run on this host.

    deep=False : imports + adapter files only (fast, no model load).
    deep=True  : also construct the real MLXLoRABackend and run one
                 short generation -- the exact path a C3 sweep uses.
    Never raises.
    """
    directory = Path(adapter_dir) if adapter_dir else DEFAULT_ADAPTER_DIR
    model_id = base_model or MLX_BASE_MODEL

    imports_ok, checks, error = check_imports()
    if not imports_ok:
        return ProbeResult(
            ok=False,
            stage_reached="import",
            detail=f"MLX stack not importable ({error})",
            checks=checks,
            remediation=(
                "Install the MLX stack in this environment "
                "(Apple Silicon only): "
                "pip install 'mlx==0.32.2' 'mlx-lm==0.31.3'."
            ),
        )

    files_ok, missing = check_adapter_files(directory)
    checks["adapter_files"] = "ok" if files_ok else f"missing {missing}"
    if not files_ok:
        return ProbeResult(
            ok=False,
            stage_reached="adapter_files",
            detail=(
                f"C3 adapter incomplete under {directory}: missing {missing}"
            ),
            checks=checks,
            remediation=(
                "Restore models/c3_adapter/ (adapter_config.json + "
                "adapters.safetensors); see docs/A11_C3_TRAINING_STATUS.md."
            ),
        )

    if not deep:
        return ProbeResult(
            ok=True,
            stage_reached="adapter_files",
            detail=(
                "MLX importable and C3 adapter files present "
                "(model not loaded; pass deep=True to load and generate)."
            ),
            checks=checks,
        )

    try:
        from src.llm.backend import MLXLoRABackend

        backend = MLXLoRABackend(
            model_name=model_id,
            adapter_path=str(directory),
        )
        text = backend.generate(
            PROBE_PROMPT,
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as exc:
        return ProbeResult(
            ok=False,
            stage_reached="model_load",
            detail=(
                "loading base model + adapter failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            checks={**checks, "model_load": "failed"},
            remediation=(
                f"Ensure the 4-bit base model '{model_id}' is in the "
                "Hugging Face cache (or the network is up for a first "
                "download) and there is enough unified memory "
                "(~6 GB peak, per A11)."
            ),
        )

    checks["model_load"] = "ok"

    if not isinstance(text, str) or text.strip() == "":
        return ProbeResult(
            ok=False,
            stage_reached="generate",
            detail=(
                "backend loaded but generation returned "
                f"{type(text).__name__}: {str(text)[:60]!r}"
            ),
            checks={**checks, "generate": "empty"},
            remediation=(
                "mlx-lm generate returned nothing; check mlx-lm "
                "compatibility with the adapter format."
            ),
        )

    checks["generate"] = "ok"
    return ProbeResult(
        ok=True,
        stage_reached="generate",
        detail=(
            f"base model + C3 adapter loaded and generated "
            f"({len(text)} chars)."
        ),
        checks=checks,
    )


if __name__ == "__main__":
    import sys

    result = probe_c3(deep=True)
    print(result.summary_line())
    for name, value in result.checks.items():
        print(f"  {name}: {value}")
    sys.exit(0 if result.ok else 1)
