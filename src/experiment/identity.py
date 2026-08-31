"""
Run identity for the A12 experiment harness (thesis 11.4).

    RunID = H(B, R, S, M, A, P, Q, s, N, Git)                (eq 11.1)

where H is a deterministic canonical hash and

    B    benchmark version          (frozen B4 10-part pilot)
    R    constraint-rule snapshot   (deterministic + classification, evaluator version)
    S    supplier / source snapshot (frozen registries + real search space)
    M    base model                 (ollama llama3.1:8b, or MLX 4-bit base, or none)
    A    fine-tune adapter          (models/c3_adapter for C3, else none)
    P    prompt version             (prompt template structure hash)
    Q    retrieval configuration    (embedder, backend, top_k, corpus)
    s    random seed
    N    evaluation budget          (deterministic objective evaluations)
    Git  code commit

Any change to any element yields a new run_id. Runs with different
run_ids must never be merged into a single result set (11.4).
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.evaluator.unified_evaluator import EVALUATOR_VERSION
from src.constraint_engine.rule_router import route_b5_rules
from src.llm.prompt_builder import (
    PROMPT_TEMPLATE_STRUCTURE,
    sha256_json,
    sha256_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HARNESS_VERSION = "A12.1"
PROMPT_VERSION = "A10.1"

CONDITIONS = ("C1", "C2", "C3", "C5")
GENERATIVE_CONDITIONS = ("C1", "C2", "C3")

# C4 condition labels (docs/A13). The run's `condition` string is the
# grouping label; backend role + ablation are also in condition_spec.
C4_CANONICAL = "C4"          # fine-tuned (C3) backend + RAG + tool-loop
C4_BASE = "C4_base"          # Ollama base backend + RAG + tool-loop
C4_ABLATION_SUFFIXES = ("no_rag", "no_schema", "no_validator")
C4_ABLATION_LABELS = tuple(f"{C4_BASE}_{s}" for s in C4_ABLATION_SUFFIXES)
C4_LABELS = (C4_CANONICAL, C4_BASE) + C4_ABLATION_LABELS

C4_LOOK_BACK_L = 10          # eq 4.2 look-back window   (LOCKED A13 section 13)
C4_EPSILON_HV = 0.1          # eq 4.2 convergence epsilon (absolute HV)
C4_RETRY_CAP_K = 3           # per-selection retry cap
C4_SELECT_POLICY = "archive_guided_v1"
C4_FEEDBACK_MODE = "prev_eval+archive+rejection"
C4_PROMPT_VERSION = "A13.C4.v1"
C4_PROMPT_TEMPLATE_TAG = "A13.C4.v1"


def is_c4(condition: str) -> bool:
    return condition in C4_LABELS


def c4_ablation(condition: str) -> str | None:
    for suffix in C4_ABLATION_SUFFIXES:
        if condition == f"{C4_BASE}_{suffix}":
            return suffix
    return None


def _c4_backend_condition(condition: str) -> str:
    # canonical C4 reuses the C3 fine-tuned backend identity;
    # C4_base* reuses C2's Ollama + RAG identity.
    return "C3" if condition == C4_CANONICAL else "C2"


def c4_loop_spec(condition: str, n_eval: int) -> dict:
    return {
        "budget_definition": "deterministic_objective_evaluations",
        "n_eval": int(n_eval),
        "convergence": {
            "look_back_L": C4_LOOK_BACK_L,
            "epsilon_hv": C4_EPSILON_HV,
            "variant": "delta",
        },
        "retry_cap_K": C4_RETRY_CAP_K,
        "proposal_attempt_cap": PROPOSAL_ATTEMPT_CAP,
        "select_policy": C4_SELECT_POLICY,
        "feedback_mode": C4_FEEDBACK_MODE,
        "ablation": c4_ablation(condition),
    }

# Full thesis protocol (11.5): ten independent seeds, the same set
# across paired conditions. Calibration (docs/A12 section 9) put the
# whole C1/C2/C3/C5 sweep at ~60 min, so no reduced-seed compromise.
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_N_EVAL = 50
PROPOSAL_ATTEMPT_CAP = 1500

DECODE = {
    "temperature": 0.2,
    "max_tokens": 512,
    "seed_supported": True,
}

DUPLICATE_POLICY = (
    "cache_by_canonical_bom_hash; a cache hit does NOT consume budget"
)

# --- frozen input paths (relative to PROJECT_ROOT) --------------------

BENCHMARK_PATH = "data/benchmark/pilot_10_parts_ground_truth.json"
SEARCH_SPACE_PATH = "data/benchmark/real_search_space.json"
DETERMINISTIC_CONSTRAINTS_PATH = (
    "data/processed/deterministic_constraints_B5.json"
)
RULE_CLASSIFICATION_PATH = "data/processed/rule_classification_B5.json"
REGISTRY_FILES = (
    "data/processed/materials.csv",
    "data/processed/processes.csv",
    "data/processed/fasteners.csv",
    "data/processed/suppliers.csv",
)
CORPUS_PATH = "data/rag/corpus.jsonl"
CORPUS_MANIFEST_PATH = "data/rag/corpus_manifest.json"

# --- model / adapter constants --------------------------------------

OLLAMA_MODEL_ID = "llama3.1:8b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

MLX_BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
C3_ADAPTER_DIR = "models/c3_adapter"
# Training-dataset digest recorded in docs/A11_C3_TRAINING_STATUS.md.
C3_TRAINING_DATA_SHA256 = (
    "e7870a5729ca1dd6ce7e713a846e48250a2967be6a2ee651d98a1780ff4169ce"
)
C3_LORA = {
    "rank": 16,
    "scale": 20.0,
    "dropout": 0.05,
    "layers": 16,
    "iters": 300,
    "checkpoint": 300,
}

RETRIEVAL_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_VECTOR_BACKEND = "numpy_cosine"
RETRIEVAL_TOP_K = 5

# The run is the full thesis protocol: 10 seeds, no reduction (see
# DEFAULT_SEEDS). There is therefore no standing seed-count deviation;
# build_run_config defaults `deviations` to []. The runner may still pass
# per-run deviations (e.g. a C3 INVALID_CONFIGURATION note if the MLX
# probe fails on the host).


# --- hashing helpers -----------------------------------------------


def _abs(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of a file's raw bytes, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha(relative_path: str) -> str:
    absolute = _abs(relative_path)
    if not absolute.is_file():
        raise FileNotFoundError(
            f"Frozen input missing: {relative_path}"
        )
    return "sha256:" + sha256_file(absolute)


def _load_json(relative_path: str) -> dict:
    with _abs(relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_run_id(identity: dict) -> str:
    """
    Canonical hash of the full identity block (eq 11.1).

    Uses the repository's canonical JSON encoder so the value is
    stable across processes and machines.
    """
    return "sha256:" + sha256_json(identity)[:16]


# --- git -----------------------------------------------------------


def git_identity() -> dict:
    """Current commit plus a coarse dirty flag. Never raises."""
    def _run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = _run(["rev-parse", "HEAD"])
    porcelain = _run(["status", "--porcelain"])

    tracked_dirty = False
    untracked_files = 0
    if porcelain:
        for line in porcelain.splitlines():
            if line.startswith("??"):
                untracked_files += 1
            elif line.strip():
                tracked_dirty = True

    return {
        "commit": commit,
        "tracked_dirty": tracked_dirty,
        "untracked_files": untracked_files,
    }


# --- ollama ------------------------------------------------------


def ollama_model_digest(
    model_id: str = OLLAMA_MODEL_ID,
    base_url: str = OLLAMA_BASE_URL,
) -> str | None:
    """
    Full content digest of an installed Ollama model, or None if the
    server is unreachable or the model is absent. A None here means the
    run_config records model_digest=null and the identity is weaker;
    the runner warns but does not abort C1/C2 on this alone.
    """
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None

    for model in payload.get("models", []):
        if model.get("name") == model_id or model.get("model") == model_id:
            digest = model.get("digest")
            return str(digest) if digest else None
    return None


# --- the ten identity elements ------------------------------------


def build_benchmark_identity() -> dict:
    """B -- frozen B4 10-part pilot."""
    document = _load_json(BENCHMARK_PATH)
    return {
        "name": "B4_pilot_10_parts",
        "path": BENCHMARK_PATH,
        "version": "v2",
        "frozen": bool(document.get("frozen", False)),
        "frozen_date": document.get("frozen_date"),
        "sha256": _file_sha(BENCHMARK_PATH),
    }


def build_ruleset_identity() -> dict:
    """R -- constraint-rule snapshot and evaluator version."""
    routed = route_b5_rules()
    return {
        "deterministic_constraints_sha256": _file_sha(
            DETERMINISTIC_CONSTRAINTS_PATH
        ),
        "rule_classification_sha256": _file_sha(
            RULE_CLASSIFICATION_PATH
        ),
        "evaluator_version": EVALUATOR_VERSION,
        "routed_rule_counts": {
            "optimizer_rules": len(routed["optimizer_rules"]),
            "special_deterministic_rules": len(
                routed["special_deterministic_rules"]
            ),
            "compliance_rules": len(routed["compliance_rules"]),
            "review_rules": len(routed["review_rules"]),
            "quality_gates": len(routed["quality_gates"]),
            "no_active_constraints": len(
                routed["no_active_constraints"]
            ),
            "total_entries": routed["total_entries"],
        },
    }


def build_source_identity() -> dict:
    """S -- frozen registries and the engineering-reviewed search space."""
    search_space = _load_json(SEARCH_SPACE_PATH)
    return {
        "registry_files": {
            Path(relative).name: _file_sha(relative)
            for relative in REGISTRY_FILES
        },
        "search_space": {
            "path": SEARCH_SPACE_PATH,
            "schema_version": search_space.get("schema_version"),
            "status": search_space.get("status"),
            "sha256": _file_sha(SEARCH_SPACE_PATH),
        },
    }


def build_model_identity(condition: str) -> dict | None:
    """M -- base model. None for C5 (no model)."""
    if is_c4(condition):
        return build_model_identity(_c4_backend_condition(condition))
    if condition in ("C1", "C2"):
        return {
            "role": "base",
            "backend_name": "ollama",
            "model_id": OLLAMA_MODEL_ID,
            "model_digest": ollama_model_digest(),
            "decode": dict(DECODE),
        }
    if condition == "C3":
        return {
            "role": "fine_tuned",
            "backend_name": "mlx_lora",
            "model_id": MLX_BASE_MODEL,
            "model_digest": None,
            "decode": dict(DECODE),
        }
    return None


def build_adapter_identity(condition: str) -> dict:
    """A -- fine-tune adapter. C3 and canonical C4 carry one."""
    if condition == C4_CANONICAL:
        return build_adapter_identity("C3")
    if condition != "C3":
        return {"adapter_id": None}

    weights = f"{C3_ADAPTER_DIR}/adapters.safetensors"
    config = f"{C3_ADAPTER_DIR}/adapter_config.json"
    return {
        "adapter_id": C3_ADAPTER_DIR,
        "adapter_sha256": _file_sha(weights),
        "adapter_config_sha256": _file_sha(config),
        "base_model": MLX_BASE_MODEL,
        "lora": dict(C3_LORA),
        "training_data_sha256": C3_TRAINING_DATA_SHA256,
    }


def build_prompt_identity(condition: str) -> dict | None:
    """P -- prompt version. None for C5."""
    if condition == "C5":
        return None
    if is_c4(condition):
        return {
            "prompt_version": C4_PROMPT_VERSION,
            "prompt_template_sha256": sha256_text(
                PROMPT_TEMPLATE_STRUCTURE + "\n" + C4_PROMPT_TEMPLATE_TAG
            ),
            "builder_module": "src.llm.prompt_builder:build_c4_prompt",
        }
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": sha256_text(
            PROMPT_TEMPLATE_STRUCTURE
        ),
        "builder_module": "src.llm.prompt_builder",
    }


def build_retrieval_identity(condition: str) -> dict | None:
    """Q -- retrieval configuration."""
    if condition == "C5":
        return None
    if condition == "C1":
        return {"rag_enabled": False}
    if is_c4(condition):
        if c4_ablation(condition) == "no_rag":
            return {"rag_enabled": False}
        return build_retrieval_identity("C2")

    manifest = _load_json(CORPUS_MANIFEST_PATH)
    return {
        "rag_enabled": True,
        "embedder": RETRIEVAL_EMBEDDER,
        "vector_backend": RETRIEVAL_VECTOR_BACKEND,
        "top_k": RETRIEVAL_TOP_K,
        "corpus_path": CORPUS_PATH,
        "corpus_version": manifest.get("corpus_version"),
        "corpus_sha256": _file_sha(CORPUS_PATH),
    }


def build_budget(condition: str, n_eval: int) -> dict:
    """N -- evaluation budget, defined as deterministic objective evals."""
    if n_eval < 1:
        raise ValueError("n_eval must be a positive integer")
    return {
        "definition": "deterministic_objective_evaluations",
        "n_eval": int(n_eval),
        "duplicate_policy": DUPLICATE_POLICY,
        "proposal_attempt_cap": (
            PROPOSAL_ATTEMPT_CAP
            if (condition in GENERATIVE_CONDITIONS or is_c4(condition))
            else None
        ),
    }


# --- assembly ----------------------------------------------------


def build_identity(condition: str, seed: int, n_eval: int) -> dict:
    """
    The full eq 11.1 identity block. This dict -- and only this dict --
    determines the run_id. `created_utc` and other bookkeeping stay out
    of it so the id is reproducible.
    """
    if condition not in CONDITIONS and not is_c4(condition):
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of "
            f"{CONDITIONS} or a C4 label {C4_LABELS}"
        )

    identity = {
        "benchmark": build_benchmark_identity(),
        "ruleset_snapshot": build_ruleset_identity(),
        "source_snapshot": build_source_identity(),
        "model": build_model_identity(condition),
        "adapter": build_adapter_identity(condition),
        "prompt": build_prompt_identity(condition),
        "retrieval": build_retrieval_identity(condition),
        "seed": int(seed),
        "budget": build_budget(condition, n_eval),
        "git": git_identity(),
    }
    if is_c4(condition):
        # the tool-loop's identity-bearing method parameters -- so an
        # ablation (no_schema / no_validator, which do not touch M/A/P/Q)
        # and any change to the frozen L / eps / K produce a new run_id
        # (11.4, and 4.9's "frozen before final runs and reported").
        identity["method"] = {
            "driver": "C4Driver",
            "ablation": c4_ablation(condition),
            "select_policy": C4_SELECT_POLICY,
            "feedback_mode": C4_FEEDBACK_MODE,
            "convergence": {
                "look_back_L": C4_LOOK_BACK_L,
                "epsilon_hv": C4_EPSILON_HV,
            },
            "retry_cap_K": C4_RETRY_CAP_K,
        }
    return identity


def _condition_spec(
    condition: str,
    *,
    target_parts: list[str],
    nsga2_spec: dict | None,
    n_eval: int,
) -> dict:
    if is_c4(condition):
        return {
            "driver": "C4Driver",
            "backend_role": (
                "fine_tuned"
                if condition == C4_CANONICAL
                else "base"
            ),
            "generator_fn": "src.experiment.c4_driver.C4Driver",
            "decision_variables": ["material_id", "process_id"],
            "target_parts": list(target_parts),
            "proposal_application": "compounding_on_working_state",
            "c4_loop": c4_loop_spec(condition, n_eval),
            "nsga2": None,
        }

    if condition in GENERATIVE_CONDITIONS:
        generator_fn = {
            "C1": "src.llm.conditions.generate_c1",
            "C2": "src.llm.conditions.generate_c2",
            "C3": "src.llm.conditions.generate_c3",
        }[condition]
        return {
            "driver": "GenerativeDriver",
            "generator_fn": generator_fn,
            "decision_variables": ["material_id", "process_id"],
            "target_parts": list(target_parts),
            "proposal_application": "atomic_vs_frozen_baseline",
            "nsga2": None,
        }

    return {
        "driver": "Nsga2Driver",
        "generator_fn": None,
        "decision_variables": ["material_id", "process_id"],
        "target_parts": list(target_parts),
        "proposal_application": None,
        "nsga2": nsga2_spec
        or {
            "population_size": 20,
            "mutation_rate": 0.35,
            "reference_point_rule": "1.2x baseline (eq 11.10)",
        },
    }


def build_run_config(
    condition: str,
    seed: int,
    *,
    n_eval: int = DEFAULT_N_EVAL,
    target_parts: list[str],
    nsga2_spec: dict | None = None,
    deviations: list[dict] | None = None,
) -> dict:
    """
    Assemble the complete run_config.json payload for one
    runs/<condition>/seed_NN/ directory (docs/A12 section 3).
    """
    identity = build_identity(condition, seed, n_eval)
    return {
        "run_id": compute_run_id(identity),
        "condition": condition,
        "seed": int(seed),
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "harness_version": HARNESS_VERSION,
        "identity": identity,
        "condition_spec": _condition_spec(
            condition,
            target_parts=target_parts,
            nsga2_spec=nsga2_spec,
            n_eval=n_eval,
        ),
        "deviations": list(deviations) if deviations is not None else [],
    }


def write_run_config(run_config: dict, seed_dir: Path) -> Path:
    """
    Write run_config.json into runs/<condition>/seed_NN/.

    Refuses to overwrite an existing run_config.json whose run_id
    differs -- that is a changed experimental identity and must not be
    silently merged (11.4). An identical run_id is treated as a
    resumable no-op and left untouched.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    target = seed_dir / "run_config.json"

    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("run_id") != run_config["run_id"]:
            raise ValueError(
                f"{target} already exists with run_id "
                f"{existing.get('run_id')!r}, refusing to overwrite "
                f"with {run_config['run_id']!r} (thesis 11.4)"
            )
        return target

    target.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def seed_dir_name(seed: int) -> str:
    """Zero-padded seed directory name (docs/A12 section 2)."""
    return f"seed_{int(seed):02d}"
