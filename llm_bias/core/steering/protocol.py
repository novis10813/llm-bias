"""Frozen model registry, population split, chat-template pinning and run provenance.

Functions copied from ``scripts/probe_concept_cone.py`` (V2) keep identical
behavior; tests assert equivalence. Protocol: docs/concept-cone-steering/confirmation-v1/proposal.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

DECISION_PREFIX = '{"decision": "'
MAX_NEW_TOKENS = 192
POPULATION_CSV = "data/sp500_constituents_2020_2025.csv"
SPLIT_SEED = 20260923
UNSPECIFIED_SECTOR = "Unspecified"
# Harmony (GPT-OSS) writes strftime_now("%Y-%m-%d") into its system message; pin it to the
# V2 smoke date so every rendering in every run is identical regardless of wall-clock date.
TEMPLATE_DATE = datetime(2026, 9, 25)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    n_layers: int
    peak: int                 # C2-427 instruction peak = primary steering layer
    band: tuple[int, ...]     # C2-427 layers with T >= 0.7 * peak (Qwen: frozen V1 rule T >= 0.25)
    suffix_tokens: int        # common instruction-suffix length K across all renderings
    dtype: str                # "bf16" or "native"
    c2_sha256: str
    mlp: str                  # "dense" or "moe"
    down_proj_key: str        # safetensors key template of the MLP down projection ({layer})
    post_mlp_norm_key: str | None  # sandwich-norm gain applied to the MLP output, if any

    @property
    def last_layer(self) -> int:
        return self.n_layers - 1


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen3.5-4b": ModelSpec(
        "qwen3.5-4b", 32, 16, (14, 15, 16, 17, 18), 100, "bf16",
        "5f5da88738c10f784357fbd26994235400cc1ccd72f46495314748738e1733f7", "dense",
        "model.language_model.layers.{layer}.mlp.down_proj.weight", None),
    "gemma4-12b-it": ModelSpec(
        "gemma4-12b-it", 48, 27, (26, 27, 28, 29, 30, 31, 32), 100, "bf16",
        "295170b9c09ca453ce7b285c37a0f128d99cf68fc1a20970f390f084fdd54635", "dense",
        "model.language_model.layers.{layer}.mlp.down_proj.weight",
        "model.language_model.layers.{layer}.post_feedforward_layernorm.weight"),
    "glm4-9b-0414": ModelSpec(
        "glm4-9b-0414", 40, 19, (17, 18, 19, 20, 21), 98, "bf16",
        "d0b7e41ac21adc1cb858fbdc6a75e34338109cab84f3630bcd6f839119b480a6", "dense",
        "model.layers.{layer}.mlp.down_proj.weight", "model.layers.{layer}.post_mlp_layernorm.weight"),
    "gpt-oss-20b": ModelSpec(
        "gpt-oss-20b", 24, 14, (12, 13, 14, 15, 16), 99, "native",
        "f8ccb7c24d25048d358a32562ba8fffd799384a482b1e17152d7f0bef543a506", "moe",
        "model.layers.{layer}.mlp.experts.down_proj", None),
}


def model_spec(slug: str) -> ModelSpec:
    if slug not in MODEL_REGISTRY:
        raise ValueError(f"model {slug!r} is not in the confirmation-v1 registry {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[slug]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def load_population(path: Path) -> dict[str, dict[str, str]]:
    """The 503 unique 2024 S&P 500 constituents with name and GICS sector."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("year") == "2024" and r.get("index_name") == "S&P 500"]
    if len(rows) != 503 or len({r["ticker"] for r in rows}) != 503:
        raise ValueError(f"expected 503 unique 2024 S&P 500 tickers, got {len(rows)}")
    if any(not r.get("company_name") or not r.get("gics_sector") for r in rows):
        raise ValueError("missing company name or sector")
    return {r["ticker"]: {"ticker": r["ticker"], "name": r["company_name"], "sector": r["gics_sector"]} for r in rows}


def split_population(path: Path, seed: int) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """Disjoint 402/101 construction/evaluation split (identical to V2 ``split_population``)."""
    companies = load_population(path)
    shuffled = sorted(companies)
    random.Random(seed).shuffle(shuffled)
    construction, evaluation = sorted(shuffled[:402]), sorted(shuffled[402:])
    assert len(construction) == 402 and len(evaluation) == 101 and not set(construction) & set(evaluation)
    return companies, construction, evaluation


def split_sha256(construction: Sequence[str], evaluation: Sequence[str]) -> str:
    """Same payload as V2 metadata ``split_sha256``."""
    return sha256_bytes(json.dumps({"construction": list(construction), "evaluation": list(evaluation)},
                                   sort_keys=True).encode())


def low_reasoning_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Harmony templates read reasoning_effort (default medium); request low. JSON-serializable."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return {"reasoning_effort": "low"} if "reasoning_effort" in template else {}


def template_uses_date(tokenizer: Any) -> bool:
    return "strftime_now" in (getattr(tokenizer, "chat_template", None) or "")


def template_render_kwargs(tokenizer: Any) -> dict[str, Any]:
    """low_reasoning_kwargs plus a render-time ``strftime_now`` that shadows the Jinja global."""
    kwargs = low_reasoning_kwargs(tokenizer)
    if template_uses_date(tokenizer):
        kwargs["strftime_now"] = TEMPLATE_DATE.strftime
    return kwargs


def template_provenance(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None) or ""
    record: dict[str, Any] = {"chat_template_sha256": sha256_bytes(template.encode()),
                              "chat_template_kwargs": low_reasoning_kwargs(tokenizer)}
    if template_uses_date(tokenizer):
        record["chat_template_date"] = TEMPLATE_DATE.date().isoformat()
    return record


def c2_427_source(slug: str, root: Path = Path("artifacts")) -> dict[str, Any]:
    """Bind the primary layer to the persisted, SHA-frozen 427-company instruction curve."""
    spec = model_spec(slug)
    path = root / slug / "balanced-evidence-gap-phase2" / "runs" / "phase2b-v2-427-01" / "analyze" / "summary.json"
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if digest != spec.c2_sha256:
        raise ValueError(f"C2-427 summary SHA for {slug} differs from the frozen registry")
    curve = json.loads(payload)["curves"]["instruction"]
    if set(curve) != {str(i) for i in range(spec.n_layers)}:
        raise ValueError("C2 instruction curve does not cover all layers")
    transfer = {int(k): float(v["mean_normalized_transfer"]) for k, v in curve.items()}
    if max(transfer, key=transfer.get) != spec.peak or any(v["n_directions"] != 854 for v in curve.values()):
        raise ValueError(f"C2-427 source does not select L{spec.peak} for {slug}")
    return {"path": str(path), "sha256": digest, "instruction_peak": spec.peak, "n_directions": 854,
            "instruction_T": {str(k): transfer[k] for k in sorted(transfer)}}


SCOPED_PATHS = ("scripts", "llm_bias", "tests", "pyproject.toml", "uv.lock")


def git_provenance(cwd: Path | None = None) -> dict[str, Any]:
    """Commit plus a dirty flag scoped to code paths; other paths are summarized only."""
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True, cwd=cwd).rstrip("\n")

    try:
        commit = run("rev-parse", "HEAD").strip()
        scoped = run("status", "--porcelain", "--", *SCOPED_PATHS)
        other = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "code_dirty": None, "code_dirty_paths": [], "other_status_lines": None}
    scoped_lines = [line for line in scoped.splitlines() if line]
    return {"git_commit": commit, "code_dirty": bool(scoped_lines),
            "code_dirty_paths": sorted(line[3:] for line in scoped_lines),
            "other_status_lines": len([line for line in other.splitlines() if line]) - len(scoped_lines)}


def runtime_versions() -> dict[str, Any]:
    import torch
    import transformers

    record: dict[str, Any] = {"torch": torch.__version__, "transformers": transformers.__version__,
                              "cuda": torch.version.cuda}
    try:
        import jlens

        record["jlens"] = getattr(jlens, "__version__", "unknown")
    except ImportError:  # pragma: no cover - jlens is a workspace member
        record["jlens"] = None
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    return record


def checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"no safetensors checkpoint under {model_dir}")
    return {
        "model": str(model_dir.resolve()), "model_slug": model_dir.resolve().name,
        "model_config_sha256": sha256_bytes((model_dir / "config.json").read_bytes()),
        "tokenizer_config_sha256": sha256_bytes((model_dir / "tokenizer_config.json").read_bytes()),
        "checkpoint_files": [{"name": p.name, "bytes": p.stat().st_size} for p in weights],
    }


def file_sha256(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_bytes(Path(path).read_bytes()) for name, path in sorted(paths.items())}
