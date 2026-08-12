"""Pinned pretrained Jacobian-lens registry and exact model identities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PRETRAINED_LENS_REGISTRY = Path("config/pretrained_lenses.json")
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelIdentity:
    base_model: str
    architecture: str
    d_model: int
    n_layers: int


@dataclass(frozen=True)
class PretrainedLensEntry:
    model: ModelIdentity
    repo_id: str
    revision: str
    filename: str
    config_filename: str | None
    license: str
    binary_sha256: str
    config_sha256: str | None
    source_layers: tuple[int, ...]
    calibration_dataset: str
    registry_digest: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _entry_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _single_architecture(config: dict[str, Any]) -> str:
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("model config must identify exactly one architecture")
    architecture = architectures[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model architecture must be a non-empty string")
    return architecture


def _nested_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def model_identity_from_config(
    config: dict[str, Any],
    *,
    requested_model: str,
    base_model: str | None = None,
) -> ModelIdentity:
    """Resolve the exact base-model identity recorded by an HF config."""
    nested = _nested_config(config)
    resolved_base_model = (
        base_model or config.get("_name_or_path") or nested.get("_name_or_path")
    )
    if (
        not isinstance(resolved_base_model, str)
        or not resolved_base_model
        or Path(resolved_base_model).exists()
    ):
        requested_path = Path(requested_model)
        resolved_base_model = (
            requested_model
            if "/" in requested_model
            and not requested_path.is_absolute()
            and not requested_model.startswith(".")
            else ""
        )
    if not resolved_base_model or "/" not in resolved_base_model:
        raise ValueError(
            "model config does not prove a canonical Hugging Face base-model identity"
        )
    d_model = nested.get("hidden_size", nested.get("d_model"))
    n_layers = nested.get("num_hidden_layers", nested.get("n_layers"))
    return ModelIdentity(
        base_model=resolved_base_model,
        architecture=_single_architecture(config),
        d_model=_positive_int(d_model, label="d_model"),
        n_layers=_positive_int(n_layers, label="n_layers"),
    )


def resolve_model_identity(model_name: str) -> ModelIdentity:
    """Read local config.json or fetch an HF config without loading model weights."""
    path = Path(model_name)
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"local model is missing config.json: {path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        from transformers import AutoConfig

        loaded = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        config = loaded.to_dict()
        config.setdefault("_name_or_path", model_name)
    if not isinstance(config, dict):
        raise ValueError("model config must be a JSON object")
    return model_identity_from_config(config, requested_model=model_name)


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _parse_entry(raw: dict[str, Any]) -> PretrainedLensEntry:
    model = raw.get("model")
    source = raw.get("source")
    expected = raw.get("expected")
    if not all(isinstance(value, dict) for value in (model, source, expected)):
        raise ValueError("registry entry requires model, source, and expected objects")
    revision = _require_string(source.get("revision"), label="source.revision")
    if not _FULL_COMMIT_RE.fullmatch(revision):
        raise ValueError("source.revision must be a full lowercase 40-character commit SHA")
    binary_sha256 = _require_string(
        expected.get("binary_sha256"), label="expected.binary_sha256"
    )
    config_sha256 = expected.get("config_sha256")
    if config_sha256 is not None:
        config_sha256 = _require_string(
            config_sha256, label="expected.config_sha256"
        )
    if not _SHA256_RE.fullmatch(binary_sha256) or (
        config_sha256 is not None and not _SHA256_RE.fullmatch(config_sha256)
    ):
        raise ValueError("expected artifact hashes must be lowercase SHA-256 values")
    d_model = _positive_int(model.get("d_model"), label="model.d_model")
    n_layers = _positive_int(model.get("n_layers"), label="model.n_layers")
    layers = expected.get("source_layers")
    if not isinstance(layers, list) or layers != list(range(n_layers - 1)):
        raise ValueError("expected.source_layers must cover every non-final model layer")
    filename = _require_string(source.get("filename"), label="source.filename")
    config_filename = source.get("config_filename")
    if config_filename is not None:
        config_filename = _require_string(
            config_filename, label="source.config_filename"
        )
    if (config_filename is None) != (config_sha256 is None):
        raise ValueError(
            "source.config_filename and expected.config_sha256 must be provided together"
        )
    for label, filename_value in (
        ("source.filename", filename),
        ("source.config_filename", config_filename),
    ):
        if filename_value is None:
            continue
        path = Path(filename_value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{label} must be a safe Hub-relative path")
    return PretrainedLensEntry(
        model=ModelIdentity(
            base_model=_require_string(model.get("base_model"), label="model.base_model"),
            architecture=_require_string(
                model.get("architecture"), label="model.architecture"
            ),
            d_model=d_model,
            n_layers=n_layers,
        ),
        repo_id=_require_string(source.get("repo_id"), label="source.repo_id"),
        revision=revision,
        filename=filename,
        config_filename=config_filename,
        license=_require_string(source.get("license"), label="source.license"),
        binary_sha256=binary_sha256,
        config_sha256=config_sha256,
        source_layers=tuple(int(layer) for layer in layers),
        calibration_dataset=_require_string(
            source.get("calibration_dataset"), label="source.calibration_dataset"
        ),
        registry_digest=_entry_digest(raw),
    )


def load_pretrained_lens_registry(
    path: str | Path = DEFAULT_PRETRAINED_LENS_REGISTRY,
) -> list[PretrainedLensEntry]:
    """Load and validate every pinned pretrained-lens registry entry."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", -1)) != 1:
        raise ValueError("pretrained lens registry has an unsupported schema_version")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("pretrained lens registry entries must be a list")
    parsed = [_parse_entry(entry) for entry in entries if isinstance(entry, dict)]
    if len(parsed) != len(entries):
        raise ValueError("pretrained lens registry entries must be JSON objects")
    identities = [entry.model for entry in parsed]
    if len(set(identities)) != len(identities):
        raise ValueError("pretrained lens registry contains duplicate model identities")
    return parsed


def find_pretrained_lens(
    identity: ModelIdentity,
    entries: list[PretrainedLensEntry],
) -> PretrainedLensEntry | None:
    """Return the exact compatible entry, without aliases or fuzzy matching."""
    matches = [entry for entry in entries if entry.model == identity]
    if len(matches) > 1:
        raise ValueError("multiple pretrained lenses match the exact model identity")
    return matches[0] if matches else None
