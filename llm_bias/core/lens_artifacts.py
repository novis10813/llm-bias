"""Canonical paths and integrity checks for model-scoped lens artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


LENS_ARTIFACT_TYPE = "jacobian_lens"
LENS_ARTIFACT_SCHEMA_VERSION = 2


def model_slug(model_name: str) -> str:
    """Return a stable artifact directory name for a local path or Hub ID."""
    normalized = model_name.rstrip("/")
    path = Path(normalized)
    if path.exists() or normalized.startswith((".", "/")):
        candidate = path.name
    else:
        candidate = normalized.replace("/", "--")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-")
    if not slug:
        raise ValueError(f"cannot derive an artifact name from model {model_name!r}")
    return slug


def lens_artifact_root(model_name: str) -> Path:
    """Return the single model-scoped directory for all lens artifacts."""
    return Path("artifacts") / model_slug(model_name) / "jacobian-lens"


def canonical_lens_path(model_name: str) -> Path:
    """Return the one active Jacobian-lens path assigned to a model."""
    return lens_artifact_root(model_name) / "jacobian_lens.pt"


def canonical_lens_checkpoint_path(
    model_name: str,
    calibration_sha256: str | None = None,
) -> Path:
    """Return resumable fitting state in the archived model scope."""
    suffix = (
        f".{calibration_sha256[:12]}" if calibration_sha256 is not None else ""
    )
    return (
        Path("artifacts")
        / "archive"
        / model_slug(model_name)
        / "jacobian-lens"
        / "checkpoints"
        / f"jacobian_lens{suffix}.checkpoint.pt"
    )


def lens_candidates_root(model_name: str) -> Path:
    """Return the model-scoped root for condition-specific candidates."""
    return lens_artifact_root(model_name) / "candidates"


def lens_candidate_path(model_name: str, condition: str) -> Path:
    """Return a condition-specific candidate path in the model lens scope."""
    if not condition or Path(condition).name != condition:
        raise ValueError(f"invalid lens candidate condition: {condition!r}")
    return lens_candidates_root(model_name) / condition / "jacobian_lens.pt"


def lens_evaluation_path(model_name: str) -> Path:
    """Return the default candidate evaluation artifact path."""
    return lens_candidates_root(model_name) / "evaluation.json"


def lens_archive_root(model_name: str) -> Path:
    """Return the model-scoped root for replaced canonical lenses."""
    return lens_artifact_root(model_name) / "archive"


def lens_selection_path(model_name: str) -> Path:
    """Return the active lens selection provenance path."""
    return lens_artifact_root(model_name) / "selection.json"


def lens_metadata_path(lens_path: str | Path) -> Path:
    path = Path(lens_path)
    return path.with_name(path.name + ".metadata.json")


def sha256_file(path: str | Path) -> str:
    """Hash a binary artifact without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_sha256(metadata: dict[str, Any]) -> str:
    """Hash metadata while excluding its self-referential digest field."""
    payload = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def complete_lens_metadata(
    *,
    metadata: dict[str, Any],
    lens_path: str | Path,
) -> dict[str, Any]:
    """Add the required artifact identity, hashes, and provenance fields."""
    result = dict(metadata)
    result["artifact_type"] = LENS_ARTIFACT_TYPE
    result["schema_version"] = LENS_ARTIFACT_SCHEMA_VERSION
    result["binary_sha256"] = sha256_file(lens_path)
    result["metadata_sha256"] = metadata_sha256(result)
    return result


def validate_lens_metadata(
    *, metadata: dict[str, Any], lens_path: str | Path
) -> None:
    if metadata.get("artifact_type") != LENS_ARTIFACT_TYPE:
        raise ValueError("lens metadata has an unsupported artifact_type")
    if int(metadata.get("schema_version", -1)) != LENS_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("lens metadata has an unsupported schema_version")
    expected_binary = metadata.get("binary_sha256")
    if not isinstance(expected_binary, str) or expected_binary != sha256_file(lens_path):
        raise ValueError("lens metadata binary_sha256 disagrees with the lens file")
    expected_metadata = metadata.get("metadata_sha256")
    if not isinstance(expected_metadata, str) or expected_metadata != metadata_sha256(metadata):
        raise ValueError("lens metadata metadata_sha256 is invalid")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("lens metadata is missing provenance")


def expected_source_layers(n_layers: int) -> list[int]:
    """All transported rows; the final block is separately read with J = I."""
    return list(range(n_layers - 1))


def load_lens_metadata(lens_path: str | Path) -> dict[str, Any] | None:
    path = lens_metadata_path(lens_path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"lens metadata must be a JSON object: {path}")
    return value


def validate_lens_for_model(
    *,
    model: Any,
    lens: Any,
    model_name: str,
    lens_path: str | Path,
    require_complete: bool = True,
) -> dict[str, Any] | None:
    """Reject wrong-shape, wrong-model, incomplete, or corrupt lenses."""
    if lens.d_model != model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} does not match "
            f"model d_model={model.d_model}"
        )

    fitted = sorted(int(layer) for layer in lens.source_layers)
    expected = expected_source_layers(model.n_layers)
    invalid = [layer for layer in fitted if layer not in expected]
    if invalid:
        raise ValueError(
            f"lens contains source layers outside model range: {invalid}"
        )
    missing = sorted(set(expected) - set(fitted))
    if require_complete and missing:
        raise ValueError(
            "interactive Jacobian-lens visualization requires complete layer "
            f"coverage; missing L{', L'.join(map(str, missing))}. Fit with "
            "--layer-stride 1 or use the canonical lens for this model."
        )

    metadata = load_lens_metadata(lens_path)
    if metadata is None:
        raise ValueError(f"lens is missing reproducibility metadata: {lens_path}")
    metadata_model = metadata.get("model")
    if metadata_model and model_slug(str(metadata_model)) != model_slug(model_name):
        raise ValueError(
            f"lens metadata model={metadata_model!r} does not match "
            f"requested model={model_name!r}"
        )
    metadata_layers = metadata.get("source_layers")
    if metadata_layers is not None and list(metadata_layers) != fitted:
        raise ValueError("lens metadata source_layers disagree with the lens file")
    metadata_d_model = metadata.get("d_model")
    if metadata_d_model is not None and int(metadata_d_model) != model.d_model:
        raise ValueError("lens metadata d_model disagrees with the loaded model")
    metadata_n_layers = metadata.get("n_layers")
    if metadata_n_layers is not None and int(metadata_n_layers) != model.n_layers:
        raise ValueError("lens metadata n_layers disagree with the loaded model")
    validate_lens_metadata(metadata=metadata, lens_path=lens_path)
    return metadata
