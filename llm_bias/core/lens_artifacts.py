"""Canonical paths and compatibility checks for model-specific lens artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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


def canonical_lens_path(model_name: str) -> Path:
    """Return the one active Jacobian-lens path assigned to a model."""
    return Path("artifacts") / "lenses" / model_slug(model_name) / "jacobian_lens.pt"


def canonical_lens_checkpoint_path(
    model_name: str,
    calibration_sha256: str | None = None,
) -> Path:
    """Keep resumable fitting state outside the active lens directory."""
    suffix = (
        f".{calibration_sha256[:12]}" if calibration_sha256 is not None else ""
    )
    return (
        Path("artifacts")
        / "archive"
        / "lens_checkpoints"
        / model_slug(model_name)
        / f"jacobian_lens{suffix}.checkpoint.pt"
    )


def lens_metadata_path(lens_path: str | Path) -> Path:
    path = Path(lens_path)
    return path.with_name(path.name + ".metadata.json")


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
    """Reject wrong-shape, wrong-model, or incomplete active lenses."""
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
        return None
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
    return metadata
