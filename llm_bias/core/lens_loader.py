"""Single validated runtime loader for local Jacobian-lens artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jlens

from llm_bias.core.lens_artifacts import (
    canonical_lens_path,
    validate_lens_for_model,
)


@dataclass(frozen=True)
class LoadedLens:
    lens: Any
    path: Path
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        provenance = self.metadata.get("provenance", {})
        return str(provenance.get("source") or provenance.get("workflow") or "unknown")


def load_validated_lens(
    *,
    model: Any,
    model_name: str,
    lens_path: str | Path | None = None,
    artifact_root: str | Path = "artifacts",
    require_complete: bool = True,
) -> LoadedLens:
    """Load a local explicit or canonical lens and enforce the artifact contract."""
    path = (
        Path(lens_path)
        if lens_path is not None
        else canonical_lens_path(model_name, artifact_root=artifact_root)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Jacobian lens is not ready at {path}. Run jacobian-lens install "
            "for an exact registry match or jacobian-lens fit for this model."
        )
    lens = jlens.JacobianLens.load(str(path))
    metadata = validate_lens_for_model(
        model=model,
        lens=lens,
        model_name=model_name,
        lens_path=path,
        require_complete=require_complete,
    )
    if metadata is None:
        raise ValueError(f"lens is missing reproducibility metadata: {path}")
    return LoadedLens(lens=lens, path=path, metadata=metadata)
