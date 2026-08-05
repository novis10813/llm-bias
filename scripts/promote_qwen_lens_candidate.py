#!/usr/bin/env python
"""Promote the preregistered winner and archive the previous canonical lens."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from llm_bias.core.lens_artifacts import (
    LENS_ARTIFACT_SCHEMA_VERSION,
    LENS_ARTIFACT_TYPE,
    canonical_lens_path,
    complete_lens_metadata,
    lens_archive_root,
    lens_evaluation_path,
    lens_metadata_path,
    lens_selection_path,
    model_slug,
    validate_lens_metadata,
)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def promote(
    *,
    model_name: str,
    evaluation_path: Path,
    archive_root: Path | None = None,
) -> dict[str, object]:
    """Promote one evaluated candidate into the model-scoped active path."""
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected = str(evaluation["selected_candidate"])
    candidate = Path(evaluation["candidates"][selected]["lens_path"])
    candidate_metadata = lens_metadata_path(candidate)
    if not candidate.is_file() or not candidate_metadata.is_file():
        raise FileNotFoundError(
            f"selected candidate or metadata is missing: {candidate}"
        )

    metadata = json.loads(candidate_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"candidate metadata must be a JSON object: {candidate_metadata}")
    validate_lens_metadata(metadata=metadata, lens_path=candidate)
    metadata_model = metadata.get("model")
    if metadata_model is not None:
        if model_slug(str(metadata_model)) != model_slug(model_name):
            raise ValueError(
                f"candidate metadata model={metadata_model!r} does not match "
                f"requested model={model_name!r}"
            )

    canonical = canonical_lens_path(model_name)
    canonical_metadata = lens_metadata_path(canonical)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = (archive_root or lens_archive_root(model_name)) / timestamp
    if canonical.is_file():
        archive.mkdir(parents=True, exist_ok=False)
        _atomic_copy(canonical, archive / canonical.name)
        if canonical_metadata.is_file():
            _atomic_copy(canonical_metadata, archive / canonical_metadata.name)

    metadata.update(
        {
            "selection_status": "canonical",
            "selected_candidate": selected,
            "selection_evaluation": str(evaluation_path),
            "selection_rule": evaluation["selection_rule"],
            "selection_score": evaluation["candidates"][selected]["summary"][
                "selection_score"
            ],
            "selection_uncertainty": evaluation.get("selection_uncertainty"),
            "promoted_at": datetime.now(UTC).isoformat(),
            "previous_canonical_archive": (
                str(archive) if archive.is_dir() else None
            ),
            "provenance": {
                **dict(metadata.get("provenance", {})),
                "workflow": "promote-jacobian-lens-candidate",
                "module": "scripts.promote_qwen_lens_candidate",
                "model": model_name,
                "evaluation": str(evaluation_path),
                "candidate": str(candidate),
            },
        }
    )
    _atomic_copy(candidate, canonical)
    metadata = complete_lens_metadata(metadata=metadata, lens_path=canonical)
    temporary_metadata = canonical_metadata.with_name(canonical_metadata.name + ".tmp")
    temporary_metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, canonical_metadata)

    selection = {
        "artifact_type": LENS_ARTIFACT_TYPE,
        "schema_version": LENS_ARTIFACT_SCHEMA_VERSION,
        "model": model_name,
        "canonical_lens": str(canonical),
        "canonical_sha256": metadata["binary_sha256"],
        "canonical_metadata_sha256": metadata["metadata_sha256"],
        "selected_candidate": selected,
        "evaluation": str(evaluation_path),
        "selection_uncertainty": evaluation.get("selection_uncertainty"),
        "archive": str(archive) if archive.is_dir() else None,
        "provenance": {
            "workflow": "promote-jacobian-lens-candidate",
            "module": "scripts.promote_qwen_lens_candidate",
            "model": model_name,
            "evaluation": str(evaluation_path),
        },
    }
    selection_path = lens_selection_path(model_name)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    evaluation = args.evaluation or lens_evaluation_path(args.model)
    print(
        json.dumps(
            promote(
                model_name=args.model,
                evaluation_path=evaluation,
                archive_root=args.archive_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
