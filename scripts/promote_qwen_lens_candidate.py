#!/usr/bin/env python
"""Promote the preregistered winner and archive the previous canonical lens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from llm_bias.core.lens_artifacts import canonical_lens_path, lens_metadata_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def promote(
    *,
    model_name: str,
    evaluation_path: Path,
    archive_root: Path,
) -> dict[str, object]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected = str(evaluation["selected_candidate"])
    candidate = Path(
        evaluation["candidates"][selected]["lens_path"]
    )
    candidate_metadata = lens_metadata_path(candidate)
    if not candidate.is_file() or not candidate_metadata.is_file():
        raise FileNotFoundError(
            f"selected candidate or metadata is missing: {candidate}"
        )

    canonical = canonical_lens_path(model_name)
    canonical_metadata = lens_metadata_path(canonical)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_root / timestamp
    if canonical.is_file():
        archive.mkdir(parents=True, exist_ok=False)
        _atomic_copy(canonical, archive / canonical.name)
        if canonical_metadata.is_file():
            _atomic_copy(
                canonical_metadata,
                archive / canonical_metadata.name,
            )

    metadata = json.loads(candidate_metadata.read_text(encoding="utf-8"))
    metadata.update(
        {
            "selection_status": "canonical",
            "selected_candidate": selected,
            "selection_evaluation": str(evaluation_path),
            "selection_rule": evaluation["selection_rule"],
            "selection_score": evaluation["candidates"][selected]["summary"][
                "selection_score"
            ],
            "selection_uncertainty": evaluation.get(
                "selection_uncertainty"
            ),
            "promoted_at": datetime.now(UTC).isoformat(),
            "candidate_sha256": _sha256(candidate),
            "previous_canonical_archive": (
                str(archive) if archive.is_dir() else None
            ),
        }
    )
    _atomic_copy(candidate, canonical)
    temporary_metadata = canonical_metadata.with_name(
        canonical_metadata.name + ".tmp"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, canonical_metadata)

    selection = {
        "schema_version": 1,
        "model": model_name,
        "canonical_lens": str(canonical),
        "canonical_sha256": _sha256(canonical),
        "selected_candidate": selected,
        "evaluation": str(evaluation_path),
        "selection_uncertainty": evaluation.get("selection_uncertainty"),
        "archive": str(archive) if archive.is_dir() else None,
    }
    selection_path = canonical.parent / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(
            "artifacts/candidate_lenses/qwen3.5-4b/evaluation.json"
        ),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(
            "artifacts/archive/replaced_lenses/qwen3.5-4b"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            promote(
                model_name=args.model,
                evaluation_path=args.evaluation,
                archive_root=args.archive_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
