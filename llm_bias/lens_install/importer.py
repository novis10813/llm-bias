"""Download, validate, and atomically install pinned pretrained lenses."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
from huggingface_hub import hf_hub_download
from jlens import JacobianLens

from llm_bias.core.artifact_paths import atomic_write_json
from llm_bias.core.lens_artifacts import (
    canonical_lens_path,
    complete_lens_metadata,
    lens_archive_root,
    lens_metadata_path,
    lens_selection_path,
    sha256_file,
    validate_lens_for_model,
    validate_lens_metadata,
)
from llm_bias.core.lens_registry import (
    DEFAULT_PRETRAINED_LENS_REGISTRY,
    PretrainedLensEntry,
    find_pretrained_lens,
    load_pretrained_lens_registry,
    model_identity_from_config,
)

DownloadFile = Callable[..., str]


def _read_model_config(
    model_name: str,
    *,
    base_model: str | None = None,
) -> dict[str, Any]:
    path = Path(model_name)
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"local model is missing config.json: {path}")
        value = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        value = config.to_dict()
        value.setdefault("_name_or_path", model_name)
    if not isinstance(value, dict):
        raise ValueError("model config must be a JSON object")
    if base_model is not None:
        from transformers import AutoConfig

        hub_config = AutoConfig.from_pretrained(
            base_model, trust_remote_code=False
        ).to_dict()
        local_identity = model_identity_from_config(
            value,
            requested_model=model_name,
            base_model=base_model,
        )
        hub_identity = model_identity_from_config(
            hub_config,
            requested_model=base_model,
            base_model=base_model,
        )
        if local_identity != hub_identity:
            raise ValueError(
                "local model config disagrees with the declared canonical base model"
            )
    return value


def _download(
    entry: PretrainedLensEntry,
    *,
    cache_dir: str | Path | None,
    offline: bool,
    download_file: DownloadFile,
) -> tuple[Path, Path | None]:
    kwargs = {
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": offline,
    }
    binary = Path(download_file(filename=entry.filename, **kwargs))
    config = (
        Path(download_file(filename=entry.config_filename, **kwargs))
        if entry.config_filename is not None
        else None
    )
    return binary, config


def _validate_downloads(
    entry: PretrainedLensEntry,
    *,
    binary_path: Path,
    config_path: Path | None,
) -> JacobianLens:
    if sha256_file(binary_path) != entry.binary_sha256:
        raise ValueError("downloaded lens binary SHA-256 disagrees with the registry")
    if config_path is not None and sha256_file(config_path) != entry.config_sha256:
        raise ValueError("downloaded lens config SHA-256 disagrees with the registry")
    lens = JacobianLens.load(str(binary_path))
    if lens.d_model != entry.model.d_model:
        raise ValueError("downloaded lens d_model disagrees with the registry")
    if tuple(lens.source_layers) != entry.source_layers:
        raise ValueError("downloaded lens source layers disagree with the registry")
    if int(lens.n_prompts) < 1:
        raise ValueError("downloaded lens n_prompts must be positive")
    for layer in entry.source_layers:
        jacobian = lens.jacobians.get(layer)
        if jacobian is None or tuple(jacobian.shape) != (
            entry.model.d_model,
            entry.model.d_model,
        ):
            raise ValueError(f"downloaded lens has an invalid Jacobian shape at L{layer}")
        if not torch.isfinite(jacobian).all().item():
            raise ValueError(f"downloaded lens contains non-finite values at L{layer}")
    return lens


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _archive_existing(canonical: Path, *, archive_root: Path) -> Path | None:
    if not canonical.is_file():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = archive_root / stamp
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(canonical, destination / canonical.name)
    metadata = lens_metadata_path(canonical)
    if metadata.is_file():
        shutil.copy2(metadata, destination / metadata.name)
    return destination


def _install_staged_pair(
    *,
    staged_binary: Path,
    staged_metadata: Path,
    canonical: Path,
) -> None:
    metadata_path = lens_metadata_path(canonical)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    rollback = Path(tempfile.mkdtemp(prefix=".lens-rollback.", dir=canonical.parent))
    old_binary = rollback / canonical.name
    old_metadata = rollback / metadata_path.name
    had_binary = canonical.is_file()
    had_metadata = metadata_path.is_file()
    try:
        if had_binary:
            shutil.copy2(canonical, old_binary)
        if had_metadata:
            shutil.copy2(metadata_path, old_metadata)
        _atomic_copy(staged_binary, canonical)
        _atomic_copy(staged_metadata, metadata_path)
    except BaseException:
        if had_binary:
            _atomic_copy(old_binary, canonical)
        else:
            canonical.unlink(missing_ok=True)
        if had_metadata:
            _atomic_copy(old_metadata, metadata_path)
        else:
            metadata_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(rollback, ignore_errors=True)


def install_pretrained_lens(
    *,
    model_name: str,
    base_model: str | None = None,
    registry_path: str | Path = DEFAULT_PRETRAINED_LENS_REGISTRY,
    artifact_root: str | Path = "artifacts",
    artifact_model_name: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    dry_run: bool = False,
    replace_existing: bool = False,
    download_file: DownloadFile = hf_hub_download,
) -> dict[str, Any]:
    """Install the exact compatible pinned Hub lens into the canonical path."""
    config = _read_model_config(model_name, base_model=base_model)
    identity = model_identity_from_config(
        config,
        requested_model=model_name,
        base_model=base_model,
    )
    entry = find_pretrained_lens(
        identity, load_pretrained_lens_registry(registry_path)
    )
    if entry is None:
        raise ValueError(
            "no pinned pretrained Jacobian lens matches the exact model identity; "
            "continue with jacobian-lens fit"
        )
    binary, source_config = _download(
        entry,
        cache_dir=cache_dir,
        offline=offline,
        download_file=download_file,
    )
    lens = _validate_downloads(
        entry, binary_path=binary, config_path=source_config
    )
    artifact_model = artifact_model_name or model_name
    canonical = canonical_lens_path(artifact_model, artifact_root=artifact_root)
    existing_sha256 = sha256_file(canonical) if canonical.is_file() else None
    if existing_sha256 == entry.binary_sha256:
        return {
            "status": "already_installed",
            "canonical_path": str(canonical),
            "binary_sha256": existing_sha256,
            "source_revision": entry.revision,
        }
    if existing_sha256 is not None and not replace_existing:
        raise FileExistsError(
            f"canonical lens already exists with SHA-256 {existing_sha256}; "
            "pass --replace-existing to archive and replace it"
        )
    if dry_run:
        return {
            "status": "validated",
            "canonical_path": str(canonical),
            "binary_sha256": entry.binary_sha256,
            "source_revision": entry.revision,
            "would_replace": existing_sha256 is not None,
        }

    canonical.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lens-import.", dir=canonical.parent) as directory:
        staging = Path(directory)
        staged_binary = staging / canonical.name
        shutil.copyfile(binary, staged_binary)
        metadata = complete_lens_metadata(
            metadata={
                "model": identity.base_model,
                "requested_model": model_name,
                "base_model_identity": {
                    "base_model": identity.base_model,
                    "architecture": identity.architecture,
                    "d_model": identity.d_model,
                    "n_layers": identity.n_layers,
                },
                "d_model": lens.d_model,
                "n_layers": identity.n_layers,
                "source_layers": lens.source_layers,
                "n_prompts": lens.n_prompts,
                "calibration_source": entry.calibration_dataset,
                "selection_basis": "pinned_huggingface_pretrained_artifact",
                "selection_status": "canonical",
                "provenance": {
                    "workflow": "install-pretrained-lens",
                    "module": "llm_bias.lens_install.importer",
                    "source": "huggingface",
                    "repo_id": entry.repo_id,
                    "revision": entry.revision,
                    "filename": entry.filename,
                    "config_filename": entry.config_filename,
                    "source_binary_sha256": entry.binary_sha256,
                    "source_config_sha256": entry.config_sha256,
                    "license": entry.license,
                    "registry_path": str(registry_path),
                    "registry_entry_sha256": entry.registry_digest,
                },
            },
            lens_path=staged_binary,
        )
        staged_metadata = staging / lens_metadata_path(canonical).name
        staged_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_lens_for_model(
            model=SimpleNamespace(d_model=identity.d_model, n_layers=identity.n_layers),
            lens=lens,
            model_name=model_name,
            lens_path=staged_binary,
            require_complete=True,
        )
        validate_lens_metadata(metadata=metadata, lens_path=staged_binary)
        archive = (
            _archive_existing(
                canonical,
                archive_root=lens_archive_root(
                    artifact_model, artifact_root=artifact_root
                ),
            )
            if existing_sha256 is not None
            else None
        )
        _install_staged_pair(
            staged_binary=staged_binary,
            staged_metadata=staged_metadata,
            canonical=canonical,
        )
    selection = {
        "schema_version": 1,
        "model": identity.base_model,
        "requested_model": model_name,
        "selection_basis": "pinned_huggingface_pretrained_artifact",
        "canonical_path": str(canonical),
        "canonical_sha256": entry.binary_sha256,
        "source": metadata["provenance"],
        "archive": str(archive) if archive is not None else None,
    }
    atomic_write_json(
        lens_selection_path(artifact_model, artifact_root=artifact_root), selection
    )
    return {
        "status": "installed",
        "canonical_path": str(canonical),
        "binary_sha256": entry.binary_sha256,
        "source_revision": entry.revision,
        "archive": str(archive) if archive is not None else None,
    }
