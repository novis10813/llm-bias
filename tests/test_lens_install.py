from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from jlens import JacobianLens

from llm_bias.core.lens_artifacts import lens_metadata_path
from llm_bias.lens_install.importer import install_pretrained_lens


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, nonfinite: bool = False):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "org/tiny-model",
                "architectures": ["TinyForCausalLM"],
                "hidden_size": 2,
                "num_hidden_layers": 3,
            }
        ),
        encoding="utf-8",
    )
    binary = tmp_path / "source.pt"
    jacobians = {0: torch.eye(2), 1: torch.eye(2)}
    if nonfinite:
        jacobians[1][0, 0] = torch.nan
    JacobianLens(jacobians, n_prompts=4, d_model=2).save(str(binary))
    source_config = tmp_path / "config.yaml"
    source_config.write_text("hf_model_name: org/tiny-model\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "model": {
                            "base_model": "org/tiny-model",
                            "architecture": "TinyForCausalLM",
                            "d_model": 2,
                            "n_layers": 3,
                        },
                        "source": {
                            "repo_id": "org/lenses",
                            "revision": "a" * 40,
                            "filename": "tiny/lens.pt",
                            "config_filename": "tiny/config.yaml",
                            "license": "MIT",
                            "calibration_dataset": "fixture",
                        },
                        "expected": {
                            "binary_sha256": _sha(binary),
                            "config_sha256": _sha(source_config),
                            "source_layers": [0, 1],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def download_file(*, filename, **_kwargs):
        return str(binary if filename.endswith(".pt") else source_config)

    return model, registry, binary, download_file


def test_install_pretrained_lens_writes_valid_canonical_and_metadata(tmp_path):
    model, registry, binary, download = _fixture(tmp_path)

    result = install_pretrained_lens(
        model_name=str(model),
        registry_path=registry,
        artifact_root=tmp_path / "artifacts",
        download_file=download,
    )

    canonical = Path(result["canonical_path"])
    assert canonical.read_bytes() == binary.read_bytes()
    metadata = json.loads(lens_metadata_path(canonical).read_text())
    assert metadata["model"] == "org/tiny-model"
    assert metadata["selection_basis"] == "pinned_huggingface_pretrained_artifact"
    assert metadata["provenance"]["source"] == "huggingface"
    assert metadata["provenance"]["revision"] == "a" * 40
    assert result["status"] == "installed"


def test_install_dry_run_does_not_create_canonical(tmp_path):
    model, registry, _binary, download = _fixture(tmp_path)
    result = install_pretrained_lens(
        model_name=str(model),
        registry_path=registry,
        artifact_root=tmp_path / "artifacts",
        download_file=download,
        dry_run=True,
    )
    assert result["status"] == "validated"
    assert not Path(result["canonical_path"]).exists()


def test_install_refuses_existing_different_canonical(tmp_path):
    model, registry, _binary, download = _fixture(tmp_path)
    artifact_root = tmp_path / "artifacts"
    canonical = artifact_root / "model" / "jacobian-lens" / "jacobian_lens.pt"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="replace-existing"):
        install_pretrained_lens(
            model_name=str(model),
            registry_path=registry,
            artifact_root=artifact_root,
            download_file=download,
        )
    assert canonical.read_bytes() == b"old"


def test_replace_existing_archives_old_canonical(tmp_path):
    model, registry, binary, download = _fixture(tmp_path)
    artifact_root = tmp_path / "artifacts"
    canonical = artifact_root / "model" / "jacobian-lens" / "jacobian_lens.pt"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"old")

    result = install_pretrained_lens(
        model_name=str(model),
        registry_path=registry,
        artifact_root=artifact_root,
        download_file=download,
        replace_existing=True,
    )

    assert canonical.read_bytes() == binary.read_bytes()
    assert Path(result["archive"]).joinpath("jacobian_lens.pt").read_bytes() == b"old"


def test_nonfinite_download_is_rejected_without_canonical(tmp_path):
    model, registry, _binary, download = _fixture(tmp_path, nonfinite=True)
    artifact_root = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="non-finite"):
        install_pretrained_lens(
            model_name=str(model),
            registry_path=registry,
            artifact_root=artifact_root,
            download_file=download,
        )
    assert not (artifact_root / "model").exists()
