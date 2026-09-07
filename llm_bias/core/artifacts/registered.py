"""Registered compact writes and verified reads for staged experiment runs."""
import json
from pathlib import Path

from llm_bias.core.artifact_paths import file_sha256
from .io import write_json, write_jsonl


def write_registered(run, relative, value):
    path = run.run_directory / relative
    if path.suffix == ".jsonl":
        write_jsonl(path, value)
    else:
        write_json(path, value)
    run.manifest.register_artifact(path=path, artifact_type=path.stem, stage=relative.split("/")[0])
    run.save()
    return path


def verified_run(source, dataset, required):
    root = Path(source).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("dataset") != dataset:
        raise ValueError("wrong or incomplete source run")
    refs = manifest.get("output_refs", [])
    names = [ref["path"] for ref in refs]
    if len(names) != len(set(names)) or not set(required) <= set(names):
        raise ValueError("missing or duplicate artifact references")
    for ref in refs:
        path = (root / ref["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != ref["sha256"]:
            raise ValueError("source artifact hash mismatch")
    return {name: json.loads((root / name).read_text()) for name in required}, file_sha256(manifest_path)
