"""Local checkpoint, tokenizer and source fingerprints for independent workflows."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from llm_bias.core.artifact_paths import file_sha256


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def local_model_identity(path: str | Path, tokenizer: Any) -> dict[str, Any]:
    root = Path(path).resolve()
    weights = sorted(set(root.glob("*.safetensors")) | set(root.glob("pytorch_model*.bin")))
    if not root.is_dir() or not (root / "config.json").is_file() or not weights:
        raise ValueError("require a local checkpoint with config and weight files")
    files = sorted(set(weights) | set(root.glob("*.json")) | set(root.glob("*.model")) | set(root.glob("*.txt")))
    return {
        "checkpoint_files": {p.name: file_sha256(p) for p in files},
        "tokenizer_sha256": object_sha256({"class": type(tokenizer).__name__, "vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map}),
    }


def source_identity(*directories: str | Path) -> dict[str, Any]:
    files = sorted({p for directory in directories for p in Path(directory).rglob("*.py")})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_sha256": object_sha256({str(p): file_sha256(p) for p in files})}
