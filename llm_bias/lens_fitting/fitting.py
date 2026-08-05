"""Standalone Jacobian-lens fitting implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import jlens

from llm_bias.core.lens_artifacts import (
    canonical_lens_checkpoint_path,
    canonical_lens_path,
    complete_lens_metadata,
    expected_source_layers,
)
from llm_bias.core.model import DEFAULT_MODEL, load_model
from llm_bias.core.prompting import format_prompt
from llm_bias.lens_fitting.calibration import (
    builtin_calibration_prompts,
    load_calibration_prompts,
)


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _prompt_digest(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fit_jacobian_lens(
    *,
    model_name: str = DEFAULT_MODEL,
    output: str | None = None,
    calibration_count: int = 16,
    calibration_file: str | None = None,
    calibration_field: str = "text",
    layer_stride: int = 1,
    dim_batch: int = 16,
    max_seq_len: int = 128,
    skip_first: int = 0,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
    checkpoint_every: int = 4,
) -> Path:
    """Fit and save a model-specific lens plus reproducibility metadata."""
    if calibration_count < 1:
        raise ValueError("calibration_count must be positive")
    if layer_stride < 1:
        raise ValueError("layer_stride must be positive")
    if dim_batch < 1:
        raise ValueError("dim_batch must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if skip_first < 0:
        raise ValueError("skip_first must be non-negative")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")

    raw_prompts = (
        load_calibration_prompts(
            calibration_file,
            field=calibration_field,
            count=calibration_count,
        )
        if calibration_file
        else builtin_calibration_prompts(calibration_count)
    )
    jlens.configure_logging()
    model, tokenizer, _device = load_model(model_name)
    if enable_thinking and not use_chat_template:
        raise ValueError("enable_thinking requires use_chat_template")
    prompts = [
        format_prompt(
            tokenizer,
            prompt,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )
        for prompt in raw_prompts
    ]
    calibration_input_sha256 = _prompt_digest(raw_prompts)
    calibration_sha256 = _prompt_digest(prompts)
    layers = list(range(0, model.n_layers - 1, layer_stride))
    canonical = canonical_lens_path(model_name)
    destination = Path(output) if output is not None else canonical
    if destination == canonical and layers != expected_source_layers(model.n_layers):
        raise ValueError(
            "the canonical model lens must cover every source layer; use "
            "--layer-stride 1 or choose a non-canonical experimental output path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        canonical_lens_checkpoint_path(model_name, calibration_sha256)
        if destination == canonical
        else destination.with_name(
            f"{destination.name}.{calibration_sha256[:12]}.checkpoint.pt"
        )
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(
        model,
        prompts,
        source_layers=layers,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=checkpoint_every,
    )
    lens.save(str(destination))

    metadata: dict[str, Any] = {
        "model": model_name,
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "source_layers": layers,
        "layer_stride": layer_stride,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "calibration_source": calibration_file or "builtin",
        "calibration_field": calibration_field if calibration_file else None,
        "calibration_count": len(prompts),
        "calibration_input_sha256": calibration_input_sha256,
        "calibration_sha256": calibration_sha256,
        "use_chat_template": use_chat_template,
        "enable_thinking": enable_thinking,
        "jlens_version": _package_version("jacobian-lens", "jlens"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_every": checkpoint_every,
        "provenance": {
            "workflow": "fit-jacobian-lens",
            "module": "llm_bias.lens_fitting.fitting",
            "model": model_name,
            "output": str(destination),
        },
    }
    metadata = complete_lens_metadata(metadata=metadata, lens_path=destination)
    metadata_path = destination.with_name(destination.name + ".metadata.json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return destination
