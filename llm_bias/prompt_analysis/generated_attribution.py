"""Backward generated-token attribution over an existing forward artifact.

The forward stage owns generation.  This module only consumes the token IDs
and input span persisted by that stage, then performs one gradient readout per
persisted generated token.  It deliberately never calls ``generate``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from jspace_viz.model import WrappedModel

from llm_bias.core.model import DEFAULT_MODEL, load_model as load_lens_model
from llm_bias.prompt_analysis import attribution as _forward_primitives
from llm_bias.core.artifact_paths import sha256_file, stable_record_id as _core_stable_record_id
from llm_bias.prompt_analysis.artifact_io import (
    ARTIFACT_SCHEMA_VERSION,
    load_parent_jsonl,
    read_jsonl,
    write_jsonl,
    write_metadata,
)


METHOD = "semantic_scope_target_logit_gradient_l2_norm"
ARTIFACT_TYPE = "generated_token_attribution"
DEFAULT_OUTPUT_DIR = "artifacts/prompt_analysis"

# Keep the primitive available at module scope.  Besides making the dependency
# explicit, this lets deterministic tests replace the expensive model call.
_attribute_generated_token = _forward_primitives._attribute_generated_token


def _resolve_forward_path(
    forward_path: str | Path | None,
    forward_artifact: str | Path | None,
) -> Path:
    if forward_path is not None and forward_artifact is not None:
        if Path(forward_path) != Path(forward_artifact):
            raise ValueError("forward_path and forward_artifact disagree")
    value = forward_path if forward_path is not None else forward_artifact
    if value is None:
        raise TypeError("one of forward_path or forward_artifact is required")
    path = Path(value)
    if path.is_dir():
        path = path / "forward" / "generated_outputs.jsonl"
    return path


def _as_int_list(value: Any, *, field: str, row_index: int) -> list[int]:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"forward row {row_index} has no {field} list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"forward row {row_index} has non-integer {field}")
        result.append(int(item))
    return result


def _token_ids(row: Mapping[str, Any], *, row_index: int) -> tuple[list[int], list[int]]:
    prompt = row.get("prompt_token_ids")
    generated = row.get("generated_token_ids")
    if generated is None and isinstance(row.get("generated_tokens"), list):
        generated = [token.get("token_id") for token in row["generated_tokens"]]
    return (
        _as_int_list(prompt, field="prompt_token_ids", row_index=row_index),
        _as_int_list(generated, field="generated_token_ids", row_index=row_index),
    )


def _input_span(value: Any, *, row_index: int, prompt_length: int) -> tuple[int, int]:
    if isinstance(value, Mapping):
        value = [value.get("start"), value.get("end")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"forward row {row_index} has no input_span [start, end]")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or end > prompt_length
    ):
        raise ValueError(f"invalid input_span in forward row {row_index}: {value!r}")
    return int(start), int(end)


def _previous_parent_hash(destination: Path) -> str | None:
    metadata = destination / "metadata.json"
    if metadata.is_file():
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid backward metadata: {metadata}") from exc
        if isinstance(value, Mapping):
            for key in ("parent_forward_sha256", "parent_forward_hash"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    return candidate
    rows_path = destination / "generated_token_attribution.jsonl"
    if rows_path.is_file():
        rows = read_jsonl(rows_path)
        if rows:
            for key in ("parent_forward_sha256", "parent_forward_hash"):
                candidate = rows[0].get(key)
                if isinstance(candidate, str):
                    return candidate
    return None


def _parent_fields(path: Path, digest: str) -> dict[str, Any]:
    # Keep both the explicit SHA-256 spelling and the shorter hash spelling so
    # consumers can use the provenance name already used by their manifests.
    return {
        "parent_forward_path": str(path),
        "parent_forward_sha256": digest,
        "parent_forward_hash": digest,
        "parent_artifact": {"path": str(path), "sha256": digest},
    }


def _record_id(row: Mapping[str, Any], *, index: int) -> str:
    existing = row.get("record_id")
    if isinstance(existing, str) and re.fullmatch(r"record_[0-9a-f]{24}", existing):
        return existing
    identity = {
        "index": index,
        "prompt_token_ids": row.get("prompt_token_ids"),
        "generated_token_ids": row.get("generated_token_ids"),
        "input_span": row.get("input_span"),
        "pair_id": row.get("pair_id"),
        "prompt_column": row.get("prompt_column"),
        "sample_index": row.get("sample_index"),
        "run_index": row.get("run_index"),
    }
    return _core_stable_record_id(identity)


def _copy_parent_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy compact, visualizer-facing row metadata, excluding token payloads."""
    excluded = {
        "schema_version",
        "artifact_type",
        "record_id",
        "method",
        "coverage",
        "parent_forward_path",
        "parent_forward_sha256",
        "parent_forward_hash",
        "parent_artifact",
        "prompt_token_ids",
        "generated_token_ids",
        "generated_tokens",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def run_backward_attribution(
    *,
    forward_path: str | Path | None = None,
    forward_artifact: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_top_k: int | None = None,
    max_seq_len: int | None = None,
    prompt_columns: Iterable[str] | None = None,
    output_token_top_k: int | None = None,
) -> Path:
    """Attribute persisted generated tokens without running generation.

    ``output_dir`` is the run root; canonical output is written below its
    ``backward/`` directory.  ``input_top_k`` changes only compact attribution
    storage and never changes the parent generated token sequence.
    """
    source = _resolve_forward_path(forward_path, forward_artifact)
    if input_top_k is not None and input_top_k < 1:
        raise ValueError("input_top_k must be positive when provided")
    if max_seq_len is not None and max_seq_len < 1:
        raise ValueError("max_seq_len must be positive when provided")
    if output_token_top_k is not None and output_token_top_k < 1:
        raise ValueError("output_token_top_k must be positive when provided")

    destination = Path(output_dir) / "backward"
    previous_hash = _previous_parent_hash(destination)
    rows, parent_hash = load_parent_jsonl(
        source,
        previous_sha256=previous_hash,
    )
    requested_columns = set(prompt_columns or ())
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if not requested_columns
        or row.get("prompt_column") in requested_columns
        or row.get("condition") in requested_columns
    ]
    if not selected:
        raise ValueError("forward artifact has no rows matching prompt_columns")

    lens_model, tokenizer, _device = load_lens_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"

    parent_fields = _parent_fields(source, parent_hash)
    output_rows: list[dict[str, Any]] = []
    total_tokens = 0
    for row_index, parent in selected:
        prompt_ids_list, generated_ids_list = _token_ids(parent, row_index=row_index)
        if not prompt_ids_list:
            raise ValueError(f"forward row {row_index} has empty prompt_token_ids")
        if not generated_ids_list:
            raise ValueError(f"forward row {row_index} has empty generated_token_ids")
        if max_seq_len is not None and len(prompt_ids_list) > max_seq_len:
            raise ValueError(
                f"forward row {row_index} prompt length {len(prompt_ids_list)} exceeds max_seq_len {max_seq_len}"
            )
        span = _input_span(
            parent.get("input_span"),
            row_index=row_index,
            prompt_length=len(prompt_ids_list),
        )
        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=model.device)
        token_records: list[dict[str, Any]] = []
        process_count = (
            len(generated_ids_list)
            if output_token_top_k is None
            else min(output_token_top_k, len(generated_ids_list))
        )
        parent_tokens = parent.get("generated_tokens")
        if not isinstance(parent_tokens, list):
            parent_tokens = []
        for position, target_id in enumerate(generated_ids_list[:process_count]):
            prefix = torch.tensor(
                [generated_ids_list[:position]], dtype=torch.long, device=model.device
            )
            scored = _attribute_generated_token(
                model=model,
                prompt_ids=prompt_ids,
                generated_prefix=prefix,
                target_id=target_id,
                input_top_k=input_top_k,
                input_span=span,
            )
            parent_token = parent_tokens[position] if position < len(parent_tokens) else {}
            if not isinstance(parent_token, Mapping):
                parent_token = {}
            token_record = {
                **dict(parent_token),
                "position": position,
                "token_id": target_id,
                **scored,
            }
            # The persisted token ID is authoritative even if a test double or
            # stale parent token record returns a different one.
            token_record["token_id"] = target_id
            token_records.append(token_record)
            total_tokens += 1

        generated_text = parent.get("generated_text")
        if not isinstance(generated_text, str):
            generated_text = tokenizer.decode(
                generated_ids_list,
                skip_special_tokens=False,
            )
        record = {
            **_copy_parent_fields(parent),
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "record_id": _record_id(parent, index=row_index),
            "method": METHOD,
            **parent_fields,
            "prompt_token_ids": prompt_ids_list,
            "generated_token_ids": generated_ids_list,
            "input_span": [span[0], span[1]],
            "generated_text": generated_text,
            "generated_tokens": token_records,
            "coverage": {
                "input_span": [span[0], span[1]],
                "input_token_count": span[1] - span[0],
                "generated_token_count": len(generated_ids_list),
                "attributed_token_count": len(token_records),
                "complete": len(token_records) == len(generated_ids_list),
                "input_top_k": input_top_k,
                "output_token_top_k": output_token_top_k,
            },
        }
        output_rows.append(record)

    output_path = write_jsonl(destination / "generated_token_attribution.jsonl", output_rows)
    output_digest = sha256_file(output_path)
    write_metadata(
        destination / "metadata.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "method": METHOD,
            "model": model_name,
            "parent_forward_path": str(source),
            "parent_forward_sha256": parent_hash,
            "parent_forward_hash": parent_hash,
            "parent_artifact": {"path": str(source), "sha256": parent_hash},
            "output_sha256": output_digest,
            "input_top_k": input_top_k,
            "output_token_top_k": output_token_top_k,
            "max_seq_len": max_seq_len,
            "records_written": len(output_rows),
            "generated_tokens_attributed": total_tokens,
            "coverage": {
                "records": len(output_rows),
                "complete_records": sum(
                    bool(row["coverage"]["complete"]) for row in output_rows
                ),
                "generated_token_count": sum(
                    int(row["coverage"]["generated_token_count"]) for row in output_rows
                ),
                "attributed_token_count": total_tokens,
            },
        },
    )
    return output_path


def attribute_generated_outputs(
    *,
    forward_artifact: str | Path,
    model_name: str = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_top_k: int | None = None,
    max_seq_len: int | None = None,
    prompt_columns: Iterable[str] | None = None,
    output_token_top_k: int | None = None,
) -> Path:
    """Coordinator-facing name for :func:`run_backward_attribution`."""
    return run_backward_attribution(
        forward_artifact=forward_artifact,
        model_name=model_name,
        output_dir=output_dir,
        input_top_k=input_top_k,
        max_seq_len=max_seq_len,
        prompt_columns=prompt_columns,
        output_token_top_k=output_token_top_k,
    )


# Name used by early coordinator drafts; keep it as a direct alias rather than
# a second implementation so both paths have identical hash validation.
analyze_generated_attribution_from_forward = run_backward_attribution
backward_generated_attribution = run_backward_attribution
