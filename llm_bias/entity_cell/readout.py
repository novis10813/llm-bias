"""Compact Jacobian-lens readout for E2-selected components.

The readout transports one selected aggregate component at a time.  It keeps
vectors and the complete vocabulary in memory only, then returns compact token
scores plus provenance.  The result describes transported representation
alignment; it does not establish attention, a reasoning path, or causality.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import model_slug, sha256_file
from llm_bias.core.continuation_scoring import fp32_next_token_logits
from llm_bias.core.lens_loader import LoadedLens, load_validated_lens
from llm_bias.core.analysis.transport import transport_residual_delta
from llm_bias.core.prompt_input.encoding import decode_token, input_ids

# The vocabulary is deliberately small and explicit.  IDs are resolved against
# the run tokenizer before scoring and duplicate IDs are retained once.
DEFAULT_SECTOR_VOCABULARY: dict[str, tuple[str, ...]] = {
    "Technology": (
        "technology",
        "software",
        "semiconductor",
        "cloud",
        "internet",
    ),
    "Financial Services": ("bank", "insurance", "finance"),
}
DEFAULT_READOUT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "top_k": 10,
    "answer_tokens": {"buy": "buy", "sell": "sell"},
    "sector_vocabulary": DEFAULT_SECTOR_VOCABULARY,
    "score_contract": "FP32 final norm and LM head; top-k ordered by score",
}
FULL_ATTENTION_LAYERS = frozenset({3, 7, 11, 15, 19, 23, 27, 31})
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "entity_cell_e2_selected_component_readout"


def _json_safe(value: Any, *, field: str = "value") -> Any:
    if torch.is_tensor(value) or hasattr(value, "detach"):
        raise ValueError(f"{field} must not contain tensors or component vectors")
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(term in normalized for term in ("activation", "gradient", "jacobian", "hidden_state", "residual_vector", "component_vector", "transported_vector", "full_vocabulary")):
                raise ValueError(f"{field}.{key} is outside the compact readout contract")
            result[str(key)] = _json_safe(item, field=f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field=field) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{field} must contain finite values")
        return value
    raise TypeError(f"{field} contains a non-serializable value: {type(value).__name__}")


def _model_revision(model: Any) -> str | None:
    for owner in (model, getattr(model, "_hf_model", None)):
        if owner is None:
            continue
        for name in ("model_revision", "revision", "_commit_hash"):
            value = getattr(owner, name, None)
            if isinstance(value, str) and value:
                return value
        config = getattr(owner, "config", None)
        for name in ("_commit_hash", "revision", "model_revision"):
            value = getattr(config, name, None) if config is not None else None
            if isinstance(value, str) and value:
                return value
    return None


def _tokenizer_identity(tokenizer: Any) -> str | None:
    for name in ("name_or_path", "_name_or_path", "tokenizer_name", "revision"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _metadata_model_revision(metadata: Mapping[str, Any]) -> str | None:
    provenance = metadata.get("provenance", {})
    for source in (metadata, provenance):
        if isinstance(source, Mapping):
            for key in ("model_revision",):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _metadata_tokenizer_identity(metadata: Mapping[str, Any]) -> str | None:
    provenance = metadata.get("provenance", {})
    for source in (metadata, provenance):
        if isinstance(source, Mapping):
            for key in ("tokenizer_identity", "tokenizer", "tokenizer_name"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _validate_selection_status(status: Mapping[str, Any] | str | bool) -> dict[str, Any]:
    if status is True:
        return {"status": "selected", "validated": True}
    if status is False or status is None:
        raise ValueError("readout requires a validated E2-selected component")
    if isinstance(status, str):
        if status != "selected":
            raise ValueError("component selection status must be 'selected'")
        return {"status": status, "validated": True}
    if not isinstance(status, Mapping):
        raise TypeError("selection status must be a mapping, 'selected', or True")
    if status.get("selection_eligible") is not True or status.get("selected_top_five") is not True:
        raise ValueError("component is not an E2-selected eligible head")
    return _json_safe(dict(status), field="selection_status")


def _single_token_id(tokenizer: Any, value: int | str, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a token ID or token string")
    if isinstance(value, int):
        token_id = value
    elif isinstance(value, str):
        ids = input_ids(tokenizer, " " + value.strip(), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{label} must resolve to one tokenizer token")
        token_id = int(ids[0])
    else:
        raise TypeError(f"{label} must be a token ID or token string")
    if token_id < 0:
        raise ValueError(f"{label} must be non-negative")
    return token_id


def _vocabulary_items(
    tokenizer: Any,
    vocabulary: Mapping[str, Sequence[int | str | Mapping[str, Any]]],
    *,
    vocab_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ValueError("sector_vocabulary must be a non-empty mapping")
    result: dict[str, list[dict[str, Any]]] = {}
    duplicates: dict[str, list[int]] = {}
    for sector, values in vocabulary.items():
        if not isinstance(sector, str) or not sector.strip():
            raise ValueError("sector vocabulary names must be non-empty strings")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"sector vocabulary {sector!r} must be a sequence")
        seen: set[int] = set()
        rows: list[dict[str, Any]] = []
        removed: list[int] = []
        for index, value in enumerate(values):
            label = f"sector_vocabulary[{sector!r}][{index}]"
            token_value: int | str
            token_text: str | None = None
            if isinstance(value, Mapping):
                if "token_id" not in value and "token" not in value and "text" not in value:
                    raise ValueError(f"{label} must contain token_id, token, or text")
                token_value = value.get("token_id", value.get("token", value.get("text")))
                token_text = str(value.get("token", value.get("text", ""))) or None
            else:
                token_value = value
            token_id = _single_token_id(tokenizer, token_value, label=label)
            if token_id >= vocab_size:
                raise ValueError(f"{label} token ID {token_id} is outside vocabulary")
            if token_id in seen:
                removed.append(token_id)
                continue
            seen.add(token_id)
            rows.append({"token_id": token_id, "token": token_text or decode_token(tokenizer, token_id)})
        if not rows:
            raise ValueError(f"sector vocabulary {sector!r} is empty after duplicate removal")
        result[sector] = rows
        if removed:
            duplicates[sector] = removed
    return result, duplicates


def _validate_loaded_lens(
    loaded: LoadedLens,
    *,
    model: Any,
    model_name: str,
    tokenizer: Any,
    source_layer: int,
    expected_lens_sha256: str | None,
    expected_model_revision: str | None,
    expected_tokenizer_identity: str | None,
) -> dict[str, Any]:
    path = Path(loaded.path)
    if not path.is_file():
        raise FileNotFoundError(f"validated lens file is missing: {path}")
    actual_sha = sha256_file(path)
    metadata = loaded.metadata
    recorded_sha = metadata.get("binary_sha256")
    if recorded_sha != actual_sha:
        raise ValueError("validated lens metadata binary_sha256 does not match the lens file")
    if expected_lens_sha256 is not None and actual_sha != expected_lens_sha256:
        raise ValueError("canonical lens SHA-256 does not match the expected digest")
    recorded_models = [metadata.get("model"), metadata.get("requested_model")]
    recorded_models = [str(value) for value in recorded_models if value]
    if recorded_models and model_slug(model_name) not in {model_slug(value) for value in recorded_models}:
        raise ValueError("lens model identity does not match the requested model")
    actual_model_revision = _model_revision(model)
    recorded_model_revision = _metadata_model_revision(metadata)
    if expected_model_revision is not None and actual_model_revision != expected_model_revision:
        raise ValueError("model revision does not match the expected revision")
    if recorded_model_revision is not None and actual_model_revision != recorded_model_revision:
        raise ValueError("lens model revision does not match the loaded model revision")
    actual_tokenizer_identity = _tokenizer_identity(tokenizer)
    recorded_tokenizer = _metadata_tokenizer_identity(metadata)
    if expected_tokenizer_identity is not None and actual_tokenizer_identity != expected_tokenizer_identity:
        raise ValueError("tokenizer identity does not match the expected tokenizer")
    if recorded_tokenizer is not None and actual_tokenizer_identity != recorded_tokenizer:
        raise ValueError("lens tokenizer identity does not match the loaded tokenizer")
    lens = loaded.lens
    if not callable(getattr(lens, "transport", None)):
        raise TypeError("validated lens must expose transport()")
    if int(getattr(lens, "d_model", -1)) != int(getattr(model, "d_model", -2)):
        raise ValueError("lens hidden size does not match the loaded model")
    source_layers = {int(layer) for layer in getattr(lens, "source_layers", ())}
    if source_layer not in source_layers:
        raise ValueError(f"canonical lens does not cover source layer L{source_layer}")
    if source_layer not in FULL_ATTENTION_LAYERS:
        raise ValueError(f"E2 selected-component readout requires a full-attention source layer, got L{source_layer}")
    provenance = metadata.get("provenance", {})
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("revision"), str) or not provenance["revision"]:
        raise ValueError("validated lens provenance is missing its pinned source revision")
    return {
        "path": path.as_posix(),
        "sha256": actual_sha,
        "metadata": _json_safe(metadata, field="lens_metadata"),
        "source": str(loaded.source),
        "model_name": str(model_name),
        "model_revision": actual_model_revision,
        "tokenizer_identity": actual_tokenizer_identity,
        "hidden_size": int(model.d_model),
        "source_layers": sorted(source_layers),
    }


def _compact_scores(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    top_k: int,
    answer_token_ids: Mapping[str, int],
    sector_items: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    scores = logits.float().flatten()
    probabilities = scores.softmax(dim=0)
    order = torch.argsort(scores, descending=True, stable=True)[: min(top_k, scores.numel())]
    top_tokens = []
    for rank, token_id in enumerate(order.tolist(), 1):
        token_id = int(token_id)
        top_tokens.append({
            "rank": rank,
            "token_id": token_id,
            "token": decode_token(tokenizer, token_id),
            "score": float(scores[token_id].detach()),
            "probability": float(probabilities[token_id].detach()),
        })
    answer_scores: dict[str, dict[str, Any]] = {}
    for label, token_id in answer_token_ids.items():
        answer_scores[str(label)] = {
            "token_id": int(token_id),
            "token": decode_token(tokenizer, int(token_id)),
            "rank": int((torch.argsort(scores, descending=True, stable=True) == int(token_id)).nonzero()[0].item()) + 1,
            "score": float(scores[int(token_id)].detach()),
            "probability": float(probabilities[int(token_id)].detach()),
        }
    sector_scores: dict[str, list[dict[str, Any]]] = {}
    for sector, items in sector_items.items():
        sector_scores[sector] = []
        for item in items:
            token_id = int(item["token_id"])
            sector_scores[sector].append({
                "token_id": token_id,
                "token": str(item["token"]),
                "rank": int((torch.argsort(scores, descending=True, stable=True) == token_id).nonzero()[0].item()) + 1,
                "score": float(scores[token_id].detach()),
                "probability": float(probabilities[token_id].detach()),
            })
    return top_tokens, answer_scores, sector_scores


def readout_selected_component(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    component_vector: torch.Tensor,
    source_layer: int,
    component_identity: Mapping[str, Any] | str,
    prompt_id: str,
    ticker: str,
    selection_status: Mapping[str, Any] | str | bool,
    loaded_lens: LoadedLens | None = None,
    artifact_root: str | Path = "artifacts",
    expected_lens_sha256: str | None = None,
    expected_model_revision: str | None = None,
    expected_tokenizer_identity: str | None = None,
    component_kind: str = "aggregate_head",
    source_group: str | None = None,
    component_provenance: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    answer_token_ids: Mapping[str, int | str] | None = None,
    sector_vocabulary: Mapping[str, Sequence[int | str | Mapping[str, Any]]] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Transport and compactly decode one validated E2 component.

    ``component_vector`` is an aggregate final-position head output in the
    primary ``aggregate_head`` mode.  A ``source_group`` mode remains available
    for diagnostics and is labelled as source-group descriptive output.
    """
    if not torch.is_tensor(component_vector) or component_vector.ndim != 1:
        raise ValueError("component_vector must have shape [d_model]")
    if not torch.isfinite(component_vector).all():
        raise ValueError("component_vector must be finite")
    if component_kind not in {"aggregate_head", "source_group"}:
        raise ValueError("component_kind must be aggregate_head or source_group")
    if component_kind == "source_group" and not source_group:
        raise ValueError("source_group is required for source-group readout")
    if component_kind == "aggregate_head" and source_group is not None:
        raise ValueError("aggregate head readout cannot carry a source_group")
    if not isinstance(prompt_id, str) or not prompt_id.strip() or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("prompt_id and ticker must be non-empty")
    source_layer = int(source_layer)
    if source_layer < 0:
        raise ValueError("source_layer must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    status = _validate_selection_status(selection_status)
    identity = _json_safe(component_identity, field="component_identity") if isinstance(component_identity, Mapping) else str(component_identity)
    if isinstance(identity, str) and not identity.strip():
        raise ValueError("component_identity must be non-empty")
    loaded = loaded_lens or load_validated_lens(
        model=model,
        model_name=model_name,
        artifact_root=artifact_root,
        require_complete=True,
    )
    if expected_lens_sha256 is not None and (len(expected_lens_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_lens_sha256)):
        raise ValueError("expected_lens_sha256 must be a lowercase SHA-256 digest")
    lens_info = _validate_loaded_lens(
        loaded,
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        source_layer=source_layer,
        expected_lens_sha256=expected_lens_sha256,
        expected_model_revision=expected_model_revision,
        expected_tokenizer_identity=expected_tokenizer_identity,
    )
    model_width = int(getattr(model, "d_model", -1))
    if component_vector.numel() != model_width:
        raise ValueError("component vector width does not match model hidden size")
    head = getattr(model, "_lm_head", None)
    if head is None or not torch.is_tensor(getattr(head, "weight", None)):
        raise TypeError("model must expose _lm_head.weight for component readout")
    vocab_size = int(head.weight.shape[0])
    answer_values = answer_token_ids or {"buy": "buy", "sell": "sell"}
    answers: dict[str, int] = {}
    for label, value in answer_values.items():
        token_id = _single_token_id(tokenizer, value, label=f"answer_token_ids[{label!r}]")
        if token_id >= vocab_size:
            raise ValueError(f"answer token {label!r} is outside vocabulary")
        if token_id in answers.values():
            raise ValueError("Buy/Sell answer token IDs must be distinct")
        answers[str(label)] = token_id
    fixed = DEFAULT_SECTOR_VOCABULARY if sector_vocabulary is None else sector_vocabulary
    sector_items, duplicate_ids = _vocabulary_items(tokenizer, fixed, vocab_size=vocab_size)
    actual_lens = loaded.lens
    final_layer = int(getattr(model, "n_layers", -1)) - 1
    zeros = torch.zeros_like(component_vector)
    transported = transport_residual_delta(
        component_vector,
        zeros,
        layer=source_layer,
        final_layer=final_layer,
        lens=actual_lens,
    )
    logits = fp32_next_token_logits(model, transported.unsqueeze(0))[0]
    if logits.numel() != vocab_size or not torch.isfinite(logits).all():
        raise ValueError("component readout produced invalid vocabulary scores")
    top_tokens, answer_scores, sector_scores = _compact_scores(
        logits,
        tokenizer,
        top_k=top_k,
        answer_token_ids=answers,
        sector_items=sector_items,
    )
    config_value = _json_safe(dict(config or DEFAULT_READOUT_CONFIG), field="config")
    config_value["resolved_answer_tokens"] = {label: {"token_id": token_id, "token": decode_token(tokenizer, token_id)} for label, token_id in answers.items()}
    config_value["resolved_sector_vocabulary"] = sector_items
    component_info: dict[str, Any] = {
        "identity": identity,
        "kind": component_kind,
        "source_layer": source_layer,
        "source_group": source_group,
        "ticker": ticker,
        "prompt_id": prompt_id,
        "selection_status": status,
        "provenance": _json_safe(dict(component_provenance or {}), field="component_provenance"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "availability": "available",
        "component": component_info,
        "readout": {
            "top_tokens": top_tokens,
            "answer_tokens": answer_scores,
            "fixed_buy_sell": answer_scores,
            "sector_vocabulary": sector_scores,
            "sector_vocabulary_duplicates_removed": duplicate_ids,
            "top_k": min(int(top_k), vocab_size),
            "score_contract": "FP32 final norm and LM head; top-k ordered by score",
        },
        "provenance": {
            "model": str(model_name),
            "model_revision": lens_info["model_revision"],
            "model_hidden_size": lens_info["hidden_size"],
            "lens": lens_info,
            "tokenizer": {
                "identity": lens_info["tokenizer_identity"],
                "vocabulary_size": vocab_size,
            },
            "config": config_value,
            "raw_runtime_payloads": False,
            "full_vocabulary_persisted": False,
            "component_vector_persisted": False,
        },
        "interpretation_limit": (
            "Jacobian-lens transported representation readout; scores and token ranks "
            "are descriptive and do not establish attention, chain-of-thought, a "
            "discrete reasoning path, or causal evidence"
        ),
    }


# Short aliases for experiment callers.
readout_component = readout_selected_component
compact_selected_component_readout = readout_selected_component


def unavailable_component_readout(
    *, layer: int, component_identity: Mapping[str, Any] | str, prompt_id: str, ticker: str,
    selection_status: Mapping[str, Any] | str | bool, reason: str,
) -> dict[str, Any]:
    """Create a compact unavailable record for an unsupported lens source layer."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "availability": "unavailable",
        "reason": str(reason),
        "component": {
            "identity": _json_safe(component_identity, field="component_identity") if isinstance(component_identity, Mapping) else str(component_identity),
            "kind": "aggregate_head",
            "source_layer": int(layer),
            "ticker": str(ticker),
            "prompt_id": str(prompt_id),
            "selection_status": _validate_selection_status(selection_status),
        },
        "provenance": {
            "raw_runtime_payloads": False,
            "full_vocabulary_persisted": False,
            "component_vector_persisted": False,
        },
    }


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_READOUT_CONFIG",
    "DEFAULT_SECTOR_VOCABULARY",
    "FULL_ATTENTION_LAYERS",
    "compact_selected_component_readout",
    "readout_component",
    "readout_selected_component",
    "unavailable_component_readout",
]
