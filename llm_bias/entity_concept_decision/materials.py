"""Strict, model-free loading of reviewed entity-concept materials."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {"schema_version", "concepts", "pairs"}
_CONCEPT_KEYS = {
    "concept_id",
    "definition",
    "positive_pole",
    "negative_pole",
    "excluded_interpretations",
}
_PAIR_KEYS = {
    "id",
    "concept_id",
    "family_id",
    "split",
    "text_positive",
    "text_negative",
    "review_status",
    "confound_tags",
}
_SPLITS = {"fit", "validation", "audit"}


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: str
    definition: str
    positive_pole: str
    negative_pole: str
    excluded_interpretations: tuple[str, ...]


@dataclass(frozen=True)
class ConceptPair:
    id: str
    concept_id: str
    family_id: str
    split: str
    text_positive: str
    text_negative: str
    review_status: str
    confound_tags: tuple[str, ...]


@dataclass(frozen=True)
class MaterialBundle:
    concepts: tuple[ConceptDefinition, ...]
    pairs: tuple[ConceptPair, ...]
    source_sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _check_finite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, path=f"{path}[{index}]")


def _keys(value: dict[str, Any], expected: set[str], *, path: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"{path} is missing field(s): {sorted(missing)!r}")


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _unique_strings(value: Any, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    result = tuple(_text(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique strings")
    return result


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_material_bundle(path: str | Path, *, expected_sha256: str) -> MaterialBundle:
    """Hash and strictly parse one reviewed material JSON document."""
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")

    source = Path(path)
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"material source hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid material JSON: {source}") from exc
    _check_finite(value)
    if not isinstance(value, dict):
        raise ValueError("material root must be a JSON object")
    _keys(value, _TOP_LEVEL_KEYS, path="root")
    schema_version = value["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ValueError("schema_version must be integer 1")
    if not isinstance(value["concepts"], list):
        raise ValueError("concepts must be a list")
    if not isinstance(value["pairs"], list):
        raise ValueError("pairs must be a list")
    if not 1 <= len(value["concepts"]) <= 3:
        raise ValueError("concepts must contain between 1 and 3 concepts")

    concepts: list[ConceptDefinition] = []
    concept_ids: set[str] = set()
    for index, raw in enumerate(value["concepts"]):
        path_prefix = f"concepts[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{path_prefix} must be an object")
        _keys(raw, _CONCEPT_KEYS, path=path_prefix)
        concept_id = _text(raw["concept_id"], field=f"{path_prefix}.concept_id")
        if concept_id in concept_ids:
            raise ValueError(f"duplicate concept id: {concept_id}")
        concept_ids.add(concept_id)
        definition = _text(raw["definition"], field=f"{path_prefix}.definition")
        positive_pole = _text(raw["positive_pole"], field=f"{path_prefix}.positive_pole")
        negative_pole = _text(raw["negative_pole"], field=f"{path_prefix}.negative_pole")
        if positive_pole == negative_pole:
            raise ValueError(f"{path_prefix}.positive_pole and negative_pole must differ")
        excluded = _unique_strings(
            raw["excluded_interpretations"],
            field=f"{path_prefix}.excluded_interpretations",
            allow_empty=False,
        )
        concepts.append(
            ConceptDefinition(
                concept_id,
                definition,
                positive_pole,
                negative_pole,
                excluded,
            )
        )

    pairs: list[ConceptPair] = []
    pair_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    normalized_splits: dict[str, str] = {}
    for index, raw in enumerate(value["pairs"]):
        path_prefix = f"pairs[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{path_prefix} must be an object")
        _keys(raw, _PAIR_KEYS, path=path_prefix)
        pair_id = _text(raw["id"], field=f"{path_prefix}.id")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair id: {pair_id}")
        pair_ids.add(pair_id)
        concept_id = _text(raw["concept_id"], field=f"{path_prefix}.concept_id")
        if concept_id not in concept_ids:
            raise ValueError(f"{path_prefix}.concept_id references unknown concept: {concept_id}")
        family_id = _text(raw["family_id"], field=f"{path_prefix}.family_id")
        split = _text(raw["split"], field=f"{path_prefix}.split")
        if split not in _SPLITS:
            raise ValueError(f"{path_prefix}.split must be one of fit, validation, audit")
        prior_split = family_splits.setdefault(family_id, split)
        if prior_split != split:
            raise ValueError(f"family_id {family_id!r} occurs in multiple splits")
        text_positive = _text(raw["text_positive"], field=f"{path_prefix}.text_positive")
        text_negative = _text(raw["text_negative"], field=f"{path_prefix}.text_negative")
        positive_key = _normalized(text_positive)
        negative_key = _normalized(text_negative)
        if not positive_key or not negative_key or positive_key == negative_key:
            raise ValueError(f"{path_prefix} positive and negative text must differ after normalization")
        for sentence_key in (positive_key, negative_key):
            previous_split = normalized_splits.setdefault(sentence_key, split)
            if previous_split != split:
                raise ValueError("normalized material text is duplicated across splits")
        review_status = _text(raw["review_status"], field=f"{path_prefix}.review_status")
        if review_status != "approved":
            raise ValueError(f"{path_prefix}.review_status must be approved")
        confound_tags = _unique_strings(
            raw["confound_tags"], field=f"{path_prefix}.confound_tags", allow_empty=True
        )
        pairs.append(
            ConceptPair(
                pair_id,
                concept_id,
                family_id,
                split,
                text_positive,
                text_negative,
                review_status,
                confound_tags,
            )
        )

    for concept_id in concept_ids:
        for split in _SPLITS:
            if not any(
                pair.concept_id == concept_id and pair.split == split for pair in pairs
            ):
                raise ValueError(f"concept {concept_id!r} lacks a family in {split} split")

    return MaterialBundle(tuple(concepts), tuple(pairs), actual_sha256)


__all__ = [
    "ConceptDefinition",
    "ConceptPair",
    "MaterialBundle",
    "load_material_bundle",
]
