"""Bounded Phase 1 V2 development-material conversion and prompt preparation."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.prompt_input import format_prompt, input_ids, token_span

_SOURCE_ID_RE = re.compile(r"^[CS]-(?:F|V|E|L|X)\d{2}-(?:P|N|PH|PL|NH|NL|A|B)$")
_ROUND2_ROW_ID_RE = re.compile(r"^[A-Z]+-(?:F|E|L|X)\d{2}-(?:P|N|PH|PL|NH|NL|A|B)$")
_ADVICE_RE = re.compile(r"\b(?:buy|sell)\b", re.IGNORECASE)

_PRIMARY_FAMILIES = {
    "C1": {"C-F01", "C-F04", "C-F05", "C-V01", "C-V03"},
    "C2": {"C-F02", "C-F03", "C-F06"},
    "S1": {"S-F01", "S-F02", "S-F03", "S-F05", "S-F06", "S-V02", "S-V04"},
    "S2": {"S-F04", "S-V01", "S-V03"},
}
_REPLACEMENT_IDS = {
    "S-F01-P",
    "S-F02-P",
    "S-F02-N",
    "S-V01-P",
    "S-V01-N",
    "S-E01-PH",
    "S-E01-PL",
}
_EXCLUDED_IDS = {"C-V02-P", "C-V02-N", "C-V04-P", "C-V04-N"}
_EXPECTED_FAMILIES = {
    f"{concept}-{kind}{number:02d}-{pole}"
    for concept in ("C", "S")
    for kind, numbers, poles in (
        ("F", range(1, 7), ("P", "N")),
        ("V", range(1, 5), ("P", "N")),
        ("E", range(1, 3), ("PH", "PL", "NH", "NL")),
        ("L", range(1, 2), ("A", "B")),
        ("X", range(1, 2), ("A", "B")),
    )
    for number in numbers
    for pole in poles
}


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _markdown_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or not _SOURCE_ID_RE.fullmatch(cells[0]):
            continue
        if len(cells) != 2:
            raise ValueError(f"{path}:{line_number} material row must have exactly two columns")
        if not cells[1]:
            raise ValueError(f"{path}:{line_number} material {cells[0]} has empty text")
        rows.append((cells[0], cells[1]))
    return rows


def _parse_source(path: Path) -> dict[str, str]:
    rows = _markdown_rows(path)
    result: dict[str, str] = {}
    for identifier, text in rows:
        if identifier in result:
            raise ValueError(f"duplicate material ID in source: {identifier}")
        if identifier not in _EXPECTED_FAMILIES:
            raise ValueError(f"unexpected material ID in source: {identifier}")
        result[identifier] = text
    if set(result) != _EXPECTED_FAMILIES:
        raise ValueError(
            "source must contain the exact 64-material ID matrix; "
            f"missing={sorted(_EXPECTED_FAMILIES - set(result))}, "
            f"extra={sorted(set(result) - _EXPECTED_FAMILIES)}"
        )
    return result


def _parse_replacements(path: Path) -> dict[str, str]:
    rows = _markdown_rows(path)
    result: dict[str, str] = {}
    for identifier, text in rows:
        if identifier not in _REPLACEMENT_IDS:
            raise ValueError(f"unexpected replacement ID in review: {identifier}")
        if identifier in result:
            raise ValueError(f"duplicate replacement ID in review: {identifier}")
        result[identifier] = text
    if set(result) != _REPLACEMENT_IDS:
        raise ValueError(
            "review must contain the exact seven replacement IDs; "
            f"missing={sorted(_REPLACEMENT_IDS - set(result))}, "
            f"extra={sorted(set(result) - _REPLACEMENT_IDS)}"
        )
    return result


def _validate_texts(texts: Mapping[str, str], *, label: str) -> None:
    normalized: dict[str, str] = {}
    for identifier, text in texts.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} {identifier} has empty text")
        if _ADVICE_RE.search(text):
            raise ValueError(f"{label} {identifier} contains investment advice")
        key = _normalized(text)
        if not key:
            raise ValueError(f"{label} {identifier} has empty normalized text")
        prior = normalized.get(key)
        if prior is not None:
            raise ValueError(f"normalized duplicate material text: {prior} and {identifier}")
        normalized[key] = identifier


def _family_id(identifier: str) -> str:
    return identifier.rsplit("-", 1)[0]


def _row_role(family_id: str) -> str:
    kind = family_id.split("-", 1)[1][0]
    return {"F": "primary", "V": "primary", "E": "evaluation", "L": "lexical", "X": "competitor"}[kind]


def _row_group(family_id: str) -> tuple[str, str | None]:
    for group_id, families in _PRIMARY_FAMILIES.items():
        if family_id in families:
            return group_id, None
    if family_id.startswith("C-E"):
        return family_id, "C2"
    if family_id.startswith("S-E"):
        return family_id, "S1"
    return family_id, None


def _comparison(identifier: str, concept_id: str, family_id: str, kind: str, positive_id: str, negative_id: str) -> dict[str, str]:
    return {
        "id": f"{family_id}-{kind}",
        "concept_id": concept_id,
        "family_id": family_id,
        "kind": kind,
        "positive_id": positive_id,
        "negative_id": negative_id,
    }


def _make_comparisons(rows_by_id: Mapping[str, dict[str, Any]]) -> list[dict[str, str]]:
    comparisons: list[dict[str, str]] = []
    for family_id in sorted({_family_id(identifier) for identifier in rows_by_id}):
        members = {identifier.rsplit("-", 1)[1]: identifier for identifier in rows_by_id if _family_id(identifier) == family_id}
        concept_id = family_id[0]
        kind = family_id.split("-", 1)[1][0]
        if kind in {"F", "V"}:
            comparisons.append(_comparison(family_id, concept_id, family_id, "primary", members["P"], members["N"]))
        elif kind == "E":
            for comparison_kind, positive, negative in (
                ("concept_at_positive_evaluation", "PH", "NH"),
                ("concept_at_negative_evaluation", "PL", "NL"),
                ("evaluation_at_positive_concept", "PH", "PL"),
                ("evaluation_at_negative_concept", "NH", "NL"),
            ):
                comparisons.append(_comparison(family_id, concept_id, family_id, comparison_kind, members[positive], members[negative]))
        elif kind in {"L", "X"}:
            comparisons.append(_comparison(family_id, concept_id, family_id, "lexical" if kind == "L" else "competitor", members["A"], members["B"]))
    return comparisons


def build_development_materials(source_path: str | Path, review_path: str | Path) -> dict[str, Any]:
    """Convert exactly the reviewed V2 Markdown sources into 60 development rows."""
    source = Path(source_path)
    review = Path(review_path)
    source_texts = _parse_source(source)
    replacements = _parse_replacements(review)
    _validate_texts(source_texts, label="source material")
    _validate_texts(replacements, label="replacement")
    for identifier, replacement in replacements.items():
        if replacement == source_texts[identifier]:
            raise ValueError(f"replacement {identifier} must differ from source")
        if _normalized(replacement) == _normalized(source_texts[identifier]):
            raise ValueError(f"replacement {identifier} duplicates source after normalization")
    all_material_texts = dict(source_texts)
    all_material_texts.update({f"replacement:{identifier}": text for identifier, text in replacements.items()})
    _validate_texts(all_material_texts, label="source/replacement material")

    selected = {identifier: text for identifier, text in source_texts.items() if identifier not in _EXCLUDED_IDS}
    selected.update(replacements)
    _validate_texts(selected, label="development material")

    rows: list[dict[str, Any]] = []
    for identifier in sorted(selected):
        family_id = _family_id(identifier)
        group_id, source_group = _row_group(family_id)
        suffix = identifier.rsplit("-", 1)[1]
        row: dict[str, Any] = {
            "id": identifier,
            "concept_id": identifier[0],
            "family_id": family_id,
            "group_id": group_id,
            "role": _row_role(family_id),
            "pole": suffix,
            "text": selected[identifier],
        }
        if source_group is not None:
            row["source_group"] = source_group
        rows.append(row)
    rows_by_id = {row["id"]: row for row in rows}
    comparisons = _make_comparisons(rows_by_id)
    if len(rows) != 60 or len(comparisons) != 38:
        raise ValueError(f"unexpected development material counts: rows={len(rows)}, comparisons={len(comparisons)}")
    return {
        "schema_version": 1,
        "protocol": "phase1-v2-development",
        "review_status": "ai_reviewed_development",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "review_path": str(review),
        "review_sha256": file_sha256(review),
        "rows": rows,
        "comparisons": comparisons,
    }


def _encoded_fields(encoded: Any) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    def get(name: str, default: Any = None) -> Any:
        if isinstance(encoded, Mapping):
            return encoded.get(name, default)
        return getattr(encoded, name, default)

    ids = get("input_ids")
    offsets = get("offset_mapping")
    specials = get("special_tokens_mask")
    if ids is None or offsets is None:
        raise ValueError("tokenizer must provide input_ids and offset_mapping")
    for name, value in (("input_ids", ids), ("offset_mapping", offsets), ("special_tokens_mask", specials)):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if name == "input_ids":
            ids = value
        elif name == "offset_mapping":
            offsets = value
        else:
            specials = value
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if specials is None:
        specials = [False] * len(offsets)
    elif specials and isinstance(specials[0], (list, tuple)):
        specials = specials[0]
    if not isinstance(ids, list) or not isinstance(offsets, list) or not isinstance(specials, list):
        raise ValueError("tokenizer encodings must be one-dimensional sequences")
    if len(ids) != len(offsets) or len(ids) != len(specials) or not ids:
        raise ValueError("tokenizer encodings must have matching non-empty lengths")
    normalized_offsets: list[tuple[int, int]] = []
    normalized_specials: list[bool] = []
    for index, (token_ids, offset, special) in enumerate(zip(ids, offsets, specials, strict=True)):
        if isinstance(token_ids, bool) or not isinstance(token_ids, int):
            raise ValueError(f"invalid token id at position {index}")
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError(f"invalid offset at position {index}")
        start, end = offset
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"invalid offset at position {index}")
        normalized_offsets.append((start, end))
        normalized_specials.append(bool(special))
    return [int(value) for value in ids], normalized_offsets, normalized_specials


def prepare_development_prompt(
    tokenizer: Any,
    text: str,
    *,
    instruction_suffix: str,
    answer_prefix: str,
    use_chat_template: bool = True,
) -> dict[str, Any]:
    """Prepare one development description with a strict instruction token span."""
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if not isinstance(instruction_suffix, str) or not instruction_suffix:
        raise ValueError("instruction_suffix must be a non-empty string")
    if not isinstance(answer_prefix, str) or not answer_prefix:
        raise ValueError("answer_prefix must be a non-empty string")
    if not isinstance(use_chat_template, bool):
        raise ValueError("use_chat_template must be a boolean")

    raw_prompt = (
        "Refer to the company description below to make a final investment decision.\n\n"
        f"Company description:\n{text}\n\n—\n\n{instruction_suffix}"
    )
    if raw_prompt.count(instruction_suffix) != 1:
        raise ValueError("instruction_suffix must occur exactly once in raw prompt")
    formatted = format_prompt(tokenizer, raw_prompt, use_chat_template=use_chat_template)
    if not isinstance(formatted, str):
        raise ValueError("formatted prompt must be text")
    occurrences = [match.start() for match in re.finditer(re.escape(instruction_suffix), formatted)]
    if len(occurrences) != 1:
        raise ValueError("instruction_suffix must occur exactly once outside answer_prefix")
    instruction_start = occurrences[0]
    instruction_end = instruction_start + len(instruction_suffix)
    formatted_with_answer = formatted + answer_prefix
    ids = input_ids(tokenizer, formatted_with_answer, add_special_tokens=False)
    encoded = tokenizer(
        formatted_with_answer,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    encoded_ids, offsets, specials = _encoded_fields(encoded)
    if encoded_ids != ids:
        raise ValueError("tokenizer input_ids and offset encoding do not match")
    span = token_span(
        tokenizer,
        formatted_with_answer,
        instruction_start,
        instruction_end,
        add_special_tokens=False,
    )
    if span is None:
        raise ValueError("instruction suffix has no token span")
    start, end = span
    selected = []
    for index, ((token_start, token_end), special) in enumerate(zip(offsets, specials, strict=True)):
        if special or token_end <= token_start:
            continue
        if token_end > instruction_start and token_start < instruction_end:
            if token_start < instruction_start or token_end > instruction_end:
                raise ValueError("token crosses instruction boundary")
            selected.append(index)
    if not selected or (start, end) != (selected[0], selected[-1] + 1) or selected != list(range(start, end)):
        raise ValueError("instruction token span is not contiguous or offset-aligned")
    if any(offsets[index][0] < instruction_start or offsets[index][1] > instruction_end for index in range(start, end)):
        raise ValueError("instruction token offsets exceed instruction suffix")
    if ids[start:end] != encoded_ids[start:end]:
        raise ValueError("instruction token IDs do not match offset encoding")
    if end >= len(ids):
        raise ValueError("answer_prefix must append at least one token")
    partition = {
        "before_instruction": [0, start],
        "instruction": [start, end],
        "after_instruction": [end, len(ids)],
    }
    if partition["before_instruction"][1] != partition["instruction"][0] or partition["instruction"][1] != partition["after_instruction"][0]:
        raise ValueError("instruction partition is not half-open and disjoint")
    return {
        "raw_prompt": raw_prompt,
        "formatted": formatted_with_answer,
        "input_ids": ids,
        "instruction_span": [start, end],
        "instruction_ids": ids[start:end],
        "final_position": len(ids) - 1,
        "partition": partition,
    }


def build_round2_materials(source_paths: Sequence[str | Path], *, expected_concepts: Sequence[str]) -> dict[str, Any]:
    """Parse round-2 table-format sources into development rows and comparisons.

    Each source row is a Markdown table line with six columns:
    ``ID | concept | group | role | pole | text``.  In round 2 each group is one
    family (one P/N primary pair, one PH/PL/NH/NL evaluation set, or one A/B
    control), so ``family_id`` equals ``group_id``.
    """
    expected = {str(c) for c in expected_concepts}
    if not expected:
        raise ValueError("expected_concepts must be non-empty")
    role_poles = {"primary": {"P", "N"}, "evaluation": {"PH", "PL", "NH", "NL"}, "lexical": {"A", "B"}, "competitor": {"A", "B"}}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    normalized: dict[str, str] = {}
    for raw_path in source_paths:
        path = Path(raw_path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "|" not in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 6:
                continue
            identifier, concept_id, group_id, role, pole, text = cells
            if not _ROUND2_ROW_ID_RE.fullmatch(identifier):
                continue
            if identifier in seen_ids:
                raise ValueError(f"{path}:{line_number} duplicate material ID {identifier}")
            if concept_id not in expected:
                raise ValueError(f"{path}:{line_number} {identifier} has unexpected concept {concept_id}")
            if role not in role_poles:
                raise ValueError(f"{path}:{line_number} {identifier} has invalid role {role}")
            if pole not in role_poles[role]:
                raise ValueError(f"{path}:{line_number} {identifier} has invalid pole {pole} for role {role}")
            if not text:
                raise ValueError(f"{path}:{line_number} {identifier} has empty text")
            if _ADVICE_RE.search(text):
                raise ValueError(f"{path}:{line_number} {identifier} contains investment advice")
            key = _normalized(text)
            prior = normalized.get(key)
            if prior is not None:
                raise ValueError(f"normalized duplicate material text: {prior} and {identifier}")
            normalized[key] = identifier
            seen_ids.add(identifier)
            rows.append({"id": identifier, "concept_id": concept_id, "family_id": group_id, "group_id": group_id, "role": role, "pole": pole, "text": text})
    if not rows:
        raise ValueError("round-2 sources produced no material rows")

    # Validate per-family pole completeness.
    by_family: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        by_family.setdefault((row["concept_id"], row["family_id"]), set()).add(row["pole"])
    for (concept_id, family_id), poles in sorted(by_family.items()):
        expected_role = next(row["role"] for row in rows if row["concept_id"] == concept_id and row["family_id"] == family_id)
        if poles != role_poles[expected_role]:
            raise ValueError(f"family {concept_id}/{family_id} ({expected_role}) must contain exactly {sorted(role_poles[expected_role])}, got {sorted(poles)}")
    groups_by_concept: dict[str, set[str]] = {}
    for row in rows:
        if row["role"] == "primary":
            groups_by_concept.setdefault(row["concept_id"], set()).add(row["group_id"])
    for concept_id in expected:
        if len(groups_by_concept.get(concept_id, set())) < 2:
            raise ValueError(f"concept {concept_id} needs at least two primary groups")

    rows_by_id = {row["id"]: row for row in rows}
    comparisons: list[dict[str, str]] = []
    for family_id in sorted({row["family_id"] for row in rows}):
        members = {row["pole"]: row["id"] for row in rows if row["family_id"] == family_id}
        concept_id = next(row["concept_id"] for row in rows if row["family_id"] == family_id)
        role = next(row["role"] for row in rows if row["family_id"] == family_id)
        if role == "primary":
            comparisons.append({"id": f"{family_id}-primary", "concept_id": concept_id, "family_id": family_id, "kind": "primary", "positive_id": members["P"], "negative_id": members["N"]})
        elif role == "evaluation":
            for kind, positive, negative in (
                ("concept_at_positive_evaluation", "PH", "NH"),
                ("concept_at_negative_evaluation", "PL", "NL"),
                ("evaluation_at_positive_concept", "PH", "PL"),
                ("evaluation_at_negative_concept", "NH", "NL"),
            ):
                comparisons.append({"id": f"{family_id}-{kind}", "concept_id": concept_id, "family_id": family_id, "kind": kind, "positive_id": members[positive], "negative_id": members[negative]})
        else:
            kind = "lexical" if role == "lexical" else "competitor"
            comparisons.append({"id": f"{family_id}-{kind}", "concept_id": concept_id, "family_id": family_id, "kind": kind, "positive_id": members["A"], "negative_id": members["B"]})

    return {
        "schema_version": 1,
        "protocol": "phase1-v2-round2",
        "review_status": "ai_reviewed_development",
        "source_paths": [str(Path(p)) for p in source_paths],
        "source_sha256": [file_sha256(Path(p)) for p in source_paths],
        "concepts": sorted(expected),
        "rows": rows,
        "comparisons": comparisons,
    }


__all__ = ["build_development_materials", "build_round2_materials", "prepare_development_prompt"]
