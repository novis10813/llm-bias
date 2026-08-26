"""Candidate concept selection from discovery-only keyword artifacts."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from llm_bias.jspace_intervention.schemas import ConceptToken, PrototypeSpec


def single_leading_space_token(tokenizer: Any, concept: str) -> tuple[int, str] | None:
    """Resolve a concept only when ``' '+concept`` is one complete token."""
    encoded = tokenizer(" " + concept, add_special_tokens=False)
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) != 1:
        return None
    token_id = int(ids[0])
    decoded = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not decoded.startswith(" ") or decoded.strip().lower() != concept.lower():
        return None
    return token_id, decoded


def select_prototype(
    rows: Iterable[dict],
    *,
    tokenizer: Any,
    sector: str,
    score_type: str,
    top_n: int = 4,
    min_logodds_z: float = 2.0,
    excluded: set[str] | None = None,
    contrast_sector: str | None = None,
) -> PrototypeSpec:
    """Select eligible single-token concepts in fixed score order.

    When ``contrast_sector`` is supplied for TF-IDF, selection uses the positive
    difference ``tfidf(sector) - tfidf(contrast_sector)``. This makes the two
    directional prototype vocabularies disjoint by construction.
    """
    if score_type not in {"tfidf", "logodds_z"}:
        raise ValueError("score_type must be tfidf or logodds_z")
    if contrast_sector is not None and score_type != "tfidf":
        raise ValueError("contrast_sector is supported only for TF-IDF")
    rows = list(rows)
    contrast_scores = {
        str(row["token"]).lower(): float(row.get("tfidf") or 0.0)
        for row in rows
        if row.get("document") == contrast_sector
    }
    excluded = {value.lower() for value in (excluded or {"buy", "sell"})}
    candidates = []
    for row in rows:
        if row.get("document") != sector:
            continue
        score = row.get(score_type)
        if score is None:
            continue
        concept = str(row["token"]).lower()
        score = float(score)
        if contrast_sector is not None:
            score -= contrast_scores.get(concept, 0.0)
        if score <= 0:
            continue
        if score_type == "logodds_z" and score < min_logodds_z:
            continue
        if concept in excluded:
            continue
        resolved = single_leading_space_token(tokenizer, concept)
        if resolved is None:
            continue
        token_id, decoded = resolved
        candidates.append((score, concept, token_id, decoded))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:top_n]
    if not selected:
        raise ValueError(f"no eligible {score_type} concepts for sector {sector!r}")
    total = sum(score for score, *_ in selected)
    tokens = tuple(
        ConceptToken(
            token=concept,
            token_id=token_id,
            weight=score / total,
            selection_score=score,
        )
        for score, concept, token_id, _decoded in selected
    )
    label = "contrastive_tfidf" if contrast_sector is not None else score_type
    return PrototypeSpec(
        name=f"{sector}:{label}", sector=sector, score_type=label, tokens=tokens
    )


__all__ = ["select_prototype", "single_leading_space_token"]
