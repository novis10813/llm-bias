"""Caller-owned Phase 1 V2 development measurement and descriptive analysis.

This module is intentionally a development runner, not a formal audit pipeline.  The
caller owns the already-loaded model, tokenizer, and validated CPU basis.  Residuals
and directions are kept in RAM only; artifacts contain compact scalar diagnostics.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.artifacts.io import write_json, write_jsonl
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.prompt_input.encoding import continuation_token_ids, input_ids
from llm_bias.entity_concept_decision.concepts import concept_scores, fit_concept_direction
from llm_bias.entity_concept_decision.development_materials import prepare_development_prompt
from llm_bias.entity_concept_decision.materials import (
    ConceptDefinition,
    ConceptPair,
    MaterialBundle,
)

_ALLOWED_MODES = {"fake_smoke", "model_smoke", "development"}
_EXPECTED_GROUPS = {"C1", "C2", "S1", "S2"}
_PRIMARY_ROLES = {"primary"}
_ROLES = {"primary", "evaluation", "lexical", "competitor"}
_POLES = {"P", "N", "PH", "PL", "NH", "NL", "A", "B"}
_COMPARISON_KINDS = {
    "primary",
    "concept_at_positive_evaluation",
    "concept_at_negative_evaluation",
    "evaluation_at_positive_concept",
    "evaluation_at_negative_concept",
    "lexical",
    "competitor",
}


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_number(value: torch.Tensor | float | int, *, name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    return _finite_float(value, name=name)


def _validate_basis(basis: Any, model: Any) -> torch.Tensor:
    if not torch.is_tensor(basis) or basis.device.type != "cpu" or not torch.is_floating_point(basis):
        raise ValueError("basis must be a CPU floating tensor")
    if basis.layout != torch.strided or basis.ndim != 2:
        raise ValueError("basis must have shape [d, k]")
    d, k = map(int, basis.shape)
    if d <= 0 or k <= 0 or k > d:
        raise ValueError("basis must have shape [d, k] with 0 < k <= d")
    if not torch.isfinite(basis).all():
        raise ValueError("basis must contain only finite values")
    gram = basis.to(torch.float64).T @ basis.to(torch.float64)
    if not torch.allclose(gram, torch.eye(k, dtype=torch.float64), atol=1e-6, rtol=0.0):
        raise ValueError("basis columns must be orthonormal")
    n_layers = getattr(model, "n_layers", None)
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 17:
        raise ValueError("model must expose at least 17 decoder layers")
    if not hasattr(model, "layers") or len(model.layers) != n_layers:
        raise ValueError("model layers and n_layers must agree")
    return basis.detach().to(dtype=torch.float32).contiguous()


def _row_validation(materials: Mapping[str, Any], concept_ids: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(materials, Mapping):
        raise ValueError("materials must be the materials builder result mapping")
    if materials.get("protocol") not in {"phase1-v2-development", "phase1-v2-round2"}:
        raise ValueError("materials protocol must be a recognized development protocol")
    if materials.get("review_status") != "ai_reviewed_development":
        raise ValueError("materials review_status must be ai_reviewed_development")
    rows = materials.get("rows")
    comparisons = materials.get("comparisons")
    if not isinstance(rows, list) or not isinstance(comparisons, list):
        raise ValueError("materials must contain rows and comparisons lists")
    if not rows or not comparisons:
        raise ValueError("materials must contain non-empty rows and comparisons")
    expected_concepts = {str(c) for c in concept_ids}
    if not expected_concepts:
        raise ValueError("concept_ids must be non-empty")
    ids: set[str] = set()
    concepts: set[str] = set()
    groups: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}] must be an object")
        required = {"id", "concept_id", "family_id", "group_id", "role", "pole", "text"}
        if not required <= set(row):
            raise ValueError(f"rows[{index}] is missing required fields")
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"rows[{index}] has an invalid or duplicate id")
        if row["concept_id"] not in expected_concepts:
            raise ValueError(f"row {identifier} has an invalid concept")
        if row["role"] not in _ROLES or row["pole"] not in _POLES:
            raise ValueError(f"row {identifier} has an invalid role or pole")
        if not isinstance(row["group_id"], str) or not row["group_id"].strip():
            raise ValueError(f"row {identifier} has an invalid group")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"row {identifier} has empty text")
        ids.add(identifier)
        concepts.add(row["concept_id"])
        if row["role"] == "primary":
            groups.add(row["group_id"])
        by_id[identifier] = dict(row)
    if concepts != expected_concepts:
        raise ValueError(f"materials must contain exactly the concepts {sorted(expected_concepts)}")
    groups_by_concept: dict[str, set[str]] = {c: set() for c in expected_concepts}
    for row in rows:
        if row["role"] == "primary":
            groups_by_concept[row["concept_id"]].add(row["group_id"])
    for concept_id, concept_groups in groups_by_concept.items():
        if len(concept_groups) < 2:
            raise ValueError(f"concept {concept_id} needs at least two primary groups for leave-one-group-out")

    comparison_ids: set[str] = set()
    covered: set[str] = set()
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            raise ValueError(f"comparisons[{index}] must be an object")
        required = {"id", "concept_id", "family_id", "kind", "positive_id", "negative_id"}
        if not required <= set(comparison):
            raise ValueError(f"comparisons[{index}] is missing required fields")
        identifier = comparison["id"]
        if not isinstance(identifier, str) or identifier in comparison_ids:
            raise ValueError("comparison IDs must be unique")
        if comparison["kind"] not in _COMPARISON_KINDS:
            raise ValueError(f"comparison {identifier} has an invalid kind")
        positive, negative = comparison["positive_id"], comparison["negative_id"]
        if positive not in by_id or negative not in by_id:
            raise ValueError(f"comparison {identifier} references an unknown row")
        if by_id[positive]["family_id"] != comparison["family_id"] or by_id[negative]["family_id"] != comparison["family_id"]:
            raise ValueError(f"comparison {identifier} family coverage is invalid")
        if by_id[positive]["concept_id"] != comparison["concept_id"] or by_id[negative]["concept_id"] != comparison["concept_id"]:
            raise ValueError(f"comparison {identifier} concept coverage is invalid")
        comparison_ids.add(identifier)
        covered.update((positive, negative))
    if len(comparison_ids) != len(comparisons):
        raise ValueError("comparison IDs must be unique")
    if set(covered) != ids:
        raise ValueError("comparisons must cover every material row exactly")
    return [dict(row) for row in rows], [dict(item) for item in comparisons]


def _bundle_for_rows(rows: Sequence[Mapping[str, Any]], concept_ids: Sequence[str] | None = None, *, source_sha256: str) -> MaterialBundle:
    """Build the in-memory adapter used by the strict family reducer.

    ``ConceptPair.id`` remains the original material family identifier so vector
    lookup and the reducer's pair-set contract are stable.  ``family_id`` is the
    development weighting unit: primary families in the same ``group_id`` are
    deliberately collapsed into one equally weighted group.
    """
    primary = [row for row in rows if row["role"] == "primary"]
    if concept_ids is None:
        concept_ids = sorted({str(row["concept_id"]) for row in primary})
    by_family: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in primary:
        family = str(row["family_id"])
        by_family.setdefault(family, {})[str(row["pole"])] = row
    if any(set(members) != {"P", "N"} for members in by_family.values()):
        raise ValueError("each primary family must contain exactly positive and negative rows")
    pairs = tuple(
        ConceptPair(
            id=family,
            concept_id=str(members["P"]["concept_id"]),
            family_id=str(members["P"]["group_id"]),
            split="fit",
            text_positive=str(members["P"]["text"]),
            text_negative=str(members["N"]["text"]),
            review_status="ai_reviewed_development",
            confound_tags=(),
        )
        for family, members in sorted(by_family.items())
    )
    concepts = tuple(
        ConceptDefinition(concept_id, "development concept", "positive", "negative", ("formal audit interpretation",))
        for concept_id in sorted(set(concept_ids))
    )
    return MaterialBundle(concepts=concepts, pairs=pairs, source_sha256=source_sha256 or "development")


def _material_prompt_records(tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, instruction_suffix: str, answer_prefix: str) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    instruction_ids: list[int] | None = None
    for row in rows:
        prompt = prepare_development_prompt(tokenizer, str(row["text"]), instruction_suffix=instruction_suffix, answer_prefix=answer_prefix)
        current = list(prompt["instruction_ids"])
        if instruction_ids is None:
            instruction_ids = current
        elif current != instruction_ids:
            raise ValueError("all material instruction_ids must be identical")
        prepared.append({**dict(row), **prompt, "kind": "material"})
    return prepared


def _company_records(tokenizer: Any, company_prompts: Any, *, instruction_suffix: str, answer_prefix: str, expected_instruction_ids: list[int]) -> list[dict[str, Any]]:
    if company_prompts is None:
        return []
    if isinstance(company_prompts, Mapping):
        source = list(company_prompts.values())
    elif isinstance(company_prompts, Sequence) and not isinstance(company_prompts, (str, bytes)):
        source = list(company_prompts)
    else:
        raise ValueError("company_prompts must be a sequence, mapping, or None")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(source):
        if not isinstance(raw, Mapping):
            raise ValueError(f"company_prompts[{index}] must be a prepared record")
        item = dict(raw)
        identifier = item.get("id", item.get("company_id"))
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("company prompt IDs must be unique non-empty strings")
        seen.add(identifier)
        if "text" in item and "formatted" not in item:
            raise ValueError(f"company prompt {identifier} must be caller-prepared; refusing to rewrite it")
        required = {"formatted", "input_ids", "instruction_span", "instruction_ids", "partition", "final_position"}
        if not required <= set(item):
            raise ValueError(f"company prompt {identifier} is not a prepared record")
        rendered_ids = input_ids(tokenizer, str(item["formatted"]), add_special_tokens=False)
        if rendered_ids != list(item["input_ids"]):
            raise ValueError(f"company prompt {identifier} input_ids do not match formatted text")
        if list(item["instruction_ids"]) != expected_instruction_ids:
            raise ValueError(f"company prompt {identifier} instruction_ids do not match materials")
        start, end = item["instruction_span"]
        if not (0 <= start < end <= len(rendered_ids)) or rendered_ids[start:end] != expected_instruction_ids:
            raise ValueError(f"company prompt {identifier} has an invalid instruction span")
        if item["final_position"] != len(rendered_ids) - 1:
            raise ValueError(f"company prompt {identifier} has an invalid final position")
        result.append({**item, "id": identifier, "kind": "company"})
    return result


def _answer_ids(tokenizer: Any, formatted: str) -> tuple[list[int], int, int]:
    prompt_ids = input_ids(tokenizer, formatted, add_special_tokens=False)
    positive = continuation_token_ids(tokenizer, formatted, "buy")
    negative = continuation_token_ids(tokenizer, formatted, "sell")
    if len(positive) != 1 or len(negative) != 1 or positive[0] == negative[0]:
        raise ValueError("buy and sell must be distinct single-token continuations")
    return prompt_ids, positive[0], negative[0]


def _prepare_inputs(model: Any, tokenizer: Any, rows: list[dict[str, Any]], companies: list[dict[str, Any]], basis: torch.Tensor, *, instruction_suffix: str, answer_prefix: str, layer: int, min_norm: float, random_count: int, provenance: Mapping[str, Any]) -> None:
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 or layer >= int(model.n_layers):
        raise ValueError("layer must be a valid decoder layer")
    if not math.isfinite(float(min_norm)) or min_norm <= 0:
        raise ValueError("min_norm must be a finite positive number")
    if not isinstance(random_count, int) or isinstance(random_count, bool) or random_count < 0:
        raise ValueError("random_count must be a non-negative integer")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    if provenance.get("mode") not in _ALLOWED_MODES:
        raise ValueError("provenance.mode must be fake_smoke, model_smoke, or development")
    if provenance.get("review_status") != "ai_reviewed_development":
        raise ValueError("provenance.review_status must be ai_reviewed_development")
    if "upstream" not in provenance or "lens" not in provenance:
        raise ValueError("provenance must explicitly contain upstream and lens references")
    last = int(model.n_layers) - 1
    for item in [*rows, *companies]:
        ids = list(item["input_ids"])
        if not ids or any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise ValueError("prepared input_ids must be a non-empty integer sequence")
        if input_ids(tokenizer, str(item["formatted"]), add_special_tokens=False) != ids:
            raise ValueError(f"prepared input_ids do not match {item.get('id')}")
        start, end = item["instruction_span"]
        if not (0 <= start < end <= len(ids)) or list(item["instruction_ids"]) != ids[start:end]:
            raise ValueError(f"invalid instruction partition for {item.get('id')}")
        _answer_ids(tokenizer, str(item["formatted"]))
    if layer == last:
        raise ValueError("development layer must precede the final decoder layer")
    if basis.shape[0] <= 0:
        raise ValueError("basis has invalid dimension")


def _margin(model: Any, residual: torch.Tensor, positive_id: int, negative_id: int) -> float:
    if residual.ndim != 2 or residual.shape[0] != 1:
        raise ValueError("final residual must have shape [1, d]")
    log_probs = fp32_next_token_log_probs(model, residual)
    value = float(log_probs[0, positive_id].detach().cpu()) - float(log_probs[0, negative_id].detach().cpu())
    if not math.isfinite(value):
        raise ValueError("non-finite continuation margin")
    return value


def _fit_one(bundle: MaterialBundle, concept_id: str, vectors: Mapping[str, torch.Tensor], basis: torch.Tensor, min_norm: float, *, excluded_group: str | None = None):
    rows = vectors["__rows__"]
    by_family: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["role"] == "primary":
            by_family.setdefault(str(row["family_id"]), {})[str(row["pole"])] = row
    selected = [
        pair
        for pair in bundle.pairs
        if pair.concept_id == concept_id
        and (excluded_group is None or pair.family_id != excluded_group)
    ]
    if not selected:
        raise ValueError(f"no fit rows remain for concept {concept_id}")
    if any(pair.id not in by_family or set(by_family[pair.id]) != {"P", "N"} for pair in selected):
        raise ValueError("fit pair lookup is incomplete")
    subbundle = MaterialBundle(bundle.concepts, tuple(selected), bundle.source_sha256)
    pair_ids = [pair.id for pair in selected]
    positive = torch.stack([vectors[by_family[pair.id]["P"]["id"]] for pair in selected])
    negative = torch.stack([vectors[by_family[pair.id]["N"]["id"]] for pair in selected])
    return fit_concept_direction(subbundle, concept_id, positive=positive, negative=negative, pair_ids=pair_ids, basis=basis, min_norm=min_norm)


def _fit_all(bundle: MaterialBundle, rows: list[dict[str, Any]], vectors: Mapping[str, torch.Tensor], basis: torch.Tensor, min_norm: float, concept_ids: Sequence[str] | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if concept_ids is None:
        concept_ids = sorted({str(row["concept_id"]) for row in rows if row["role"] == "primary"})
    vector_map = dict(vectors)
    vector_map["__rows__"] = rows  # in-memory adapter only
    fits: dict[str, Any] = {}
    loo: dict[str, dict[str, Any]] = {}
    for concept_id in concept_ids:
        fits[concept_id] = _fit_one(bundle, concept_id, vector_map, basis, min_norm)
        loo[concept_id] = {}
        groups = sorted({row["group_id"] for row in rows if row["concept_id"] == concept_id and row["role"] == "primary"})
        for group in groups:
            loo[concept_id][group] = _fit_one(bundle, concept_id, vector_map, basis, min_norm, excluded_group=group)
    return fits, loo


def _fit_payload(fit: Any) -> dict[str, Any]:
    return {"status": fit.status, "reason": fit.reason, "source_norm": fit.source_norm, "projected_norm": fit.projected_norm, "retained_fraction": fit.retained_fraction, "n_pairs": fit.n_pairs, "n_families": fit.n_families}


def _analyze(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], companies: list[dict[str, Any]], vectors: Mapping[str, torch.Tensor], margins: Mapping[str, float], basis: torch.Tensor, bundle: MaterialBundle, min_norm: float, random_seed: int, random_count: int, concept_ids: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if concept_ids is None:
        concept_ids = sorted({str(row["concept_id"]) for row in rows if row["role"] == "primary"})
    fits, loo = _fit_all(bundle, rows, vectors, basis, min_norm, concept_ids)
    scalar_by_concept: dict[str, dict[str, float | None]] = {concept: {} for concept in concept_ids}
    for concept_id, fit in fits.items():
        ids = [row["id"] for row in rows]
        ids.extend(identifier for identifier in vectors if identifier not in {row["id"] for row in rows} and identifier != "__rows__")
        if fit.status == "ok":
            values = torch.stack([vectors[identifier] for identifier in ids])
            scores = concept_scores(values, fit.direction)
            for identifier, score in zip(ids, scores, strict=True):
                scalar_by_concept[concept_id][identifier] = _json_number(score, name="scalar score")
        else:
            for identifier in ids:
                scalar_by_concept[concept_id][identifier] = None
    output_comparisons: list[dict[str, Any]] = []
    for comparison in comparisons:
        positive, negative = comparison["positive_id"], comparison["negative_id"]
        concept_id = comparison["concept_id"]
        fit = fits[concept_id]
        record: dict[str, Any] = {**comparison, "margin_positive": margins[positive], "margin_negative": margins[negative], "margin_delta": _json_number(margins[positive] - margins[negative], name="margin delta"), "scalar_positive": scalar_by_concept[concept_id][positive], "scalar_negative": scalar_by_concept[concept_id][negative], "scalar_delta": None, "claim_status": "in_sample" if comparison["kind"] == "primary" else "descriptive"}
        if fit.status == "ok" and scalar_by_concept[concept_id][positive] is not None and scalar_by_concept[concept_id][negative] is not None:
            record["scalar_delta"] = _json_number(float(scalar_by_concept[concept_id][positive]) - float(scalar_by_concept[concept_id][negative]), name="scalar delta")
        elif fit.status != "ok":
            record["degenerate_reason"] = fit.reason
        output_comparisons.append(record)

    loo_summary: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["role"] == "primary":
            by_family.setdefault(str(row["family_id"]), {})[str(row["pole"])] = row
    for concept_id, groups in loo.items():
        for group_id, fit in groups.items():
            families = sorted(
                family for family, members in by_family.items()
                if members["P"]["concept_id"] == concept_id and members["P"]["group_id"] == group_id
            )
            item: dict[str, Any] = {"concept_id": concept_id, "excluded_group": group_id, "status": fit.status, "reason": fit.reason, "claim_status": "descriptive_non_independent", "pair_deltas": [], "group_mean_delta": None}
            if fit.status == "ok":
                deltas: list[float] = []
                for family in families:
                    members = by_family[family]
                    values = torch.stack([vectors[members["P"]["id"]], vectors[members["N"]["id"]]])
                    score = concept_scores(values, fit.direction)
                    delta = _json_number(score[0] - score[1], name="LO-group delta")
                    deltas.append(delta)
                    item["pair_deltas"].append({"family_id": family, "pair_id": family, "delta": delta})
                item["group_mean_delta"] = _json_number(sum(deltas) / len(deltas), name="LO-group mean") if deltas else None
            loo_summary.append(item)

    def _quantiles(values: Sequence[float]) -> dict[str, Any]:
        ordered = sorted(values)
        quantile = lambda q: None if not ordered else ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
        return {"n_deltas": len(ordered), "q05": quantile(0.05), "q50": quantile(0.50), "q95": quantile(0.95)}

    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    random_deltas: dict[str, list[float]] = {str(comparison["id"]): [] for comparison in comparisons}
    if random_count:
        coeffs = torch.randn((random_count, basis.shape[1]), generator=generator, dtype=torch.float32)
        coeffs = coeffs / torch.linalg.vector_norm(coeffs, dim=1, keepdim=True)
        directions = [(basis @ coeff).contiguous() for coeff in coeffs]
        for direction in directions:
            for comparison in comparisons:
                difference = (vectors[comparison["positive_id"]] - vectors[comparison["negative_id"]]).unsqueeze(0)
                delta = concept_scores(difference, direction)[0]
                random_deltas[str(comparison["id"])].append(_json_number(delta, name="random delta"))

    random_comparison_rows = []
    for comparison in comparisons:
        deltas = random_deltas[str(comparison["id"])]
        random_comparison_rows.append({"comparison_id": comparison["id"], "concept_id": comparison["concept_id"], "kind": comparison["kind"], "aggregation": "per-comparison empirical quantiles across random unit directions in Q", **_quantiles(deltas)})
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted(_COMPARISON_KINDS):
        kind_comparisons = [item for item in comparisons if item["kind"] == kind]
        pooled = [delta for comparison in kind_comparisons for delta in random_deltas[str(comparison["id"])] ]
        by_kind[kind] = {"comparison_count": len(kind_comparisons), "comparison_ids": [str(item["id"]) for item in kind_comparisons], "aggregation": "all per-comparison deltas pooled; empirical quantiles", **_quantiles(pooled)}
    random_summary: dict[str, Any] = {
        "random_count": random_count,
        "comparison_count": len(comparisons),
        "aggregation": "per-comparison empirical quantiles; kind aggregates pool all comparison deltas",
        "comparisons": random_comparison_rows,
        "by_kind": by_kind,
    }
    for concept_id in concept_ids:
        selected = [item for item in random_comparison_rows if item["concept_id"] == concept_id]
        selected_kinds = {kind: value for kind, value in by_kind.items() if any(item["kind"] == kind for item in comparisons if item["concept_id"] == concept_id)}
        random_summary[concept_id] = {"random_count": random_count, "comparison_count": len(selected), "aggregation": "per-comparison empirical quantiles across random unit directions in Q", "comparisons": selected, "by_kind": selected_kinds, "n_deltas": sum(int(item["n_deltas"]) for item in selected)}

    direction_summary: dict[str, Any] = {}
    for concept_id, fit in fits.items():
        group_metrics = []
        for group_id in sorted({row["group_id"] for row in rows if row["concept_id"] == concept_id and row["role"] == "primary"}):
            group = [row for row in rows if row["concept_id"] == concept_id and row["role"] == "primary" and row["group_id"] == group_id]
            families = sorted({str(row["family_id"]) for row in group})
            source = torch.stack([
                vectors[next(r["id"] for r in rows if r["family_id"] == family and r["pole"] == "P")]
                - vectors[next(r["id"] for r in rows if r["family_id"] == family and r["pole"] == "N")]
                for family in families
            ]).mean(dim=0)
            projected = basis @ (basis.T @ source)
            source_norm = torch.linalg.vector_norm(source)
            projected_norm = torch.linalg.vector_norm(projected)
            source_norm_value = _json_number(source_norm, name="source norm")
            projected_norm_value = _json_number(projected_norm, name="projected norm")
            metric = {"group_id": group_id, "source_norm": source_norm_value, "projected_norm": projected_norm_value, "retained_fraction": None if source_norm_value == 0.0 else _json_number(projected_norm / source_norm, name="retained fraction")}
            if fit.status == "ok":
                metric["similarity_to_full"] = _json_number(torch.dot(projected, fit.direction) / projected_norm, name="direction similarity") if projected_norm_value > 0 else None
            else:
                metric["similarity_to_full"] = None
            group_metrics.append(metric)
        direction_summary[concept_id] = {"fit": _fit_payload(fit), "groups": group_metrics}

    sentiment_residualization = _sentiment_residualization(rows, vectors, basis, fits, comparisons, companies, min_norm)

    summary = {"schema_version": 1, "scientific_status": "not_evaluated", "purpose": "development", "n_independent_families": None, "fits": {concept: _fit_payload(fit) for concept, fit in fits.items()}, "leave_one_group_out": loo_summary, "direction_diagnostics": direction_summary, "sentiment_residualization": sentiment_residualization, "random_controls": random_summary, "company_margins": [{"id": company["id"], "margin": margins[company["id"]], "scalar_scores": {concept: scalar_by_concept[concept][company["id"]] for concept in concept_ids}, "claim_status": "descriptive"} for company in companies], "comparison_count": len(output_comparisons), "primary_comparisons_in_sample": True}
    return output_comparisons, summary


def _sentiment_residualization(rows: list[dict[str, Any]], vectors: Mapping[str, torch.Tensor], basis: torch.Tensor, fits: Mapping[str, Any], comparisons: list[dict[str, Any]], companies: list[dict[str, Any]], min_norm: float) -> dict[str, Any]:
    """Fit a general-stance direction from evaluation rows and residualize each concept direction.

    Development-only descriptive control: the stance direction is the mean of the
    evaluation-effect pairs (PH-PL, NH-NL) pooled across evaluation families and
    both concepts, projected onto Q. Concept directions are orthogonalized against
    it and re-scored so we can report how much of the concept and evaluation response
    survives with general stance removed. Not a calibrated gate.
    """
    eval_pairs: list[torch.Tensor] = []
    eval_families: list[str] = []
    by_eval: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for row in rows:
        if row.get("role") != "evaluation":
            continue
        by_eval.setdefault((str(row["concept_id"]), str(row["family_id"])), {})[str(row["pole"])] = vectors[str(row["id"])]
    for (concept_id, family_id), poles in by_eval.items():
        for high, low in (("PH", "PL"), ("NH", "NL")):
            if high in poles and low in poles:
                eval_pairs.append(poles[high] - poles[low])
                eval_families.append(family_id)
    if not eval_pairs:
        return {"status": "no_evaluation_rows", "reason": "no PH/PL/NH/NL evaluation pairs found"}
    source = torch.stack(eval_pairs, dim=0).to(dtype=torch.float32).mean(dim=0)
    projected = basis @ (basis.T @ source)
    source_norm = float(torch.linalg.vector_norm(source).item())
    projected_norm = float(torch.linalg.vector_norm(projected).item())
    if source_norm <= min_norm or projected_norm <= min_norm:
        return {"status": "degenerate", "reason": "stance direction below min_norm", "source_norm": source_norm, "projected_norm": projected_norm}
    stance = (projected / projected_norm).contiguous()

    per_concept: dict[str, Any] = {}
    for concept_id, fit in fits.items():
        entry: dict[str, Any] = {"status": fit.status}
        if fit.status != "ok":
            entry["reason"] = fit.reason
            per_concept[concept_id] = entry
            continue
        direction = fit.direction.to(dtype=torch.float32)
        cosine = float(torch.dot(direction, stance).item())
        residual = (direction - cosine * stance).contiguous()
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if residual_norm <= min_norm:
            entry.update({"cosine_with_stance": cosine, "orthogonalized_status": "degenerate", "reason": "orthogonalized direction below min_norm"})
            per_concept[concept_id] = entry
            continue
        residual_dir = (residual / residual_norm).contiguous()
        entry.update({"cosine_with_stance": cosine, "orthogonalized_status": "ok", "orthogonalized_norm_before_renorm": residual_norm})

        def _rescore(direction_to_use: torch.Tensor) -> dict[str, float]:
            deltas: dict[str, float] = {}
            for comparison in comparisons:
                if comparison["concept_id"] != concept_id:
                    continue
                positive, negative = comparison["positive_id"], comparison["negative_id"]
                diff = (vectors[positive] - vectors[negative]).unsqueeze(0).to(dtype=torch.float32)
                deltas[str(comparison["id"])] = _json_number(concept_scores(diff, direction_to_use)[0], name="rescored delta")
            return deltas

        before = _rescore(direction)
        after = _rescore(residual_dir)

        def _mean_by_kind(kind: str) -> tuple[float | None, float | None]:
            before_vals = [before[c["id"]] for c in comparisons if c["concept_id"] == concept_id and c["kind"] == kind]
            after_vals = [after[c["id"]] for c in comparisons if c["concept_id"] == concept_id and c["kind"] == kind]
            b = sum(before_vals) / len(before_vals) if before_vals else None
            a = sum(after_vals) / len(after_vals) if after_vals else None
            return (None if b is None else _json_number(b, name="mean before"), None if a is None else _json_number(a, name="mean after"))

        concept_b, concept_a = _mean_by_kind("concept_at_positive_evaluation")
        concept_b2, concept_a2 = _mean_by_kind("concept_at_negative_evaluation")
        eval_b, eval_a = _mean_by_kind("evaluation_at_positive_concept")
        eval_b2, eval_a2 = _mean_by_kind("evaluation_at_negative_concept")
        entry["response"] = {
            "concept_mean_abs_before": None if None in (concept_b, concept_b2) else _json_number((abs(concept_b) + abs(concept_b2)) / 2, name="concept before"),
            "concept_mean_abs_after": None if None in (concept_a, concept_a2) else _json_number((abs(concept_a) + abs(concept_a2)) / 2, name="concept after"),
            "evaluation_mean_abs_before": None if None in (eval_b, eval_b2) else _json_number((abs(eval_b) + abs(eval_b2)) / 2, name="evaluation before"),
            "evaluation_mean_abs_after": None if None in (eval_a, eval_a2) else _json_number((abs(eval_a) + abs(eval_a2)) / 2, name="evaluation after"),
        }
        # Re-score company scalar on the residualized direction.
        company_scores: dict[str, float] = {}
        for company in companies:
            company_scores[str(company["id"])] = _json_number(concept_scores(vectors[str(company["id"])].unsqueeze(0).to(dtype=torch.float32), residual_dir)[0], name="residual company score")
        entry["company_orthogonalized_scores"] = company_scores
        per_concept[concept_id] = entry

    return {"status": "ok", "stance_n_pairs": len(eval_pairs), "stance_n_families": len(eval_families), "stance_source_norm": source_norm, "stance_projected_norm": projected_norm, "claim_status": "descriptive", "concepts": per_concept}


def _reference_values(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        return list(value)
    return [value]


def _validate_reference_values(provenance: Mapping[str, Any]) -> None:
    for role in ("upstream", "lens"):
        for reference in _reference_values(provenance[role]):
            path = reference.get("path") if isinstance(reference, Mapping) else reference
            if not isinstance(path, (str, Path)) or not Path(path).is_file():
                raise ValueError(f"provenance reference path does not exist: {path!r}")
            if isinstance(reference, Mapping) and "sha256" in reference:
                supplied = reference["sha256"]
                if not isinstance(supplied, str) or file_sha256(path) != supplied:
                    raise ValueError(f"provenance reference digest mismatch: {path}")


def _register_reference(run: ArtifactRun, value: Any, *, role: str, stage: str) -> None:
    for reference in _reference_values(value):
        if isinstance(reference, Mapping):
            path = reference.get("path")
            supplied = reference.get("sha256")
        else:
            path, supplied = reference, None
        metadata = {"sha256": supplied} if supplied is not None else None
        run.manifest.register_artifact(path, artifact_type=f"development_{role}_reference", stage=stage, role="lens" if role == "lens" else "input", metadata=metadata)


def run_development(*, model: Any, tokenizer: Any, device: Any, basis: torch.Tensor, materials: Mapping[str, Any], instruction_suffix: str, answer_prefix: str, company_prompts: Any, run_id: str, model_name: str, artifact_root: str | Path, provenance: Mapping[str, Any], layer: int = 15, min_norm: float = 1e-6, random_seed: int = 1729, random_count: int = 16, concept_ids: Sequence[str] | None = None) -> Path:
    """Run prepare → forward → analyze, then finalize the development artifact."""
    if concept_ids is None:
        concept_ids = sorted({str(row["concept_id"]) for row in materials.get("rows", [])})
    concept_ids = [str(c) for c in concept_ids]
    rows, comparisons = _row_validation(materials, concept_ids)
    source_sha_value = materials.get("source_sha256", "development")
    if isinstance(source_sha_value, (list, tuple)):
        source_sha_value = ";".join(str(item) for item in source_sha_value)
    source_meta = {key: materials.get(key) for key in ("source_path", "source_paths", "source_sha256", "review_path", "review_sha256", "concepts") if materials.get(key) is not None}
    basis = _validate_basis(basis, model)
    prepared_rows = _material_prompt_records(tokenizer, rows, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix)
    expected_instruction_ids = list(prepared_rows[0]["instruction_ids"])
    prepared_companies = _company_records(tokenizer, company_prompts, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix, expected_instruction_ids=expected_instruction_ids)
    _prepare_inputs(model, tokenizer, prepared_rows, prepared_companies, basis, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix, layer=layer, min_norm=min_norm, random_count=random_count, provenance=provenance)
    try:
        random_seed = int(random_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("random_seed must be an integer") from exc
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    _validate_reference_values(provenance)

    run = ArtifactRun.create(model_name, "entity-concept-decision-development", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            prepare_dir = run.run_directory / "prepare"
            materials_path = prepare_dir / "materials.json"
            prompts_path = prepare_dir / "prompts.jsonl"
            write_json(materials_path, dict(materials), overwrite=False)
            prompt_count = write_jsonl(prompts_path, [*prepared_rows, *prepared_companies], overwrite=False)
            parameters = list(model.parameters()) if hasattr(model, "parameters") else []
            metadata = {"purpose": "development", "provenance": dict(provenance), "sources": source_meta, "seed": random_seed, "dtype": str(parameters[0].dtype) if parameters else "unknown", "layer": layer, "basis_shape": list(basis.shape), "min_norm": min_norm}
            metadata_path = prepare_dir / "metadata.json"
            write_json(metadata_path, metadata, overwrite=False)
            run.manifest.register_artifact(materials_path, artifact_type="development_materials", stage="prepare", record_count=len(rows))
            run.manifest.register_artifact(prompts_path, artifact_type="development_prompts", stage="prepare", record_count=prompt_count)
            run.manifest.register_artifact(metadata_path, artifact_type="development_prepare_metadata", stage="prepare")
            _register_reference(run, provenance["upstream"], role="upstream", stage="prepare")
            _register_reference(run, provenance["lens"], role="lens", stage="prepare")
            stage.count(prompt_count)

        vectors: dict[str, torch.Tensor] = {}
        margins: dict[str, float] = {}
        records: list[dict[str, Any]] = []
        forward_inputs = [*prepared_rows, *prepared_companies]
        with run.stage("forward") as stage:
            for item in forward_inputs:
                started = time.perf_counter()
                ids, buy_id, sell_id = _answer_ids(tokenizer, item["formatted"])
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                captured = record_residuals(model, tensor, [layer, int(model.n_layers) - 1])
                if layer not in captured or int(model.n_layers) - 1 not in captured:
                    raise ValueError(f"missing residual capture for {item['id']}")
                instruction = captured[layer][:, item["instruction_span"][0]:item["instruction_span"][1], :]
                final = captured[int(model.n_layers) - 1][:, -1, :]
                if instruction.ndim != 3 or instruction.shape[0] != 1 or instruction.shape[1] <= 0 or final.shape != (1, basis.shape[0]):
                    raise ValueError(f"invalid residual capture shape for {item['id']}")
                mean = instruction.float().mean(dim=1).detach().cpu().reshape(-1)
                final_cpu = final.float().detach().cpu()
                if mean.shape != (basis.shape[0],) or not torch.isfinite(mean).all() or not torch.isfinite(final_cpu).all():
                    raise ValueError(f"non-finite or invalid residual capture for {item['id']}")
                margin = _margin(model, final.float(), buy_id, sell_id)
                vectors[item["id"]] = mean
                margins[item["id"]] = margin
                records.append({"id": item["id"], "kind": item["kind"], "margin": margin, "elapsed_seconds": time.perf_counter() - started})
                print(f"forward {len(records)}/{len(forward_inputs) + 1}: {item['id']}", flush=True)
                del captured, instruction, final, final_cpu
            first = prepared_rows[0]
            started = time.perf_counter()
            ids, buy_id, sell_id = _answer_ids(tokenizer, first["formatted"])
            captured = record_residuals(model, torch.tensor([ids], dtype=torch.long, device=device), [layer, int(model.n_layers) - 1])
            repeated_mean = captured[layer][:, first["instruction_span"][0]:first["instruction_span"][1], :].float().mean(dim=1).detach().cpu().reshape(-1)
            repeated_margin = _margin(model, captured[int(model.n_layers) - 1][:, -1, :].float(), buy_id, sell_id)
            mean_diff = float((repeated_mean - vectors[first["id"]]).abs().max().item())
            margin_diff = abs(repeated_margin - margins[first["id"]])
            records.append({"id": first["id"] + "__repeat", "kind": "material_repeat", "margin": repeated_margin, "elapsed_seconds": time.perf_counter() - started, "max_abs_mean_diff": mean_diff, "margin_diff": margin_diff})
            records_path = run.run_directory / "forward" / "records.jsonl"
            count = write_jsonl(records_path, records, overwrite=False)
            forward_meta = run.run_directory / "forward" / "metadata.json"
            write_json(forward_meta, {"dtype": "fp32_analysis_tail", "layer": layer, "final_layer": int(model.n_layers) - 1, "record_count": count, "repeat_max_abs_mean_diff": mean_diff, "repeat_margin_diff": margin_diff, "total_forward_seconds": sum(record["elapsed_seconds"] for record in records), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0}, overwrite=False)
            run.manifest.register_artifact(records_path, artifact_type="development_forward_records", stage="forward", record_count=count)
            run.manifest.register_artifact(forward_meta, artifact_type="development_forward_metadata", stage="forward")
            stage.count(count)

        with run.stage("analyze") as stage:
            vectors_for_fit = dict(vectors)
            comparisons_output, summary = _analyze(prepared_rows, comparisons, prepared_companies, vectors_for_fit, margins, basis, _bundle_for_rows(prepared_rows, concept_ids, source_sha256=source_sha_value), min_norm, random_seed, random_count, concept_ids)
            comparisons_path = run.run_directory / "analyze" / "comparisons.jsonl"
            summary_path = run.run_directory / "analyze" / "summary.json"
            count = write_jsonl(comparisons_path, comparisons_output, overwrite=False)
            write_json(summary_path, summary, overwrite=False)
            run.manifest.register_artifact(comparisons_path, artifact_type="development_comparisons", stage="analyze", record_count=count)
            run.manifest.register_artifact(summary_path, artifact_type="development_summary", stage="analyze")
            stage.count(count)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = ["run_development"]
