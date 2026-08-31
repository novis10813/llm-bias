"""Direction C V1: frozen-family L16 instruction-context readout.

This workflow reads complete vocabulary distributions in memory and writes only
compact diagnostics.  It is a descriptive transported-representation readout,
not causal evidence.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import fp32_next_token_logits
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import decode_token, format_prompt, input_ids
from llm_bias.jspace_intervention.activation_patching import resolve_prompt_spans
from llm_bias.jspace_intervention.candidates import single_leading_space_token

SCHEMA_VERSION = 1
CONFIG_ARTIFACT_TYPE = "l16_context_readout_config"
PAIRS_ARTIFACT_TYPE = "valence_pairs"
RECORD_ARTIFACT_TYPE = "l16_context_readout"
SUMMARY_ARTIFACT_TYPE = "l16_context_readout_analysis"

PRIMARY_LAYER = 16
PRIMARY_SPAN = "instruction_context"
DEFAULT_TOP_K = 30
DEFAULT_DATASET = "l16-context-readout"
DEFAULT_DECISION_PREFIX = '{\n  "decision": "'
REQUIRED_SITES = (
    (6, "all_evidence", "l6_all_evidence"),
    (16, "instruction_context", "l16_instruction_context"),
    (30, "final_position", "l30_final_position"),
    (16, "header", "l16_header"),
)
FAMILY_NAMES = (
    "financial_evidence",
    "technology_sector",
    "financial_services_sector",
)

READOUT_CONTRACT = (
    "FP32 final norm and LM head, full vocabulary softmax at every span position, "
    "then mean the complete per-position probabilities before top-k selection"
)
INTERPRETATION_LIMIT = (
    "Jacobian-lens transported representation readout; token mass, rank, and top-k "
    "describe vocabulary alignment and are not causal effect, attention map, "
    "chain-of-thought, discrete reasoning path, or complete mediation path"
)


def _json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} not found: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_token_contract(tokenizer: Any, token_id: int, token: str, *, label: str) -> dict[str, Any]:
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"{label} token_id must be a non-negative integer")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"{label} token must be a non-empty string")
    resolved = single_leading_space_token(tokenizer, token.strip())
    if resolved is None:
        raise ValueError(f"{label} is not one complete leading-space token")
    resolved_id, decoded = resolved
    if resolved_id != token_id:
        raise ValueError(
            f"{label} token id mismatch: artifact={token_id}, tokenizer={resolved_id}"
        )
    if decoded.strip().lower() != token.strip().lower():
        raise ValueError(f"{label} decoded token does not match artifact token")
    return {"token_id": token_id, "token": decoded}


def _candidate_tokens(payload: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    if payload.get("artifact_type") != "frozen_candidate_suggestions":
        raise ValueError("frozen candidates must be a frozen_candidate_suggestions artifact")
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("frozen candidates must use schema_version 2")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 12:
        raise ValueError("frozen candidates must contain exactly 12 valence candidates")
    result = []
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            raise ValueError(f"candidate {index} is not an object")
        token_id = item.get("token_id")
        token = item.get("token")
        checked = _validate_token_contract(
            tokenizer, token_id, token, label=f"financial_evidence candidate {index}"
        )
        side = item.get("side")
        if side not in {"positive", "negative"}:
            raise ValueError(f"financial_evidence candidate {index} has invalid side")
        result.append({
            **checked,
            "side": side,
            "nomination_rule": "all 12 candidates from frozen_candidate_suggestions.json",
        })
    return result


def _prototype_tokens(
    payload: Mapping[str, Any], key: str, tokenizer: Any, *, model: str
) -> list[dict[str, Any]]:
    if payload.get("artifact_type") != "jspace_intervention_config" or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("TF-IDF input must be a schema_version 1 jspace_intervention_config artifact")
    if payload.get("source", {}).get("score_type") != "contrastive_tfidf" or payload.get("target", {}).get("score_type") != "contrastive_tfidf":
        raise ValueError("TF-IDF prototypes must be contrastive_tfidf")
    model_name = payload.get("model")
    if model_name is not None and model_name != model:
        raise ValueError("TF-IDF config model does not match requested model")
    prototype = payload.get(key)
    if not isinstance(prototype, Mapping):
        raise ValueError(f"TF-IDF config is missing {key} prototype")
    tokens = prototype.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"TF-IDF {key} prototype has no tokens")
    result = []
    for index, item in enumerate(tokens):
        if not isinstance(item, Mapping):
            raise ValueError(f"TF-IDF {key} token {index} is not an object")
        checked = _validate_token_contract(
            tokenizer, item.get("token_id"), str(item.get("token", "")),
            label=f"{key} prototype {index}",
        )
        result.append({
            **checked,
            "selection_score": item.get("selection_score"),
            "weight": item.get("weight"),
            "nomination_rule": "all frozen contrastive-TFIDF prototype tokens",
        })
    return result


def _validate_families(families: Mapping[str, Any]) -> None:
    missing = set(FAMILY_NAMES) - set(families)
    if missing:
        raise ValueError(f"config is missing families: {sorted(missing)}")
    if len(families["financial_evidence"].get("tokens", [])) != 12:
        raise ValueError("financial_evidence family must contain all 12 frozen candidates")
    seen: dict[int, str] = {}
    for family in FAMILY_NAMES:
        items = families[family].get("tokens") if isinstance(families[family], Mapping) else None
        if not isinstance(items, list) or not items:
            raise ValueError(f"family {family!r} is incomplete")
        for item in items:
            token_id = int(item.get("token_id", -1)) if isinstance(item, Mapping) else -1
            previous = seen.get(token_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate token id {token_id} across families {previous!r} and {family!r}"
                )
            seen[token_id] = family


def prepare_context_readout_config(
    frozen_candidates_path: str | Path | None = None,
    tfidf_config_path: str | Path | None = None,
    model: str | None = None,
    output_path: str | Path | None = None,
    *,
    candidates_path: str | Path | None = None,
) -> Path:
    """Freeze Direction C V1 vocabulary families without running a model."""
    if frozen_candidates_path is None:
        frozen_candidates_path = candidates_path
    if frozen_candidates_path is None or tfidf_config_path is None or model is None or output_path is None:
        raise TypeError("frozen candidates, TF-IDF config, model, and output are required")
    candidates_path = Path(frozen_candidates_path)
    tfidf_path = Path(tfidf_config_path)
    output = Path(output_path)
    candidate_payload = _json_object(candidates_path, "frozen candidates")
    tfidf_payload = _json_object(tfidf_path, "TF-IDF config")
    tokenizer = load_tokenizer(model)
    financial = _candidate_tokens(candidate_payload, tokenizer)
    technology = _prototype_tokens(tfidf_payload, "source", tokenizer, model=model)
    financial_sector = _prototype_tokens(tfidf_payload, "target", tokenizer, model=model)
    families = {
        "financial_evidence": {
            "nomination_rule": "all 12 candidates from frozen_candidate_suggestions.json",
            "source_artifact": str(candidates_path),
            "tokens": financial,
        },
        "technology_sector": {
            "nomination_rule": "source contrastive-TFIDF prototype tokens",
            "source_artifact": str(tfidf_path),
            "tokens": technology,
        },
        "financial_services_sector": {
            "nomination_rule": "target contrastive-TFIDF prototype tokens",
            "source_artifact": str(tfidf_path),
            "tokens": financial_sector,
        },
    }
    _validate_families(families)
    config = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": CONFIG_ARTIFACT_TYPE,
        "model": str(model),
        "sources": {
            "frozen_candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path), "artifact_type": "frozen_candidate_suggestions", "schema_version": 2},
            "tfidf_config": {"path": str(tfidf_path), "sha256": sha256_file(tfidf_path), "artifact_type": "jspace_intervention_config", "schema_version": 1},
        },
        "families": families,
        "primary": {"layer": PRIMARY_LAYER, "span": PRIMARY_SPAN},
        "comparisons": [
            {"layer": 6, "span": "all_evidence", "label": "l6_all_evidence"},
            {"layer": 30, "span": "final_position", "label": "l30_final_position"},
            {"layer": 16, "span": "header", "label": "l16_header"},
        ],
        "top_k": DEFAULT_TOP_K,
        "readout_contract": READOUT_CONTRACT,
        "formal_success_gate": False,
        "formal_gate_status": "not_defined_for_Direction_C_V1",
    }
    write_json(output, config, overwrite=False)
    return output


def validate_context_config(
    config: Mapping[str, Any], *, tokenizer: Any | None = None, model: str | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    if config.get("artifact_type") != CONFIG_ARTIFACT_TYPE or int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("invalid L16 context readout config artifact type/version")
    if model is not None and config.get("model") != model:
        raise ValueError("context readout config model does not match requested model")
    if config.get("formal_success_gate") is not False:
        raise ValueError("Direction C V1 formal_success_gate must be false")
    if config.get("top_k") != DEFAULT_TOP_K:
        raise ValueError("Direction C V1 top_k must remain 30")
    primary = config.get("primary")
    if primary != {"layer": PRIMARY_LAYER, "span": PRIMARY_SPAN}:
        raise ValueError("context readout primary site must be L16 instruction_context")
    families = config.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("context readout config is missing families")
    _validate_families(families)
    expected_comparisons = [
        {"layer": 6, "span": "all_evidence", "label": "l6_all_evidence"},
        {"layer": 30, "span": "final_position", "label": "l30_final_position"},
        {"layer": 16, "span": "header", "label": "l16_header"},
    ]
    if config.get("comparisons") != expected_comparisons:
        raise ValueError("context readout comparisons do not match the frozen Direction C V1 sites")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("context readout config is missing source bindings")
    for name in ("frozen_candidates", "tfidf_config"):
        ref = sources.get(name)
        if not isinstance(ref, Mapping):
            raise ValueError(f"config is missing source binding {name}")
        path = Path(str(ref.get("path", "")))
        digest = _require_digest(ref.get("sha256"), f"sources.{name}.sha256")
        if verify_sources:
            if not path.is_file():
                raise FileNotFoundError(f"bound config source is missing: {path}")
            actual = sha256_file(path)
            if actual != digest:
                raise ValueError(f"bound config source hash mismatch for {name}")
    if tokenizer is not None:
        for family in FAMILY_NAMES:
            for index, item in enumerate(families[family]["tokens"]):
                checked = _validate_token_contract(
                    tokenizer, int(item["token_id"]), str(item["token"]),
                    label=f"{family} token {index}",
                )
                if checked["token"] != item["token"]:
                    raise ValueError(f"config decoded token changed for {family} token {index}")
    return dict(config)


@torch.no_grad()
def context_readout_vector(
    model: Any, lens: Any, residual: torch.Tensor, layer: int,
) -> torch.Tensor:
    """Return the mean complete-vocabulary softmax over a residual span.

    ``lens`` may be a LoadedLens wrapper or the underlying lens.  The vector
    stays in memory and callers must not serialize it.
    """
    if not torch.is_tensor(residual) or residual.ndim != 2:
        raise ValueError("residual must have shape [positions, d_model]")
    if residual.shape[0] < 1:
        raise ValueError("residual span must contain at least one position")
    actual_lens = getattr(lens, "lens", lens)
    final_layer = int(model.n_layers) - 1
    layer = int(layer)
    if layer < 0 or layer > final_layer:
        raise ValueError(f"requested layer {layer} is outside model range 0..{final_layer}")
    if int(layer) != final_layer:
        source_layers = getattr(actual_lens, "source_layers", None)
        if source_layers is None:
            source_layers = tuple(getattr(actual_lens, "jacobians", {}).keys())
        if int(layer) not in {int(value) for value in source_layers}:
            raise ValueError(f"lens does not cover requested layer {layer}")
        head_device = model._lm_head.weight.device
        transported = actual_lens.transport(residual.float().to(head_device), int(layer))
    else:
        transported = residual.float().to(model._lm_head.weight.device)
    logits = fp32_next_token_logits(model, transported)
    probabilities = torch.softmax(logits.float(), dim=-1)
    return probabilities.mean(dim=0).detach()


# Descriptive aliases used by callers that name the aggregation contract.
mean_softmax_readout = context_readout_vector
readout_vector = context_readout_vector


def _prepare_scoring_condition(tokenizer: Any, pair: Mapping[str, Any], condition: str, decision_prefix: str) -> tuple[str, dict[str, tuple[int, int]], list[int]]:
    from llm_bias.jspace_intervention.activation_patching import _prepare_scoring_condition as prepare
    scoring_prompt, shifted = prepare(
        tokenizer, pair["prompts"][condition], pair["evidence_char_spans"][condition],
        decision_prefix=decision_prefix,
    )
    return scoring_prompt, {key: tuple(value) for key, value in shifted.items()}, input_ids(tokenizer, scoring_prompt)


def _rank(probabilities: torch.Tensor, token_id: int) -> int:
    return int((torch.argsort(probabilities, descending=True) == int(token_id)).nonzero(as_tuple=False)[0].item()) + 1


def _compact_vector(
    probabilities: torch.Tensor, *, tokenizer: Any, top_k: int, families: Mapping[str, Any],
) -> dict[str, Any]:
    vector = probabilities.float()
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("readout vector must be a finite vocabulary vector")
    order = torch.argsort(vector, descending=True)
    k = min(int(top_k), vector.numel())
    token_cache: dict[int, str] = {}
    top_tokens = []
    for rank, token_id in enumerate(order[:k].tolist(), 1):
        token_id = int(token_id)
        top_tokens.append({
            "rank": rank, "token_id": token_id,
            "token": token_cache.setdefault(token_id, decode_token(tokenizer, token_id)),
            "probability": float(vector[token_id]),
        })
    family_rows: dict[str, Any] = {}
    for family in FAMILY_NAMES:
        tokens = families[family]["tokens"]
        details = []
        mass = 0.0
        for item in tokens:
            token_id = int(item["token_id"])
            probability = float(vector[token_id])
            mass += probability
            details.append({
                "token_id": token_id,
                "token": str(item["token"]),
                "rank": _rank(vector, token_id),
                "probability": probability,
            })
        family_rows[family] = {"mass": mass, "tokens": details}
    entropy = float(-(vector.clamp_min(1e-12).log() * vector).sum())
    return {
        "top_tokens": top_tokens,
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(max(vector.numel(), 2)),
        "families": family_rows,
    }


def _aggregate_vectors(vectors: Mapping[tuple[str, str], list[torch.Tensor]]) -> dict[str, Any]:
    return {
        f"{condition}:{site}": torch.stack(values).double().mean(dim=0)
        for (condition, site), values in vectors.items()
        if values
    }


def _sector_contrast(records: Sequence[Mapping[str, Any]], families: Mapping[str, Any]) -> dict[str, Any]:
    """Compute descriptive equal-ticker Technology minus Financial Services masses."""
    by_ticker: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in records:
        sector = str(row.get("sector", ""))
        if sector not in {"Technology", "Financial Services"}:
            continue
        for family in FAMILY_NAMES:
            by_ticker[(str(row["ticker"]), str(row["site"]), str(row["condition"]), sector, family)].append(
                float(row["readout"]["families"][family]["mass"])
            )
    result: dict[str, Any] = {}
    for site in sorted({str(row["site"]) for row in records}):
        result[site] = {}
        for condition in ("positive", "negative"):
            result[site][condition] = {}
            for family in FAMILY_NAMES:
                technology = {
                    ticker: statistics.fmean(values)
                    for (ticker, row_site, row_condition, sector, row_family), values in by_ticker.items()
                    if row_site == site and row_condition == condition and sector == "Technology" and row_family == family
                }
                financial = {
                    ticker: statistics.fmean(values)
                    for (ticker, row_site, row_condition, sector, row_family), values in by_ticker.items()
                    if row_site == site and row_condition == condition and sector == "Financial Services" and row_family == family
                }
                technology_values = list(technology.values())
                financial_values = list(financial.values())
                result[site][condition][family] = {
                    "technology_mean_mass": statistics.fmean(technology_values) if technology_values else None,
                    "financial_services_mean_mass": statistics.fmean(financial_values) if financial_values else None,
                    "technology_minus_financial_services": (
                        statistics.fmean(technology_values) - statistics.fmean(financial_values)
                        if technology_values and financial_values else None
                    ),
                    "technology_ticker_count": len(technology_values),
                    "financial_services_ticker_count": len(financial_values),
                    "equal_ticker_aggregation": bool(technology_values and financial_values),
                }
    return result


def _aggregate_site(
    *, site: str, condition_means: Mapping[str, torch.Tensor], records: Sequence[Mapping[str, Any]],
    tokenizer: Any, top_k: int, families: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"site": site, "conditions": {}}
    for condition in ("positive", "negative"):
        vector = condition_means.get(condition)
        if vector is None:
            raise ValueError(f"missing {condition} vectors for {site}")
        compact = _compact_vector(vector, tokenizer=tokenizer, top_k=top_k, families=families)
        condition_records = [row for row in records if row["site"] == site and row["condition"] == condition]
        rank_stats = {}
        masses = compact["families"]
        for family in FAMILY_NAMES:
            rank_values = [
                float(token["rank"])
                for row in condition_records
                for token in row["readout"]["families"][family]["tokens"]
            ]
            rank_stats[family] = {
                "mean_rank": statistics.fmean(rank_values) if rank_values else None,
                "median_rank": statistics.median(rank_values) if rank_values else None,
            }
        result["conditions"][condition] = {
            **compact,
            "family_rank_statistics": rank_stats,
            "prompt_count": len(condition_records),
        }
    result["positive_minus_negative_family_mass"] = {
        family: result["conditions"]["positive"]["families"][family]["mass"]
        - result["conditions"]["negative"]["families"][family]["mass"]
        for family in FAMILY_NAMES
    }
    return result


def run_context_readout_pipeline(
    pairs_path: str | Path | None = None,
    config_path: str | Path | None = None,
    model_name: str | None = None,
    run_id: str | None = None,
    split_name: str = "test",
    dataset: str = DEFAULT_DATASET,
    artifact_root: str | Path = "artifacts",
    lens_path: str | Path | None = None,
    max_records: int | None = None,
    max_seq_len: int = 1024,
    top_k: int | None = None,
    *,
    input_path: str | Path | None = None,
) -> Path:
    """Run prepare, forward, analyze, and finalize for Direction C V1."""
    if pairs_path is None:
        pairs_path = input_path
    if pairs_path is None or config_path is None or model_name is None or run_id is None:
        raise TypeError("pairs, config, model, and run_id are required")
    if split_name not in {"discovery", "calibration", "test"}:
        raise ValueError("split must be discovery, calibration, or test")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    pairs_path = Path(pairs_path)
    config_path = Path(config_path)
    pairs = read_jsonl(pairs_path)
    if max_records is not None:
        pairs = pairs[:max_records]
    if not pairs:
        raise ValueError("context readout requires at least one pair")
    config = _json_object(config_path, "context readout config")
    preflight_tokenizer = load_tokenizer(model_name)
    validate_context_config(config, tokenizer=preflight_tokenizer, model=model_name)
    for number, pair in enumerate(pairs, 1):
        if pair.get("artifact_type") != PAIRS_ARTIFACT_TYPE:
            raise ValueError(f"{pairs_path}:{number} is not a valence_pairs record")
        if not isinstance(pair.get("prompts"), Mapping) or not isinstance(pair.get("evidence_char_spans"), Mapping):
            raise ValueError(f"{pairs_path}:{number} lacks prompts or evidence spans")
        for condition in ("positive", "negative"):
            if not isinstance(pair["prompts"].get(condition), str):
                raise ValueError(f"{pairs_path}:{number} lacks {condition} prompt")
            scoring_prompt, _spans, ids = _prepare_scoring_condition(
                preflight_tokenizer, pair, condition, DEFAULT_DECISION_PREFIX
            )
            if len(ids) + 1 > max_seq_len:
                raise ValueError(f"formatted {condition} prompt exceeds max_seq_len={max_seq_len}")
            if not scoring_prompt:
                raise ValueError("empty scoring prompt")
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(pairs_path, artifact_type=PAIRS_ARTIFACT_TYPE, stage="prepare", role="input")
    run.manifest.register_artifact(config_path, artifact_type=CONFIG_ARTIFACT_TYPE, stage="prepare", role="input")
    run.manifest.save()
    try:
        prepare_dir = run.run_directory / "prepare"
        prepare_metadata_path = prepare_dir / "metadata.json"
        with run.stage("prepare") as stage:
            write_metadata(prepare_metadata_path, {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "l16_context_readout_prepare_metadata",
                "model": model_name, "split": split_name,
                "pairs_path": str(pairs_path), "pairs_sha256": sha256_file(pairs_path),
                "config_path": str(config_path), "config_sha256": sha256_file(config_path),
                "pair_count": len(pairs), "ticker_count": len({str(p.get("ticker")) for p in pairs}),
                "state_persistence": "no raw residuals, activations, or full-vocabulary vectors",
            }, overwrite=False)
            stage.count(len(pairs))
        run.manifest.register_artifact(prepare_metadata_path, artifact_type="l16_context_readout_prepare_metadata", stage="prepare", role="output")
        run.manifest.save()

        model, tokenizer, fallback_device = load_model(model_name)
        loaded_lens = load_validated_lens(model=model, model_name=model_name, lens_path=lens_path, artifact_root=artifact_root)
        run.manifest.register_artifact(loaded_lens.path, artifact_type="jacobian_lens", stage="prepare", role="lens", metadata={"source": loaded_lens.source, "metadata": loaded_lens.metadata})
        run.manifest.save()
        n_layers = int(model.n_layers)
        required_layers = [6, 16, 30]
        if max(required_layers) >= n_layers:
            raise ValueError(f"model has no required context-readout layer L30: n_layers={n_layers}")
        k = int(top_k if top_k is not None else config.get("top_k", DEFAULT_TOP_K))
        if k < 1:
            raise ValueError("top_k must be positive")
        device = getattr(model, "input_device", fallback_device)
        families = config["families"]
        records: list[dict[str, Any]] = []
        vectors: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
        ticker_vectors: dict[tuple[str, str, str], list[torch.Tensor]] = defaultdict(list)
        forward_dir = run.run_directory / "forward"
        readout_path = forward_dir / "l16_context_readout.jsonl"
        with run.stage("forward") as stage:
            for pair in pairs:
                for condition in ("positive", "negative"):
                    scoring_prompt, shifted, ids = _prepare_scoring_condition(tokenizer, pair, condition, DEFAULT_DECISION_PREFIX)
                    encoded = torch.tensor([ids], dtype=torch.long, device=device)
                    residuals = record_residuals(model, encoded, required_layers)
                    spans = resolve_prompt_spans(tokenizer, scoring_prompt, shifted)
                    for layer, span, label in REQUIRED_SITES:
                        start, end = spans[span]
                        if end <= start:
                            raise ValueError(f"span {span} is empty for {pair.get('ticker')}/{condition}")
                        vector = context_readout_vector(
                            model=model, lens=loaded_lens, residual=residuals[layer][0, start:end, :], layer=layer,
                        )
                        compact = _compact_vector(vector, tokenizer=tokenizer, top_k=k, families=families)
                        record = {
                            "schema_version": SCHEMA_VERSION, "artifact_type": RECORD_ARTIFACT_TYPE,
                            "record_id": stable_record_id(str(pair.get("record_id")), condition, label),
                            "pair_record_id": pair.get("record_id"), "ticker": pair.get("ticker"),
                            "name": pair.get("name", ""), "sector": pair.get("sector", ""),
                            "source_trial_key": pair.get("source_trial_key"), "condition": condition,
                            "site": label, "layer": layer, "span": span,
                            "readout_positions": list(range(start, end)), "position_count": end - start,
                            "sequence_length": len(ids), "readout": compact,
                        }
                        records.append(record)
                        vectors[(condition, label)].append(vector.double().cpu())
                        ticker_vectors[(str(pair.get("ticker")), condition, label)].append(vector.double().cpu())
            count = write_jsonl(readout_path, records, overwrite=False)
            write_metadata(forward_dir / "metadata.json", {
                "schema_version": SCHEMA_VERSION, "artifact_type": "l16_context_readout_metadata",
                "model": model_name, "split": split_name, "layers": required_layers,
                "sites": [dict(layer=l, span=s, label=label) for l, s, label in REQUIRED_SITES],
                "top_k": k, "decision_prefix": DEFAULT_DECISION_PREFIX,
                "aggregation_contract": READOUT_CONTRACT, "records_written": count,
                "full_vectors_persisted": False, "lens_path": str(loaded_lens.path),
                "lens_metadata": loaded_lens.metadata,
            }, overwrite=False)
            stage.count(count)
        run.manifest.register_artifact(readout_path, artifact_type=RECORD_ARTIFACT_TYPE, stage="forward", role="output", record_count=count)
        forward_metadata_path = forward_dir / "metadata.json"
        run.manifest.register_artifact(forward_metadata_path, artifact_type="l16_context_readout_metadata", stage="forward", role="output")
        run.manifest.save()

        analyze_dir = run.run_directory / "analyze"
        summary_path = analyze_dir / "summary.json"
        with run.stage("analyze") as stage:
            condition_means = _aggregate_vectors(vectors)
            site_summaries = []
            for _layer, _span, label in REQUIRED_SITES:
                site_summaries.append(_aggregate_site(
                    site=label,
                    condition_means={condition: condition_means[f"{condition}:{label}"] for condition in ("positive", "negative")},
                    records=records, tokenizer=tokenizer, top_k=k, families=families,
                ))
            ticker_summary: dict[str, Any] = {}
            tickers = sorted({key[0] for key in ticker_vectors})
            for _layer, _span, label in REQUIRED_SITES:
                per_ticker = {}
                for ticker in tickers:
                    for condition in ("positive", "negative"):
                        values = ticker_vectors.get((ticker, condition, label), [])
                        if values:
                            per_ticker.setdefault(condition, {})[ticker] = torch.stack(values).double().mean(dim=0)
                if per_ticker.get("positive") and per_ticker.get("negative"):
                    common = sorted(set(per_ticker["positive"]) & set(per_ticker["negative"]))
                    if common:
                        differences = {
                            family: statistics.fmean(
                                float(sum(per_ticker["positive"][ticker][int(item["token_id"])] for item in families[family]["tokens"]))
                                - float(sum(per_ticker["negative"][ticker][int(item["token_id"])] for item in families[family]["tokens"]))
                                for ticker in common
                            )
                            for family in FAMILY_NAMES
                        }
                        ticker_summary[label] = {"ticker_count": len(common), "positive_minus_negative_family_mass_equal_ticker": differences}
            summary = {
                "schema_version": SCHEMA_VERSION, "artifact_type": SUMMARY_ARTIFACT_TYPE,
                "model": model_name, "split": split_name, "primary_site": {"layer": PRIMARY_LAYER, "span": PRIMARY_SPAN},
                    "sites": site_summaries, "equal_ticker_aggregates": ticker_summary,
                "sector_contrast": _sector_contrast(records, families),
                "formal_success_gate": False, "interpretation_limit": INTERPRETATION_LIMIT,
                "config": str(config_path), "config_sha256": sha256_file(config_path),
                "pairs": str(pairs_path), "pairs_sha256": sha256_file(pairs_path),
                "lens": str(loaded_lens.path), "lens_sha256": sha256_file(loaded_lens.path),
            }
            write_json(summary_path, summary, overwrite=False)
            write_metadata(analyze_dir / "metadata.json", {
                "schema_version": SCHEMA_VERSION, "artifact_type": "l16_context_readout_analysis_metadata",
                "aggregation": "condition full-vocabulary means before top-k; equal-ticker means where feasible",
                "summary_path": str(summary_path), "formal_success_gate": False,
            }, overwrite=False)
            stage.count(len(site_summaries))
        run.manifest.register_artifact(summary_path, artifact_type=SUMMARY_ARTIFACT_TYPE, stage="analyze", role="output")
        analysis_metadata_path = analyze_dir / "metadata.json"
        run.manifest.register_artifact(analysis_metadata_path, artifact_type="l16_context_readout_analysis_metadata", stage="analyze", role="output")
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


# Explicit aliases make the small pure helper convenient for callers/tests.
run_l16_context_readout_pipeline = run_context_readout_pipeline
run_l16_context_readout = run_context_readout_pipeline
prepare_l16_context_readout_config = prepare_context_readout_config

__all__ = [
    "CONFIG_ARTIFACT_TYPE", "DEFAULT_TOP_K", "FAMILY_NAMES", "PRIMARY_LAYER", "PRIMARY_SPAN",
    "REQUIRED_SITES", "SCHEMA_VERSION", "context_readout_vector", "mean_softmax_readout", "prepare_context_readout_config",
    "prepare_l16_context_readout_config", "readout_vector", "run_context_readout_pipeline",
    "run_l16_context_readout", "run_l16_context_readout_pipeline", "validate_context_config",
]
