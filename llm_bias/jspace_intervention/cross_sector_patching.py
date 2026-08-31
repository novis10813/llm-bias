"""V1 cross-sector identity-header residual patching.

The preparation side consumes structured attribute-trial rows and never parses a
legacy rendered prompt. The forward side reuses the activation-patching hooks,
keeps residual caches in memory, and emits only compact margin records.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import sha256_file, sha256_json, stable_record_id
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import CandidateMargin, continuation_token_ids, score_single_token_margin_fp32
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids
from llm_bias.jspace_intervention.activation_patching import (
    cache_source_residuals,
    run_activation_patching_record,
)
from llm_bias.jspace_intervention.valence import (
    CONDITIONS,
    CONDITION_SIDES,
    PROMPT_TEMPLATE_VERSION,
    build_valence_pair,
    render_valence_prompt_with_spans,
    has_balanced_valence_evidence,
    validate_raw_trial_row,
)

ARTIFACT_SCHEMA_VERSION = 1
PAIRS_ARTIFACT_TYPE = "cross_sector_header_pairs"
RECORD_ARTIFACT_TYPE = "cross_sector_header_patch_record"
ANALYSIS_ARTIFACT_TYPE = "cross_sector_header_patch_analysis"
DEFAULT_DATASET = "cross-sector-header-patching"
DEFAULT_DECISION_PREFIX = '{\n  "decision": "'
DEFAULT_POSITIVE_CANDIDATE = "buy"
DEFAULT_NEGATIVE_CANDIDATE = "sell"
SECTORS = ("Technology", "Financial Services")
CONTROL_TYPES = ("cross_sector", "same_sector_peer", "name_form", "self_source")


def _selection_key(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"raw trial row is missing field {key!r}")
    return value


def load_raw_trial_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_split_assignments(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "jspace_intervention_splits":
        raise ValueError("split manifest artifact_type must be jspace_intervention_splits")
    assignments = payload.get("assignments")
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("split manifest is missing ticker assignments")
    return {str(ticker): str(split) for ticker, split in assignments.items()}


def _origin_sector(row: Mapping[str, Any]) -> str:
    for key in ("evidence_origin_sector", "origin_sector", "evidence_sector"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(row["sector"])


def _evidence_texts(row: Mapping[str, Any]) -> list[str]:
    identity = validate_raw_trial_row(row)
    return list(identity["evidence"].values())


def _has_identity_leak(row: Mapping[str, Any], identities: Sequence[Mapping[str, Any]]) -> bool:
    body = "\n".join(_evidence_texts(row)).casefold()
    return any(
        str(identity["ticker"]).casefold() in body
        or str(identity["name"]).casefold() in body
        for identity in identities
    )


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_raw_trial_row(row)
    return {
        "ticker": checked["ticker"],
        "name": checked["name"],
        "sector": checked["sector"],
    }


def _record_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_raw_trial_row(row)
    return {
        "record_id": str(row.get("record_id") or row["trial_key"]),
        "trial_key": checked["trial_key"],
        "trial_index": checked["trial_index"],
        "set_index": checked["set_index"],
        "trial_row_sha256": sha256_json(dict(row)),
        "evidence_item_hashes": {
            f"{side}_{kind}": sha256_json({"side": side, "kind": kind, "text": text})
            for (side, kind), text in sorted(checked["evidence"].items())
        },
    }


def _render(identity: Mapping[str, Any], row: Mapping[str, Any], valence: str) -> tuple[str, dict[str, list[int]]]:
    checked = validate_raw_trial_row(row)
    side = CONDITION_SIDES[valence]
    evidence = checked["evidence"]
    return render_valence_prompt_with_spans(
        ticker=str(identity["ticker"]),
        name=str(identity["name"]),
        qual_text=evidence[(side, "qual")],
        quant_text=evidence[(side, "quant")],
    )


def _make_prepared_record(
    *,
    pair_id: str,
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    valence: str,
    control_type: str,
    origin_sector: str,
) -> dict[str, Any]:
    if control_type not in CONTROL_TYPES:
        raise ValueError(f"unknown control type {control_type!r}")
    source_prompt, source_spans = _render(source, row, valence)
    target_prompt, target_spans = _render(target, row, valence)
    if source_prompt[source_prompt.index("--- Evidence ---") :] != target_prompt[target_prompt.index("--- Evidence ---") :]:
        raise ValueError("matched prompts do not share byte-identical evidence/instruction text")
    provenance = _record_identity(row)
    record_id = stable_record_id(
        "cross-sector-header",
        pair_id,
        valence,
        control_type,
        str(source["ticker"]),
        str(target["ticker"]),
        provenance["record_id"],
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": PAIRS_ARTIFACT_TYPE,
        "record_id": record_id,
        "pair_id": pair_id,
        "control_type": control_type,
        "evidence_valence": valence,
        "evidence_origin_sector": origin_sector,
        "source_identity": dict(source),
        "target_identity": dict(target),
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "source_evidence_char_spans": source_spans,
        "target_evidence_char_spans": target_spans,
        "source_trial_record_id": provenance["record_id"],
        "source_trial_key": provenance["trial_key"],
        "source_trial_index": provenance["trial_index"],
        "source_set_index": provenance["set_index"],
        "trial_row_sha256": provenance["trial_row_sha256"],
        "evidence_item_hashes": provenance["evidence_item_hashes"],
    }


def _name_form_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    """Apply ROT13 while preserving length, case, punctuation, and character class."""
    return {
        "ticker": codecs.decode(str(identity["ticker"]), "rot_13"),
        "name": codecs.decode(str(identity["name"]), "rot_13"),
        "sector": str(identity["sector"]),
    }


def prepare_cross_sector_pairs(
    rows: Sequence[Mapping[str, Any]],
    split_assignments: Mapping[str, str],
    *,
    split_name: str,
    seed: int = 0,
    max_pairs: int | None = None,
    max_records: int | None = None,
    source_ticker: str | None = None,
    target_ticker: str | None = None,
    allow_cross_split_smoke: bool = False,
    sectors: Sequence[str] = SECTORS,
    input_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare deterministic cross-sector pairs and bounded controls.

    Each identity pair receives one positive and one negative record for each
    available evidence-origin stratum. The same-sector and name-form controls
    use the same evidence row; self-source records patch a header back into the
    prompt that produced it.
    """
    if len(tuple(sectors)) != 2 or set(sectors) != set(SECTORS):
        raise ValueError("sectors must be Technology and Financial Services")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if (source_ticker is None) != (target_ticker is None):
        raise ValueError("source_ticker and target_ticker must be supplied together")
    by_sector_ticker: dict[str, dict[str, list[dict[str, Any]]]] = {sector: defaultdict(list) for sector in SECTORS}
    for raw in rows:
        if raw.get("condition") != "attribute":
            continue
        sector = raw.get("sector")
        ticker = raw.get("ticker")
        if sector not in SECTORS or not isinstance(ticker, str):
            continue
        explicit_smoke_ticker = ticker in {source_ticker, target_ticker}
        if split_assignments.get(ticker) != split_name and not (
            allow_cross_split_smoke and explicit_smoke_ticker
        ):
            continue
        if not has_balanced_valence_evidence(raw.get("evidence")):
            continue
        validate_raw_trial_row(raw)
        by_sector_ticker[str(sector)][ticker].append(dict(raw))
    if not all(by_sector_ticker[sector] for sector in SECTORS):
        raise ValueError(f"no balanced attribute trials for {SECTORS} in split {split_name!r}")

    ordered_tickers: dict[str, list[str]] = {}
    for sector in SECTORS:
        ordered_tickers[sector] = sorted(
            by_sector_ticker[sector], key=lambda ticker: _selection_key(seed, sector, ticker)
        )
    if allow_cross_split_smoke and (source_ticker is None or target_ticker is None):
        raise ValueError("allow_cross_split_smoke requires an explicit ticker pair")
    if source_ticker is not None and target_ticker is not None:
        if source_ticker not in by_sector_ticker[SECTORS[0]]:
            raise ValueError(f"source ticker {source_ticker!r} is not an eligible Technology ticker")
        if target_ticker not in by_sector_ticker[SECTORS[1]]:
            raise ValueError(
                f"target ticker {target_ticker!r} is not an eligible Financial Services ticker"
            )
        ticker_pairs = [(source_ticker, target_ticker)]
    else:
        pair_count = min(len(ordered_tickers[SECTORS[0]]), len(ordered_tickers[SECTORS[1]]))
        if max_pairs is not None:
            pair_count = min(pair_count, max_pairs)
        if pair_count < 1:
            raise ValueError("could not form a Technology/Financial Services ticker pair")
        ticker_pairs = list(
            zip(
                ordered_tickers[SECTORS[0]][:pair_count],
                ordered_tickers[SECTORS[1]][:pair_count],
                strict=True,
            )
        )

    # A pair gets one row from each canonical origin sector. Dropping a pair
    # when either origin lacks a leakage-free row keeps the two strata balanced.
    origin_values = list(SECTORS)
    if not origin_values:
        raise ValueError("no evidence-origin strata found")
    prepared: list[dict[str, Any]] = []
    for pair_index, (tech_ticker, fin_ticker) in enumerate(ticker_pairs):
        tech_rows = by_sector_ticker[SECTORS[0]][tech_ticker]
        fin_rows = by_sector_ticker[SECTORS[1]][fin_ticker]
        tech_identity = _identity(tech_rows[0])
        fin_identity = _identity(fin_rows[0])
        pair_id = stable_record_id("cross-sector-pair", split_name, seed, tech_ticker, fin_ticker)
        selected_rows: list[dict[str, Any]] = []
        for origin in origin_values:
            candidates = [
                row for sector_rows in by_sector_ticker.values()
                for rows_for_ticker in sector_rows.values()
                for row in rows_for_ticker
                if _origin_sector(row) == origin
                and not _has_identity_leak(row, (tech_identity, fin_identity))
            ]
            if not candidates:
                continue
            selected_rows.append(sorted(candidates, key=lambda row: _selection_key(seed, pair_id, origin, row["trial_key"]))[0])
        if len(selected_rows) != len(origin_values):
            continue
        for row in selected_rows:
            origin = _origin_sector(row)
            for valence in CONDITIONS:
                primary = _make_prepared_record(
                    pair_id=pair_id, row=row, source=tech_identity, target=fin_identity,
                    valence=valence, control_type="cross_sector", origin_sector=origin,
                )
                prepared.append(primary)
                # Self-source controls are guaranteed by the same leakage check.
                prepared.append(_make_prepared_record(
                    pair_id=pair_id, row=row, source=tech_identity, target=tech_identity,
                    valence=valence, control_type="self_source", origin_sector=origin,
                ))
                prepared.append(_make_prepared_record(
                    pair_id=pair_id, row=row, source=fin_identity, target=fin_identity,
                    valence=valence, control_type="self_source", origin_sector=origin,
                ))
                # Same-sector peer controls use a distinct ticker when one is
                # available; the record is omitted for a one-ticker sector.
                tech_peer = next((ticker for ticker in ordered_tickers[SECTORS[0]] if ticker != tech_ticker), None)
                fin_peer = next((ticker for ticker in ordered_tickers[SECTORS[1]] if ticker != fin_ticker), None)
                if tech_peer:
                    peer_identity = _identity(by_sector_ticker[SECTORS[0]][tech_peer][0])
                    if not _has_identity_leak(row, (tech_identity, peer_identity)):
                        prepared.append(_make_prepared_record(
                            pair_id=pair_id, row=row, source=tech_identity, target=peer_identity,
                            valence=valence, control_type="same_sector_peer", origin_sector=origin,
                        ))
                if fin_peer:
                    peer_identity = _identity(by_sector_ticker[SECTORS[1]][fin_peer][0])
                    if not _has_identity_leak(row, (fin_identity, peer_identity)):
                        prepared.append(_make_prepared_record(
                            pair_id=pair_id, row=row, source=fin_identity, target=peer_identity,
                            valence=valence, control_type="same_sector_peer", origin_sector=origin,
                        ))
                name_form = _name_form_identity(tech_identity)
                prepared.append(_make_prepared_record(
                    pair_id=pair_id, row=row, source=tech_identity, target=name_form,
                    valence=valence, control_type="name_form", origin_sector=origin,
                ))
        if max_pairs is not None and len({row["pair_id"] for row in prepared}) >= max_pairs:
            break
    if not prepared:
        raise ValueError("all candidate pairs failed evidence leakage exclusion")
    prepared.sort(key=lambda row: (row["pair_id"], row["evidence_origin_sector"], row["evidence_valence"], row["control_type"], row["record_id"]))
    for row in prepared:
        row["preparation_split"] = split_name
        row["source_identity_split"] = split_assignments.get(
            str(row["source_identity"]["ticker"])
        )
        row["target_identity_split"] = split_assignments.get(
            str(row["target_identity"]["ticker"])
        )
        row["cross_split_smoke"] = bool(
            row["source_identity_split"] != row["target_identity_split"]
        )
        if input_sha256 is not None:
            row["input_sha256"] = str(input_sha256)
        if split_manifest_sha256 is not None:
            row["split_manifest_sha256"] = str(split_manifest_sha256)
    if max_records is not None:
        prepared = prepared[:max_records]
    return prepared


def target_complete_header_mapping(
    source_header_span: tuple[int, int], target_header_span: tuple[int, int]
) -> dict[int, int]:
    """Map every target header token to a nearest-normalized source token."""
    source_start, source_end = source_header_span
    target_start, target_end = target_header_span
    source_width = source_end - source_start
    target_width = target_end - target_start
    if source_width <= 0 or target_width <= 0:
        return {}
    return {
        target_start + offset: source_start + (0 if target_width == 1 else round(offset * (source_width - 1) / (target_width - 1)))
        for offset in range(target_width)
    }


def _prepare_scoring_prompt(tokenizer: Any, prompt: str, spans: Mapping[str, Sequence[int]], decision_prefix: str) -> tuple[str, dict[str, list[int]]]:
    formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
    offset = formatted.find(prompt)
    if offset < 0:
        raise ValueError("formatted prompt does not contain the prepared raw prompt")
    return formatted + decision_prefix, {key: [offset + int(value[0]), offset + int(value[1])] for key, value in spans.items()}


def _normalized_transfer(source: float, target: float, delta: float) -> float | None:
    denominator = source - target
    if denominator == 0.0:
        return None
    value = delta / denominator
    return value if math.isfinite(value) else None


def run_cross_sector_header_record(
    *,
    model: Any,
    tokenizer: Any,
    prepared_record: Mapping[str, Any],
    source_identity: Mapping[str, Any] | None = None,
    target_identity: Mapping[str, Any] | None = None,
    source_prompt: str,
    target_prompt: str,
    source_spans: Mapping[str, Sequence[int]],
    target_spans: Mapping[str, Sequence[int]],
    layer: int,
    source_residuals: Mapping[int, torch.Tensor],
    source_clean_margin: CandidateMargin,
    target_clean_margin: CandidateMargin,
    positive_candidate: str,
    negative_candidate: str,
    device: Any,
    direction: str,
) -> dict[str, Any]:
    result = run_activation_patching_record(
        model=model, tokenizer=tokenizer, source_prompt=source_prompt, target_prompt=target_prompt,
        source_evidence_char_spans={key: list(value) for key, value in source_spans.items()},
        target_evidence_char_spans={key: list(value) for key, value in target_spans.items()},
        layers=[layer], span_condition="header", positive_candidate=positive_candidate,
        negative_candidate=negative_candidate, device=device,
        source_residuals={int(layer): source_residuals[int(layer)]},
        source_clean_margin=source_clean_margin, target_clean_margin=target_clean_margin,
    )
    source_value = float(source_clean_margin.value)
    target_value = float(target_clean_margin.value)
    delta = float(result["delta_margin"])
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": RECORD_ARTIFACT_TYPE,
        "record_id": stable_record_id(
            "cross-sector-header-patch",
            prepared_record["record_id"],
            str((source_identity or prepared_record["source_identity"])["ticker"]),
            str((target_identity or prepared_record["target_identity"])["ticker"]),
            direction,
            layer,
        ),
        "pair_id": prepared_record["pair_id"],
        "prepared_record_id": prepared_record["record_id"],
        "control_type": prepared_record["control_type"],
        "patching_direction": direction,
        "layer": int(layer),
        "layers": [int(layer)],
        "evidence_valence": prepared_record["evidence_valence"],
        "evidence_origin_sector": prepared_record["evidence_origin_sector"],
        "source_identity": dict(source_identity or prepared_record["source_identity"]),
        "target_identity": dict(target_identity or prepared_record["target_identity"]),
        "source_trial_record_id": prepared_record["source_trial_record_id"],
        "source_trial_key": prepared_record["source_trial_key"],
        "source_trial_index": prepared_record.get("source_trial_index"),
        "source_set_index": prepared_record.get("source_set_index"),
        "trial_row_sha256": prepared_record["trial_row_sha256"],
        "input_sha256": prepared_record.get("input_sha256"),
        "split_manifest_sha256": prepared_record.get("split_manifest_sha256"),
        "evidence_item_hashes": dict(prepared_record["evidence_item_hashes"]),
        "source_clean_margin": source_value,
        "target_clean_margin": target_value,
        "patched_margin": float(result["patched_margin"]),
        "delta_margin": delta,
        "normalized_transfer": _normalized_transfer(source_value, target_value, delta),
        "flip": bool(result["flip"]),
        "positions_patched": list(result["positions_patched"]),
        "position_count": int(result["position_count"]),
    }


def analyze_cross_sector_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute equal-pair means by layer, direction, stratum, and control."""
    grouped: dict[tuple[int, str, str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (int(record["layer"]), str(record["patching_direction"]), str(record["evidence_origin_sector"]), str(record["control_type"]))
        grouped[key][str(record["pair_id"])].append(record)
    rows = []
    for (layer, direction, origin, control), by_pair in sorted(grouped.items()):
        pair_delta = [sum(float(row["delta_margin"]) for row in values) / len(values) for values in by_pair.values()]
        pair_flip = [sum(bool(row["flip"]) for row in values) / len(values) for values in by_pair.values()]
        transfers = [float(row["normalized_transfer"]) for values in by_pair.values() for row in values if row.get("normalized_transfer") is not None]
        rows.append({
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "cross_sector_header_patch_summary",
            "layer": layer,
            "patching_direction": direction,
            "evidence_origin_sector": origin,
            "control_type": control,
            "pair_count": len(by_pair),
            "record_count": sum(len(values) for values in by_pair.values()),
            "equal_pair_mean_delta_margin": sum(pair_delta) / len(pair_delta),
            "equal_pair_flip_rate": sum(pair_flip) / len(pair_flip),
            "equal_pair_mean_normalized_transfer": (sum(transfers) / len(transfers)) if transfers else None,
        })
    return rows


def _load_prepared(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        if row.get("artifact_type") != PAIRS_ARTIFACT_TYPE:
            raise ValueError(f"prepared record is not {PAIRS_ARTIFACT_TYPE}")
    return rows


def run_cross_sector_header_patching_pipeline(
    *,
    prepared_pairs: str | Path,
    model_name: str,
    run_id: str,
    layers: Sequence[int] = tuple(range(31)),
    dataset: str = DEFAULT_DATASET,
    artifact_root: str | Path = "artifacts",
    decision_prefix: str = DEFAULT_DECISION_PREFIX,
    positive_candidate: str = DEFAULT_POSITIVE_CANDIDATE,
    negative_candidate: str = DEFAULT_NEGATIVE_CANDIDATE,
    max_records: int | None = None,
    max_seq_len: int = 1024,
) -> Path:
    """Run A V1 over prepared pairs through prepare/forward/analyze/finalize."""
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    prepared_path = Path(prepared_pairs)
    prepared = _load_prepared(prepared_path)
    if max_records is not None:
        prepared = prepared[:max_records]
    if not prepared:
        raise ValueError("prepared pair input is empty")

    preflight_tokenizer = load_tokenizer(model_name)
    formatted_records = []
    for row in prepared:
        source_prompt, source_spans = _prepare_scoring_prompt(preflight_tokenizer, row["source_prompt"], row["source_evidence_char_spans"], decision_prefix)
        target_prompt, target_spans = _prepare_scoring_prompt(preflight_tokenizer, row["target_prompt"], row["target_evidence_char_spans"], decision_prefix)
        for prompt, candidate in ((source_prompt, positive_candidate), (source_prompt, negative_candidate), (target_prompt, positive_candidate), (target_prompt, negative_candidate)):
            if len(input_ids(preflight_tokenizer, prompt + candidate)) > max_seq_len:
                raise ValueError(f"prepared prompt exceeds max_seq_len={max_seq_len}")
            continuation_token_ids(preflight_tokenizer, prompt, candidate)
        formatted_records.append((row, source_prompt, target_prompt, source_spans, target_spans))
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(prepared_path, artifact_type=PAIRS_ARTIFACT_TYPE, stage="prepare", role="input")
    run.manifest.save()
    try:
        prepare_metadata_path = run.run_directory / "prepare" / "metadata.json"
        with run.stage("prepare") as stage:
            write_metadata(prepare_metadata_path, {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "cross_sector_header_prepare_metadata",
                "model": model_name,
                "prepared_pairs": str(prepared_path),
                "prepared_pairs_sha256": sha256_file(prepared_path),
                "pair_record_count": len(prepared),
                "layers": [int(layer) for layer in layers],
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "state_cache_persisted": False,
            }, overwrite=False)
            stage.count(len(prepared))
        run.manifest.register_artifact(prepare_metadata_path, artifact_type="cross_sector_header_prepare_metadata", stage="prepare", role="output")
        run.manifest.save()

        model, tokenizer, fallback_device = load_model(model_name)
        device = getattr(model, "input_device", fallback_device)
        resolved_layers = sorted({int(layer) for layer in layers})
        if not resolved_layers:
            raise ValueError("at least one layer is required")
        n_layers = int(model.n_layers)
        if resolved_layers[0] < 0 or resolved_layers[-1] >= n_layers:
            raise ValueError(f"layers must be in model range 0..{n_layers - 1}")
        forward_path = run.run_directory / "forward" / "header_patch_records.jsonl"
        records: list[dict[str, Any]] = []
        clean: dict[str, CandidateMargin] = {}
        with run.stage("forward") as stage:
            for row, source_prompt, target_prompt, source_spans, target_spans in formatted_records:
                source_key = sha256_json([source_prompt, row["source_identity"]])
                target_key = sha256_json([target_prompt, row["target_identity"]])
                if source_key not in clean:
                    clean[source_key] = score_single_token_margin_fp32(model, tokenizer, source_prompt, positive_candidate, negative_candidate, device=device)
                if target_key not in clean:
                    clean[target_key] = score_single_token_margin_fp32(model, tokenizer, target_prompt, positive_candidate, negative_candidate, device=device)
                row_cache = {
                    source_key: cache_source_residuals(
                        model, tokenizer, source_prompt, resolved_layers, device
                    )
                }
                if target_key != source_key:
                    row_cache[target_key] = cache_source_residuals(
                        model, tokenizer, target_prompt, resolved_layers, device
                    )
                directions = [(source_prompt, target_prompt, source_spans, target_spans, clean[source_key], clean[target_key], source_key, "source_to_target")]
                if row["control_type"] != "self_source":
                    directions.append((target_prompt, source_prompt, target_spans, source_spans, clean[target_key], clean[source_key], target_key, "target_to_source"))
                for src_prompt, tgt_prompt, src_spans, tgt_spans, src_margin, tgt_margin, src_key, direction in directions:
                    direction_name = f"{row['source_identity']['sector']}_to_{row['target_identity']['sector']}" if direction == "source_to_target" else f"{row['target_identity']['sector']}_to_{row['source_identity']['sector']}"
                    for layer in resolved_layers:
                        records.append(run_cross_sector_header_record(
                            model=model, tokenizer=tokenizer, prepared_record=row,
                            source_identity=(row["source_identity"] if direction == "source_to_target" else row["target_identity"]),
                            target_identity=(row["target_identity"] if direction == "source_to_target" else row["source_identity"]),
                            source_prompt=src_prompt, target_prompt=tgt_prompt,
                            source_spans=src_spans, target_spans=tgt_spans, layer=layer,
                            source_residuals=row_cache[src_key], source_clean_margin=src_margin,
                            target_clean_margin=tgt_margin, positive_candidate=positive_candidate,
                            negative_candidate=negative_candidate, device=device,
                            direction=direction_name,
                        ))
                del row_cache
            count = write_jsonl(forward_path, records, overwrite=False)
            stage.count(count)
        run.manifest.register_artifact(forward_path, artifact_type=RECORD_ARTIFACT_TYPE, stage="forward", role="output", record_count=count)
        run.manifest.save()

        summary_path = run.run_directory / "analyze" / "summary.json"
        with run.stage("analyze") as stage:
            summary = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": ANALYSIS_ARTIFACT_TYPE,
                "model": model_name,
                "prepared_pairs_sha256": sha256_file(prepared_path),
                "record_count": len(records),
                "equal_pair_means": analyze_cross_sector_records(records),
                "formal_success_gate": False,
                "interpretation_limit": "header resample-patching measures state sufficiency for the fixed Buy/Sell margin; it does not establish necessity or a unique information route",
            }
            write_json(summary_path, summary, overwrite=False)
            stage.count(len(summary["equal_pair_means"]))
        run.manifest.register_artifact(summary_path, artifact_type=ANALYSIS_ARTIFACT_TYPE, stage="analyze", role="output")
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = [
    "ANALYSIS_ARTIFACT_TYPE",
    "ARTIFACT_SCHEMA_VERSION",
    "CONTROL_TYPES",
    "DEFAULT_DATASET",
    "PAIRS_ARTIFACT_TYPE",
    "analyze_cross_sector_records",
    "load_raw_trial_jsonl",
    "load_split_assignments",
    "prepare_cross_sector_pairs",
    "run_cross_sector_header_patching_pipeline",
    "run_cross_sector_header_record",
    "target_complete_header_mapping",
]
