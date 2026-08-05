"""Evaluate return predictions from forward generated-output artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from llm_bias.prompt_analysis.artifact_io import read_jsonl, sha256_file

CONDITIONS = ("original", "counterfactual")
LABELS = ("very bearish", "bearish", "neutral", "bullish", "very bullish")
FORWARD_ARTIFACT_TYPE = "generated_outputs"
REQUIRED_FIELDS = (
    "pair_id",
    "filing_date",
    "ticker",
    "peer_ticker",
    "condition",
    "target_label",
    "fwd_return_1d",
    "generated_text",
)


def _metadata(path: Path) -> dict[str, Any]:
    candidate = path.parent / "metadata.json"
    if not candidate.is_file():
        candidate = path.with_suffix(path.suffix + ".metadata.json")
    if not candidate.is_file():
        return {}
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact metadata must be an object: {candidate}")
    return value


def _artifact_type(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str | None:
    value = metadata.get("artifact_type")
    if isinstance(value, str):
        return value
    values = {row.get("artifact_type") for row in rows if row.get("artifact_type")}
    if len(values) == 1:
        value = next(iter(values))
        return value if isinstance(value, str) else None
    return None


def _require_forward(source: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    artifact_type = _artifact_type(rows, metadata)
    if artifact_type is None and rows and all("predicted_label" in row for row in rows):
        # A forward record may carry already-parsed prediction fields; it still
        # has forward semantics even when a tiny hand-written fixture omits metadata.
        artifact_type = FORWARD_ARTIFACT_TYPE
    if artifact_type != FORWARD_ARTIFACT_TYPE:
        raise ValueError(
            f"return evaluation requires a forward generated-output artifact; "
            f"got {artifact_type or 'unknown'} from {source}"
        )
    return artifact_type


def parse_return_prediction(generated_text: Any) -> dict[str, Any]:
    """Parse the strict categorical answer while retaining invalid generations."""
    invalid = {
        "predicted_label": None,
        "predicted_confidence": None,
        "parse_status": "invalid",
        "parse_reason": None,
    }
    if not isinstance(generated_text, str):
        invalid["parse_reason"] = "generated_text_not_string"
        return invalid
    start = generated_text.find("{")
    if start < 0:
        invalid["parse_reason"] = "json_object_not_found"
        return invalid
    try:
        payload, _ = json.JSONDecoder().raw_decode(generated_text[start:])
    except json.JSONDecodeError:
        invalid["parse_reason"] = "malformed_json"
        return invalid
    if not isinstance(payload, dict):
        invalid["parse_reason"] = "json_payload_not_object"
        return invalid
    label = payload.get("label")
    confidence = payload.get("confidence")
    if label not in LABELS:
        invalid["parse_reason"] = "invalid_label"
        return invalid
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        invalid["parse_reason"] = "invalid_confidence"
        return invalid
    return {
        "predicted_label": label,
        "predicted_confidence": confidence,
        "parse_status": "valid",
        "parse_reason": None,
    }


def _valid_confidence(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100


def _score(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["prediction_valid"]]
    confusion = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for row in valid:
        confusion[row["target_label"]][row["predicted_label"]] += 1
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[target][label] for target in LABELS if target != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    result = {
        "records": len(records),
        "valid_predictions": len(valid),
        "invalid_predictions": len(records) - len(valid),
        "valid_rate": len(valid) / len(records) if records else 0.0,
        "accuracy": sum(r["predicted_label"] == r["target_label"] for r in valid) / len(valid) if valid else None,
        "macro_f1": sum(f1s) / len(LABELS),
        "mean_confidence": sum(r["predicted_confidence"] for r in valid) / len(valid) if valid else None,
        "confusion_counts": json.dumps(confusion, sort_keys=True),
    }
    result.update({f"confusion_{target}_{pred}": confusion[target][pred] for target in LABELS for pred in LABELS})
    return result


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_return_predictions(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    settings: dict[str, Any] | None = None,
) -> Path:
    """Validate and score a forward generated-output JSONL artifact."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = read_jsonl(source)
    source_metadata = _metadata(source)
    artifact_type = _require_forward(source, rows, source_metadata)
    source_hash = sha256_file(source)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    identities: dict[str, tuple[Any, ...]] = {}
    for line_number, row in enumerate(rows, 1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"record {line_number}: missing fields {missing}")
        pair_id, condition = str(row["pair_id"]), row["condition"]
        if not pair_id or condition not in CONDITIONS:
            raise ValueError(f"record {line_number}: invalid pair identity or condition")
        key = (pair_id, condition)
        if key in seen:
            raise ValueError(f"duplicate pair_id+condition: {pair_id}/{condition}")
        seen.add(key)
        identity = (
            row["filing_date"], row["ticker"], row["peer_ticker"],
            row["target_label"], row["fwd_return_1d"],
        )
        if pair_id in identities and identities[pair_id] != identity:
            raise ValueError(f"inconsistent identity for pair_id: {pair_id}")
        identities[pair_id] = identity
        parsed = parse_return_prediction(row.get("generated_text"))
        if parsed["parse_status"] != "valid" and "predicted_label" in row:
            parsed = {"predicted_label": row.get("predicted_label"), "predicted_confidence": row.get("predicted_confidence"), "parse_status": row.get("parse_status", "invalid"), "parse_reason": row.get("parse_reason")}
        try:
            target_ok = row["target_label"] in LABELS
            return_ok = isinstance(row["fwd_return_1d"], (int, float)) and math.isfinite(float(row["fwd_return_1d"]))
            label_ok = parsed["predicted_label"] in LABELS
            confidence_ok = _valid_confidence(parsed["predicted_confidence"])
        except (TypeError, ValueError):
            target_ok = return_ok = label_ok = confidence_ok = False
        reasons = []
        if parsed["parse_status"] != "valid":
            reasons.append(parsed["parse_reason"] or "parse_status")
        if not target_ok:
            reasons.append("unknown_target_label")
        if not return_ok:
            reasons.append("invalid_return")
        if not label_ok:
            reasons.append("unknown_predicted_label")
        if not confidence_ok:
            reasons.append("invalid_confidence")
        records.append({
            **row,
            **parsed,
            "pair_id": pair_id,
            "prediction_valid": bool(not reasons),
            "invalid_reason": ";".join(reasons),
        })

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
    incomplete = [pair for pair, conditions in by_pair.items() if set(conditions) != set(CONDITIONS)]
    if incomplete:
        raise ValueError(f"pairs must contain original and counterfactual: {incomplete}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sample_fields = list(REQUIRED_FIELDS) + ["predicted_label", "predicted_confidence", "parse_status", "parse_reason", "prediction_valid", "invalid_reason"]
    _write_csv(destination / "prediction_samples.csv", records, sample_fields)
    summary_rows = [{"condition": "all", **_score(records)}]
    summary_rows.extend({"condition": condition, **_score([r for r in records if r["condition"] == condition])} for condition in CONDITIONS)
    summary_fields = ["condition", "records", "valid_predictions", "invalid_predictions", "valid_rate", "accuracy", "macro_f1", "mean_confidence", "confusion_counts"] + [f"confusion_{target}_{pred}" for target in LABELS for pred in LABELS]
    _write_csv(destination / "prediction_summary.csv", summary_rows, summary_fields)
    pair_rows = []
    for pair_id, conditions in sorted(by_pair.items()):
        original, counterfactual = conditions["original"], conditions["counterfactual"]
        valid_pair = original["prediction_valid"] and counterfactual["prediction_valid"]
        pair_rows.append({
            "pair_id": pair_id,
            "original_label": original["predicted_label"],
            "counterfactual_label": counterfactual["predicted_label"],
            "original_valid": original["prediction_valid"],
            "counterfactual_valid": counterfactual["prediction_valid"],
            "valid_pair": valid_pair,
            "flip": valid_pair and original["predicted_label"] != counterfactual["predicted_label"],
            "same_label": valid_pair and original["predicted_label"] == counterfactual["predicted_label"],
        })
    _write_csv(destination / "pair_flip_summary.csv", pair_rows, ["pair_id", "original_label", "counterfactual_label", "original_valid", "counterfactual_valid", "valid_pair", "flip", "same_label"])
    coverage = {
        "records": len(records),
        "pairs": len(pair_rows),
        "complete_pairs": sum(row["valid_pair"] for row in pair_rows),
        "invalid_predictions": sum(not row["prediction_valid"] for row in records),
        "conditions": list(CONDITIONS),
    }
    metadata = {
        "input": str(source),
        "input_sha256": source_hash,
        "artifact_contract": {"source_stage": "forward", "artifact_type": artifact_type, "pairing_key": "pair_id", "generated_output_field": "generated_text"},
        "source": {"artifact": str(source), "sha256": source_hash, "coverage": coverage},
        "source_artifact": str(source),
        "source_artifact_sha256": source_hash,
        "coverage": coverage,
        "records": len(records), "pairs": len(pair_rows), "conditions": list(CONDITIONS), "labels": list(LABELS),
        "invalid_predictions": coverage["invalid_predictions"],
        "valid_pairs": sum(row["valid_pair"] for row in pair_rows),
        "flips": sum(row["flip"] for row in pair_rows),
        "same_label_pairs": sum(row["same_label"] for row in pair_rows),
        "settings": settings or {},
        "interpretation": "Prediction flips are descriptive comparisons of generated outputs, not causal estimates.",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


__all__ = ["CONDITIONS", "LABELS", "REQUIRED_FIELDS", "evaluate_return_predictions", "parse_return_prediction"]
