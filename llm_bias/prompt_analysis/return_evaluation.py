"""Evaluate five-class return predictions from attribution JSONL artifacts.

The evaluator consumes one top-level JSON object per prediction.  The input
contract is intentionally independent of model generation code so artifacts can
be validated and scored without loading a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CONDITIONS = ("original", "counterfactual")
LABELS = ("very bearish", "bearish", "neutral", "bullish", "very bullish")
REQUIRED_FIELDS = (
    "pair_id", "filing_date", "ticker", "peer_ticker", "condition",
    "target_label", "fwd_return_1d", "generated_text", "predicted_label",
    "predicted_confidence", "parse_status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_confidence(value: Any) -> bool:
    # bool is an int subclass but is not a prediction confidence.
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100


def _score(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r["prediction_valid"]]
    correct = sum(r["predicted_label"] == r["target_label"] for r in valid)
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
        "records": len(records), "valid_predictions": len(valid),
        "invalid_predictions": len(records) - len(valid),
        "valid_rate": len(valid) / len(records) if records else 0.0,
        "accuracy": correct / len(valid) if valid else None,
        "macro_f1": sum(f1s) / len(LABELS),
        "mean_confidence": sum(r["predicted_confidence"] for r in valid) / len(valid) if valid else None,
        "confusion_counts": json.dumps(confusion, sort_keys=True),
    }
    result.update({
        f"confusion_{target}_{pred}": confusion[target][pred]
        for target in LABELS for pred in LABELS
    })
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
    """Validate and score an attribution JSONL, writing three CSVs and metadata."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    identities: dict[str, tuple[Any, ...]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"line {line_number}: missing fields {missing}")
            pair_id, condition = str(row["pair_id"]), row["condition"]
            if not pair_id or condition not in CONDITIONS:
                raise ValueError(f"line {line_number}: invalid pair identity or condition")
            key = (pair_id, condition)
            if key in seen:
                raise ValueError(f"duplicate pair_id+condition: {pair_id}/{condition}")
            seen.add(key)
            identity = (row["filing_date"], row["ticker"], row["peer_ticker"], row["target_label"], row["fwd_return_1d"])
            if pair_id in identities and identities[pair_id] != identity:
                raise ValueError(f"inconsistent identity for pair_id: {pair_id}")
            identities[pair_id] = identity
            try:
                target_ok = row["target_label"] in LABELS
                return_ok = isinstance(row["fwd_return_1d"], (int, float)) and math.isfinite(float(row["fwd_return_1d"]))
                label_ok = row["predicted_label"] in LABELS
                confidence_ok = _valid_confidence(row["predicted_confidence"])
            except (TypeError, ValueError):
                target_ok = return_ok = label_ok = confidence_ok = False
            parse_ok = row["parse_status"] == "valid"
            valid = bool(parse_ok and target_ok and return_ok and label_ok and confidence_ok)
            reasons = []
            if not parse_ok: reasons.append("parse_status")
            if not target_ok: reasons.append("unknown_target_label")
            if not return_ok: reasons.append("invalid_return")
            if not label_ok: reasons.append("unknown_predicted_label")
            if not confidence_ok: reasons.append("invalid_confidence")
            records.append({**row, "pair_id": pair_id, "prediction_valid": valid, "invalid_reason": ";".join(reasons)})
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records: by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
    incomplete = [pair for pair, conditions in by_pair.items() if set(conditions) != set(CONDITIONS)]
    if incomplete:
        raise ValueError(f"pairs must contain original and counterfactual: {incomplete}")

    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    sample_fields = list(REQUIRED_FIELDS) + ["prediction_valid", "invalid_reason"]
    _write_csv(destination / "prediction_samples.csv", records, sample_fields)
    summary_rows = [{"condition": "all", **_score(records)}]
    for condition in CONDITIONS: summary_rows.append({"condition": condition, **_score([r for r in records if r["condition"] == condition])})
    summary_fields = ["condition", "records", "valid_predictions", "invalid_predictions", "valid_rate", "accuracy", "macro_f1", "mean_confidence", "confusion_counts"] + [f"confusion_{target}_{pred}" for target in LABELS for pred in LABELS]
    _write_csv(destination / "prediction_summary.csv", summary_rows, summary_fields)
    pair_rows = []
    for pair_id, conditions in sorted(by_pair.items()):
        original, counterfactual = conditions["original"], conditions["counterfactual"]
        valid_pair = original["prediction_valid"] and counterfactual["prediction_valid"]
        pair_rows.append({"pair_id": pair_id, "original_label": original["predicted_label"], "counterfactual_label": counterfactual["predicted_label"], "original_valid": original["prediction_valid"], "counterfactual_valid": counterfactual["prediction_valid"], "valid_pair": valid_pair, "flip": valid_pair and original["predicted_label"] != counterfactual["predicted_label"], "same_label": valid_pair and original["predicted_label"] == counterfactual["predicted_label"]})
    _write_csv(destination / "pair_flip_summary.csv", pair_rows, ["pair_id", "original_label", "counterfactual_label", "original_valid", "counterfactual_valid", "valid_pair", "flip", "same_label"])
    metadata = {"input": str(source), "input_sha256": _sha256(source), "records": len(records), "pairs": len(pair_rows), "conditions": list(CONDITIONS), "labels": list(LABELS), "invalid_predictions": sum(not r["prediction_valid"] for r in records), "valid_pairs": sum(r["valid_pair"] for r in pair_rows), "flips": sum(r["flip"] for r in pair_rows), "same_label_pairs": sum(r["same_label"] for r in pair_rows), "settings": settings or {}}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination
