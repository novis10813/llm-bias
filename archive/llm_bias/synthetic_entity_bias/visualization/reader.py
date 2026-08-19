"""Strict artifact-only reader for complete synthetic entity-bias runs."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_manifest import RunManifest
from llm_bias.core.artifact_paths import file_sha256
from llm_bias.synthetic_entity_bias.spec import (
    BASELINE_ENTITY,
    LABEL_HASH,
    SCORES,
    TEMPLATE_HASH,
    TEMPLATES,
)

from .contract import REQUIRED_OUTPUTS, REQUIRED_STAGES, TEMPLATE_ORDER, TIER_ORDER

_HEX64 = re.compile(r"[0-9a-f]{64}")


class ArtifactContractError(ValueError):
    """Raised when a run is incomplete, tampered with, or schema-invalid."""


@dataclass(frozen=True)
class ValidatedRun:
    root: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    tokenization: dict[str, Any]
    entity_pool: tuple[dict[str, str], ...]
    baselines: tuple[dict[str, str], ...]
    results: tuple[dict[str, str], ...]
    localization: tuple[dict[str, str], ...]
    source_artifacts: tuple[dict[str, Any], ...]

    @property
    def raw(self) -> tuple[dict[str, str], ...]:
        return self.results


ArtifactBundle = ValidatedRun


def _fail(message: str) -> None:
    raise ArtifactContractError(message)


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        _fail(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{field} must be an integer") from exc
    if str(result) != str(value).strip() or result < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return result


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{path.name} must contain a JSON object")
    return value


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                _fail(f"{path.name} schema mismatch")
            return list(reader)
    except OSError as exc:
        raise ArtifactContractError(f"cannot read {path.name}: {exc}") from exc


def _probabilities(value: Any, field: str) -> list[float]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(f"{field} must be a JSON array") from exc
    if not isinstance(parsed, list) or len(parsed) != len(SCORES):
        _fail(f"{field} must contain nine probabilities")
    values = [_finite(item, field) for item in parsed]
    if any(item < 0 for item in values) or not math.isclose(
        sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        _fail(f"{field} must be non-negative and sum to one")
    return values


def _expected(probabilities: list[float]) -> float:
    return sum(probability * score for probability, score in zip(probabilities, SCORES))


def _resolve_output(root: Path, ref: dict[str, Any], expected_path: str) -> Path:
    if ref.get("role") != "output" or ref.get("status") != "complete":
        _fail(f"{ref.get('artifact_type')} must be a complete output")
    relative = Path(str(ref.get("path", "")))
    if relative.is_absolute() or relative.as_posix() != expected_path:
        _fail(f"invalid path for {ref.get('artifact_type')}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactContractError(f"artifact path escapes run root: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _digest(ref.get("sha256"), f"{ref.get('artifact_type')}.sha256")
    if file_sha256(resolved) != digest:
        _fail(f"artifact SHA-256 mismatch: {relative}")
    return resolved


def _validate_config(config: dict[str, Any], entity_count: int) -> None:
    if config.get("template_hash") != TEMPLATE_HASH or config.get("label_hash") != LABEL_HASH:
        _fail("config immutable protocol hash mismatch")
    if config.get("templates") != TEMPLATES:
        _fail("config templates disagree with the immutable protocol")
    if config.get("score_mapping") != {str(label): score for label, score in zip(range(9), SCORES)}:
        _fail("config score_mapping disagrees with the immutable protocol")
    if _integer(config.get("pool_count"), "config.pool_count", minimum=1) != entity_count:
        _fail("config pool_count mismatch")
    for field in ("model_config_sha256", "chat_template_sha256", "lens_binary_sha256"):
        _digest(config.get(field), f"config.{field}")
    inputs = config.get("input_hashes")
    if not isinstance(inputs, dict) or not inputs:
        _fail("config input_hashes must be non-empty")
    for name, value in inputs.items():
        _digest(value, f"config.input_hashes.{name}")


def _entities(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = row["ticker"].strip()
        if not ticker or ticker in output or not row["company_name"].strip():
            _fail("entity ticker/company identity is invalid or duplicated")
        if row["split"] not in {"train", "eval"} or row["familiarity_tier"] not in TIER_ORDER:
            _fail(f"invalid split/tier for {ticker}")
        latest = _integer(row["latest_year"], "latest_year", minimum=1900)
        years = [_integer(item, "years", minimum=1900) for item in row["years"].split("|") if item]
        if not years or latest != max(years):
            _fail(f"year provenance mismatch for {ticker}")
        _integer(row["source_row_count"], "source_row_count", minimum=1)
        output[ticker] = row
    return output


def _baselines(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    if len(rows) != len(TEMPLATE_ORDER):
        _fail("baseline record count mismatch")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        template = row["template"]
        if template not in TEMPLATE_ORDER or template in output or row["entity"] != BASELINE_ENTITY:
            _fail("baseline template/entity identity mismatch")
        probabilities = _probabilities(row["probabilities"], "baseline.probabilities")
        if not math.isclose(_finite(row["expected_score"], "baseline.expected_score"), _expected(probabilities), abs_tol=1e-5):
            _fail("baseline expected score mismatch")
        if _finite(row["entropy_nats"], "baseline.entropy_nats") < 0:
            _fail("baseline entropy must be non-negative")
        if _finite(row["effective_temperature"], "baseline.effective_temperature") <= 0:
            _fail("baseline effective temperature must be positive")
        output[template] = row
    return output


def _validate_results(rows, entities, baselines) -> None:
    expected_keys = {(ticker, template) for ticker in entities for template in TEMPLATE_ORDER}
    seen = set()
    for row in rows:
        key = (row["ticker"], row["template"])
        if key in seen or key not in expected_keys:
            _fail("result ticker/template coverage is invalid")
        seen.add(key)
        entity, baseline = entities[row["ticker"]], baselines[row["template"]]
        for field in ("company_name", "split", "familiarity_tier"):
            if row[field] != entity[field]:
                _fail(f"result/entity mismatch for {key}: {field}")
        entity_probs = _probabilities(row["entity_probabilities"], "entity_probabilities")
        baseline_probs = _probabilities(row["baseline_probabilities"], "baseline_probabilities")
        stored_baseline = _probabilities(baseline["probabilities"], "baseline.probabilities")
        if any(not math.isclose(a, b, abs_tol=1e-8) for a, b in zip(baseline_probs, stored_baseline)):
            _fail(f"result baseline probabilities mismatch for {key}")
        entity_expected = _finite(row["entity_expected_score"], "entity_expected_score")
        baseline_expected = _finite(row["baseline_expected_score"], "baseline_expected_score")
        if not math.isclose(entity_expected, _expected(entity_probs), abs_tol=1e-5):
            _fail(f"entity expected score mismatch for {key}")
        if not math.isclose(baseline_expected, _expected(baseline_probs), abs_tol=1e-5):
            _fail(f"baseline expected score mismatch for {key}")
        if not math.isclose(_finite(row["delta_expected_score"], "delta_expected_score"), entity_expected - baseline_expected, abs_tol=1e-5):
            _fail(f"delta expected score mismatch for {key}")
        for field in ("entity_entropy_nats", "baseline_entropy_nats"):
            if _finite(row[field], field) < 0:
                _fail(f"{field} must be non-negative")
        for field in ("entity_effective_temperature", "baseline_effective_temperature"):
            if _finite(row[field], field) <= 0:
                _fail(f"{field} must be positive")
        start = _integer(row["entity_span_start"], "entity_span_start")
        end = _integer(row["entity_span_end"], "entity_span_end")
        answer = _integer(row["answer_position"], "answer_position")
        if not start < end <= answer:
            _fail(f"invalid entity/answer positions for {key}")
    if seen != expected_keys:
        _fail("result ticker/template grid is incomplete")


def _validate_localization(rows, entities) -> None:
    if not rows:
        _fail("localization artifact is empty")
    layers = {_integer(row["layer"], "layer") for row in rows}
    if layers != set(range(max(layers) + 1)):
        _fail("localization layers must be contiguous from zero")
    expected = {(layer, template) for layer in layers for template in TEMPLATE_ORDER}
    seen = set()
    train_count = sum(row["split"] == "train" for row in entities.values())
    eval_count = len(entities) - train_count
    for row in rows:
        key = (_integer(row["layer"], "layer"), row["template"])
        if key in seen or key not in expected or row["fit_split"] != "train":
            _fail("localization layer/template grid is invalid")
        seen.add(key)
        for field in ("mean_cosine", "pearson_r", "spearman_r", "linear_r2", "q25", "q75"):
            _finite(row[field], field)
        if _integer(row["n_train"], "n_train") != train_count or _integer(row["n_eval"], "n_eval") != eval_count:
            _fail("localization train/eval counts mismatch")
        for field in ("n_high", "n_low"):
            count = _integer(row[field], field, minimum=1)
            if count > train_count:
                _fail("localization high/low count mismatch")
        for field in ("high_ids_sha256", "low_ids_sha256", "direction_sha256"):
            _digest(row[field], field)
        if not row["statistic_flag"].strip():
            _fail("localization statistic_flag must be explicit")
    if seen != expected:
        _fail("localization grid is incomplete")


def validate_run(run_root: str | Path) -> ValidatedRun:
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "manifest.json"
    manifest_object = _read_json(manifest_path)
    try:
        manifest = RunManifest.load(manifest_path)
    except (OSError, ValueError) as exc:
        raise ArtifactContractError(f"invalid manifest: {exc}") from exc
    if manifest.status != "complete" or manifest.error is not None:
        _fail("visualization requires a complete, error-free run")
    stored_root = Path(str(manifest_object.get("run_root", "")))
    if stored_root.is_absolute() and stored_root.resolve() != root:
        _fail("manifest run_root identity mismatch")
    if stored_root.name != root.name or manifest.run_id != root.name:
        _fail("manifest run identity mismatch")
    for stage in REQUIRED_STAGES:
        if manifest.stages.get(stage, {}).get("status") != "complete":
            _fail(f"manifest stage is incomplete: {stage}")

    refs_by_type: dict[str, list[dict[str, Any]]] = {}
    for ref in manifest.artifacts:
        if not isinstance(ref, dict):
            _fail("manifest artifact references must be objects")
        refs_by_type.setdefault(str(ref.get("artifact_type")), []).append(ref)
    paths, sources = {}, []
    for artifact_type, (filename, stage, _fields) in REQUIRED_OUTPUTS.items():
        refs = refs_by_type.get(artifact_type, [])
        if len(refs) != 1:
            _fail(f"manifest must contain exactly one {artifact_type} reference")
        ref = refs[0]
        if ref.get("stage") != stage:
            _fail(f"manifest stage mismatch for {artifact_type}")
        paths[artifact_type] = _resolve_output(root, ref, filename)
        sources.append({"artifact_type": artifact_type, "path": filename, "sha256": ref["sha256"], "record_count": ref.get("record_count")})

    config = _read_json(paths["config"])
    tokenization = _read_json(paths["tokenization_validation"])
    entities = _read_csv(paths["entity_pool"], REQUIRED_OUTPUTS["entity_pool"][2])
    baselines = _read_csv(paths["no_entity_baselines"], REQUIRED_OUTPUTS["no_entity_baselines"][2])
    results = _read_csv(paths["raw_entity_template_results"], REQUIRED_OUTPUTS["raw_entity_template_results"][2])
    localization = _read_csv(paths["layer_template_localization"], REQUIRED_OUTPUTS["layer_template_localization"][2])

    entity_by_ticker = _entities(entities)
    _validate_config(config, len(entity_by_ticker))
    if tokenization.get("label_token_ids") != config.get("label_token_ids") or tokenization.get("decoded") != config.get("label_decoded"):
        _fail("tokenization/config label identity mismatch")
    if _integer(tokenization.get("n_prompts"), "tokenization.n_prompts", minimum=1) != len(entities) * len(TEMPLATE_ORDER) + len(TEMPLATE_ORDER):
        _fail("tokenization prompt count mismatch")
    if tokenization.get("anomalies") != []:
        _fail("tokenization validation contains anomalies")
    expected_splits = {split: sum(row["split"] == split for row in entity_by_ticker.values()) for split in ("train", "eval")}
    if config.get("split_counts") != expected_splits:
        _fail("config split_counts mismatch")
    expected_tiers = {tier: sum(row["familiarity_tier"] == tier for row in entity_by_ticker.values()) for tier in TIER_ORDER if any(row["familiarity_tier"] == tier for row in entity_by_ticker.values())}
    if config.get("tier_counts") != expected_tiers:
        _fail("config tier_counts mismatch")

    baseline_by_template = _baselines(baselines)
    _validate_results(results, entity_by_ticker, baseline_by_template)
    _validate_localization(localization, entity_by_ticker)
    actual_counts = {"entity_pool": len(entities), "no_entity_baselines": len(baselines), "raw_entity_template_results": len(results), "layer_template_localization": len(localization)}
    for source in sources:
        artifact_type = source["artifact_type"]
        if artifact_type in actual_counts:
            if source["record_count"] != actual_counts[artifact_type] or manifest.record_counts.get(artifact_type) != actual_counts[artifact_type]:
                _fail(f"manifest record count mismatch: {artifact_type}")

    return ValidatedRun(root, manifest_object, config, tokenization, tuple(entities), tuple(baselines), tuple(results), tuple(localization), tuple(sources))


def read_run(run_root: str | Path) -> ValidatedRun:
    return validate_run(run_root)
