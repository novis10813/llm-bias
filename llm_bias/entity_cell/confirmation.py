"""Frozen entity-cell confirmation configuration and compact evaluation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_bias.core.artifact_paths import sha256_file, sha256_json
from llm_bias.core.artifacts.io import read_jsonl, write_json
from llm_bias.core.analysis.statistics import bootstrap_mean_ci, holm_bonferroni, sign_flip_pvalue

from .attention_attribution import FULL_ATTENTION_LAYERS, PRIMARY_ATTENTION_LAYERS
from .mlp_cells import CANDIDATE_LAYERS
from .preparation import E2_DONOR_CONDITIONS
from .suppression import E3_ALPHA_GRID, E3_BETA_GRID

SCHEMA_VERSION = 1
CONFIG_ARTIFACT_TYPE = "entity_cell_confirmation_config"
RECORD_ARTIFACT_TYPE = "entity_cell_confirmation_record"
RESULT_ARTIFACT_TYPE = "entity_cell_confirmation_evaluation"
PROTOCOL = "entity-cell-localization-v1"
HOLM_ALPHA = 0.05
PRIMARY_FAMILY = (
    "upstream_wrong_cell",
    "upstream_random_neuron",
    "downstream_evidence_control",
    "downstream_random_source_control",
    "mediation",
)


def default_confirmation_config(
    *,
    selected_heads: Sequence[Sequence[int] | tuple[int, int]],
    minimum_eligible_tickers: int = 8,
    discovery_run: str | None = None,
    discovery_run_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return the machine-readable V1 freeze after discovery selects heads/doses."""
    heads = [[int(layer), int(head)] for layer, head in selected_heads]
    if not heads or len({tuple(item) for item in heads}) != len(heads):
        raise ValueError("selected_heads must contain distinct layer/head pairs")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": CONFIG_ARTIFACT_TYPE,
        "protocol": PROTOCOL,
        "model": model,
        "discovery_run": discovery_run,
        "discovery_run_sha256": discovery_run_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "e1_rule": {
            "candidate_layers": list(CANDIDATE_LAYERS),
            "epsilon": 1e-6,
            "top_k": 5,
            "localization_variant_ids": list(range(8)),
            "held_variant_ids": list(range(8, 12)),
            "eligible": "held_variant_top5_overlap_gt_0_and_alpha_minus3_target_exceeds_wrong_and_random_on_at_least_two_eligible_prompts",
        },
        "e2_selection": {
            "full_attention_layers": list(FULL_ATTENTION_LAYERS),
            "primary_layers": list(PRIMARY_ATTENTION_LAYERS),
            "selected_heads": heads,
            "selection_rule": "discovery_frozen_top_five_equal_ticker_mean_abs_identity_and_sign_consistency",
        },
        "e3_primary": {
            "upstream_alpha": -3.0,
            "downstream_beta": 0.0,
            "alpha_grid": list(E3_ALPHA_GRID),
            "beta_grid": list(E3_BETA_GRID),
            "upstream_scope": "all_positions",
            "downstream_grouping": "single",
        },
        "conditions": {
            "identity": list(E2_DONOR_CONDITIONS),
            "upstream_controls": ["wrong_entity", "matched_random"],
            "downstream_controls": ["evidence", "random_subset", "whole_head"],
        },
        "minimum_eligible_tickers": int(minimum_eligible_tickers),
        "thresholds": {
            "upstream_contrast_minimum": 0.0,
            "downstream_contrast_minimum": 0.0,
            "mediation_minimum": 0.0,
            "evidence_gap_minimum": 0.0,
            "direction_consistency_minimum": 0.5,
            "ci_lower_strictly_gt_zero": True,
        },
        "statistics": {
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260901,
            "confidence": 0.95,
            "sign_flip_alternative": "greater",
            "sign_flip_alpha": HOLM_ALPHA,
            "holm_family": list(PRIMARY_FAMILY),
            "aggregation": "average_prompts_within_ticker_then_equal_ticker_mean",
        },
        "authorization": {
            "calibration_authorizes_test": True,
            "failed_calibration_blocks_test": True,
            "test_authorizes_nothing": True,
        },
    }


def _require_digest(value: Any, name: str, *, optional: bool = True) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _config_mapping(config: Mapping[str, Any] | "ConfirmationConfig") -> Mapping[str, Any]:
    if isinstance(config, ConfirmationConfig):
        return config.value
    return config


def validate_confirmation_config(config: Mapping[str, Any] | "ConfirmationConfig") -> dict[str, Any]:
    """Validate the frozen V1 confirmation schema and return a copy."""
    config = _config_mapping(config)
    if not isinstance(config, Mapping):
        raise TypeError("confirmation config must be a mapping")
    value = json.loads(json.dumps(dict(config)))
    if value.get("artifact_type") != CONFIG_ARTIFACT_TYPE or value.get("schema_version") != SCHEMA_VERSION or value.get("protocol") != PROTOCOL:
        raise ValueError("confirmation config schema or protocol mismatch")
    _require_digest(value.get("discovery_run_sha256"), "discovery_run_sha256")
    _require_digest(value.get("split_manifest_sha256"), "split_manifest_sha256")
    e1 = value.get("e1_rule")
    if not isinstance(e1, Mapping) or tuple(e1.get("candidate_layers", ())) != CANDIDATE_LAYERS or e1.get("epsilon") != 1e-6 or e1.get("top_k") != 5 or e1.get("localization_variant_ids") != list(range(8)) or e1.get("held_variant_ids") != list(range(8, 12)):
        raise ValueError("confirmation config E1 rule is not the frozen V1 rule")
    e2 = value.get("e2_selection")
    if not isinstance(e2, Mapping) or tuple(e2.get("full_attention_layers", ())) != FULL_ATTENTION_LAYERS or tuple(e2.get("primary_layers", ())) != PRIMARY_ATTENTION_LAYERS:
        raise ValueError("confirmation config E2 layer freeze is invalid")
    heads = e2.get("selected_heads")
    if not isinstance(heads, list) or not heads or any(not isinstance(item, list) or len(item) != 2 for item in heads):
        raise ValueError("confirmation config must contain selected E2 heads")
    if len({tuple(map(int, item)) for item in heads}) != len(heads) or any(int(item[0]) not in FULL_ATTENTION_LAYERS for item in heads):
        raise ValueError("confirmation config selected heads are invalid")
    e3 = value.get("e3_primary")
    if not isinstance(e3, Mapping) or e3.get("upstream_alpha") != -3.0 or e3.get("downstream_beta") != 0.0 or tuple(e3.get("alpha_grid", ())) != E3_ALPHA_GRID or tuple(e3.get("beta_grid", ())) != E3_BETA_GRID or e3.get("upstream_scope") != "all_positions" or e3.get("downstream_grouping") != "single":
        raise ValueError("confirmation config E3 dose freeze is invalid")
    conditions = value.get("conditions")
    if not isinstance(conditions, Mapping) or tuple(conditions.get("identity", ())) != E2_DONOR_CONDITIONS or tuple(conditions.get("upstream_controls", ())) != ("wrong_entity", "matched_random") or tuple(conditions.get("downstream_controls", ())) != ("evidence", "random_subset", "whole_head"):
        raise ValueError("confirmation config source/control freeze is invalid")
    minimum = value.get("minimum_eligible_tickers")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_eligible_tickers must be a positive integer")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("confirmation config thresholds are missing")
    for key in ("upstream_contrast_minimum", "downstream_contrast_minimum", "mediation_minimum", "evidence_gap_minimum", "direction_consistency_minimum"):
        if not isinstance(thresholds.get(key), (int, float)) or not math.isfinite(float(thresholds[key])):
            raise ValueError(f"confirmation threshold {key} must be finite")
    if not 0 <= float(thresholds["direction_consistency_minimum"]) <= 1 or thresholds.get("ci_lower_strictly_gt_zero") is not True:
        raise ValueError("confirmation direction and CI gates are invalid")
    statistics = value.get("statistics")
    if not isinstance(statistics, Mapping) or statistics.get("bootstrap_samples") != 10000 or statistics.get("confidence") != 0.95 or statistics.get("sign_flip_alternative") != "greater" or statistics.get("sign_flip_alpha") != HOLM_ALPHA or tuple(statistics.get("holm_family", ())) != PRIMARY_FAMILY or statistics.get("aggregation") != "average_prompts_within_ticker_then_equal_ticker_mean":
        raise ValueError("confirmation statistics are not the frozen V1 family")
    if not isinstance(statistics.get("bootstrap_seed"), int):
        raise ValueError("confirmation bootstrap_seed must be an integer")
    authorization = value.get("authorization")
    if authorization != {"calibration_authorizes_test": True, "failed_calibration_blocks_test": True, "test_authorizes_nothing": True}:
        raise ValueError("confirmation authorization rules are invalid")
    return value


def confirmation_config_hash(config: Mapping[str, Any] | "ConfirmationConfig") -> str:
    return sha256_json(validate_confirmation_config(config))


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


_METRIC_ALIASES = {
    "upstream_wrong_cell": ("upstream_wrong_cell", "upstream_target_vs_wrong_cell", "target_wrong_cell_contrast"),
    "upstream_random_neuron": ("upstream_random_neuron", "upstream_target_vs_random_neuron", "target_random_neuron_contrast"),
    "downstream_evidence_control": ("downstream_evidence_control", "downstream_identity_vs_evidence", "identity_evidence_contrast"),
    "downstream_random_source_control": ("downstream_random_source_control", "downstream_identity_vs_random_source", "identity_random_source_contrast"),
    "mediation": ("mediation", "mediation_contrast", "upstream_to_downstream_mediation"),
    "evidence_gap": ("evidence_gap", "evidence_gap_preservation", "evidence_preservation"),
}


def _metric(record: Mapping[str, Any], name: str) -> float:
    metrics = record.get("metrics")
    aliases = _METRIC_ALIASES.get(name, (name,))
    value = None
    for alias in aliases:
        if isinstance(metrics, Mapping) and alias in metrics:
            value = metrics[alias]
            break
        if alias in record:
            value = record[alias]
            break
    if value is None:
        raise ValueError(f"confirmation record is missing metric {name}")
    if isinstance(value, Mapping):
        for key in ("contrast", "value", "mean", "delta"):
            if key in value:
                value = value[key]
                break
    return _finite(value, f"metrics.{name}")


def _direction(record: Mapping[str, Any]) -> float:
    metrics = record.get("metrics")
    value = metrics.get("direction_consistency") if isinstance(metrics, Mapping) else record.get("direction_consistency")
    if value is None:
        value = record.get("direction_consistency")
    if value is None:
        value = record.get("direction_consistency_fraction", record.get("direction_consistent"))
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _finite(value, "direction_consistency")


def _eligible(record: Mapping[str, Any], config: Mapping[str, Any], *, config_file_sha256: str | None = None) -> tuple[bool, str | None]:
    e1 = record.get("e1")
    e2 = record.get("e2")
    e3 = record.get("e3")
    if isinstance(e1, Mapping):
        eligible = e1.get("eligible", e1.get("cell_eligible", False))
    else:
        eligible = record.get("cell_eligible", record.get("eligible", False))
    if eligible is not True:
        return False, "e1_cell_not_eligible"
    heads = config["e2_selection"]["selected_heads"]
    observed = e2.get("selected_heads") if isinstance(e2, Mapping) else record.get("selected_heads")
    if observed is not None and {tuple(map(int, item)) for item in observed} != {tuple(item) for item in heads}:
        return False, "e2_selected_heads_mismatch"
    if isinstance(e3, Mapping):
        if e3.get("alpha", -3.0) != -3.0 or e3.get("beta", 0.0) != 0.0:
            return False, "e3_primary_dose_mismatch"
    expected_config_hash = config_file_sha256 or confirmation_config_hash(config)
    if isinstance(record.get("config"), Mapping):
        if record["config"].get("sha256") and record["config"]["sha256"] != expected_config_hash:
            return False, "config_hash_mismatch"
    if record.get("config_sha256") and record.get("config_sha256") != expected_config_hash:
        return False, "config_hash_mismatch"
    for key, expected in (("model", config.get("model")), ("split_manifest_sha256", config.get("split_manifest_sha256"))):
        if expected is not None and record.get(key) is not None and record.get(key) != expected:
            return False, f"{key}_mismatch"
    return True, None


def _stat(values: Sequence[float], *, seed: int, samples: int, alternative: str = "greater") -> dict[str, Any]:
    if not values:
        raise ValueError("confirmation metric has no eligible tickers")
    summary = bootstrap_mean_ci(values, seed=seed, n_resamples=samples, confidence=0.95)
    return {
        "ticker_count": len(values),
        "mean": summary["mean"],
        "bootstrap_95_ci": summary["bootstrap_ci"],
        "sign_flip_p": sign_flip_pvalue(values, seed=seed, n_resamples=samples, alternative=alternative),
    }


def evaluate_confirmation(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | ConfirmationConfig,
    *,
    split: str,
    calibration_result: Mapping[str, Any] | None = None,
    config_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate calibration or test records; never selects a cell, head, or dose."""
    frozen = validate_confirmation_config(config)
    if split not in {"calibration", "test"}:
        raise ValueError("split must be calibration or test")
    if not records:
        raise ValueError("confirmation records are empty")
    seen: set[str] = set()
    eligible: list[Mapping[str, Any]] = []
    exclusions: dict[str, int] = {}
    for record in records:
        if record.get("artifact_type", RECORD_ARTIFACT_TYPE) != RECORD_ARTIFACT_TYPE:
            raise ValueError("confirmation record artifact_type is invalid")
        ticker = str(record.get("ticker", ""))
        if not ticker or ticker in seen:
            raise ValueError("confirmation records require unique ticker identities")
        seen.add(ticker)
        if record.get("split", split) != split:
            raise ValueError("confirmation record split does not match evaluator split")
        ok, reason = _eligible(record, frozen, config_file_sha256=config_file_sha256)
        if ok:
            eligible.append(record)
        else:
            exclusions[reason or "excluded"] = exclusions.get(reason or "excluded", 0) + 1
    blocked = False
    block_reason = None
    if split == "test":
        if calibration_result is None:
            blocked, block_reason = True, "calibration_result_required_for_test"
        elif (
            calibration_result.get("split") != "calibration"
            or calibration_result.get("success") is not True
            or calibration_result.get("test_authorized") is not True
            or calibration_result.get("frozen_config_sha256") != confirmation_config_hash(frozen)
        ):
            blocked, block_reason = True, "calibration_did_not_authorize_test"
    minimum = int(frozen["minimum_eligible_tickers"])
    if len(eligible) < minimum:
        blocked = True
        block_reason = block_reason or "minimum_eligible_tickers_not_met"
    names = list(PRIMARY_FAMILY)
    values = {name: [_metric(record, name) for record in eligible] for name in names}
    stats = {name: _stat(items, seed=int(frozen["statistics"]["bootstrap_seed"]) + index, samples=int(frozen["statistics"]["bootstrap_samples"])) for index, (name, items) in enumerate(values.items())} if eligible else {}
    raw = [stats[name]["sign_flip_p"] for name in names] if stats else []
    adjusted = holm_bonferroni(raw) if raw else []
    for name, value in zip(names, adjusted, strict=True):
        stats[name]["holm_adjusted_p"] = value
    thresholds = frozen["thresholds"]
    gates: list[dict[str, Any]] = [{"gate": "minimum_eligible_tickers", "observed": len(eligible), "threshold": minimum, "pass": len(eligible) >= minimum}]
    gate_thresholds = {
        "upstream_wrong_cell": thresholds["upstream_contrast_minimum"],
        "upstream_random_neuron": thresholds["upstream_contrast_minimum"],
        "downstream_evidence_control": thresholds["downstream_contrast_minimum"],
        "downstream_random_source_control": thresholds["downstream_contrast_minimum"],
        "mediation": thresholds["mediation_minimum"],
    }
    for name in names:
        item = stats.get(name)
        gate_ok = bool(item) and float(item["mean"]) > float(gate_thresholds[name]) and float(item["bootstrap_95_ci"][0]) > 0 and float(item["holm_adjusted_p"]) < HOLM_ALPHA
        gates.append({"gate": name, "observed": None if item is None else item["mean"], "threshold": gate_thresholds[name], "pass": gate_ok, "holm_adjusted_p": None if item is None else item["holm_adjusted_p"]})
    direction_values = [_direction(record) for record in eligible]
    direction_fraction = sum(direction_values) / len(direction_values) if direction_values else 0.0
    gates.append({"gate": "evidence_gap_preservation", "observed": None if not eligible else sum(_metric(record, "evidence_gap") for record in eligible) / len(eligible), "threshold": thresholds["evidence_gap_minimum"], "pass": bool(eligible) and sum(_metric(record, "evidence_gap") for record in eligible) / len(eligible) >= float(thresholds["evidence_gap_minimum"])})
    gates.append({"gate": "direction_consistency", "observed": direction_fraction, "threshold": thresholds["direction_consistency_minimum"], "pass": direction_fraction >= float(thresholds["direction_consistency_minimum"])})
    success = not blocked and all(bool(gate["pass"]) for gate in gates)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RESULT_ARTIFACT_TYPE,
        "protocol": PROTOCOL,
        "split": split,
        "eligible_ticker_count": len(eligible),
        "observed_ticker_count": len(seen),
        "excluded_ticker_count": len(seen) - len(eligible),
        "exclusions": exclusions,
        "statistics": stats,
        "direction_consistency": {"fraction": direction_fraction, "ticker_count": len(direction_values)},
        "gate_checks": gates,
        "blocked": blocked,
        "block_reason": block_reason,
        "success": bool(success),
        "test_authorized": bool(split == "calibration" and success),
        "frozen_config_sha256": confirmation_config_hash(frozen),
        "raw_runtime_payloads": False,
    }
    if split == "test":
        result["formal"] = not blocked
        result["test_authorized"] = False
    return result


def evaluate_confirmation_artifact(
    records_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate files and publish one immutable provenance-bound JSON result."""
    records_path, config_path, output_path = map(Path, (records_path, config_path, output_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration = None if calibration_path is None else json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration_path is not None and not isinstance(calibration, Mapping):
        raise ValueError("calibration artifact must be a JSON object")
    result = evaluate_confirmation(
        read_jsonl(records_path), config, split=split, calibration_result=calibration,
        config_file_sha256=sha256_file(config_path),
    )
    result.update({"config": str(config_path), "config_sha256": sha256_file(config_path), "parent_records": str(records_path), "parent_sha256": sha256_file(records_path)})
    if calibration_path is not None:
        result.update({"calibration": str(calibration_path), "calibration_sha256": sha256_file(calibration_path)})
    # Validate the parent lifecycle before publishing the immutable derivative.
    run_root = records_path.parent.parent
    manifest_path = run_root / "manifest.json"
    parent_run = None
    if manifest_path.is_file() and output_path.parent == run_root / "analyze":
        from llm_bias.core.artifacts.lifecycle import ArtifactRun
        parent_run = ArtifactRun.open(manifest_path)
        if parent_run.status != "complete":
            raise ValueError("confirmation provenance requires a complete parent run")
    write_json(output_path, result, overwrite=False)
    # When records belong to a completed entity-cell run, bind the derivative
    # result and its parents into that run's manifest without reopening stages.
    if parent_run is not None:
        parent_run.manifest.register_artifact(config_path, artifact_type=CONFIG_ARTIFACT_TYPE, stage="analyze", role="input")
        parent_run.manifest.register_artifact(records_path, artifact_type=RECORD_ARTIFACT_TYPE, stage="analyze", role="input")
        parent_run.manifest.register_artifact(output_path, artifact_type=RESULT_ARTIFACT_TYPE, stage="analyze", role="output")
        parent_run.manifest.save()
    return result


@dataclass(frozen=True)
class ConfirmationConfig:
    """Typed facade for the machine-readable V1 confirmation schema."""

    value: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConfirmationConfig":
        return cls(validate_confirmation_config(value))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.value))

    def __getitem__(self, key: str) -> Any:
        return self.value[key]


evaluate_entity_cell_confirmation = evaluate_confirmation
evaluate_entity_cell_confirmation_artifact = evaluate_confirmation_artifact


def summarize_discovery(
    *,
    e1: Mapping[str, Any],
    e2: Mapping[str, Any],
    e3: Mapping[str, Any],
) -> dict[str, Any]:
    """Join discovery counts and exclusions without producing a confirmation verdict."""
    e1_amnesia = e1.get("amnesia", {}) if isinstance(e1, Mapping) else {}
    e1_tickers = {str(key).split(":", 1)[0] for key in e1_amnesia}
    trusted = {ticker for ticker, value in e1_amnesia.items() if isinstance(value, Mapping) and value.get("trusted_candidate_entity_cell") is True}
    e2_rows = e2.get("head_ranking", []) if isinstance(e2, Mapping) else []
    e3_groups = e3.get("groups", []) if isinstance(e3, Mapping) else []
    e3_tickers = {str(row.get("group", "")).split(":", 1)[0] for row in e3_groups if isinstance(row, Mapping)}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "entity_cell_discovery_summary",
        "phase": "discovery",
        "counts": {
            "e1_observed_tickers": len(e1_tickers),
            "e1_trusted_tickers": len(trusted),
            "e1_excluded_tickers": len(e1_tickers - trusted),
            "e2_ranked_components": len(e2_rows),
            "e2_selected_components": sum(bool(row.get("selection_eligible")) for row in e2_rows if isinstance(row, Mapping)),
            "e3_observed_tickers": len(e3_tickers),
            "e3_trusted_tickers": len(e3_tickers & trusted),
        },
        "exclusions": {"e1": sorted(e1_tickers - trusted), "e2": [row for row in e2_rows if isinstance(row, Mapping) and not row.get("selection_eligible")], "e3": sorted(trusted - e3_tickers)},
        "confirmatory_verdict": None,
        "test_authorized": False,
        "raw_runtime_payloads": False,
    }


__all__ = [
    "CONFIG_ARTIFACT_TYPE", "ConfirmationConfig", "HOLM_ALPHA", "PRIMARY_FAMILY", "PROTOCOL", "RECORD_ARTIFACT_TYPE", "RESULT_ARTIFACT_TYPE", "confirmation_config_hash", "default_confirmation_config", "evaluate_confirmation", "evaluate_confirmation_artifact", "evaluate_entity_cell_confirmation", "evaluate_entity_cell_confirmation_artifact", "summarize_discovery", "validate_confirmation_config",
]
