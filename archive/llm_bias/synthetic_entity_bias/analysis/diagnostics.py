"""Derived diagnostics for completed synthetic entity-bias runs."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy import optimize, stats

from llm_bias.synthetic_entity_bias.visualization.contract import TEMPLATE_ORDER

SCORES = np.arange(-4, 5, dtype=np.float64)
TAIL_THRESHOLDS = (0.2, 0.5, 0.75, 1.0)
NULL_TEMPERATURE_BOUNDS = (0.25, 4.0)
_PROBABILITY_EPSILON = np.finfo(np.float64).tiny
LOCALIZATION_METRICS = ("mean_cosine", "pearson_r", "spearman_r", "linear_r2")


def _values(run: Any, template: str, field: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in run.results if row["template"] == template],
        dtype=np.float64,
    )


def _baseline(run: Any, template: str) -> dict[str, str]:
    return next(row for row in run.baselines if row["template"] == template)


def baseline_statistics(run: Any) -> list[dict[str, Any]]:
    rows = []
    for template in TEMPLATE_ORDER:
        baseline = _baseline(run, template)
        entity = _values(run, template, "entity_expected_score")
        baseline_score = float(baseline["expected_score"])
        entity_mean = float(np.mean(entity))
        entity_median = float(np.median(entity))
        rows.append({
            "template": template,
            "n": len(entity),
            "baseline_expected_score": baseline_score,
            "baseline_entropy_nats": float(baseline["entropy_nats"]),
            "baseline_effective_temperature": float(baseline["effective_temperature"]),
            "entity_expected_score_mean": entity_mean,
            "entity_expected_score_median": entity_median,
            "entity_expected_score_q05": float(np.quantile(entity, 0.05)),
            "entity_expected_score_q95": float(np.quantile(entity, 0.95)),
            "mean_movement_from_baseline": entity_mean - baseline_score,
            "median_movement_from_baseline": entity_median - baseline_score,
            "mean_crosses_score_zero": bool(baseline_score * entity_mean < 0),
            "median_crosses_score_zero": bool(baseline_score * entity_median < 0),
        })
    return rows


def entity_distribution_diagnostics(run: Any) -> list[dict[str, Any]]:
    rows = []
    for template in TEMPLATE_ORDER:
        values = _values(run, template, "delta_expected_score")
        mean = float(np.mean(values))
        item: dict[str, Any] = {
            "template": template,
            "n": len(values),
            "mean": mean,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "negative_fraction": float(np.mean(values < 0)),
            "zero_fraction": float(np.mean(values == 0)),
            "positive_fraction": float(np.mean(values > 0)),
        }
        if len(values) >= 3 and np.std(values) > 0:
            item.update({
                "skewness": float(stats.skew(values, bias=False)),
                "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
                "shape_status": "ok",
                "shape_reason": "",
            })
        else:
            item.update({
                "skewness": None,
                "excess_kurtosis": None,
                "shape_status": "degenerate",
                "shape_reason": "requires n >= 3 and non-constant values",
            })
        for threshold in TAIL_THRESHOLDS:
            key = str(threshold).replace(".", "_")
            negative = int(np.sum(values <= -threshold))
            positive = int(np.sum(values >= threshold))
            item[f"negative_tail_count_ge_{key}"] = negative
            item[f"negative_tail_fraction_ge_{key}"] = negative / len(values)
            item[f"positive_tail_count_ge_{key}"] = positive
            item[f"positive_tail_fraction_ge_{key}"] = positive / len(values)
        central = values[np.abs(values) < TAIL_THRESHOLDS[0]]
        item["central_abs_lt_0_2_count"] = len(central)
        item["central_abs_lt_0_2_mean"] = float(np.mean(central)) if len(central) else None
        item["mean_minus_median"] = mean - float(np.median(values))
        rows.append(item)
    return rows


def _probabilities(value: str) -> np.ndarray:
    data = np.asarray(json.loads(value), dtype=np.float64)
    if data.shape != (9,) or not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError("temperature null requires nine finite non-negative probabilities")
    total = float(np.sum(data))
    if total <= 0:
        raise ValueError("temperature null requires positive probability mass")
    return data / total


def temperature_scaled_probabilities(baseline: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(baseline, _PROBABILITY_EPSILON, None)
    logits = np.log(clipped) / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def _fit_temperature(baseline: np.ndarray, observed: np.ndarray) -> tuple[float, np.ndarray]:
    lower, upper = NULL_TEMPERATURE_BOUNDS
    result = optimize.minimize_scalar(
        lambda temperature: float(np.sum((temperature_scaled_probabilities(baseline, temperature) - observed) ** 2)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise ValueError(f"temperature fit failed: {result.message}")
    temperature = float(result.x)
    return temperature, temperature_scaled_probabilities(baseline, temperature)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= 0:
        return None
    return 1 - float(np.sum((observed - predicted) ** 2)) / denominator


def temperature_null_entity_rows(run: Any) -> list[dict[str, Any]]:
    output = []
    for row in run.results:
        baseline = _probabilities(row["baseline_probabilities"])
        observed = _probabilities(row["entity_probabilities"])
        try:
            temperature, predicted = _fit_temperature(baseline, observed)
            predicted_score = float(predicted @ SCORES)
            predicted_entropy = float(-np.sum(predicted * np.log(np.clip(predicted, _PROBABILITY_EPSILON, None))))
            status, reason = "ok", ""
        except ValueError as exc:
            temperature = predicted_score = predicted_entropy = None
            status, reason = "degenerate", str(exc)
        baseline_score = float(row["baseline_expected_score"])
        entity_score = float(row["entity_expected_score"])
        entity_entropy = float(row["entity_entropy_nats"])
        output.append({
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "template": row["template"],
            "observed_delta_expected_score": float(row["delta_expected_score"]),
            "null_delta_expected_score": predicted_score - baseline_score if predicted_score is not None else None,
            "expected_score_difference_from_null": entity_score - predicted_score if predicted_score is not None else None,
            "observed_entity_entropy_nats": entity_entropy,
            "null_entity_entropy_nats": predicted_entropy,
            "entropy_difference_from_null": entity_entropy - predicted_entropy if predicted_entropy is not None else None,
            "fitted_probability_temperature": temperature,
            "fit_status": status,
            "fit_reason": reason,
        })
    return output


def temperature_null_diagnostics(run: Any) -> list[dict[str, Any]]:
    entity_rows = temperature_null_entity_rows(run)
    output = []
    for template in TEMPLATE_ORDER:
        rows = [row for row in entity_rows if row["template"] == template]
        valid = [row for row in rows if row["fit_status"] == "ok"]
        if not valid:
            output.append({
                "template": template,
                "n": len(rows),
                "n_valid": 0,
                "fit_status": "degenerate",
                "fit_reason": "no valid entity temperature fits",
            })
            continue
        observed_delta = np.asarray([row["observed_delta_expected_score"] for row in valid])
        null_delta = np.asarray([row["null_delta_expected_score"] for row in valid])
        score_difference = observed_delta - null_delta
        observed_entropy = np.asarray([row["observed_entity_entropy_nats"] for row in valid])
        null_entropy = np.asarray([row["null_entity_entropy_nats"] for row in valid])
        entropy_difference = observed_entropy - null_entropy
        temperatures = np.asarray([row["fitted_probability_temperature"] for row in valid])
        output.append({
            "template": template,
            "n": len(rows),
            "n_valid": len(valid),
            "fit_status": "ok" if len(valid) == len(rows) else "partial",
            "fit_reason": "" if len(valid) == len(rows) else f"{len(rows) - len(valid)} fits invalid",
            "temperature_bounds_lower": NULL_TEMPERATURE_BOUNDS[0],
            "temperature_bounds_upper": NULL_TEMPERATURE_BOUNDS[1],
            "fitted_probability_temperature_mean": float(np.mean(temperatures)),
            "fitted_probability_temperature_sample_std": float(np.std(temperatures, ddof=1)) if len(temperatures) > 1 else None,
            "fitted_probability_temperature_median": float(np.median(temperatures)),
            "observed_delta_expected_score_mean": float(np.mean(observed_delta)),
            "null_delta_expected_score_mean": float(np.mean(null_delta)),
            "expected_score_difference_from_null_mean": float(np.mean(score_difference)),
            "expected_score_difference_from_null_mae": float(np.mean(np.abs(score_difference))),
            "expected_score_difference_from_null_rmse": float(np.sqrt(np.mean(score_difference ** 2))),
            "expected_score_null_r2": _r_squared(observed_delta, null_delta),
            "entropy_difference_from_null_mean": float(np.mean(entropy_difference)),
            "entropy_difference_from_null_mae": float(np.mean(np.abs(entropy_difference))),
            "entropy_difference_from_null_rmse": float(np.sqrt(np.mean(entropy_difference ** 2))),
            "entropy_null_r2": _r_squared(observed_entropy, null_entropy),
        })
    return output


def _sign_changes(values: np.ndarray, layers: np.ndarray) -> tuple[int, str]:
    pairs = []
    for index in range(len(values) - 1):
        if values[index] * values[index + 1] < 0:
            pairs.append(f"{int(layers[index])}->{int(layers[index + 1])}")
    return len(pairs), "|".join(pairs)


def localization_transition_diagnostics(run: Any) -> list[dict[str, Any]]:
    output = []
    for template in TEMPLATE_ORDER:
        source = sorted(
            (row for row in run.localization if row["template"] == template),
            key=lambda row: int(row["layer"]),
        )
        layers = np.asarray([int(row["layer"]) for row in source], dtype=np.int64)
        max_layer = int(np.max(layers))
        for metric in LOCALIZATION_METRICS:
            values = np.asarray([float(row[metric]) for row in source], dtype=np.float64)
            peak = int(np.argmax(np.abs(values)))
            jumps = np.diff(values)
            jump = int(np.argmax(np.abs(jumps))) if len(jumps) else None
            crossing_count, crossing_pairs = _sign_changes(values, layers)
            output.append({
                "template": template,
                "metric": metric,
                "n_layers": len(values),
                "absolute_peak_layer": int(layers[peak]),
                "absolute_peak_normalized_depth": int(layers[peak]) / max_layer if max_layer else 0.0,
                "absolute_peak_value": float(values[peak]),
                "final_layer_value": float(values[-1]),
                "sign_change_count": crossing_count,
                "sign_change_layer_pairs": crossing_pairs,
                "maximum_absolute_jump": float(abs(jumps[jump])) if jump is not None else None,
                "maximum_jump_delta": float(jumps[jump]) if jump is not None else None,
                "maximum_jump_from_layer": int(layers[jump]) if jump is not None else None,
                "maximum_jump_to_layer": int(layers[jump + 1]) if jump is not None else None,
                "positive_layer_fraction": float(np.mean(values > 0)),
                "negative_layer_fraction": float(np.mean(values < 0)),
                "zero_layer_fraction": float(np.mean(values == 0)),
            })
    return output
