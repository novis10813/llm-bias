"""Create artifact-only statistical tables for a synthetic entity-bias run."""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from llm_bias.synthetic_entity_bias.visualization.contract import TEMPLATE_ORDER, TIER_ORDER
from llm_bias.synthetic_entity_bias.visualization.reader import validate_run

from .diagnostics import (
    baseline_statistics,
    entity_distribution_diagnostics,
    localization_transition_diagnostics,
    temperature_null_diagnostics,
)
from .statistics import (
    bootstrap_interval,
    bootstrap_mean_difference,
    descriptive_statistics,
    holm_adjust,
    independent_tests,
    kruskal_test,
    mean_median_intervals,
    one_sample_tests,
    paired_tests,
)

SECTOR_MINIMUM_COUNT = 20
LOCALIZATION_METRICS = ("mean_cosine", "pearson_r", "spearman_r", "linear_r2")


def _values(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["delta_expected_score"]) for row in rows]


def _description(values: list[float], key: str) -> dict[str, Any]:
    return {**descriptive_statistics(values), **mean_median_intervals(values, key=key)}


def _adjust(rows: list[dict[str, Any]], raw: str, adjusted: str) -> None:
    for row, value in zip(rows, holm_adjust(row.get(raw) for row in rows), strict=True):
        row[adjusted] = value


def _template_statistics(run: Any) -> list[dict[str, Any]]:
    rows = []
    for template in TEMPLATE_ORDER:
        values = _values([row for row in run.results if row["template"] == template])
        rows.append({"template": template, **_description(values, f"template:{template}"), **one_sample_tests(values)})
    _adjust(rows, "t_p_value", "t_p_value_holm")
    _adjust(rows, "wilcoxon_p_value", "wilcoxon_p_value_holm")
    return rows


def _template_pairwise(run: Any) -> list[dict[str, Any]]:
    by_ticker = defaultdict(dict)
    for row in run.results:
        by_ticker[row["ticker"]][row["template"]] = float(row["delta_expected_score"])
    rows = []
    for left, right in combinations(TEMPLATE_ORDER, 2):
        tickers = sorted(ticker for ticker, values in by_ticker.items() if left in values and right in values)
        a = np.asarray([by_ticker[ticker][left] for ticker in tickers])
        b = np.asarray([by_ticker[ticker][right] for ticker in tickers])
        differences = a - b
        mean_lower, mean_upper = bootstrap_interval(differences, np.mean, key=f"pair:{left}:{right}:mean")
        median_lower, median_upper = bootstrap_interval(differences, np.median, key=f"pair:{left}:{right}:median")
        tests = paired_tests(a, b)
        rows.append({
            "template_a": left, "template_b": right, "difference_definition": "template_a_minus_template_b",
            "n_paired": len(tickers), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "mean_paired_difference": float(np.mean(differences)), "mean_difference_ci95_lower": mean_lower,
            "mean_difference_ci95_upper": mean_upper, "median_paired_difference": float(np.median(differences)),
            "median_difference_ci95_lower": median_lower, "median_difference_ci95_upper": median_upper,
            "sign_agreement_fraction": float(np.mean(np.sign(a) == np.sign(b))),
            "sign_reversal_fraction": float(np.mean(np.sign(a) == -np.sign(b))), **tests,
        })
    for raw in ("t_p_value", "wilcoxon_p_value", "pearson_p_value", "spearman_p_value"):
        _adjust(rows, raw, f"{raw}_holm")
    return rows


def _tier_groups(run: Any) -> dict[tuple[str, str], list[float]]:
    groups = defaultdict(list)
    for row in run.results:
        groups[(row["template"], row["familiarity_tier"])].append(float(row["delta_expected_score"]))
    return groups


def _tier_statistics(run: Any) -> list[dict[str, Any]]:
    groups = _tier_groups(run)
    rows = []
    for template in TEMPLATE_ORDER:
        present = [(tier, groups[(template, tier)]) for tier in TIER_ORDER if groups.get((template, tier))]
        omnibus = kruskal_test(values for _, values in present)
        for tier, values in present:
            rows.append({
                "template": template, "familiarity_tier": tier,
                **_description(values, f"tier:{template}:{tier}"),
                "kruskal_statistic": omnibus["statistic"], "kruskal_p_value": omnibus["p_value"],
                "kruskal_status": omnibus["status"], "kruskal_reason": omnibus["reason"],
            })
    return rows


def _tier_pairwise(run: Any) -> list[dict[str, Any]]:
    groups = _tier_groups(run)
    rows = []
    for template in TEMPLATE_ORDER:
        tiers = [tier for tier in TIER_ORDER if groups.get((template, tier))]
        for left, right in combinations(tiers, 2):
            a, b = groups[(template, left)], groups[(template, right)]
            difference = float(np.mean(a) - np.mean(b))
            lower, upper = bootstrap_mean_difference(
                a, b, key=f"tier-pair:{template}:{left}:{right}"
            )
            rows.append({
                "template": template, "tier_a": left, "tier_b": right, "difference_definition": "tier_a_minus_tier_b",
                "n_a": len(a), "n_b": len(b), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "mean_difference": difference, "mean_difference_ci95_lower": float(lower),
                "mean_difference_ci95_upper": float(upper), "median_difference": float(np.median(a) - np.median(b)),
                **independent_tests(a, b),
            })
    _adjust(rows, "welch_t_p_value", "welch_t_p_value_holm")
    _adjust(rows, "mann_whitney_p_value", "mann_whitney_p_value_holm")
    return rows


def _sector_statistics(run: Any) -> list[dict[str, Any]]:
    sectors = {row["ticker"]: tuple(filter(None, row["sectors"].split("|"))) or ("Unknown",) for row in run.entity_pool}
    groups = defaultdict(list)
    for row in run.results:
        for sector in dict.fromkeys(sectors[row["ticker"]]):
            groups[(row["template"], sector)].append(float(row["delta_expected_score"]))
    rows = []
    for template, sector in sorted(groups, key=lambda key: (TEMPLATE_ORDER.index(key[0]), key[1])):
        values = groups[(template, sector)]
        included = len(values) >= SECTOR_MINIMUM_COUNT
        tests = one_sample_tests(values) if included else {
            "cohens_dz": None, "t_statistic": None, "t_p_value": None, "t_status": "excluded_small_group",
            "t_reason": f"requires n >= {SECTOR_MINIMUM_COUNT}", "wilcoxon_statistic": None,
            "wilcoxon_p_value": None, "wilcoxon_status": "excluded_small_group",
            "wilcoxon_reason": f"requires n >= {SECTOR_MINIMUM_COUNT}",
        }
        rows.append({"template": template, "sector": sector, "included_in_primary_analysis": included,
                     **_description(values, f"sector:{template}:{sector}"), **tests})
    _adjust(rows, "t_p_value", "t_p_value_holm")
    _adjust(rows, "wilcoxon_p_value", "wilcoxon_p_value_holm")
    return rows


def _zero_crossings(values: np.ndarray) -> int:
    signs = np.sign(values)
    nonzero = signs[signs != 0]
    return int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0


def _localization_statistics(run: Any) -> list[dict[str, Any]]:
    rows = []
    for template in TEMPLATE_ORDER:
        source = sorted((row for row in run.localization if row["template"] == template), key=lambda row: int(row["layer"]))
        layers = np.asarray([int(row["layer"]) for row in source], dtype=np.float64)
        max_layer = float(np.max(layers))
        depth = layers / max_layer if max_layer else np.zeros_like(layers)
        degenerate = sum(row["statistic_flag"] != "ok" for row in source)
        for metric in LOCALIZATION_METRICS:
            values = np.asarray([float(row[metric]) for row in source])
            peak_index = int(np.argmax(np.abs(values)))
            rows.append({
                "template": template, "metric": metric, "n_layers": len(values),
                "absolute_peak_layer": int(layers[peak_index]), "absolute_peak_normalized_depth": float(depth[peak_index]),
                "absolute_peak_value": float(values[peak_index]), "final_layer_value": float(values[-1]),
                "mean_across_layers": float(np.mean(values)), "median_across_layers": float(np.median(values)),
                "signed_auc": float(np.trapezoid(values, depth)), "absolute_auc": float(np.trapezoid(np.abs(values), depth)),
                "zero_crossing_count": _zero_crossings(values), "degenerate_layer_count": degenerate,
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty analysis table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent analysis table fields: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def analyze_run(run_root: str | Path, *, output_dir: str | Path | None = None, replace_existing: bool = False) -> Path:
    run = validate_run(run_root)
    destination = Path(output_dir).resolve() if output_dir else run.root / "analysis"
    if destination == run.root:
        raise ValueError("analysis output cannot replace the source run root")
    if destination.exists() and any(destination.iterdir()) and not replace_existing:
        raise FileExistsError(f"analysis output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        tables = {
            "template_statistics.csv": _template_statistics(run),
            "template_pairwise_tests.csv": _template_pairwise(run),
            "familiarity_tier_statistics.csv": _tier_statistics(run),
            "familiarity_tier_pairwise_tests.csv": _tier_pairwise(run),
            "sector_statistics.csv": _sector_statistics(run),
            "localization_statistics.csv": _localization_statistics(run),
            "baseline_statistics.csv": baseline_statistics(run),
            "entity_distribution_diagnostics.csv": entity_distribution_diagnostics(run),
            "temperature_null_diagnostics.csv": temperature_null_diagnostics(run),
            "localization_transition_diagnostics.csv": localization_transition_diagnostics(run),
        }
        for filename, rows in tables.items():
            _write_csv(staging / filename, rows)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
