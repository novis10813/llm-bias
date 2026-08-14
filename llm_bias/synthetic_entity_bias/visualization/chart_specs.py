"""Renderer-independent specifications for paper and interactive figures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contract import FIGURE_IDS, TEMPLATE_ORDER, TIER_ORDER
from .summaries import summarize_all
from .theme import template_style

_FORBIDDEN = ("activation", "residual", "gradient", "hidden_state")
_CONTEXT_LABELS = {template: f"{template.title()} context" for template in TEMPLATE_ORDER}


def _series(template: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    return {"key": template, "label": _CONTEXT_LABELS[template], "style": template_style(template), "points": points}


def _spec(identifier: str, title: str, subtitle: str, note: str, mark: str, series: list[dict[str, Any]], table: list[dict[str, Any]], panels: list[dict[str, Any]], supporting_table: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "title": title,
        "subtitle": subtitle,
        "figure_note": note,
        "description": subtitle,
        "mark": mark,
        "tooltip_fields": list(table[0]) if table else [],
        "series": series,
        "table": table,
        "panels": panels,
        "supporting_table": supporting_table,
    }


def _distribution(run: Any) -> dict[str, Any]:
    series, table = [], []
    counts_by_template = {}
    for template in TEMPLATE_ORDER:
        values = np.asarray([float(row["delta_expected_score"]) for row in run.results if row["template"] == template])
        counts_by_template[template] = len(values)
        counts, edges = np.histogram(values, bins=30)
        points = []
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True):
            point = {"x": float((left + right) / 2), "y": int(count), "bin_start": float(left), "bin_end": float(right)}
            points.append(point)
            table.append({"template": template, **point})
        series.append(_series(template, points))
    n = min(counts_by_template.values())
    return _spec(
        "entity_effect_distribution",
        "Entity names shift expected scores differently across prompt contexts",
        f"ΔE = entity expected score − matched ‘The company’ baseline; n = {n:,} entities per context.",
        "Expected scores use the restricted nine-label distribution. ΔE is descriptive and is not, by itself, a standalone causal effect.",
        "line", series, table,
        [
            {"label": "(a)", "title": "Distribution", "sample_size": n, "reference": "ΔE = 0: no entity effect"},
            {"label": "(b)", "title": "Empirical cumulative distribution", "sample_size": n, "reference": "ΔE = 0: no entity effect"},
        ],
        "template_summary.csv",
    )


def _tail_diagnostics(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    series, table, panels = [], [], []
    for index, template in enumerate(TEMPLATE_ORDER):
        row = next(item for item in summaries["tail_diagnostics"] if item["template"] == template)
        points = [
            {"x": row["q05"], "y": 0, "statistic": "q05"},
            {"x": row["median"], "y": 1, "statistic": "median"},
            {"x": row["mean"], "y": 2, "statistic": "mean"},
            {"x": row["q95"], "y": 3, "statistic": "q95"},
        ]
        series.append(_series(template, points))
        table.append(row)
        shape = (
            f"skew={row['skewness']:.2f}; excess kurtosis={row['excess_kurtosis']:.2f}"
            if row["skewness"] is not None
            else f"shape unavailable ({row['shape_reason']})"
        )
        panels.append({"label": f"({chr(97 + index)})", "title": f"{template.title()} context", "sample_size": row["n"], "statistic": shape, "reference": "ΔE = 0: no entity effect"})
    return _spec(
        "entity_effect_tail_diagnostics",
        "Entity-effect distributions contain asymmetric and heavy tails",
        "Mean, median, central quantiles, skewness, kurtosis, and sparse tails are reported for each context.",
        "Shape diagnostics are descriptive. Histogram peaks alone do not establish bimodality or a latent mixture.",
        "scatter", series, table, panels, "entity_effect_tail_diagnostics.csv",
    )


def _baseline_movement(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    series, table, panels = [], [], []
    for index, template in enumerate(TEMPLATE_ORDER):
        row = next(item for item in summaries["baseline_movement"] if item["template"] == template)
        points = [
            {"x": 0, "y": row["baseline_expected_score"], "stage": "baseline"},
            {"x": 1, "y": row["entity_expected_score_mean"], "stage": "entity_mean"},
        ]
        series.append(_series(template, points)); table.append(row)
        panels.append({"label": f"({chr(97 + index)})", "title": f"{template.title()} context", "sample_size": row["n"], "statistic": f"mean movement={row['mean_movement_from_baseline']:+.3f}", "reference": "Score 0 is the midpoint of the restricted −4…+4 mapping"})
    return _spec(
        "baseline_entity_movement",
        "Named entities move expected scores away from matched generic baselines",
        "Each context compares its ‘The company’ baseline with the entity-level score distribution.",
        "Movement is a matched prompt contrast under the restricted label distribution, not a standalone causal effect.",
        "line", series, table, panels, "baseline_entity_movement.csv",
    )


def _temperature_null(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    series, table, panels = [], [], []
    for index, template in enumerate(TEMPLATE_ORDER):
        row = next(item for item in summaries["temperature_null"] if item["template"] == template)
        points = [
            {"x": row["null_delta_expected_score_mean"], "y": row["observed_delta_expected_score_mean"], "comparison": "template_mean"},
        ]
        series.append(_series(template, points)); table.append(row)
        r2 = row["expected_score_null_r2"]
        statistic = f"R²={r2:.3f}; mean difference={row['expected_score_difference_from_null_mean']:+.3f}" if r2 is not None else "R² undefined"
        panels.append({"label": f"({chr(97 + index)})", "title": f"{template.title()} context", "sample_size": row["n_valid"], "statistic": statistic, "reference": "Observed = null indicates a temperature-only fit"})
    return _spec(
        "temperature_null_diagnostics",
        "A one-dimensional temperature null does not explain every entity effect",
        "The null fits pᵢ(T) ∝ p₀ᵢ^(1/T) to persisted nine-point probabilities and compares predicted with observed ΔE.",
        "The fitted probability temperature is not the artifact effective temperature, is not recovered model-logit temperature, and is not a new model experiment.",
        "scatter", series, table, panels, "temperature_null_diagnostics.csv",
    )


def _tier(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    series, table, panels = [], [], []
    for index, template in enumerate(TEMPLATE_ORDER):
        points = []
        rows = {row["familiarity_tier"]: row for row in summaries["tier"] if row["template"] == template}
        for tier_index, tier in enumerate(TIER_ORDER):
            if tier not in rows:
                continue
            row = rows[tier]
            point = {"x": tier_index, "category": tier, "y": row["median_delta_expected_score"], "q25": row["q25_delta_expected_score"], "q75": row["q75_delta_expected_score"], "count": row["count"]}
            points.append(point)
            table.append({"template": template, **point})
        series.append(_series(template, points))
        panels.append({"label": f"({chr(97 + index)})", "title": _CONTEXT_LABELS[template], "sample_size": sum(point["count"] for point in points), "reference": "ΔE = 0: no entity effect"})
    return _spec(
        "entity_effect_by_tier",
        "Entity effects vary by familiarity tier and prompt context",
        "Boxes show entity-level ΔE distributions; labels report the number of entities in each tier.",
        "The train/eval field is a deterministic analysis split, not an independently repeated sample.",
        "interval", series, table, panels, "familiarity_tier_summary.csv",
    )


def _relationships(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pairs = (("negative", "positive"), ("negative", "neutral"), ("positive", "neutral"))
    series, table, panels = [], [], []
    for index, (left, right) in enumerate(pairs):
        points = []
        for row in summaries["ticker"]:
            point = {"x": row[f"{left}_delta_expected_score"], "y": row[f"{right}_delta_expected_score"], "ticker": row["ticker"], "company_name": row["company_name"], "pair": f"{left}_vs_{right}"}
            points.append(point)
            table.append(point)
        x = np.asarray([point["x"] for point in points]); y = np.asarray([point["y"] for point in points])
        pearson = float(np.corrcoef(x, y)[0, 1]) if len(points) > 1 and np.std(x) and np.std(y) else 0.0
        style = template_style(left)
        series.append({"key": f"{left}_vs_{right}", "label": f"{_CONTEXT_LABELS[left]} vs {_CONTEXT_LABELS[right]}", "style": style, "points": points})
        panels.append({"label": f"({chr(97 + index)})", "title": f"{left.title()} vs {right.title()}", "sample_size": len(points), "statistic": f"Pearson r = {pearson:.2f}", "reference": "x = 0 and y = 0: no entity effect"})
    return _spec(
        "template_relationships",
        "Entity effects are related across prompt contexts",
        f"Each point is one ticker; n = {len(summaries['ticker']):,} tickers in every panel.",
        "Axes show ΔE for the named contexts. Correlations are descriptive and do not establish causality.",
        "scatter", series, table, panels, "ticker_template_effects.csv",
    )


def _localization(summaries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    series, table = [], []
    for template in TEMPLATE_ORDER:
        points = []
        for row in summaries["localization"]:
            if row["template"] != template:
                continue
            point = {"x": row["normalized_depth"], "y": row["mean_cosine"], "pearson_r": row["pearson_r"], "spearman_r": row["spearman_r"], "linear_r2": row["linear_r2"], "layer": row["layer"], "statistic_flag": row["statistic_flag"]}
            points.append(point); table.append({"template": template, **point})
        series.append(_series(template, points))
    first = summaries["localization"][0]
    metrics = (("Mean cosine", "mean_cosine"), ("Pearson r", "pearson_r"), ("Spearman ρ", "spearman_r"), ("Linear R²", "linear_r2"))
    panels = [{"label": f"({chr(97 + index)})", "title": title, "metric": metric, "sample_size": f"train n={first['n_train']:,}; eval n={first['n_eval']:,}", "reference": "Depth 1.0 is the final model layer"} for index, (title, metric) in enumerate(metrics)]
    return _spec(
        "localization_profiles",
        "Entity-sensitive localization changes across model depth",
        f"Depth is layer/final layer; train n = {first['n_train']:,}, eval n = {first['n_eval']:,} entities per context.",
        "Localization is Jacobian-transported representation evidence, not chain-of-thought or a standalone causal proof.",
        "line", series, table, panels, "localization_summary.csv",
    )


def _sector(summaries: dict[str, list[dict[str, Any]]], minimum_count: int) -> dict[str, Any]:
    all_rows = summaries["sector"]
    included = [row for row in all_rows if row["count"] >= minimum_count]
    sectors = sorted({row["sector"] for row in included})
    table = [{"template": row["template"], "category": row["sector"], "mean_delta_expected_score": row["mean_delta_expected_score"], "count": row["count"], "included_in_plot": row["count"] >= minimum_count} for row in all_rows]
    series = []
    for template in TEMPLATE_ORDER:
        indexed = {row["sector"]: row for row in included if row["template"] == template}
        points = [{"x": indexed[sector]["mean_delta_expected_score"], "y": index, "category": sector, "count": indexed[sector]["count"]} for index, sector in enumerate(sectors) if sector in indexed]
        series.append(_series(template, points))
    excluded = len(all_rows) - len(included)
    return _spec(
        "sector_effects",
        "Mean entity effects differ across reported sectors",
        f"Only sector–context groups with n ≥ {minimum_count} are shown; {excluded} groups are excluded.",
        "Pipe-delimited sector memberships are exploded; missing sectors are Unknown. Source years are provenance, not independently verified historical membership evidence.",
        "bar", series, table,
        [{"label": "(a)", "title": "Included sector–context groups", "sample_size": f"n ≥ {minimum_count}", "reference": "ΔE = 0: no entity effect"}],
        "sector_summary.csv",
    )


def validate_chart_specs(specs: list[dict[str, Any]]) -> None:
    if tuple(spec["id"] for spec in specs) != FIGURE_IDS:
        raise ValueError("chart specifications are incomplete or out of order")
    for spec in specs:
        if not all(spec.get(key) for key in ("title", "subtitle", "figure_note", "description", "table", "tooltip_fields", "panels", "supporting_table")):
            raise ValueError(f"chart specification lacks a paper or accessible data path: {spec['id']}")
        labels = [panel["label"] for panel in spec["panels"]]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate panel labels: {spec['id']}")
        for field in spec["tooltip_fields"]:
            if any(word in field.lower() for word in _FORBIDDEN):
                raise ValueError(f"forbidden chart field: {field}")
        for row in spec["table"]:
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"non-finite chart value: {spec['id']}")


def build_chart_specs(run: Any, *, sector_minimum_count: int = 20) -> list[dict[str, Any]]:
    summaries = summarize_all(run)
    specs = [
        _distribution(run),
        _tail_diagnostics(summaries),
        _baseline_movement(summaries),
        _temperature_null(summaries),
        _tier(summaries),
        _relationships(summaries),
        _localization(summaries),
        _sector(summaries, sector_minimum_count),
    ]
    validate_chart_specs(specs)
    return specs
