"""Deterministic summaries for validated synthetic entity-bias runs."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Iterable

from .contract import TEMPLATE_ORDER, TIER_ORDER


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("summary inputs must be finite")
    return result


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _statistics(values: Iterable[float]) -> dict[str, Any]:
    data = list(values)
    if not data:
        raise ValueError("cannot summarize an empty group")
    mean = sum(data) / len(data)
    variance = sum((value - mean) ** 2 for value in data) / len(data)
    return {
        "count": len(data),
        "mean_delta_expected_score": mean,
        "median_delta_expected_score": median(data),
        "std_delta_expected_score": math.sqrt(variance),
        "q25_delta_expected_score": _quantile(data, 0.25),
        "q75_delta_expected_score": _quantile(data, 0.75),
        "min_delta_expected_score": min(data),
        "max_delta_expected_score": max(data),
        "positive_fraction": sum(value > 0 for value in data) / len(data),
        "negative_fraction": sum(value < 0 for value in data) / len(data),
        "zero_fraction": sum(value == 0 for value in data) / len(data),
    }


def summarize_template(run: Any) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in run.results:
        grouped[row["template"]].append(row)
    output = []
    for template in TEMPLATE_ORDER:
        rows = grouped[template]
        item = {"template": template, **_statistics(_number(row["delta_expected_score"]) for row in rows)}
        for field in (
            "entity_expected_score", "baseline_expected_score", "entity_entropy_nats",
            "baseline_entropy_nats", "entity_effective_temperature",
            "baseline_effective_temperature",
        ):
            item[f"mean_{field}"] = sum(_number(row[field]) for row in rows) / len(rows)
        output.append(item)
    return output


def summarize_tier(run: Any) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in run.results:
        grouped[(row["template"], row["familiarity_tier"])].append(row)
    output = []
    for template in TEMPLATE_ORDER:
        for tier in TIER_ORDER:
            rows = grouped.get((template, tier), [])
            if not rows:
                continue
            output.append({
                "template": template,
                "familiarity_tier": tier,
                "train_count": sum(row["split"] == "train" for row in rows),
                "eval_count": sum(row["split"] == "eval" for row in rows),
                **_statistics(_number(row["delta_expected_score"]) for row in rows),
            })
    return output


def summarize_sector(run: Any) -> list[dict[str, Any]]:
    sectors = {row["ticker"]: tuple(filter(None, row["sectors"].split("|"))) or ("Unknown",) for row in run.entity_pool}
    grouped = defaultdict(list)
    for row in run.results:
        for sector in sectors[row["ticker"]]:
            grouped[(row["template"], sector)].append(_number(row["delta_expected_score"]))
    return [
        {"template": template, "sector": sector, **_statistics(grouped[(template, sector)])}
        for template, sector in sorted(grouped, key=lambda key: (TEMPLATE_ORDER.index(key[0]), key[1]))
    ]


def summarize_ticker(run: Any) -> list[dict[str, Any]]:
    pool = {row["ticker"]: row for row in run.entity_pool}
    grouped = defaultdict(dict)
    for row in run.results:
        grouped[row["ticker"]][row["template"]] = _number(row["delta_expected_score"])
    output = []
    for ticker in sorted(grouped):
        effects = grouped[ticker]
        values = [effects[template] for template in TEMPLATE_ORDER]
        signs = {0 if value == 0 else (1 if value > 0 else -1) for value in values}
        entity = pool[ticker]
        output.append({
            "ticker": ticker,
            "company_name": entity["company_name"],
            "split": entity["split"],
            "familiarity_tier": entity["familiarity_tier"],
            **{f"{template}_delta_expected_score": effects[template] for template in TEMPLATE_ORDER},
            "mean_delta_expected_score": sum(values) / len(values),
            "range_delta_expected_score": max(values) - min(values),
            "sign_consistent": len(signs) == 1,
        })
    return output


def summarize_localization(run: Any) -> list[dict[str, Any]]:
    max_layer = max(int(row["layer"]) for row in run.localization)
    output = []
    for row in run.localization:
        layer = int(row["layer"])
        output.append({
            "layer": layer,
            "normalized_depth": layer / max_layer if max_layer else 0.0,
            "template": row["template"],
            "mean_cosine": _number(row["mean_cosine"]),
            "pearson_r": _number(row["pearson_r"]),
            "spearman_r": _number(row["spearman_r"]),
            "linear_r2": _number(row["linear_r2"]),
            "n_train": int(row["n_train"]),
            "n_eval": int(row["n_eval"]),
            "q25": _number(row["q25"]),
            "q75": _number(row["q75"]),
            "n_high": int(row["n_high"]),
            "n_low": int(row["n_low"]),
            "statistic_flag": row["statistic_flag"],
            "direction_sha256": row["direction_sha256"],
        })
    return sorted(output, key=lambda row: (TEMPLATE_ORDER.index(row["template"]), row["layer"]))


def summarize_ticker_halo(run: Any) -> list[dict[str, Any]]:
    from llm_bias.synthetic_entity_bias.spec import TEMPLATE_SENTIMENTS
    pool = {row["ticker"]: row for row in run.entity_pool}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in run.results:
        sentiment = TEMPLATE_SENTIMENTS.get(row["template"], "neutral")
        grouped[row["ticker"]][sentiment].append(_number(row["delta_expected_score"]))

    output = []
    for ticker, entity in sorted(pool.items()):
        scores_by_sentiment = grouped[ticker]
        all_scores = [score for scores in scores_by_sentiment.values() for score in scores]
        if not all_scores:
            continue
        pos_scores = scores_by_sentiment.get("positive", [0.0])
        neu_scores = scores_by_sentiment.get("neutral", [0.0])
        neg_scores = scores_by_sentiment.get("negative", [0.0])

        pos_mean = sum(pos_scores) / len(pos_scores)
        neu_mean = sum(neu_scores) / len(neu_scores)
        neg_mean = sum(neg_scores) / len(neg_scores)
        halo_mean = sum(all_scores) / len(all_scores)

        pos_var = sum((s - pos_mean) ** 2 for s in pos_scores) / len(pos_scores)
        neg_var = sum((s - neg_mean) ** 2 for s in neg_scores) / len(neg_scores)

        output.append({
            "ticker": ticker,
            "company_name": entity["company_name"],
            "familiarity_tier": entity["familiarity_tier"],
            "sector": entity.get("sectors", "").split("|")[0] if entity.get("sectors") else "",
            "split": entity["split"],
            "halo_mean": halo_mean,
            "sentiment_sensitivity": pos_mean - neg_mean,
            "pos_mean": pos_mean,
            "neu_mean": neu_mean,
            "neg_mean": neg_mean,
            "pos_std": math.sqrt(pos_var),
            "neg_std": math.sqrt(neg_var),
        })
    return output


def summarize_tier_sector_sentiment(run: Any) -> list[dict[str, Any]]:
    from llm_bias.synthetic_entity_bias.spec import TEMPLATE_SENTIMENTS
    pool = {row["ticker"]: row for row in run.entity_pool}
    grouped = defaultdict(list)
    for row in run.results:
        ticker = row["ticker"]
        entity = pool.get(ticker, {})
        sector = entity.get("sectors", "").split("|")[0] if entity.get("sectors") else ""
        if not sector:
            sector = "Unclassified"
        tier = row["familiarity_tier"]
        sentiment = TEMPLATE_SENTIMENTS.get(row["template"], "neutral")
        grouped[(sector, tier, sentiment)].append(_number(row["delta_expected_score"]))

    output = []
    for (sector, tier, sentiment), scores in sorted(grouped.items()):
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        sem = std / math.sqrt(n) if n > 0 else 0.0
        output.append({
            "sector": sector,
            "familiarity_tier": tier,
            "sentiment": sentiment,
            "count": n,
            "mean_delta_expected_score": mean,
            "std_delta_expected_score": std,
            "sem_delta_expected_score": sem,
            "ci95_lower": mean - 1.96 * sem,
            "ci95_upper": mean + 1.96 * sem,
        })
    return output


def summarize_layer_sentiment_ribbon(run: Any) -> list[dict[str, Any]]:
    from llm_bias.synthetic_entity_bias.spec import TEMPLATE_SENTIMENTS
    max_layer = max(int(row["layer"]) for row in run.localization)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in run.localization:
        layer = int(row["layer"])
        sentiment = TEMPLATE_SENTIMENTS.get(row["template"], "neutral")
        for metric in ("mean_cosine", "pearson_r", "spearman_r", "linear_r2"):
            grouped[(layer, sentiment)][metric].append(_number(row[metric]))

    output = []
    for (layer, sentiment), metrics in sorted(grouped.items()):
        row_dict = {
            "layer": layer,
            "normalized_depth": layer / max_layer if max_layer else 0.0,
            "sentiment": sentiment,
        }
        for metric, values in metrics.items():
            n = len(values)
            m = sum(values) / n
            var = sum((v - m) ** 2 for v in values) / n
            std = math.sqrt(var)
            sem = std / math.sqrt(n) if n > 0 else 0.0
            row_dict[f"{metric}_mean"] = m
            row_dict[f"{metric}_sem"] = sem
            row_dict[f"{metric}_min"] = min(values)
            row_dict[f"{metric}_max"] = max(values)
            row_dict[f"{metric}_ci95_lower"] = m - 1.96 * sem
            row_dict[f"{metric}_ci95_upper"] = m + 1.96 * sem
        output.append(row_dict)
    return output


def summarize_all(run: Any) -> dict[str, list[dict[str, Any]]]:
    from llm_bias.synthetic_entity_bias.analysis.diagnostics import (
        baseline_statistics,
        entity_distribution_diagnostics,
        localization_transition_diagnostics,
        temperature_null_diagnostics,
    )

    return {
        "template": summarize_template(run),
        "tier": summarize_tier(run),
        "sector": summarize_sector(run),
        "ticker": summarize_ticker(run),
        "localization": summarize_localization(run),
        "tail_diagnostics": entity_distribution_diagnostics(run),
        "baseline_movement": baseline_statistics(run),
        "temperature_null": temperature_null_diagnostics(run),
        "localization_transitions": localization_transition_diagnostics(run),
        "ticker_halo": summarize_ticker_halo(run),
        "tier_sector_sentiment": summarize_tier_sector_sentiment(run),
        "layer_sentiment_ribbon": summarize_layer_sentiment_ribbon(run),
    }
