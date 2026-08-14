"""Compact, content-level summaries for the binary-association workflow."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch
from llm_bias.core.analysis.statistics import bootstrap_mean_ci, paired_bootstrap_ci, sign_flip_pvalue


def _mean_ci(values: list[float], *, seed: int, n_resamples: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "bootstrap_95_ci": [None, None]}
    result = bootstrap_mean_ci(values, seed=seed, n_resamples=n_resamples)
    return {"count": result["count"], "mean": result["mean"], "bootstrap_95_ci": result["bootstrap_ci"]}


def _paired_summary(
    pairs: dict[str, tuple[float, float]], *, seed: int, n_resamples: int
) -> dict[str, Any]:
    if not pairs:
        return {
            "pair_count": 0,
            "order_effect_mom_minus_dad": None,
            "paired_bootstrap_95_ci": [None, None],
            "sign_flip_two_sided_p": None,
            "same_sign_ratio": None,
            "margin_correlation": None,
        }
    ordered = [pairs[key] for key in sorted(pairs)]
    dad = torch.tensor([item[0] for item in ordered], dtype=torch.float64)
    mom = torch.tensor([item[1] for item in ordered], dtype=torch.float64)
    differences = mom - dad
    ci = paired_bootstrap_ci(differences.tolist(), seed=seed, n_resamples=n_resamples)
    observed = differences.mean()
    p_value = sign_flip_pvalue(differences.tolist(), seed=seed, n_resamples=n_resamples)
    nonzero = (dad != 0) & (mom != 0)
    same_sign = ((dad[nonzero] > 0) == (mom[nonzero] > 0)).float()
    correlation = None
    if len(ordered) >= 2 and float(dad.std()) > 0 and float(mom.std()) > 0:
        correlation = float(torch.corrcoef(torch.stack([dad, mom]))[0, 1])
    return {
        "pair_count": len(ordered),
        "order_effect_mom_minus_dad": float(observed),
        "paired_bootstrap_95_ci": ci,
        "sign_flip_two_sided_p": p_value,
        "same_sign_ratio": float(same_sign.mean()) if len(same_sign) else None,
        "margin_correlation": correlation,
    }


def summarize_baseline(
    rows: Iterable[dict[str, Any]], *, seed: int = 0, n_resamples: int = 10_000
) -> dict[str, Any]:
    """Summarize baseline margins by order without treating order rows as independent."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    materialized = list(rows)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_split[str(row["split"])].append(row)
    result: dict[str, Any] = {"task_type": "binary_association", "splits": {}}
    for split, split_rows in sorted(by_split.items()):
        by_order: dict[str, list[float]] = defaultdict(list)
        paired: dict[str, dict[str, float]] = defaultdict(dict)
        for row in split_rows:
            order = str(row["prompt_order"])
            margin = float(row["margin"])
            by_order[order].append(margin)
            paired[str(row["career_id"])][order] = margin
        complete = {
            career_id: (values["dad_first"], values["mom_first"])
            for career_id, values in paired.items()
            if set(values) == {"dad_first", "mom_first"}
        }
        result["splits"][split] = {
            "order_margins": {
                order: _mean_ci(values, seed=seed + index, n_resamples=n_resamples)
                for index, (order, values) in enumerate(sorted(by_order.items()))
            },
            "paired_orders": _paired_summary(
                complete, seed=seed + 100, n_resamples=n_resamples
            ),
            "incomplete_career_ids": sorted(
                career_id for career_id, values in paired.items() if set(values) != {"dad_first", "mom_first"}
            ),
        }
    return result


def _group_means(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], metric: str) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(float(row[metric]))
    return {
        "|".join(group): {"count": len(values), "mean": sum(values) / len(values)}
        for group, values in sorted(groups.items())
    }


def summarize_binary(
    baseline_rows: Iterable[dict[str, Any]],
    *,
    patch_rows: Iterable[dict[str, Any]] | None = None,
    steering_rows: Iterable[dict[str, Any]] | None = None,
    seed: int = 0,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    result = summarize_baseline(baseline_rows, seed=seed, n_resamples=n_resamples)
    if patch_rows is not None:
        result["patch"] = {
            metric: _group_means(patch_rows, ("split", "prompt_order"), metric)
            for metric in (
                "direct_effect",
                "causal_patch_effect",
                "control_patch_effect",
                "permuted_target_effect",
                "matched_random_effect",
                "corrected_effect",
            )
        }
    if steering_rows is not None:
        result["steering"] = {
            "margin": _group_means(
                steering_rows,
                ("split", "prompt_order", "direction_variant", "span_policy", "alpha"),
                "mother_minus_father_logprob_margin",
            ),
            "effect": _group_means(
                steering_rows,
                ("split", "prompt_order", "direction_variant", "span_policy", "alpha"),
                "steering_effect",
            ),
        }
    return result
