"""Compact deterministic E1 summaries and eligibility rules."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .mlp_cells import EPSILON


def _cell(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["layer"]), int(row["neuron"])


def held_variant_metrics(
    localization_candidates: Sequence[Mapping[str, Any]],
    held_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare a held ranking with the localization ranking for one ticker."""
    if not localization_candidates or not held_candidates:
        raise ValueError("both localization and held candidate rankings are required")
    local = [_cell(row) for row in localization_candidates]
    held = [_cell(row) for row in held_candidates]
    if len(set(local)) != len(local) or len(set(held)) != len(held):
        raise ValueError("candidate rankings must not contain duplicate cells")
    held_set = set(held)
    ranks = {
        f"{layer}:{neuron}": index + 1
        for index, (layer, neuron) in enumerate(held)
        if (layer, neuron) in set(local)
    }
    retention = {
        f"{layer}:{neuron}": {
            "localization_rank": local.index((layer, neuron)) + 1,
            "held_rank": index + 1,
        }
        for index, (layer, neuron) in enumerate(held)
        if (layer, neuron) in set(local)
    }
    return {
        "top1_agreement": bool(local[0] == held[0]),
        "top5_overlap": len(set(local[:5]).intersection(held_set)),
        "localization_top1_held_rank": ranks.get(f"{local[0][0]}:{local[0][1]}"),
        "candidate_rank_retention": retention,
    }


def collision_selectivity_summary(
    candidates_by_ticker: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    scores_by_ticker: Mapping[str, Mapping[tuple[int, int], float]] | None = None,
    epsilon: float = EPSILON,
) -> dict[str, dict[str, Any]]:
    """Summarize same-neuron collisions and optional score selectivity z-scores."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    selected = {ticker: _cell(rows[0]) for ticker, rows in candidates_by_ticker.items() if rows}
    collisions: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    for ticker, cell in selected.items():
        collisions[cell].append(ticker)
    result: dict[str, dict[str, Any]] = {}
    for ticker, cell in selected.items():
        peers = [name for name in collisions[cell] if name != ticker]
        item: dict[str, Any] = {
            "cell": {"layer": cell[0], "neuron": cell[1]},
            "same_neuron_other_ticker_count": len(peers),
            "same_neuron_other_tickers": sorted(peers),
            "entity_selectivity_z": None,
        }
        if scores_by_ticker is not None:
            values = [float(scores.get(cell, 0.0)) for name, scores in scores_by_ticker.items() if name != ticker]
            target = float(scores_by_ticker.get(ticker, {}).get(cell, 0.0))
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                item["entity_selectivity_z"] = (target - mean) / (math.sqrt(variance) + epsilon)
        result[ticker] = item
    return result


def surface_control_summary(
    candidate_cells: Sequence[Mapping[str, Any]],
    control_rankings: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Report candidate retention in real forms and exclusion from name-form control."""
    cells = {_cell(row) for row in candidate_cells}
    result: dict[str, Any] = {}
    for control, rows in control_rankings.items():
        ranked = [_cell(row) for row in rows]
        result[control] = {
            "top5_overlap": len(cells.intersection(ranked[:5])),
            "candidate_in_top5": any(cell in ranked[:5] for cell in cells),
        }
    real_forms = [
        bool(any(cell in [_cell(row) for row in rows[:5]] for cell in cells))
        for name, rows in control_rankings.items()
        if name != "name_form_control"
    ]
    result["form_robust"] = sum(real_forms) >= 2 and not result.get("name_form_control", {}).get("candidate_in_top5", False)
    return result


def anonymous_progress(
    margin: float, clean_margin: float, anonymous_margin: float, *, epsilon: float = EPSILON
) -> float:
    """Return frozen signed progress toward the same-prompt anonymous margin."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    gap = float(anonymous_margin) - float(clean_margin)
    return (float(margin) - float(clean_margin)) * gap / (gap * gap + epsilon)


def denominator_eligibility(
    clean_margin: float, anonymous_margin: float, *, threshold: float = 0.1
) -> tuple[bool, str | None]:
    """Apply the frozen anonymous-gap eligibility rule."""
    gap = abs(float(anonymous_margin) - float(clean_margin))
    if gap < threshold:
        return False, f"anonymous_gap_below_{threshold:g}"
    return True, None


def summarize_amnesia(
    rows: Iterable[Mapping[str, Any]],
    *,
    endpoint_alpha: float = -3.0,
    minimum_eligible_prompts: int = 2,
) -> dict[str, dict[str, Any]]:
    """Aggregate curves and issue only the frozen trusted-candidate eligibility."""
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ticker"]), str(row.get("scope", "all_positions")))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for (ticker, scope), values in grouped.items():
        target = [row for row in values if row.get("candidate") == "target"]
        eligible = [row for row in target if bool(row.get("eligible", False))]
        endpoint = [row for row in eligible if float(row["alpha"]) == endpoint_alpha]
        control_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        target_progress: list[float] = []
        for row in endpoint:
            prompt_id = str(row.get("prompt_id"))
            target_progress.append(float(row.get("anonymous_progress", 0.0)))
            for control in ("wrong_entity", "matched_random"):
                matches = [candidate for candidate in values if candidate.get("candidate") == control and str(candidate.get("prompt_id")) == prompt_id and float(candidate["alpha"]) == endpoint_alpha]
                if matches:
                    control_values[control]["progress"].append(float(matches[0].get("anonymous_progress", 0.0)))
        passed_prompt_ids = []
        for row in endpoint:
            prompt_id = str(row.get("prompt_id"))
            target_value = float(row.get("anonymous_progress", 0.0))
            controls = {
                control: next((float(candidate.get("anonymous_progress", 0.0)) for candidate in values if candidate.get("candidate") == control and str(candidate.get("prompt_id")) == prompt_id and float(candidate["alpha"]) == endpoint_alpha), None)
                for control in ("wrong_entity", "matched_random")
            }
            # E1 V2: a degenerate wrong-entity control (identical cell to the
            # target) is uninformative and drops out of the pass rule.
            if bool(row.get("wrong_entity_degenerate")):
                controls.pop("wrong_entity", None)
            if target_value > 0 and all(value is not None and target_value > value for value in controls.values()):
                passed_prompt_ids.append(prompt_id)
        reasons: list[str] = []
        if not target:
            reasons.append("missing_target_curve")
        target_prompt_ids = {str(row.get("prompt_id")) for row in target}
        eligible_prompt_ids = {str(row.get("prompt_id")) for row in eligible}
        if len(eligible_prompt_ids) < minimum_eligible_prompts:
            reasons.append(f"eligible_prompt_count_below_{minimum_eligible_prompts}")
        if len(set(passed_prompt_ids)) < minimum_eligible_prompts:
            reasons.append(f"endpoint_control_gate_below_{minimum_eligible_prompts}")
        entry: dict[str, Any] = {
            "ticker": ticker,
            "scope": scope,
            "eligible_prompt_count": len(eligible_prompt_ids),
            "denominator_eligible_prompt_ids": sorted(eligible_prompt_ids),
            "excluded_prompt_ids": sorted(target_prompt_ids - eligible_prompt_ids),
            "endpoint_pass_count": len(set(passed_prompt_ids)),
            "endpoint_pass_prompt_ids": sorted(set(passed_prompt_ids)),
            "mean_endpoint_anonymous_progress": sum(target_progress) / len(target_progress) if target_progress else None,
            "control_endpoint_progress": {name: dict(metrics) for name, metrics in control_values.items()},
            "trusted_candidate_entity_cell": bool(not reasons and scope == "all_positions"),
            "eligibility": "eligible" if not reasons and scope == "all_positions" else "excluded",
            "exclusion_reasons": reasons if scope == "all_positions" else ["secondary_header_only_scope"],
        }
        # Only V2 rows carry the degenerate flag; V1 output stays byte-identical.
        if any("wrong_entity_degenerate" in row for row in values):
            entry["degraded_control"] = bool(any(bool(row.get("wrong_entity_degenerate")) for row in endpoint))
        result[f"{ticker}:{scope}"] = entry
    return result


def frame_surface_control_summary(
    candidate_cells: Sequence[Mapping[str, Any]],
    control_rankings: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """E1 V2 form-robust: the candidate must be absent from both frame-family control top-5s."""
    cells = {_cell(row) for row in candidate_cells}
    result: dict[str, Any] = {}
    for control, rows in control_rankings.items():
        ranked = [_cell(row) for row in rows]
        result[control] = {
            "top5_overlap": len(cells.intersection(ranked[:5])),
            "candidate_in_top5": any(cell in ranked[:5] for cell in cells),
        }
    result["form_robust"] = not any(result[name]["candidate_in_top5"] for name in control_rankings)
    return result


def select_wrong_entity_cell(
    ticker: str,
    tickers: Sequence[str],
    cells_by_ticker: Mapping[str, Sequence[Mapping[str, Any]]],
    target_cell: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    """Deterministic non-degenerate wrong-entity control (E1 V2 rule).

    The wrong entity is the alphabetically next ticker in the run (wrap-around).
    Its candidate list is scanned in rank order for the first cell that differs
    from the target cell. If all five match, the control is degenerate.
    """
    ordered = sorted(str(value) for value in tickers)
    if str(ticker) not in ordered:
        raise ValueError("wrong-entity selection requires the target ticker in the run")
    wrong_ticker = ordered[(ordered.index(str(ticker)) + 1) % len(ordered)]
    candidates = list(cells_by_ticker[wrong_ticker])
    if not candidates:
        raise ValueError("wrong entity has no candidates")
    target = (int(target_cell["layer"]), int(target_cell["neuron"]))
    for candidate in candidates:
        if (int(candidate["layer"]), int(candidate["neuron"])) != target:
            return candidate, False
    return candidates[0], True


def v2_robustness_reasons(
    *,
    held_metrics: Mapping[str, Any],
    surface_control_summary: Mapping[str, Any],
    candidate_cells: Sequence[Mapping[str, Any]],
    template_signature: Sequence[Mapping[str, Any]],
) -> list[str]:
    """E1 V2 Gates 1-3 (held overlap, form-robust, template-robust) reasons only."""
    reasons: list[str] = []
    if int(held_metrics.get("top5_overlap", 0)) <= 0:
        reasons.append("held_variant_top5_overlap_zero")
    if not bool(surface_control_summary.get("form_robust", False)):
        reasons.append("not_form_robust")
    cells = {_cell(row) for row in candidate_cells}
    signature = {(int(row["layer"]), int(row["neuron"])) for row in template_signature}
    if cells.intersection(signature):
        reasons.append("in_template_signature")
    return reasons


def v2_candidate_eligibility(
    *,
    held_metrics: Mapping[str, Any],
    surface_control_summary: Mapping[str, Any],
    candidate_cells: Sequence[Mapping[str, Any]],
    template_signature: Sequence[Mapping[str, Any]],
    amnesia_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the frozen E1 V2 gates: held overlap, form-robust, template-robust, amnesia.

    Cell membership follows the V1 convention: the candidate top-5 set, not only the top-1.
    """
    reasons = v2_robustness_reasons(
        held_metrics=held_metrics,
        surface_control_summary=surface_control_summary,
        candidate_cells=candidate_cells,
        template_signature=template_signature,
    )
    if not bool(amnesia_summary.get("trusted_candidate_entity_cell", False)):
        reasons.extend(str(reason) for reason in amnesia_summary.get("exclusion_reasons", []))
    return {
        "eligible": not reasons,
        "label": "trusted candidate entity cell" if not reasons else None,
        "exclusion_reasons": sorted(set(reasons)),
    }


def trusted_candidate_eligibility(
    held_metrics: Mapping[str, Any],
    amnesia_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind held-variant and primary all-position gates with explicit reasons."""
    reasons: list[str] = []
    if int(held_metrics.get("top5_overlap", 0)) <= 0:
        reasons.append("held_variant_top5_overlap_zero")
    if not bool(amnesia_summary.get("trusted_candidate_entity_cell", False)):
        reasons.extend(str(reason) for reason in amnesia_summary.get("exclusion_reasons", []))
    return {
        "eligible": not reasons,
        "label": "trusted candidate entity cell" if not reasons else None,
        "exclusion_reasons": sorted(set(reasons)),
    }


__all__ = [
    "anonymous_progress", "collision_selectivity_summary", "denominator_eligibility",
    "frame_surface_control_summary", "held_variant_metrics", "select_wrong_entity_cell",
    "summarize_amnesia", "surface_control_summary", "trusted_candidate_eligibility",
    "v2_candidate_eligibility", "v2_robustness_reasons",
]
