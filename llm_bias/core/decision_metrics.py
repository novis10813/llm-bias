"""Outcome-only ITT metrics and paired issuer-cluster uncertainty.

All planned rows, gates and issuer metadata are validated before efficacy. Failed
interventions retain their baseline-source denominator. No text, scoring inputs,
models or implicit research bootstrap budgets are consumed.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.experiment_contract import (
    ExecutionRow,
    ExperimentPlan,
    progress,
    validate_rows,
)

# Public GenerationOutcome failure vocabulary, alphabetically ordered; keep zero
# counts for every canonical type without importing contract-private constants.
_FAILURE_TYPES = (
    "exception", "invalid_json", "invalid_reason", "invalid_schema",
    "no_legal_token", "timeout", "truncated", "unsupported_channel",
)
_DIRECTIONS = {"buy_to_sell": ("buy", "sell"), "sell_to_buy": ("sell", "buy")}


def _plan_copy(plan: ExperimentPlan) -> ExperimentPlan:
    if not isinstance(plan, ExperimentPlan):
        raise ValueError("plan must be an ExperimentPlan")
    return ExperimentPlan(plan.identity, plan.keys, plan.gate_names)


def _gate_states(plan, gates, *, allow_missing=False):
    if gates is None and allow_missing:
        return {name: None for name in plan.gate_names}
    if not isinstance(gates, Mapping) or set(gates) != set(plan.gate_names):
        raise ValueError("gates must exactly match plan gate_names")
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError("gates must be strictly bool")
    return {name: gates[name] for name in plan.gate_names}


def _failure_counts(outcomes):
    counts = dict.fromkeys(_FAILURE_TYPES, 0)
    for outcome in outcomes:
        if outcome.failure_type is not None:
            counts[outcome.failure_type] += 1
    return counts


def metrics_progress(
    plan: ExperimentPlan, rows: Iterable[ExecutionRow], *, gates=None,
) -> dict[str, Any]:
    """Coverage and failures only; absent gates are unknown, not passed.

    Explicit gate mappings must be exact. Partial row coverage (including a
    baseline-only plan) is permitted, but malformed/foreign/duplicate rows fail.
    """
    plan = _plan_copy(plan)
    rows = validate_rows(plan, rows)
    states = _gate_states(plan, gates, allow_missing=True)
    state = progress(plan, rows, None if gates is None else states)
    return {
        "schema_version": 1, "plan_hash": plan.plan_hash, **state,
        "generation_failures": _failure_counts(row.outcome for row in rows),
    }


def _issuer_mapping(plan, rows, supplied):
    if not isinstance(supplied, Mapping):
        raise ValueError("issuer_by_ticker must be a mapping")
    for ticker, issuer in supplied.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("issuer mapping tickers must be nonblank strings")
        if not isinstance(issuer, str) or not issuer.strip():
            raise ValueError("issuer mapping issuers must be nonblank strings")
    if set(supplied) != {key.ticker for key in plan.keys}:
        raise ValueError("issuer mapping must exactly match all planned tickers")
    mapping = dict(sorted(supplied.items()))
    for row in rows:
        if row.outcome.issuer_id != mapping[row.key.ticker]:
            raise ValueError("outcome issuer_id does not match issuer mapping")
    return mapping


def _paired_groups(rows):
    """Validate stage-wide full populations before splitting by condition."""
    stages = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = row.key
        if key.arm == "baseline" and key.dose != "0":
            raise ValueError("baseline selector must be arm='baseline', dose='0'")
        unit = (key.ticker, key.condition, key.trial_id)
        stages[key.stage][(key.arm, key.dose)][unit] = row.outcome
    groups = []
    for stage, arms in sorted(stages.items()):
        baseline = arms.get(("baseline", "0"))
        if baseline is None:
            raise ValueError(f"stage {stage!r} has no baseline")
        if len(arms) < 2:
            raise ValueError(f"stage {stage!r} has no nonbaseline arm/dose")
        for (arm, dose), outcomes in sorted(arms.items()):
            if arm == "baseline":
                continue
            if set(outcomes) != set(baseline):
                raise ValueError("nonbaseline arm/dose must cover exactly the baseline unit set")
            for condition in sorted({unit[1] for unit in baseline}):
                pairs = [(unit, baseline[unit], outcomes[unit]) for unit in sorted(baseline)
                         if unit[1] == condition]
                groups.append((stage, condition, arm, dose, pairs))
    return sorted(groups, key=lambda group: group[:4])


def _direction(pairs, mapping, source, target):
    ticker_counts = {unit[0]: {"source_units": 0, "flips": 0} for unit, _, _ in pairs}
    flips = retentions = invalid = off_target = 0
    for (ticker, _, _), baseline, intervention in pairs:
        if not baseline.primary_valid:
            continue
        if baseline.decision == source:
            ticker_counts[ticker]["source_units"] += 1
            if not intervention.primary_valid:
                invalid += 1
            elif intervention.decision == target:
                flips += 1
                ticker_counts[ticker]["flips"] += 1
            else:
                retentions += 1
        elif intervention.primary_valid and intervention.decision == source:
            off_target += 1
    tickers = [
        {"ticker": ticker, "issuer_id": mapping[ticker], **counts,
         "rate": counts["flips"] / counts["source_units"] if counts["source_units"] else None}
        for ticker, counts in sorted(ticker_counts.items())
    ]
    source_units = sum(record["source_units"] for record in tickers)
    rates = [record["rate"] for record in tickers if record["rate"] is not None]
    return {
        "source_units": source_units, "flips": flips, "retentions": retentions,
        "invalid_interventions": invalid, "off_target_flips": off_target,
        "source_tickers": len(rates), "source_free_tickers": len(tickers) - len(rates),
        "company_first_rate": sum(rates) / len(rates) if rates else None,
        "micro_rate": flips / source_units if source_units else None, "tickers": tickers,
    }


def summarize_decisions(
    plan: ExperimentPlan, rows: Iterable[ExecutionRow], *, issuer_by_ticker, gates,
) -> dict[str, Any]:
    """Complete fixed-denominator accounting, sorted by stage/condition/arm/dose.

    Dose order is lexical on canonical strings. Company-first rates give each
    source-eligible ticker equal weight, while micro rates weight source units.
    Every failed baseline is unknown; every failed source intervention is a
    denominator member with zero flips. Returned dictionaries retain no inputs.
    """
    plan = _plan_copy(plan)
    rows = validate_rows(plan, rows, require_complete=True)
    states = _gate_states(plan, gates)
    if not all(states.values()):
        raise ValueError("efficacy requires every declared gate true")
    mapping = _issuer_mapping(plan, rows, issuer_by_ticker)
    paired = _paired_groups(rows)
    groups = []
    for stage, condition, arm, dose, pairs in paired:
        baseline_buy = sum(b.primary_valid and b.decision == "buy" for _, b, _ in pairs)
        baseline_sell = sum(b.primary_valid and b.decision == "sell" for _, b, _ in pairs)
        observed_flips = sum(b.primary_valid and i.primary_valid and b.decision != i.decision
                             for _, b, i in pairs)
        tickers = {unit[0] for unit, _, _ in pairs}
        groups.append({
            "stage": stage, "condition": condition, "arm": arm, "dose": dose,
            "planned_units": len(pairs), "planned_tickers": len(tickers),
            "planned_issuers": len({mapping[ticker] for ticker in tickers}),
            "distinct_trial_ids": len({unit[2] for unit, _, _ in pairs}),
            "baseline_buy": baseline_buy, "baseline_sell": baseline_sell,
            "baseline_unknown": len(pairs) - baseline_buy - baseline_sell,
            "baseline_primary_valid": baseline_buy + baseline_sell,
            "intervention_primary_valid": sum(i.primary_valid for _, _, i in pairs),
            "baseline_failures": _failure_counts(b for _, b, _ in pairs),
            "intervention_failures": _failure_counts(i for _, _, i in pairs),
            "observed_flips": observed_flips, "observed_flips_over_planned": observed_flips / len(pairs),
            "directions": {name: _direction(pairs, mapping, source, target)
                           for name, (source, target) in _DIRECTIONS.items()},
        })
    return {
        "schema_version": 1, "plan_hash": plan.plan_hash, "complete": True, "eligible": True,
        "gates": states, "issuer_mapping_sha256": sha256_json(mapping),
        "planned_rows": len(plan.keys), "executed_rows": len(rows), "groups": groups,
    }


def _interval(values, confidence):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    q = (1 - confidence) / 2
    return [percentile(q), percentile(1 - q)]


def paired_cluster_bootstrap(
    plan: ExperimentPlan, rows: Iterable[ExecutionRow], *, issuer_by_ticker, gates,
    stage, condition, direction, left_arm, left_dose, right_arm, right_dose,
    seed, samples, confidence,
) -> dict[str, Any]:
    """Left-minus-right company-first contrast with shared issuer draws.

    Use random.Random(seed). In replicate order, draw n issuer occurrences in
    occurrence order using randrange(n), indexing lexically sorted planned
    issuer IDs for this stage/condition. An occurrence repeats ALL that issuer's
    source-eligible ticker trial aggregates, not its issuer mean or individual
    trials. Source-free issuers remain in the universe. Empty-source draws are
    undefined for both methods and the difference, counted without redrawing.

    Percentile intervals sort finite draws and linearly interpolate at position
    (n_finite-1)*q for q=(1-confidence)/2 and 1-q. No defaults are supplied.
    """
    # Entire input validation precedes even selecting a contrast.
    summary = summarize_decisions(plan, rows, issuer_by_ticker=issuer_by_ticker, gates=gates)
    selectors = {
        "stage": stage, "condition": condition, "direction": direction,
        "left_arm": left_arm, "left_dose": left_dose,
        "right_arm": right_arm, "right_dose": right_dose,
    }
    if any(not isinstance(value, str) or not value.strip() for value in selectors.values()):
        raise ValueError("bootstrap selection fields must be nonblank strings")
    if direction not in _DIRECTIONS:
        raise ValueError("invalid direction")
    if (left_arm, left_dose) == (right_arm, right_dose):
        raise ValueError("bootstrap groups must differ")
    if type(seed) is not int:
        raise ValueError("seed must be an integer, never bool")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer, never bool")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1), never bool")
    confidence = float(confidence)

    def select(arm, dose):
        for group in summary["groups"]:
            if (group["stage"], group["condition"], group["arm"], group["dose"]) == (stage, condition, arm, dose):
                return group["directions"][direction]
        raise ValueError("bootstrap group absent or not nonbaseline")

    left, right = select(left_arm, left_dose), select(right_arm, right_dose)
    # Ticker records include source-free tickers, so the universe is planned,
    # never restricted to issuers that happened to supply a valid source class.
    issuers = sorted({record["issuer_id"] for record in left["tickers"]})
    rates = {issuer: [] for issuer in issuers}
    for lrecord, rrecord in zip(left["tickers"], right["tickers"], strict=True):
        if lrecord["rate"] is not None:
            rates[lrecord["issuer_id"]].append((lrecord["rate"], rrecord["rate"]))
    rng = random.Random(seed)
    left_draws, right_draws, differences = [], [], []
    undefined = 0
    for _ in range(samples):
        drawn_rates = []
        for _ in issuers:
            drawn_rates.extend(rates[issuers[rng.randrange(len(issuers))]])
        if not drawn_rates:
            undefined += 1
            continue
        lmean = sum(pair[0] for pair in drawn_rates) / len(drawn_rates)
        rmean = sum(pair[1] for pair in drawn_rates) / len(drawn_rates)
        left_draws.append(lmean)
        right_draws.append(rmean)
        differences.append(lmean - rmean)
    left_estimate, right_estimate = left["company_first_rate"], right["company_first_rate"]
    difference = left_estimate - right_estimate if left_estimate is not None else None
    return {
        "schema_version": 1, "plan_hash": summary["plan_hash"],
        "issuer_mapping_sha256": summary["issuer_mapping_sha256"], **selectors,
        "seed": seed, "samples": samples, "confidence": confidence,
        "planned_issuers": len(issuers), "finite_draws": len(differences), "undefined_draws": undefined,
        "left": {"estimate": left_estimate, "interval": _interval(left_draws, confidence)},
        "right": {"estimate": right_estimate, "interval": _interval(right_draws, confidence)},
        "difference": {"estimate": difference, "interval": _interval(differences, confidence)},
    }
