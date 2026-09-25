"""Pre-registered estimators for confirmation-v1 (flip rates, CAL grid rule, bootstrap, C5/C7/C8/C10).

Conventions: alpha > 0 pushes toward buy, alpha < 0 toward sell. The on-target transition at alpha > 0
is sell->buy among companies whose own alpha-0 decision is sell (and buy->sell for alpha < 0). The
primary ITT denominator is every company of that alpha-0 class (an unparsed steered row counts as
"not flipped"); the conditional rate over rows parsed at both ends is secondary. A zero denominator
is reported as None, never as 0%.
"""
from __future__ import annotations

import math
import random
from statistics import fmean, median
from typing import Any, Callable, Mapping, Sequence

PARSED = ("buy", "sell")

CAL_LADDER = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
GRID_CAP = 14
V2_ANCHORS = (2.0,)


def on_target(alpha: float) -> tuple[str, str]:
    """(alpha-0 class, flipped class) that the sign of alpha aims at."""
    if alpha == 0:
        raise ValueError("alpha 0 has no target direction")
    return ("sell", "buy") if alpha > 0 else ("buy", "sell")


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def flip_stats(baseline: Mapping[str, Mapping[str, Any]], steered: Mapping[str, Mapping[str, Any]],
               alpha: float) -> dict[str, Any]:
    """ITT and conditional on-/off-target transitions for one alpha over the companies in ``steered``."""
    source, goal = on_target(alpha)
    companies = sorted(steered)
    base = {t: baseline[t]["decision"] for t in companies}
    rows = {t: steered[t] for t in companies}
    on_class = [t for t in companies if base[t] == source]
    off_class = [t for t in companies if base[t] == goal]
    on_flip = sum(rows[t]["decision"] == goal for t in on_class)
    off_flip = sum(rows[t]["decision"] == source for t in off_class)
    on_parsed = [t for t in on_class if rows[t]["decision"] in PARSED]
    off_parsed = [t for t in off_class if rows[t]["decision"] in PARSED]
    unparsed = [t for t in companies if rows[t]["decision"] not in PARSED]
    margins = [rows[t]["margin"] - baseline[t]["margin"] for t in companies]
    realized = [rows[t]["realized_margin"] for t in companies if rows[t].get("realized_margin") is not None]
    return {
        "alpha": alpha, "n": len(companies), "on_target": f"{source}->{goal}",
        "on_class_n": len(on_class), "on_flip": on_flip, "on_flip_itt": _rate(on_flip, len(on_class)),
        "on_flip_conditional": _rate(on_flip, len(on_parsed)), "on_parsed_n": len(on_parsed),
        "off_class_n": len(off_class), "off_flip": off_flip, "off_flip_itt": _rate(off_flip, len(off_class)),
        "off_flip_conditional": _rate(off_flip, len(off_parsed)),
        "any_flip": on_flip + off_flip,
        "any_flip_itt": _rate(on_flip + off_flip, len(on_class) + len(off_class)),
        "parsed": len(companies) - len(unparsed), "parse_rate": _rate(len(companies) - len(unparsed), len(companies)),
        "collapsed": sum(rows[t]["unparsed_kind"] == "collapsed" for t in unparsed),
        "truncated": sum(rows[t]["unparsed_kind"] == "truncated" for t in unparsed),
        "mean_delta_margin": fmean(margins) if margins else None,
        "median_margin": median(rows[t]["margin"] for t in companies) if companies else None,
        "median_realized_margin": median(realized) if realized else None, "realized_n": len(realized),
    }


def baseline_stats(baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(baseline.values())
    parsed = sum(r["decision"] in PARSED for r in rows)
    return {"n": len(rows), "buy": sum(r["decision"] == "buy" for r in rows),
            "sell": sum(r["decision"] == "sell" for r in rows), "parsed": parsed,
            "parse_rate": _rate(parsed, len(rows)),
            "collapsed": sum(r["unparsed_kind"] == "collapsed" for r in rows),
            "truncated": sum(r["unparsed_kind"] == "truncated" for r in rows),
            "path_class": dict(sorted(_count(r["path_class"] for r in rows).items())),
            "realized_ok": sum(r["realized_status"] == "ok" for r in rows)}


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------------
# CAL rule (proposal §CAL)


def cal_rule(baseline: Mapping[str, Mapping[str, Any]], ladder_rows: Mapping[str, Mapping[float, Mapping[str, Any]]],
             fallback: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Derive alpha_lo / alpha_50 / alpha_90 / alpha_hi and the full, reduced and random grids.

    ``ladder_rows[ticker][alpha]`` holds the balanced CAL rows (missing = adaptively stopped; counted as
    unparsed and not flipped). ``fallback[direction] = {"baseline": ..., "rows": ...}`` supplies the
    cross-condition half-ladder for a direction whose balanced class has fewer than 6 companies.
    """
    fallback = fallback or {}
    companies = sorted(baseline)
    per_direction: dict[str, dict[str, Any]] = {}
    for sign, name in ((1.0, "sell->buy"), (-1.0, "buy->sell")):
        source, goal = on_target(sign)
        base, rows, cross = baseline, ladder_rows, False
        if sum(baseline[t]["decision"] == source for t in companies) < 6 and name in fallback:
            base, rows, cross = fallback[name]["baseline"], fallback[name]["rows"], True
        klass = [t for t in sorted(base) if base[t]["decision"] == source]
        curve = []
        for magnitude in CAL_LADDER:
            alpha = sign * magnitude
            flips = sum(rows.get(t, {}).get(alpha, {}).get("decision") == goal for t in klass)
            curve.append({"alpha": alpha, "class_n": len(klass), "flips": flips, "itt": _rate(flips, len(klass))})
        per_direction[name] = {"class_n": len(klass), "cross_condition": cross, "curve": curve}

    def first(direction: str, threshold: float) -> float | None:
        for point in per_direction[direction]["curve"]:
            if point["itt"] is not None and point["itt"] >= threshold:
                return abs(point["alpha"])
        return None

    def parse_ok(magnitude: float) -> bool:
        for sign in (1.0, -1.0):
            rows = [ladder_rows.get(t, {}).get(sign * magnitude) for t in companies]
            parsed = sum(r is not None and r["decision"] in PARSED for r in rows)
            if not companies or parsed / len(companies) < 0.9:
                return False
        return True

    flip50 = {name: first(name, 0.5) for name in per_direction}
    flip90 = {name: first(name, 0.9) for name in per_direction}
    usable = [v for v in flip50.values() if v is not None]
    hi_ok = [m for m in CAL_LADDER if parse_ok(m)]
    alpha_hi = max(hi_ok) if hi_ok else None
    # alpha_lo: upper edge of the contiguous low-dose region (from 0.0625 upward) where both
    # directions stay at <= 10% ITT on-target flips; left-censored if 0.0625 already exceeds 10%.
    alpha_lo, left_censored = None, False
    for magnitude in CAL_LADDER:
        rates = [p["itt"] for d in per_direction.values() for p in d["curve"]
                 if abs(p["alpha"]) == magnitude and p["itt"] is not None]
        if rates and max(rates) > 0.10:
            break
        alpha_lo = magnitude
    if alpha_lo is None:
        left_censored = True
        alpha_lo = CAL_LADDER[0]

    def nearest(value: float) -> float:
        return min(CAL_LADDER, key=lambda m: (abs(math.log(m) - math.log(value)), m))

    alpha_50 = nearest(math.exp(fmean(math.log(v) for v in usable))) if usable else None
    usable90 = [v for v in flip90.values() if v is not None]
    alpha_90 = nearest(math.exp(fmean(math.log(v) for v in usable90))) if usable90 else alpha_hi
    abort = []
    if alpha_hi is None:
        abort.append("no ladder point keeps parse >= 90% on both signs")
    if alpha_50 is None:
        abort.append("no direction reaches 50% ITT flips on the ladder (right-censored > 64)")
    if alpha_hi is not None and alpha_50 is not None and alpha_hi < alpha_50:
        abort.append("alpha_hi < alpha_50")
    grid = full_grid(alpha_lo, alpha_hi, alpha_50) if not abort else None
    return {
        "directions": per_direction, "alpha_flip50": flip50, "alpha_flip90": flip90,
        "alpha_lo": alpha_lo, "left_censored": left_censored, "alpha_50": alpha_50, "alpha_90": alpha_90,
        "alpha_hi": alpha_hi, "right_censored": {k: v is None for k, v in flip50.items()},
        "collapse_dose": _first_dose(ladder_rows, companies, "collapsed"),
        "truncation_dose": _first_dose(ladder_rows, companies, "truncated"),
        "abort": abort, "full_grid": grid,
        "reduced_grid": sorted({0.0, alpha_50, -alpha_50, alpha_hi, -alpha_hi}) if not abort else None,
        "random_grid": sorted({alpha_50, -alpha_50, alpha_90, -alpha_90, alpha_hi, -alpha_hi}) if not abort else None,
    }


def _first_dose(ladder_rows: Mapping[str, Mapping[float, Mapping[str, Any]]], companies: Sequence[str],
                kind: str) -> float | None:
    for magnitude in CAL_LADDER:
        for sign in (1.0, -1.0):
            if any(ladder_rows.get(t, {}).get(sign * magnitude, {}).get("unparsed_kind") == kind for t in companies):
                return magnitude
    return None


def full_grid(alpha_lo: float, alpha_hi: float, alpha_50: float) -> list[float]:
    """{0, +-2} U +-(ladder in [alpha_lo, alpha_hi]) U {+-alpha_next}; at most 14 nonzero points."""
    inside = [m for m in CAL_LADDER if alpha_lo <= m <= alpha_hi]
    above = [m for m in CAL_LADDER if m > alpha_hi]
    magnitudes = sorted(set(inside) | set(V2_ANCHORS) | ({above[0]} if above else set()))
    while 2 * len(magnitudes) > GRID_CAP:
        low = [m for m in magnitudes if m < alpha_50 / 2 and m not in V2_ANCHORS]
        if len(low) < 2:
            removable = [m for m in magnitudes if m not in V2_ANCHORS and m not in (alpha_50, alpha_hi)]
            if not removable:
                break
            magnitudes.remove(removable[0])
            continue
        for m in low[::2]:
            if 2 * len(magnitudes) <= GRID_CAP:
                break
            magnitudes.remove(m)
    return sorted([0.0] + magnitudes + [-m for m in magnitudes])


# ---------------------------------------------------------------------------------------------------
# bootstrap and pre-registered decision rules


def bootstrap_ci(units: Sequence[Any], statistic: Callable[[Sequence[Any]], float | None], *, samples: int = 2000,
                 seed: int = 20260925, level: float = 0.95) -> dict[str, Any]:
    """Percentile CI of ``statistic`` over units resampled with replacement (None draws are dropped)."""
    point = statistic(units)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        value = statistic([units[rng.randrange(len(units))] for _ in units])
        if value is not None:
            draws.append(value)
    if not draws:
        return {"point": point, "lower": None, "upper": None, "valid_draws": 0}
    draws.sort()
    lo = draws[int(math.floor((1 - level) / 2 * (len(draws) - 1)))]
    hi = draws[int(math.ceil((1 + level) / 2 * (len(draws) - 1)))]
    return {"point": point, "lower": lo, "upper": hi, "valid_draws": len(draws)}


def c5_contrast(baseline: Mapping[str, Mapping[str, Any]], dim_rows: Mapping[str, Mapping[str, Any]],
                random_rows: Mapping[str, Mapping[str, Mapping[str, Any]]], alpha: float, **boot: Any) -> dict[str, Any]:
    """DIM ITT on-target flip rate minus the max over random seeds of their any-direction flip rate."""
    source, goal = on_target(alpha)
    companies = sorted(dim_rows)

    def stat(sample: Sequence[str]) -> float | None:
        klass = [t for t in sample if baseline[t]["decision"] == source]
        if not klass:
            return None
        dim = sum(dim_rows[t]["decision"] == goal for t in klass) / len(klass)
        parsed = [t for t in sample if baseline[t]["decision"] in PARSED]
        if not parsed:
            return None
        rand = max(sum(rows[t]["decision"] in PARSED and rows[t]["decision"] != baseline[t]["decision"]
                       for t in parsed) / len(parsed) for rows in random_rows.values())
        return dim - rand

    ci = bootstrap_ci(companies, stat, **boot)
    return {"alpha": alpha, **ci, "supports": ci["lower"] is not None and ci["lower"] > 0}


def smoothness(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """C8 metrics on (alpha_eff, mean delta-M) points for one sign, ordered by |alpha_eff|."""
    pts = sorted(points, key=lambda p: abs(p[0]))
    ys = [y for _, y in pts]
    steps = [b - a for a, b in zip(ys, ys[1:])]
    sign = 1 if pts and pts[-1][0] > 0 else -1
    increasing = [s * sign > 0 for s in steps]
    reversals = sum(1 for a, b in zip(steps, steps[1:]) if (a > 0) != (b > 0) and a != 0 and b != 0)
    half = len(pts) // 2
    def slope(seg: Sequence[tuple[float, float]]) -> float | None:
        if len(seg) < 2 or seg[-1][0] == seg[0][0]:
            return None
        return (seg[-1][1] - seg[0][1]) / (seg[-1][0] - seg[0][0])
    low, high = slope(pts[:half + 1]), slope(pts[half:])
    return {"n_points": len(pts), "monotone_step_fraction": fmean(increasing) if increasing else None,
            "spearman": spearman([abs(x) for x, _ in pts], [y * sign for y in ys]), "reversals": reversals,
            "slope_high_over_low": (high / low) if low not in (None, 0) and high is not None else None}


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = fmean(rx), fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def flip_dose(baseline_row: Mapping[str, Any], rows_by_alpha: Mapping[float, Mapping[str, Any]], sign: float
              ) -> dict[str, Any]:
    """Smallest |alpha| on the grid (one sign) that flips this company on target; censoring kind otherwise.

    ``blocked``: every steered row of that sign parsed, none flipped. ``collapsed``: an unparsed row occurs
    before any flip. Only an actual flip yields a dose.
    """
    source, goal = on_target(sign)
    if baseline_row["decision"] != source:
        return {"eligible": False}
    for alpha in sorted((a for a in rows_by_alpha if a * sign > 0), key=abs):
        decision = rows_by_alpha[alpha]["decision"]
        if decision == goal:
            return {"eligible": True, "dose": abs(alpha), "censor": None}
        if decision not in PARSED:
            return {"eligible": True, "dose": None, "censor": "collapsed", "at": abs(alpha)}
    return {"eligible": True, "dose": None, "censor": "blocked"}
