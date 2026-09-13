"""Gate A/B/C statistics for entity-to-dial (protocol Rev 1 §4.2–4.4).

Pure functions with no model or artifact access. Bootstrap and Spearman
conventions match balanced-evidence-gap Phase 2 (n=2000, seed=42).
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .template import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, C2_RATIO_MIN


def toward_source_delta(patched: float, source: float, target: float) -> float:
    """sign(M_src − M_tgt) · (M_patched − M_tgt); positive = toward source."""
    if source == target:
        raise ValueError("source and target margins are identical")
    return math.copysign(1, source - target) * (patched - target)


def normalized_transfer(patched: float, source: float, target: float) -> float:
    """T = (M_patched − M_tgt) / (M_src − M_tgt); 1 = full transfer."""
    if source == target:
        raise ValueError("source and target margins are identical")
    return (patched - target) / (source - target)


def bootstrap_ci(
    values: Sequence[float],
    n: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = random.Random(seed)
    sample = list(values)
    if len(sample) < 4:
        raise ValueError("bootstrap CI requires at least four values")
    means = sorted(statistics.fmean(rng.choices(sample, k=len(sample))) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("spearman requires equal-length series of at least 2")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranked: list[float] = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranked[order[k]] = average
            i = j + 1
        return ranked

    rx = ranks(x)
    ry = ranks(y)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0.0 or sy == 0.0:
        raise ValueError("degenerate spearman input (constant series)")
    return cov / (sx * sy)


def _ci_excludes_zero(ci: tuple[float, float]) -> bool:
    return ci[0] > 0 or ci[1] < 0


def _direction_companies(direction: str) -> tuple[str, str]:
    source, target = direction.split("->")
    return source, target


def evaluate_gate_a(records: Sequence[Mapping[str, Any]], early_layers: Sequence[int]) -> dict:
    """Gate A1 (protocol §4.2).

    Units are the 8 directions; per-direction value = mean ticker-group
    toward-source ΔM over the early layers (L0–5). Pass = bootstrap 95%
    CI of the directional means excludes 0. Smoke grids (<4 directions)
    are recorded as not_evaluated.
    """
    early = set(early_layers)
    directions = sorted({r["direction"] for r in records if r["token_group"] == "ticker"})
    per_direction: dict[str, float] = {}
    for direction in directions:
        values = [
            float(r["toward_source_delta_m"])
            for r in records
            if r["token_group"] == "ticker" and r["direction"] == direction and r["layer"] in early
        ]
        if not values:
            raise ValueError(f"missing ticker records for direction {direction}")
        per_direction[direction] = statistics.fmean(values)
    if len(per_direction) < 4:
        return {
            "status": "not_evaluated",
            "reason": f"directions {len(per_direction)} < 4 (smoke grid)",
        }
    ci = bootstrap_ci([per_direction[d] for d in directions])
    return {
        "status": "evaluated",
        "a1": {
            "pass": _ci_excludes_zero(ci),
            "per_direction_mean": per_direction,
            "ci_95": list(ci),
            "early_layers": sorted(early),
        },
    }


def _block_sweep_gate(
    records: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    sector_of: Mapping[str, str],
    companies: Sequence[str],
    prefix: str,
) -> dict:
    """Shared block-sweep gate arithmetic for Gate B1/B3 and Gate D1/D3.

    {prefix}1: at least one layer in the interval whose MLP-block
toward-source ΔM bootstrap 95% CI (over directions) excludes 0.
    Strongest layer = max |mean| among qualifying layers. {prefix}3:
    at the strongest layer, the per-sector means (mean over the
directions each company participates in) are strictly 4/4 same sign.
    """
    n_directions = len({r["direction"] for r in records if r["component"] == "mlp"})
    if n_directions < 4:
        return {
            "status": "not_evaluated",
            "reason": f"directions {n_directions} < 4 (smoke grid)",
        }
    per_layer: dict[int, dict] = {}
    qualifying: list[int] = []
    for layer in layers:
        values = [
            float(r["toward_source_delta_m"])
            for r in records
            if r["component"] == "mlp" and r["layer"] == layer
        ]
        if len(values) != n_directions:
            raise ValueError(f"missing MLP records at L{layer}: {len(values)}/{n_directions}")
        mean = statistics.fmean(values)
        ci = bootstrap_ci(values)
        per_layer[layer] = {"mean": mean, "ci_95": list(ci), "n_directions": len(values)}
        if _ci_excludes_zero(ci):
            qualifying.append(layer)
    strongest = (
        max(qualifying, key=lambda layer: abs(per_layer[layer]["mean"]))
        if qualifying
        else None
    )
    sector_gate = None
    if strongest is not None:
        sector_means: dict[str, float] = {}
        for company in companies:
            sector = sector_of[company]
            values = [
                float(r["toward_source_delta_m"])
                for r in records
                if r["component"] == "mlp"
                and r["layer"] == strongest
                and company in _direction_companies(r["direction"])
            ]
            if not values:
                raise ValueError(f"no directions involve {company}")
            sector_means[sector] = statistics.fmean(values)
        same_sign = (
            all(value > 0 for value in sector_means.values())
            or all(value < 0 for value in sector_means.values())
        )
        sector_gate = {"pass": same_sign, "sector_means": sector_means, "n_sectors": len(sector_means)}
    return {
        "status": "evaluated",
        f"{prefix}1": {
            "pass": bool(qualifying),
            "qualifying_layers": qualifying,
            "per_layer": per_layer,
        },
        "strongest_layer": strongest,
        f"{prefix}3": sector_gate,
        "pass": bool(qualifying) and (sector_gate is not None and sector_gate["pass"]),
    }


def evaluate_gate_b(
    records: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    sector_of: Mapping[str, str],
    companies: Sequence[str],
) -> dict:
    """Gate B1+B3 (protocol §4.3)."""
    return _block_sweep_gate(records, layers, sector_of, companies, "b")


def evaluate_gate_d(
    records: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    sector_of: Mapping[str, str],
    companies: Sequence[str],
) -> dict:
    """Gate D1+D3 (proposal-phase-d.md §4.2).

    Identical criterion to B1/B3 applied to the L12–31 instruction-span
    block sweep (Gate D = D1 + D3).
    """
    return _block_sweep_gate(records, layers, sector_of, companies, "d")


def evaluate_gate_c(delta_m_dials: Sequence[float], gaps: Sequence[float]) -> dict:
    """Gate C1+C2 (protocol §4.4).

    Units are the 16 companies. C1: bootstrap 95% CI of the mean ΔM_dial
    has both endpoints on the same side of 0 as the mean gap (fail-closed
    when the mean gap is exactly 0). C2: mean|ΔM_dial| / mean|gap| ≥ 0.25.
    """
    if len(delta_m_dials) != len(gaps) or not delta_m_dials:
        raise ValueError("gate C requires paired per-company values")
    if len(delta_m_dials) < 4:
        return {
            "status": "not_evaluated",
            "reason": f"companies {len(delta_m_dials)} < 4 (smoke grid)",
        }
    mean_gap = statistics.fmean(gaps)
    mean_delta = statistics.fmean(delta_m_dials)
    if mean_gap == 0.0:
        c1: dict = {
            "pass": False,
            "degenerate_gap": True,
            "mean_gap": 0.0,
            "mean_delta_m": mean_delta,
            "ci_95": None,
        }
    else:
        ci = bootstrap_ci(delta_m_dials)
        c1 = {
            "pass": (ci[0] > 0 and ci[1] > 0 and mean_gap > 0)
            or (ci[0] < 0 and ci[1] < 0 and mean_gap < 0),
            "degenerate_gap": False,
            "mean_gap": mean_gap,
            "mean_delta_m": mean_delta,
            "ci_95": list(ci),
        }
    mean_abs_gap = statistics.fmean(abs(value) for value in gaps)
    if mean_abs_gap == 0.0:
        c2: dict = {"pass": False, "ratio": None, "degenerate_gap": True}
    else:
        ratio = statistics.fmean(abs(value) for value in delta_m_dials) / mean_abs_gap
        c2 = {"pass": ratio >= C2_RATIO_MIN, "ratio": ratio, "degenerate_gap": False}
    return {
        "status": "evaluated",
        "c1": c1,
        "c2": c2,
        "pass": c1["pass"] and c2["pass"],
    }


def c1_descriptive(
    dial_deltas_entity: Mapping[str, float],
    dial_deltas_final: Mapping[str, float],
    pure_margins: Mapping[str, float],
) -> dict:
    """Step C1 descriptive readout (not gated).

    Spearman ρ of the named-minus-anonymous dial activation difference
    (per position) against the 2A pure entity margin. Requires ≥3
    companies; None otherwise.
    """
    def rho(values: Mapping[str, float]) -> float | None:
        tickers = sorted(values)
        if len(tickers) < 3:
            return None
        try:
            return spearman(
                [values[t] for t in tickers], [pure_margins[t] for t in tickers]
            )
        except ValueError:
            # Degenerate (constant) series: no rank correlation defined.
            return None

    return {
        "rho_entity_position": rho(dial_deltas_entity),
        "rho_final_position": rho(dial_deltas_final),
        "descriptive_only": True,
    }


# ── Phase E (proposal-phase-e.md) ────────────────────────────────────────────


def _effective_excluded(full_delta_m: float, min_abs: float) -> bool:
    """R1 denominator rule: a direction is excluded when |ΔM_full| < min_abs.

    The boundary |ΔM_full| == min_abs is INCLUDED (≥).
    """
    if not math.isfinite(full_delta_m):
        raise ValueError("non-finite full-arm delta")
    return abs(full_delta_m) < min_abs


def _ratio_median(ratios: Sequence[float]) -> float | None:
    if not ratios:
        return None
    return statistics.median(ratios)


def evaluate_gate_e(
    e1_records: Sequence[Mapping[str, Any]],
    e2_records: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    gate_layer: int,
    k_values: Sequence[int],
    min_full_dm: float,
    e1_min: float,
    e2b_min: float,
    falsifier_max: float,
    smoke: bool,
) -> dict:
    """Gate E1 + Gate E2b (proposal-phase-e.md §3, §4.2, §4.3).

    Ratios are per-direction (joint or arm ΔM) / (in-run full-arm ΔM at the
    gate layer); medians are over the R1-effective directions only
    (|ΔM_full| ≥ min_full_dm). All arms share the same denominator.
    """
    if smoke:
        return {"status": "not_evaluated", "reason": "smoke grid (mechanism validation only)"}

    full_dm: dict[tuple[str, int], float] = {}
    joint_dm: dict[tuple[str, int], float] = {}
    for r in e1_records:
        if r["arm"] == "full":
            full_dm[(r["direction"], r["layer"])] = r["toward_source_delta_m"]
        elif r["arm"] == "joint":
            joint_dm[(r["direction"], r["layer"])] = r["toward_source_delta_m"]
    missing = [k for k in joint_dm if k not in full_dm]
    if missing:
        raise ValueError(f"joint records without a matched full arm: {sorted(missing)[:3]}")

    directions = sorted({d for d, _ in joint_dm})
    if not directions:
        raise ValueError("no E1 joint records")

    # ── E1 curve + gate (per layer; gate at gate_layer) ────────────────────
    e1_curves: dict[str, dict] = {}
    for layer in layers:
        per = []
        for d in directions:
            full = full_dm.get((d, layer))
            joint = joint_dm.get((d, layer))
            if full is None or joint is None:
                raise ValueError(f"missing full/joint record for {d} at L{layer}")
            excluded = _effective_excluded(full, min_full_dm)
            per.append(
                {
                    "direction": d,
                    "joint_dm": joint,
                    "full_dm": full,
                    "ratio": (joint / full) if not excluded else None,
                    "excluded": excluded,
                }
            )
        ratios = [p["ratio"] for p in per if not p["excluded"]]
        e1_curves[str(layer)] = {
            "median_ratio": _ratio_median(ratios),
            "n_effective": len(ratios),
            "excluded_directions": [p["direction"] for p in per if p["excluded"]],
            "mean_joint_dm": statistics.fmean(p["joint_dm"] for p in per),
            "mean_full_dm": statistics.fmean(p["full_dm"] for p in per),
        }

    gate_per = e1_curves[str(gate_layer)]
    gate_e1_pass = (
        gate_per["median_ratio"] is not None and gate_per["median_ratio"] >= e1_min
    )
    gate_e1 = {
        "layer": gate_layer,
        "per_direction": [
            p for p in (
                {
                    "direction": d,
                    "joint_dm": joint_dm[(d, gate_layer)],
                    "full_dm": full_dm[(d, gate_layer)],
                    "ratio": (
                        joint_dm[(d, gate_layer)] / full_dm[(d, gate_layer)]
                        if not _effective_excluded(full_dm[(d, gate_layer)], min_full_dm)
                        else None
                    ),
                    "excluded": _effective_excluded(full_dm[(d, gate_layer)], min_full_dm),
                }
                for d in directions
            )
        ],
        "median_ratio": gate_per["median_ratio"],
        "n_effective": gate_per["n_effective"],
        "excluded_directions": gate_per["excluded_directions"],
        "threshold": e1_min,
        "pass": gate_e1_pass,
    }

    # ── E2: dial transplant (gate E2b) + PCA sweep (E2a descriptive) ──────
    l15_full = {d: full_dm[(d, gate_layer)] for d in directions}

    def _arm_stats(arm: str) -> dict:
        per = []
        for d in directions:
            match = [r for r in e2_records if r["arm"] == arm and r["direction"] == d]
            if len(match) != 1:
                raise ValueError(f"expected exactly one {arm} record for {d}, got {len(match)}")
            r = match[0]
            excluded = _effective_excluded(l15_full[d], min_full_dm)
            ratio = (
                r["toward_source_delta_m"] / l15_full[d]
                if not excluded and l15_full[d] != 0.0
                else None
            )
            per.append(
                {
                    "direction": d,
                    "arm_dm": r["toward_source_delta_m"],
                    "full_dm": l15_full[d],
                    "ratio": ratio,
                    "excluded": excluded,
                }
            )
        ratios = [p["ratio"] for p in per if not p["excluded"]]
        return {
            "per_direction": per,
            "median_ratio": _ratio_median(ratios),
            "n_effective": len(ratios),
            "excluded_directions": [p["direction"] for p in per if p["excluded"]],
        }

    dial = _arm_stats("dial_transplant")
    dial_median = dial["median_ratio"]
    gate_e2b_pass = dial_median is not None and dial_median >= e2b_min
    gate_e2b = {
        "layer": gate_layer,
        **dial,
        "threshold": e2b_min,
        "falsifier_max": falsifier_max,
        "falsified": dial_median is not None and dial_median < falsifier_max,
        "pass": gate_e2b_pass,
    }

    e2a_curve: dict[str, dict] = {}
    for k in k_values:
        stats = _arm_stats(f"pca_k{k}")
        e2a_curve[str(k)] = stats
    full_k = (
        _arm_stats("pca_full")
        if any(r["arm"] == "pca_full" for r in e2_records)
        else None
    )
    if full_k is not None:
        e2a_curve["full"] = full_k
    footprint = _arm_stats("footprint") if any(r["arm"] == "footprint" for r in e2_records) else None

    return {
        "status": "evaluated",
        "gate_e1": gate_e1,
        "e1_curves": e1_curves,
        "gate_e2b": gate_e2b,
        "e2a_curve": e2a_curve,
        "e2_pca_full": full_k,
        "e2_footprint": footprint,
        "gate": {
            "pass": gate_e1_pass and gate_e2b_pass,
            "e1": gate_e1_pass,
            "e2b": gate_e2b_pass,
        },
    }


# ── Phase F: dual-path additivity + directional push ──────────────────────────


def classify_f2_push(
    plus: Mapping[float, float],
    minus: Mapping[float, float],
    *,
    jitter_band: float,
) -> str:
    """Four-way classification of one F2 push arm (protocol §4.3).

    ``plus``/``minus`` map α ∈ {1.0, 2.0} → ΔM (nats) for the +/− push.
    A point is decisive only when it is strictly beyond the jitter band
    (|ΔM| = band exactly is jitter-band, not decisive). Returns one of:
    ``signed_sell_axis`` (+ push moves margin sell at both α, − push buy),
    ``signed_buy_axis`` (mirror), ``attractor`` (both signs push the same
    direction decisively), or ``context_dependent_or_null`` (mixed / all
    within the band).
    """
    if set(plus) != {1.0, 2.0} or set(minus) != {1.0, 2.0}:
        raise ValueError("verdict requires α ∈ {1.0, 2.0} for both signs")

    def all_sell(values: Mapping[float, float]) -> bool:
        return all(v < -jitter_band for v in values.values())

    def all_buy(values: Mapping[float, float]) -> bool:
        return all(v > jitter_band for v in values.values())

    plus_sell, plus_buy = all_sell(plus), all_buy(plus)
    minus_sell, minus_buy = all_sell(minus), all_buy(minus)
    if plus_sell and minus_buy:
        return "signed_sell_axis"
    if plus_buy and minus_sell:
        return "signed_buy_axis"
    if (plus_sell and minus_sell) or (plus_buy and minus_buy):
        return "attractor"
    return "context_dependent_or_null"


def evaluate_gate_f(
    f1_records: Sequence[Mapping[str, Any]],
    f2_records: Sequence[Mapping[str, Any]],
    *,
    directions: Sequence[str],
    top_group: Sequence[str],
    bottom_group: Sequence[str],
    e01_ref: Mapping[str, Mapping[str, float]],
    min_full_dm: float,
    f1_min: float,
    jitter_band: float,
    consistency_tolerance: float,
    smoke: bool,
) -> dict:
    """Gate F1 + F2 descriptive verdicts (proposal-phase-f.md §3, §4.2, §4.3).

    Additivity ratio = per-direction ΔM_combined / ΔM_full (in-run full
    arm, R1 convention inherited from Phase E E2b via the same
    ``_effective_excluded`` rule); gate = median over R1-effective
    directions ≥ f1_min. F2 verdicts use the §4.3 four-way rule on the
    α ∈ {1.0, 2.0} points. Gate F = Gate F1 (F2 is descriptive).
    """
    if smoke:
        return {"status": "not_evaluated", "reason": "smoke grid (mechanism validation only)"}

    arms = ("full", "v1", "k8", "dial", "combined")
    per: dict[str, dict] = {}
    for d in directions:
        vals: dict[str, float] = {}
        for arm in arms:
            match = [r for r in f1_records if r["direction"] == d and r["arm"] == arm]
            if len(match) != 1:
                raise ValueError(f"expected exactly one {arm} record for {d}, got {len(match)}")
            vals[arm] = match[0]["toward_source_delta_m"]
        full = vals["full"]
        excluded = _effective_excluded(full, min_full_dm)
        ratio = (vals["combined"] / full) if not excluded else None
        per[d] = {
            "full_dm": full,
            "v1_dm": vals["v1"],
            "k8_dm": vals["k8"],
            "dial_dm": vals["dial"],
            "combined_dm": vals["combined"],
            "additivity_ratio": ratio,
            "interaction_delta_m": vals["combined"] - vals["v1"] - vals["dial"],
            "excluded": excluded,
        }

    ratios = [p["additivity_ratio"] for p in per.values() if p["additivity_ratio"] is not None]
    median = statistics.median(ratios) if ratios else None
    gate_f1_pass = median is not None and median >= f1_min

    def group_median(group: Sequence[str]) -> float | None:
        rs = [
            per[d]["additivity_ratio"]
            for d in directions
            if d.split("->")[0] in group and per[d]["additivity_ratio"] is not None
        ]
        return statistics.median(rs) if rs else None

    secondary = {
        "bottom_to_top": group_median(bottom_group),
        "top_to_bottom": group_median(top_group),
    }

    consistency: dict[str, float | None] = {}
    for arm in ("v1", "k8", "dial", "full"):
        diffs = [
            abs(per[d][f"{arm}_dm"] - e01_ref[d][arm])
            for d in directions
            if d in e01_ref and arm in e01_ref[d]
        ]
        consistency[arm] = max(diffs) if diffs else None

    f2_verdict: dict[str, str] = {}
    dose: dict[str, list] = {}
    for arm in ("v1", "dial_fp"):
        arm_records = [r for r in f2_records if r["arm"] == arm]
        plus: dict[float, float] = {}
        minus: dict[float, float] = {}
        for r in arm_records:
            if r["alpha"] not in (0.5, 1.0, 2.0) or r["sign"] not in (1, -1):
                raise ValueError(f"unexpected F2 point for {arm}: {r}")
            if r["sign"] > 0:
                plus[r["alpha"]] = r["delta_m"]
            else:
                minus[r["alpha"]] = r["delta_m"]
        if set(plus) != {0.5, 1.0, 2.0} or set(minus) != {0.5, 1.0, 2.0}:
            raise ValueError(f"incomplete F2 dose grid for {arm}: plus={sorted(plus)} minus={sorted(minus)}")
        f2_verdict[arm] = classify_f2_push(
            {a: plus[a] for a in (1.0, 2.0)},
            {a: minus[a] for a in (1.0, 2.0)},
            jitter_band=jitter_band,
        )
        dose[arm] = sorted(
            ((r["alpha"], r["sign"], r["delta_m"]) for r in arm_records),
            key=lambda x: (x[0], x[1]),
        )

    return {
        "status": "evaluated",
        "gate_f1": {
            "per_direction": [
                {"direction": d, **per[d]} for d in directions
            ],
            "median_ratio": median,
            "n_effective": len(ratios),
            "excluded_directions": [d for d in directions if per[d]["excluded"]],
            "threshold": f1_min,
            "pass": gate_f1_pass,
        },
        "f1_secondary": secondary,
        "f1_consistency": {
            **{arm: v for arm, v in consistency.items()},
            "tolerance": consistency_tolerance,
        },
        "f2_verdict": f2_verdict,
        "f2_dose_response": dose,
        "gate": {"pass": gate_f1_pass, "f1": gate_f1_pass},
    }
