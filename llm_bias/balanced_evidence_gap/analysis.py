"""Estimators and pre-registered gates for Phase 2 (pure functions).

Gate definitions are frozen in docs/balanced-evidence-gap/proposal-phase2.md
§4.3 and §4.5. These functions are deterministic and side-effect free.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

GATE_2A = {
    "iqr_threshold_nats": 0.5,
    "spearman_threshold": 0.5,
    "framing_max_median_nats": 1.5,
    "bootstrap_samples": 2000,
    "bootstrap_seed": 42,
}

GATE_2A_REV2 = {
    "iqr_threshold_nats": 0.5,
    "framing_max_median_nats": 1.5,
    "spearman_gap_min": 0.3,
    "group_size": 2,
}

GATE_2C = {
    "holm_alpha": 0.05,
    "sector_agreement_min": 0.75,
    "matched_control_samples": 10,
    "control_seed": 42,
}

# Qwen3.5-4B hybrid architecture: full-attention layers only (config
# full_attention_interval=4). Head-level attention interventions are
# restricted to these layers (same set as the entity-cell line).
FULL_ATTENTION_LAYERS: tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31)


# ── basic statistics ─────────────────────────────────────────────────────────

def iqr(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if len(ordered) < 4:
        raise ValueError("IQR requires at least four values")
    quarter = (len(ordered) - 1) / 4
    def quantile(p: float) -> float:
        position = p * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        fraction = position - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction
    return quantile(0.75) - quantile(0.25)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("pearson requires equal-length sequences of length >= 3")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a - mx) ** 2 for a in x))
    vy = math.sqrt(sum((b - my) ** 2 for b in y))
    if vx == 0 or vy == 0:
        raise ValueError("zero-variance input to pearson")
    return cov / (vx * vy)


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_rankdata(x), _rankdata(y))


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    sample = list(values)
    if len(sample) < 4:
        raise ValueError("bootstrap CI requires at least four values")
    means = sorted(statistics.fmean(rng.choices(sample, k=len(sample))) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """One-sided exact test that all effects share the sign of their mean."""
    positive = sum(1 for v in values if v > 0)
    negative = sum(1 for v in values if v < 0)
    n = positive + negative
    if n == 0:
        raise ValueError("sign-flip test requires nonzero effects")
    k = max(positive, negative)
    # p = P(Binomial(n, 0.5) >= k), two-sided-consistent (worst tail).
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        multiplier = len(p_values) - rank
        running = max(running, multiplier * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


# ── patching estimators (causal-tracing conventions) ─────────────────────────

def toward_source_delta(patched: float, source: float, target: float) -> float:
    """sign(M_src − M_tgt) · (M_patched − M_tgt); positive = toward source."""
    if source == target:
        raise ValueError("source and target margins are identical; no contrast")
    return math.copysign(1, source - target) * (patched - target)


def normalized_transfer(patched: float, source: float, target: float) -> float:
    """T = (M_patched − M_tgt) / (M_src − M_tgt); 1 = full transfer."""
    if source == target:
        raise ValueError("source and target margins are identical; no contrast")
    return (patched - target) / (source - target)


# ── gate 2A ──────────────────────────────────────────────────────────────────

def evaluate_gate_2a(
    *,
    pure_entity_margins: dict[str, float],
    phase1_named_margins: dict[str, float],
    framing_pair_deltas: Sequence[float],
    valid_rate: float,
) -> dict:
    margins = [pure_entity_margins[t] for t in sorted(pure_entity_margins)]
    iqr_value = iqr(margins)
    common = sorted(set(pure_entity_margins) & set(phase1_named_margins))
    if len(common) < 8:
        raise ValueError(f"gate 2A needs >=8 companies in both phases, got {len(common)}")
    rho = spearman(
        [pure_entity_margins[t] for t in common],
        [phase1_named_margins[t] for t in common],
    )
    framing_median = statistics.median(abs(v) for v in framing_pair_deltas)
    criteria = {
        "iqr": {
            "value": iqr_value,
            "threshold": GATE_2A["iqr_threshold_nats"],
            "pass": iqr_value > GATE_2A["iqr_threshold_nats"],
        },
        "spearman_vs_phase1": {
            "value": rho,
            "threshold": GATE_2A["spearman_threshold"],
            "pass": abs(rho) > GATE_2A["spearman_threshold"],
            "n_companies": len(common),
        },
        "framing_stability": {
            "value": framing_median,
            "threshold": GATE_2A["framing_max_median_nats"],
            "pass": framing_median < GATE_2A["framing_max_median_nats"],
        },
        "schema_valid_rate": {
            "value": valid_rate,
            "threshold": 1.0,
            "pass": valid_rate >= 1.0,
        },
    }
    return {
        "gate": "2A",
        "criteria": criteria,
        "pass": all(c["pass"] for c in criteria.values()),
        "phase2b_authorized": all(c["pass"] for c in criteria.values()),
    }


# ── gate 2A Rev 2 (docs/balanced-evidence-gap/proposal-phase2-rev2.md) ───

def select_margin_groups(
    pure_entity_margins: dict[str, float],
    group_size: int = GATE_2A_REV2["group_size"],
) -> tuple[list[str], list[str]]:
    """(bottom, top) tickers by pure entity margin rank; frozen top-N/bottom-N rule."""
    if len(pure_entity_margins) < 2 * group_size + 1:
        raise ValueError(
            f"margin grouping needs >= {2 * group_size + 1} tickers, got {len(pure_entity_margins)}"
        )
    ordered = sorted(pure_entity_margins, key=lambda t: pure_entity_margins[t])
    return ordered[:group_size], ordered[-group_size:]


def evaluate_gate_2a_rev2(
    *,
    pure_entity_margins: dict[str, float],
    phase1_gaps: dict[str, float],
    framing_pair_deltas: Sequence[float],
    valid_rate: float,
) -> dict:
    """Gate 2A Rev 2: IQR + framing + schema + Spearman vs Phase 1 gap + group construct check."""
    margins = [pure_entity_margins[t] for t in sorted(pure_entity_margins)]
    iqr_value = iqr(margins)
    common = sorted(set(pure_entity_margins) & set(phase1_gaps))
    if len(common) < 8:
        raise ValueError(f"gate 2A rev2 needs >=8 companies in both phases, got {len(common)}")
    rho_gap = spearman(
        [pure_entity_margins[t] for t in common],
        [phase1_gaps[t] for t in common],
    )
    bottom, top = select_margin_groups(pure_entity_margins)
    pairwise = {
        f"{t}>{b}": phase1_gaps[t] - phase1_gaps[b] for t in top for b in bottom
    }
    n_positive = sum(1 for d in pairwise.values() if d > 0)
    framing_median = statistics.median(abs(v) for v in framing_pair_deltas)
    criteria = {
        "iqr": {
            "value": iqr_value,
            "threshold": GATE_2A_REV2["iqr_threshold_nats"],
            "pass": iqr_value > GATE_2A_REV2["iqr_threshold_nats"],
        },
        "spearman_vs_phase1_gap": {
            "value": rho_gap,
            "threshold": GATE_2A_REV2["spearman_gap_min"],
            "pass": rho_gap > GATE_2A_REV2["spearman_gap_min"],
            "n_companies": len(common),
        },
        "group_construct_check": {
            "top": list(top),
            "bottom": list(bottom),
            "pairwise_gap_diffs": pairwise,
            "n_positive": n_positive,
            "n_pairs": len(pairwise),
            "threshold": "all pairs positive",
            "pass": n_positive == len(pairwise),
        },
        "framing_stability": {
            "value": framing_median,
            "threshold": GATE_2A_REV2["framing_max_median_nats"],
            "pass": framing_median < GATE_2A_REV2["framing_max_median_nats"],
        },
        "schema_valid_rate": {
            "value": valid_rate,
            "threshold": 1.0,
            "pass": valid_rate >= 1.0,
        },
    }
    return {
        "gate": "2A-rev2",
        "criteria": criteria,
        "pass": all(c["pass"] for c in criteria.values()),
        "phase2b_authorized": all(c["pass"] for c in criteria.values()),
    }


# ── handoff detection (2B discovery) ────────────────────────────────────────

def detect_handoff(
    layers: Sequence[int],
    entity_t: dict[int, float],
    context_t: dict[int, float],
) -> dict:
    """First contiguous band where context transfer >= entity transfer.

    Descriptive (not a gate): returns the crossover band and the layers 2C
    should use (band ± 1), split by architecture.
    """
    ordered = sorted(set(layers))
    band: list[int] = []
    for layer in ordered:
        if entity_t[layer] is None or context_t[layer] is None:
            continue
        if context_t[layer] >= entity_t[layer]:
            band.append(layer)
    result: dict = {
        "crossover_band": band,
        "crossover": bool(band),
    }
    if band:
        low, high = band[0], band[-1]
        interval = [l for l in ordered if low - 1 <= l <= high + 1]
        result["interval"] = interval
        result["attention_layers"] = [l for l in interval if l in FULL_ATTENTION_LAYERS]
        result["mlp_layers"] = list(interval)
    else:
        # No crossover: entity state stays at the header. 2C falls back to
        # the layers of maximal entity-span transfer (top-3, ± 1), per
        # proposal-phase2.md revision note.
        ranked = sorted(
            (l for l in ordered if entity_t[l] is not None),
            key=lambda l: entity_t[l],
            reverse=True,
        )
        peaks = ranked[:3]
        interval = sorted({l for l in ordered for p in peaks if abs(l - p) <= 1})
        result["fallback"] = "maximal_entity_transfer"
        result["peaks"] = peaks
        result["interval"] = interval
        result["attention_layers"] = [l for l in interval if l in FULL_ATTENTION_LAYERS]
        result["mlp_layers"] = list(interval)
    return result


# ── gate 2C ──────────────────────────────────────────────────────────────────

def evaluate_gate_2c(
    *,
    attention_arm: dict | None,
    mlp_arm: dict,
) -> dict:
    """Formal gate over the two 2C arms (proposal §4.5).

    attention_arm: {"status": "run"|"not_applicable", "head_effects":
    {key: {"mean_delta", "direction_deltas"}}, "holm_adjusted_p": {key: p},
    "top_head": key} where direction_deltas are paired differences
    (entity zeroing − matched position zeroing) per direction.
    mlp_arm: {"top_attribution": float, "control_mean": float,
    "sector_agreement": float, "sign_flip_p": float}.
    """
    attention: dict
    if attention_arm is None or attention_arm.get("status") == "not_applicable":
        attention = {"status": "not_applicable", "pass": None}
    else:
        effects = attention_arm["head_effects"]
        if not effects:
            raise ValueError("attention arm requires head effects")
        top_head = attention_arm.get("top_head") or max(effects, key=lambda k: effects[k]["mean_delta"])
        p = attention_arm["holm_adjusted_p"][top_head]
        passed = effects[top_head]["mean_delta"] > 0 and p < GATE_2C["holm_alpha"]
        attention = {
            "status": "run",
            "top_head": top_head,
            "top_effect": effects[top_head]["mean_delta"],
            "sign_flip_p_adjusted": p,
            "pass": passed,
        }

    mlp_top = mlp_arm["top_attribution"]
    mlp_control = mlp_arm["control_mean"]
    mlp_sector_agreement = mlp_arm["sector_agreement"]
    mlp_p = mlp_arm["sign_flip_p"]
    mlp_passed = (
        mlp_top > mlp_control
        and mlp_sector_agreement >= GATE_2C["sector_agreement_min"]
        and mlp_p < GATE_2C["holm_alpha"]
    )
    attention_passed = attention["pass"] if attention["pass"] is not None else False
    return {
        "gate": "2C",
        "attention_arm": attention,
        "mlp_arm": {
            "top_attribution": mlp_top,
            "control_mean": mlp_control,
            "sector_agreement": mlp_sector_agreement,
            "sign_flip_p": mlp_p,
            "per_layer": mlp_arm.get("per_layer", {}),
            "pass": mlp_passed,
        },
        "pass": bool(attention_passed or mlp_passed),
        "formal": True,
    }
