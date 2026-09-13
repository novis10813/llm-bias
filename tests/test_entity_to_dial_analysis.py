"""Regression tests for entity-to-dial gate statistics (pure functions)."""
from __future__ import annotations

import pytest

from llm_bias.entity_to_dial.analysis import (
    bootstrap_ci,
    c1_descriptive,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gate_d,
    normalized_transfer,
    spearman,
    toward_source_delta,
)

TOP = ("NSC", "BLK")
BOTTOM = ("IT", "BDX")
COMPANIES = ("NSC", "BLK", "IT", "BDX")
SECTORS = {"NSC": "S1", "BLK": "S2", "IT": "S3", "BDX": "S4"}
EARLY_LAYERS = (0, 2, 4)
B_LAYERS = (12, 13, 14, 15)


def _directions() -> list[tuple[str, str]]:
    pairs = [(s, t) for s in TOP for t in BOTTOM]
    return pairs + [(t, s) for s, t in pairs]


def _a_records(directions, value_fn) -> list[dict]:
    records = []
    for direction in directions:
        for layer in EARLY_LAYERS:
            records.append(
                {
                    "direction": f"{direction[0]}->{direction[1]}",
                    "layer": layer,
                    "token_group": "ticker",
                    "toward_source_delta_m": value_fn(direction, layer),
                }
            )
    return records


def test_gate_a_pass_with_consistent_positive_transfer():
    directions = _directions()
    records = _a_records(directions, lambda d, layer: 0.5 + 0.01 * layer)
    gate = evaluate_gate_a(records, early_layers=EARLY_LAYERS)
    assert gate["status"] == "evaluated"
    assert gate["a1"]["pass"] is True
    assert gate["a1"]["ci_95"][0] > 0
    assert len(gate["a1"]["per_direction_mean"]) == 8


def test_gate_a_fails_when_ci_crosses_zero():
    directions = _directions()
    records = _a_records(
        directions,
        lambda d, layer: 1.0 if d[0] in TOP else -1.0,
    )
    gate = evaluate_gate_a(records, early_layers=EARLY_LAYERS)
    assert gate["status"] == "evaluated"
    assert gate["a1"]["pass"] is False
    assert gate["a1"]["ci_95"][0] < 0 < gate["a1"]["ci_95"][1]


def test_gate_a_smoke_grid_not_evaluated():
    directions = _directions()[:2]
    records = _a_records(directions, lambda d, layer: 0.5)
    gate = evaluate_gate_a(records, early_layers=EARLY_LAYERS)
    assert gate["status"] == "not_evaluated"


def test_gate_a_ignores_name_group_and_fails_on_missing_direction():
    directions = _directions()
    records = _a_records(directions, lambda d, layer: 0.5)
    # Name-group records for the same directions carry different values and
    # must be ignored entirely.
    for direction in directions:
        for layer in EARLY_LAYERS:
            records.append(
                {
                    "direction": f"{direction[0]}->{direction[1]}",
                    "layer": layer,
                    "token_group": "name",
                    "toward_source_delta_m": -0.5,
                }
            )
    gate = evaluate_gate_a(records, early_layers=EARLY_LAYERS)
    assert gate["a1"]["pass"] is True
    assert all(v == pytest.approx(0.5) for v in gate["a1"]["per_direction_mean"].values())
    # A direction whose ticker records exist only outside the early layers
    # fails closed (present but incomplete, not absent).
    incomplete = [
        r for r in records
        if not (r["token_group"] == "ticker" and r["direction"] == "NSC->IT" and r["layer"] in EARLY_LAYERS)
    ]
    incomplete.append(
        {
            "direction": "NSC->IT",
            "layer": 9,
            "token_group": "ticker",
            "toward_source_delta_m": 0.5,
        }
    )
    with pytest.raises(ValueError, match="missing ticker records"):
        evaluate_gate_a(incomplete, early_layers=EARLY_LAYERS)


def _b_records(value_fn, layers=B_LAYERS) -> list[dict]:
    records = []
    for direction in _directions():
        for layer in layers:
            for component in ("mlp", "attn"):
                records.append(
                    {
                        "direction": f"{direction[0]}->{direction[1]}",
                        "layer": layer,
                        "component": component,
                        "toward_source_delta_m": value_fn(direction, layer, component),
                    }
                )
    return records


def test_gate_b_pass_with_strong_mlp_layer():
    def value(direction, layer, component):
        if component != "mlp":
            return 0.0
        if layer == 12:
            return 0.5
        # Layers 13-15 straddle zero so they never qualify.
        return 0.01 if direction[0] in TOP else -0.01

    gate = evaluate_gate_b(
        _b_records(value), layers=B_LAYERS, sector_of=SECTORS, companies=COMPANIES
    )
    assert gate["status"] == "evaluated"
    assert gate["b1"]["pass"] is True
    assert gate["b1"]["qualifying_layers"] == [12]
    assert gate["strongest_layer"] == 12
    assert gate["b3"]["pass"] is True
    assert gate["b3"]["n_sectors"] == 4
    assert gate["pass"] is True


def test_gate_b_fails_when_no_layer_qualifies():
    gate = evaluate_gate_b(
        _b_records(
            lambda d, layer, component:
            0.01 if (component == "mlp" and d[0] in TOP) else -0.01
        ),
        layers=B_LAYERS,
        sector_of=SECTORS,
        companies=COMPANIES,
    )
    assert gate["b1"]["pass"] is False
    assert gate["b1"]["qualifying_layers"] == []
    assert gate["strongest_layer"] is None
    assert gate["b3"] is None
    assert gate["pass"] is False


def test_gate_b_sector_means_use_participating_directions():
    # All-positive layer; each sector mean must equal the mean of the four
    # directions its company participates in (each direction is shared by
    # exactly two companies, one per group).
    direction_values = {
        d: 1.0 + 0.1 * index for index, d in enumerate(_directions())
    }

    def value(direction, layer, component):
        if component != "mlp" or layer != 12:
            return 0.01 if direction[0] in TOP else -0.01
        return direction_values[direction]

    gate = evaluate_gate_b(
        _b_records(value), layers=B_LAYERS, sector_of=SECTORS, companies=COMPANIES
    )
    assert gate["strongest_layer"] == 12
    assert gate["b3"]["pass"] is True
    for company in COMPANIES:
        participating = [
            direction_values[d]
            for d in _directions()
            if company in d
        ]
        assert len(participating) == 4
        assert gate["b3"]["sector_means"][SECTORS[company]] == pytest.approx(
            sum(participating) / 4
        )


def test_gate_b_b3_same_sign_logic():
    # B3's same-sign check is defensive: with the frozen 8-direction
    # geometry, a flipped sector mean implies the global CI crosses 0, so
    # the failure branch is exercised through a zero sector mean only via
    # direct construction. Verify the arithmetic branch itself by feeding
    # the gate a layer whose sector means are computed over distinct signs.
    # All-constant-negative layer: B1 passes (CI < 0) and B3 passes with
    # all-negative sector means.
    def value(direction, layer, component):
        if component != "mlp" or layer != 12:
            return 0.01 if direction[0] in TOP else -0.01
        return -0.5

    gate = evaluate_gate_b(
        _b_records(value), layers=B_LAYERS, sector_of=SECTORS, companies=COMPANIES
    )
    assert gate["b1"]["qualifying_layers"] == [12]
    assert gate["strongest_layer"] == 12
    assert gate["b3"]["pass"] is True
    assert all(mean < 0 for mean in gate["b3"]["sector_means"].values())
    assert gate["pass"] is True


def test_gate_b_smoke_grid_not_evaluated_and_missing_records_fail():
    records = _b_records(lambda d, layer, component: 0.5)
    smoke = [r for r in records if r["direction"] in ("NSC->IT", "IT->NSC")]
    assert evaluate_gate_b(
        smoke, layers=B_LAYERS, sector_of=SECTORS, companies=COMPANIES
    )["status"] == "not_evaluated"
    incomplete = [r for r in records if not (r["layer"] == 13 and r["component"] == "mlp")]
    with pytest.raises(ValueError, match="missing MLP records"):
        evaluate_gate_b(
            incomplete, layers=B_LAYERS, sector_of=SECTORS, companies=COMPANIES
        )


def test_gate_d_matches_gate_b_arithmetic():
    # Gate D1/D3 reuses the B1/B3 criterion over the L12–31 interval: the
    # numeric output must be identical, only the label prefix differs.
    def value(direction, layer, component):
        if component != "mlp":
            return 0.0
        if layer == 12:
            return 0.5
        return 0.01 if direction[0] in TOP else -0.01

    d_layers = tuple(range(12, 32))
    records = _b_records(value, layers=d_layers)
    gate_b = evaluate_gate_b(
        records, layers=d_layers, sector_of=SECTORS, companies=COMPANIES
    )
    gate_d = evaluate_gate_d(
        records, layers=d_layers, sector_of=SECTORS, companies=COMPANIES
    )
    assert gate_d["status"] == "evaluated"
    assert gate_d["d1"] == gate_b["b1"]
    assert gate_d["strongest_layer"] == gate_b["strongest_layer"] == 12
    assert gate_d["d3"] == gate_b["b3"]
    assert gate_d["pass"] == gate_b["pass"] is True


def test_gate_d_smoke_grid_not_evaluated():
    records = _b_records(lambda d, layer, component: 0.5)
    smoke = [r for r in records if r["direction"] in ("NSC->IT", "IT->NSC")]
    assert evaluate_gate_d(
        smoke, layers=B_LAYERS, sector_of=SECTORS, companies=COMPANIES
    )["status"] == "not_evaluated"


def _gate_c_values(n: int = 16) -> tuple[list[float], list[float]]:
    gaps = [1.0 + 0.01 * (i % 3) for i in range(n)]
    deltas = [0.4 + 0.01 * (i % 3) for i in range(n)]
    return deltas, gaps


def test_gate_c_pass():
    deltas, gaps = _gate_c_values()
    gate = evaluate_gate_c(deltas, gaps)
    assert gate["status"] == "evaluated"
    assert gate["c1"]["pass"] is True
    assert gate["c2"]["pass"] is True
    assert gate["c2"]["ratio"] == pytest.approx(0.4, abs=0.01)
    assert gate["pass"] is True


def test_gate_c2_ratio_boundary_inclusive():
    n = 8
    gate = evaluate_gate_c([0.25] * n, [1.0] * n)
    assert gate["c2"]["pass"] is True and gate["c2"]["ratio"] == 0.25
    gate = evaluate_gate_c([0.249] * n, [1.0] * n)
    assert gate["c2"]["pass"] is False


def test_gate_c1_degenerate_gap_fails_closed():
    n = 8
    gate = evaluate_gate_c([0.4] * n, [0.5] * (n // 2) + [-0.5] * (n // 2))
    assert gate["c1"]["degenerate_gap"] is True
    assert gate["c1"]["pass"] is False
    assert gate["pass"] is False


def test_gate_c1_sign_mismatch_fails():
    n = 8
    gate = evaluate_gate_c([-0.4] * n, [1.0] * n)
    assert gate["c1"]["pass"] is False
    assert gate["pass"] is False
    # Opposite-side gaps with matching negative dials pass C1.
    gate = evaluate_gate_c([-0.4] * n, [-1.0] * n)
    assert gate["c1"]["pass"] is True


def test_gate_c_smoke_and_shape_failures():
    assert evaluate_gate_c([0.4] * 3, [1.0] * 3)["status"] == "not_evaluated"
    with pytest.raises(ValueError, match="paired"):
        evaluate_gate_c([0.4] * 4, [1.0] * 3)
    with pytest.raises(ValueError, match="paired"):
        evaluate_gate_c([], [])


def test_c1_descriptive_rho_and_minimum_companies():
    entity = {t: float(i) for i, t in enumerate(sorted(COMPANIES))}
    pure = {t: float(i) for i, t in enumerate(sorted(COMPANIES))}
    out = c1_descriptive(entity, {t: 0.1 for t in entity}, pure)
    assert out["rho_entity_position"] == pytest.approx(1.0)
    assert out["rho_final_position"] is None  # constant series rejected
    assert out["descriptive_only"] is True
    small = {t: 1.0 for t in list(COMPANIES)[:2]}
    out = c1_descriptive(small, small, {t: 1.0 for t in small})
    assert out["rho_entity_position"] is None


def test_toward_source_delta_and_normalized_transfer_semantics():
    assert toward_source_delta(0.5, 2.0, -1.0) > 0
    assert toward_source_delta(-1.5, 2.0, -1.0) < 0
    assert normalized_transfer(-0.5, 2.0, -1.0) == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="identical"):
        toward_source_delta(0.0, 1.0, 1.0)


def test_bootstrap_ci_and_spearman_conventions():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    lower, upper = bootstrap_ci(values, n=500, seed=42)
    assert 2.5 < lower < upper < 6.5
    # Deterministic under the frozen seed.
    assert bootstrap_ci(values, n=500, seed=42) == (lower, upper)
    with pytest.raises(ValueError, match="at least four"):
        bootstrap_ci([1.0, 2.0, 3.0])
    assert spearman([1, 2, 3, 4], [4, 9, 16, 25]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [25, 16, 9, 4]) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="degenerate"):
        spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
