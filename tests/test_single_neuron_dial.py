import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("single_neuron_dial", ROOT / "scripts/single_neuron_dial.py")
dial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dial)


def company(ticker, split):
    return {"ticker": ticker, "name": f"{ticker} Inc.", "split": split,
            "evidence_pairs": [{"positive": f"{ticker} P{i}", "negative": f"{ticker} N{i}"} for i in range(4)]}


def test_trials_are_balanced_counterbalanced_and_deterministic():
    data = {"companies": [company("AAA", "A"), company("BBB", "B"), company("CCC", "A")]}
    rows = dial.build_trials(data, ["A"])
    assert rows == dial.build_trials(data, ["A"])
    assert {r["ticker"] for r in rows} == {"AAA", "CCC"}
    assert len(rows) == 2 * dial.DRAWS * 2
    for r in rows:
        assert r["prompt"].count("P") - r["prompt"].count("Pick") >= 0
        body = r["prompt"].split("— Evidence —")[1].split("\n\n—")[0]
        items = [line[2:] for line in body.strip().splitlines()]
        assert sum("P" in i for i in items) == 2 and sum("N" in i for i in items) == 2
        # bullish and bearish items come from different pairs
        assert not {i[-1] for i in items if " P" in i} & {i[-1] for i in items if " N" in i}
    for ticker in ("AAA", "CCC"):
        mine = [r for r in rows if r["ticker"] == ticker]
        evidence = [r["prompt"].split("— Evidence —")[1].split("\n\n—")[0] for r in mine if not r["reverse_options"]]
        assert len(set(evidence)) == dial.DRAWS
        for draw in range(dial.DRAWS):
            pair = [r for r in mine if r["draw"] == draw]
            assert sorted(r["reverse_options"] for r in pair) == [False, True]
            ev = {r["prompt"].split("— Evidence —")[1].split("\n\n—")[0] for r in pair}
            assert len(ev) == 1
    assert '"decision": "sell" or "buy"' in next(r for r in rows if r["reverse_options"])["prompt"]


def test_parse_output():
    assert dial.parse_output('{"decision": "buy", "reason": "x"}') == {"parsed": True, "decision": "buy"}
    assert dial.parse_output('```json\n{"decision": "sell", "reason": "x"}\n```') == {"parsed": True, "decision": "sell"}
    assert dial.parse_output('{"decision": "hold"}') == {"parsed": True, "decision": None}
    assert dial.parse_output('Sure! {"decision": "buy"}') == {"parsed": False, "decision": None}
    assert dial.parse_output("[1]") == {"parsed": False, "decision": None}


def rec(decision, parsed=True, i=0, ticker="T"):
    return {"id": f"{ticker}:{i}", "ticker": ticker, "decision": decision, "parsed": parsed}


def test_summarize_counts_only_parsed_buy_sell_in_pi():
    rows = [rec("buy"), rec("buy"), rec("sell"), rec(None), rec(None, parsed=False)]
    s = dial.summarize(rows)
    assert s["pi"] == pytest.approx(1 / 3)
    assert s["valid_decision_rate"] == pytest.approx(0.6)
    assert s["parse_rate"] == pytest.approx(0.8)


def pt(delta, pi, parse=1.0, valid=1.0):
    return {"delta": delta, "pi": pi, "parse_rate": parse, "valid_decision_rate": valid}


def test_isotonic_pools_violators():
    assert dial.isotonic([0, 2, 1, 3], True) == [0, 1.5, 1.5, 3]
    assert dial.isotonic([3, 1, 2, 0], False) == [3, 1.5, 1.5, 0]


def test_invert_interpolates_on_monotone_fit_without_extrapolation():
    curve = [pt(-2, -0.8), pt(-1, -0.4), pt(0, 0.2), pt(1, 0.6), pt(2, 0.9)]
    assert dial.invert(curve, 0.0) == pytest.approx(-1 + 0.4 / 0.6)
    assert dial.invert(curve, 0.6) == pytest.approx(1.0)
    assert dial.invert(curve, 0.95) is None
    # decreasing dial
    down = [pt(-1, 0.8), pt(0, 0.5), pt(1, -0.5)]
    assert dial.invert(down, 0.0) == pytest.approx(0.5)


def test_invert_uses_only_the_usable_run_around_zero():
    curve = [pt(-4, 0.9, parse=0.2), pt(-2, -0.6), pt(0, 0.2), pt(2, 0.7), pt(4, -1.0, valid=0.5)]
    assert dial.usable_curve(curve) == [(-2, -0.6), (0, 0.2), (2, 0.7)]
    assert dial.invert(curve, -0.8) is None
    assert dial.safe_invert([pt(-1, 0.3), pt(0, 0.3), pt(1, 0.3)], 0.0) is None
    assert dial.safe_invert([pt(-1, 0.3), pt(0, None), pt(1, 0.3)], 0.0) is None


def test_flat_run_equal_to_target_returns_its_midpoint():
    curve = [pt(-2, -0.5), pt(-1, 0.0), pt(0, 0.0), pt(1, 0.5)]
    assert dial.invert(curve, 0.0) == pytest.approx(-0.5)


def test_refinement_points_bisect_bracketing_intervals():
    curve = [pt(-2, -0.8), pt(-1, -0.4), pt(0, 0.2), pt(1, 0.6), pt(2, 0.9)]
    assert dial.refinement_points(curve, (-0.3, 0.0, 0.3)) == [-0.5, 0.5]
    curve.append(pt(-0.5, -0.1))
    assert -0.5 not in dial.refinement_points(curve, (-0.3, 0.0, 0.3))


def test_curve_shape_and_rmse():
    curve = [pt(-2, -0.8), pt(-1, -0.4), pt(0, 0.2), pt(1, 0.1), pt(2, 0.9)]
    shape = dial.curve_shape(curve)
    assert shape["reversals"] == 1
    assert shape["range"] == pytest.approx(1.7)
    assert 0.8 < shape["spearman"] < 1
    assert dial.rmse({-0.3: -0.2, 0.0: 0.0, 0.3: 0.4}, (-0.3, 0.0, 0.3)) == pytest.approx((0.02 / 3) ** 0.5)
    assert dial.rmse({-0.3: None, 0.0: 0.0, 0.3: 0.4}, (-0.3, 0.0, 0.3)) is None


def test_flip_summary_is_paired_and_itt():
    base = [rec("buy", i=0), rec("buy", i=1), rec("sell", i=2)]
    steered = [rec("sell", i=0), rec(None, parsed=False, i=1), rec("sell", i=2)]
    f = dial.flip_summary(base, steered)
    assert f["buy_to_sell"] == {"n": 2, "flips": 1, "rate": 0.5}
    assert f["sell_to_buy"] == {"n": 1, "flips": 0, "rate": 0.0}
    assert f["changed"] == 2
    with pytest.raises(ValueError):
        dial.flip_summary(base, steered[:2])


def test_bootstrap_pi_brackets_point_estimate():
    rows = [rec("buy" if i % 4 else "sell", i=i, ticker=f"T{i % 10}") for i in range(200)]
    lo, hi = dial.bootstrap_pi(rows, resamples=500)
    assert lo <= dial.summarize(rows)["pi"] <= hi


def test_select_picks_lowest_rmse_feasible(tmp_path):
    def cand(rank, neuron, r, ok):
        return {"rank": rank, "layer": 1, "neuron": neuron, "G": 1.0, "signed": 1.0, "rmse": r, "feasible": ok}
    (tmp_path / "a.json").write_text(json.dumps({"candidates": [cand(1, 10, 0.01, False), cand(2, 11, 0.05, True)]}))
    (tmp_path / "b.json").write_text(json.dumps({"candidates": [cand(3, 12, 0.04, True), cand(4, 13, None, False)]}))
    out = tmp_path / "sel.json"
    dial.main(["select", "--results", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--output", str(out)])
    assert json.loads(out.read_text())["selected"]["neuron"] == 12
    (tmp_path / "c.json").write_text(json.dumps({"candidates": [cand(3, 12, 0.04, True)]}))
    with pytest.raises(ValueError):
        dial.main(["select", "--results", str(tmp_path / "b.json"), str(tmp_path / "c.json"), "--output", str(out)])


def test_random_controls_exclude_candidates_and_are_deterministic():
    picks = dial.random_controls(50, {1, 2, 3}, 5)
    assert picks == dial.random_controls(50, {1, 2, 3}, 5)
    assert len(set(picks)) == 5 and not set(picks) & {1, 2, 3}


def test_real_data_gives_twenty_prompts_per_company():
    data = json.loads((ROOT / dial.DATA).read_text())
    rows = dial.build_trials(data, ["screen", "A", "B", "test"])
    assert len(rows) == 427 * 20
    splits = {}
    for c in data["companies"]:
        splits[c["split"]] = splits.get(c["split"], 0) + 1
    assert splits == {"screen": 171, "A": 85, "B": 85, "test": 86}
