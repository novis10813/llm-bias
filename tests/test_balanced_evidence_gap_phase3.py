"""Regression tests for the Phase 3 (neuron causal validation) package.

Pure-function and tmp_path artifact tests only; no checkpoint is loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_bias.balanced_evidence_gap import neuron_causal as nc
from llm_bias.balanced_evidence_gap.neuron_causal import (
    CANDIDATES,
    activation_scales,
    analyze_3a_records,
    check_c2c_reproduction,
    coord_name,
    derive_control_neurons,
    evaluate_gate_3a,
    one_sign_flip_p,
    prepare_stage,
)
from llm_bias.balanced_evidence_gap.template import ALL_TICKERS
from llm_bias.core.artifacts.lifecycle import ArtifactRun


# ── frozen constants ─────────────────────────────────────────────────────────


def test_derive_control_neurons_matches_frozen_sets():
    # proposal §4.2 — deterministic re-derivation of the 2C control sets.
    expected = {
        19: [454, 2977, 3551, 4805, 5252, 5258, 5856, 7965, 8101, 9120],
        20: [1067, 2646, 2803, 2834, 3065, 3901, 5077, 5828, 7614, 8323],
        26: [1818, 3987, 4349, 5396, 7075, 7178, 7640, 7687, 8204, 9064],
    }
    tops = {c["layer"]: c["neuron"] for c in CANDIDATES}
    for layer, want in expected.items():
        got = derive_control_neurons(layer)
        assert sorted(got) == sorted(want)
        assert tops[layer] not in got


def test_candidates_match_2c_reanalysis_coordinates():
    assert [(c["layer"], c["neuron"]) for c in CANDIDATES] == [
        (19, 6334), (20, 6520), (26, 2394),
    ]
    for c in CANDIDATES:
        # predicted direction is the sign of the 2C spearman
        assert c["predicted_sign"] == (1 if c["rho_2c"] > 0 else -1)


# ── gate helpers ─────────────────────────────────────────────────────────────


def test_one_sign_flip_p_boundaries():
    assert one_sign_flip_p(16, 0) == 1.0
    assert one_sign_flip_p(16, 16) == pytest.approx(2 ** -16)
    # 13/16 same direction (n=16): the Holm-surviving threshold
    assert one_sign_flip_p(16, 13) == pytest.approx((560 + 120 + 16 + 1) / 2 ** 16)
    # 12/16 is not significant after Holm x3
    assert one_sign_flip_p(16, 12) * 3 > 0.05
    assert one_sign_flip_p(16, 13) * 3 < 0.05


def _spec(name, sign, mean, n_same, n, ctl):
    return {
        "name": name, "predicted_sign": sign, "mean_delta_at_gate": mean,
        "n_same_direction": n_same, "n_tickers": n,
        "control_layer_max_abs_mean_delta": ctl,
    }


def test_evaluate_gate_3a_pass_with_two_of_three():
    # C1: 16/16 same dir, correct sign, beats controls -> pass
    # C2: 16/16 same dir, correct sign, beats controls -> pass
    # C3: wrong sign -> fail
    gate = evaluate_gate_3a(candidates=[
        _spec("A", -1, -0.5, 16, 16, 0.01),
        _spec("B", +1, +0.4, 16, 16, 0.01),
        _spec("C", -1, +0.3, 0, 16, 0.01),
    ])
    assert [c["pass"] for c in gate["candidates"]] == [True, True, False]
    assert gate["confirmed"] == 2
    assert gate["pass"] is True


def test_evaluate_gate_3a_fails_with_one_confirmed():
    gate = evaluate_gate_3a(candidates=[
        _spec("A", -1, -0.5, 16, 16, 0.01),
        _spec("B", +1, +0.4, 12, 16, 0.01),   # 12/16 -> Holm-adjusted p > 0.05
        _spec("C", -1, +0.3, 0, 16, 0.01),
    ])
    assert gate["candidates"][0]["pass"] is True
    assert gate["candidates"][1]["pass"] is False
    assert gate["confirmed"] == 1
    assert gate["pass"] is False


def test_evaluate_gate_3a_control_superiority_is_necessary():
    # correct sign and 16/16 consistency, but |mean| <= control max -> fail
    gate = evaluate_gate_3a(candidates=[
        _spec("A", -1, -0.005, 16, 16, 0.01),
        _spec("B", +1, +0.005, 16, 16, 0.01),
        _spec("C", -1, -0.5, 16, 16, 0.01),
    ])
    assert gate["candidates"][0]["criteria"]["control_superiority"] is False
    assert gate["candidates"][0]["pass"] is False
    assert gate["pass"] is False


def test_evaluate_gate_3a_zero_mean_fails_direction():
    gate = evaluate_gate_3a(candidates=[
        _spec("A", -1, 0.0, 16, 16, 0.01),
        _spec("B", +1, +0.5, 16, 16, 0.01),
        _spec("C", -1, -0.5, 16, 16, 0.01),
    ])
    assert gate["candidates"][0]["criteria"]["direction"] is False
    assert gate["candidates"][0]["pass"] is False


def test_evaluate_gate_3a_requires_candidates():
    with pytest.raises(ValueError):
        evaluate_gate_3a(candidates=[])


# ── prepare stage (tmp_path, no model) ───────────────────────────────────────


def _fake_source_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    a2a = tmp_path / "phase2a"
    (a2a / "forward").mkdir(parents=True)
    (a2a / "analyze").mkdir(parents=True)
    rows = []
    for i, ticker in enumerate(ALL_TICKERS):
        for order in (0, 1):
            for reverse in (False, True):
                rows.append({
                    "ticker": ticker, "order": order, "reverse": reverse,
                    "margin": -(1.0 + 0.1 * i) + 0.05 * order + 0.02 * int(reverse),
                    "formatted": f"prompt {ticker}",
                    "entity_position": 10, "final_position": 100,
                })
    with open(a2a / "forward" / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (a2a / "analyze" / "summary.json").write_text(json.dumps({"x": 1}), encoding="utf-8")

    a2c = tmp_path / "phase2c"
    (a2c / "mlp").mkdir(parents=True)
    (a2c / "mlp" / "layer_summaries.json").write_text(
        json.dumps({"layers": []}), encoding="utf-8"
    )

    rean = tmp_path / "phase2c-reanalysis"
    (rean / "analyze").mkdir(parents=True)
    rean_summary = {
        "gate_2c": {"mlp_arm": {
            "passing_layers": [19, 20, 26],
            "per_layer": {
                str(c["layer"]): {
                    "top_neuron": c["neuron"], "top_spearman": c["rho_2c"],
                }
                for c in CANDIDATES
            },
        }}
    }
    (rean / "analyze" / "summary.json").write_text(
        json.dumps(rean_summary), encoding="utf-8"
    )
    return a2a, a2c, rean


def test_prepare_stage_provenance(tmp_path):
    a2a, a2c, rean = _fake_source_runs(tmp_path)
    run = ArtifactRun.create("fake-model", nc.DATASET, "prep-test",
                             artifact_root=tmp_path / "artifacts")
    try:
        prepare_stage(run, phase2a_run=a2a, phase2c_run=a2c,
                      phase2c_reanalysis_run=rean)
    finally:
        run.finalize(required_stages={"prepare"})
    prov = json.loads((run.run_directory / "prepare" / "provenance.json").read_text())
    assert set(prov["controls"]) == {"19", "20", "26"}
    assert len(prov["controls"]["19"]) == 10
    assert set(prov["baseline_margins"]) == set(ALL_TICKERS)
    # baseline = canonical variant margin (order 0, reverse False)
    assert prov["baseline_margins"]["AMAT"] == pytest.approx(-1.0)
    assert len(prov["source_runs"]["phase2a"]["analyze_summary_sha256"]) == 64


def test_prepare_stage_fail_closed_on_drifted_candidates(tmp_path):
    a2a, a2c, rean = _fake_source_runs(tmp_path)
    summary = json.loads((rean / "analyze" / "summary.json").read_text())
    summary["gate_2c"]["mlp_arm"]["per_layer"]["19"]["top_neuron"] = 9999
    (rean / "analyze" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    run = ArtifactRun.create("fake-model", nc.DATASET, "prep-drift",
                             artifact_root=tmp_path / "artifacts")
    with pytest.raises(ValueError, match="top neuron"):
        prepare_stage(run, phase2a_run=a2a, phase2c_run=a2c,
                      phase2c_reanalysis_run=rean)


# ── analyze (pure function over synthetic records) ───────────────────────────


def _synthetic_3a_inputs(n: int = 16) -> tuple[list[dict], dict, dict]:
    tickers = [f"T{i:02d}" for i in range(n)]
    m0 = {t: -(1.0 + 0.1 * i) for i, t in enumerate(tickers)}
    controls = {str(c["layer"]): derive_control_neurons(c["layer"]) for c in CANDIDATES}

    records = []
    for t in tickers:
        records.append({
            "kind": "baseline", "layer": None, "neuron": None, "delta": 0.0,
            "ticker": t, "sector": "S", "margin": m0[t], "delta_m": 0.0,
            "decision": "sell",
        })

    grid = {}
    for c in CANDIDATES:
        name = coord_name(c["layer"], c["neuron"])
        s = 1.0
        deltas = [m * s * sign for m in (1, 2, 4) for sign in (1, -1)]
        grid[name] = {"s": s, "deltas": deltas,
                      "gate_delta": c["predicted_sign"] * 4.0 * s}

    for c in CANDIDATES:
        name = coord_name(c["layer"], c["neuron"])
        for d in grid[name]["deltas"]:
            mult = abs(d)
            for i, t in enumerate(tickers):
                if name == coord_name(19, 6334):
                    dm = -(0.5 + 0.01 * i) * mult            # always sell-shift
                elif name == coord_name(20, 6520):
                    dm = (0.6 - 0.02 * i) * mult              # always buy-shift
                else:
                    dm = (0.2 + 0.05 * i) * mult              # opposite of predicted
                records.append({
                    "kind": "candidate", "layer": c["layer"], "neuron": c["neuron"],
                    "delta": d, "ticker": t, "sector": "S",
                    "margin": m0[t] + dm, "delta_m": dm,
                    "decision": "buy" if m0[t] + dm > 0 else "sell",
                })

    for layer in sorted({c["layer"] for c in CANDIDATES}):
        for neuron in controls[str(layer)]:
            for d in (4.0, -4.0):
                for t in tickers:
                    records.append({
                        "kind": "control", "layer": layer, "neuron": neuron,
                        "delta": d, "ticker": t, "sector": "S",
                        "margin": m0[t] + 0.01, "delta_m": 0.01,
                        "decision": "sell",
                    })

    for d in (4.0, -4.0):
        for t in tickers:
            records.append({
                "kind": "dial", "layer": 15, "neuron": 8490, "delta": d,
                "ticker": t, "sector": "S", "margin": m0[t] + 0.9,
                "delta_m": 0.9, "decision": "sell",
            })

    pilot = {
        "grid": grid,
        "dial_scale_s": 1.0,
        "derivative_diagnostic": {
            coord_name(c["layer"], c["neuron"]): {
                "mean_summed_dM_da": 0.1 * c["predicted_sign"],
                "predicted_sign": c["predicted_sign"], "sign_match": True,
            } for c in CANDIDATES
        },
        "c2c_reproduction": {"status": "run"},
        "margin_baseline_max_abs_diff": 0.0,
    }
    provenance = {"controls": controls, "baseline_margins": m0}
    return records, pilot, provenance


def test_analyze_3a_records_gate_pass_two_of_three():
    records, pilot, provenance = _synthetic_3a_inputs()
    summary = analyze_3a_records(records, pilot, provenance)
    gate = summary["gate_3a"]
    assert gate["pass"] is True
    assert gate["confirmed"] == 2
    by_name = {c["name"]: c for c in gate["candidates"]}
    assert by_name["L19_n6334"]["pass"] is True
    assert by_name["L20_n6520"]["pass"] is True
    c26 = by_name["L26_n2394"]
    assert c26["pass"] is False
    assert c26["criteria"]["direction"] is False
    assert c26["n_same_direction"] == 0
    # descriptive blocks populated
    assert summary["descriptive"]["decision_flips"]["L20_n6520"]
    assert summary["n_tickers"] == 16


def test_analyze_3a_records_requires_all_gate_records():
    records, pilot, provenance = _synthetic_3a_inputs()
    records = [
        r for r in records
        if not (r["kind"] == "candidate" and r["layer"] == 19
                and abs(r["delta"] - (-4.0)) < 1e-9)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        analyze_3a_records(records, pilot, provenance)


def test_coord_name():
    assert coord_name(19, 6334) == "L19_n6334"


def test_check_c2c_reproduction_rev2_semantics():
    import numpy as np

    layer = 19
    candidate = 6334
    # synthetic 2C stored summary
    ref = {
        "layer": layer,
        "top_neuron": candidate,
        "top_spearman": -0.8971,
        "control_rhos": [0.1, -0.2, 0.3, 0.05, -0.1, 0.15, -0.05, 0.2, 0.0, -0.15],
    }
    rng_ctrls = derive_control_neurons(layer)
    assert len(rng_ctrls) == len(ref["control_rhos"])

    # exact reproduction -> pass
    rho = np.zeros(9216)
    rho[candidate] = -0.8971
    for n, c in zip(rng_ctrls, ref["control_rhos"]):
        rho[n] = c
    res = check_c2c_reproduction(rho, ref, candidate)
    assert res["pass"] is True
    assert res["top_neuron"] == candidate
    assert res["control_rho_max_abs_diff"] == 0.0

    # knife-edge mirror: argmax flips to an equal-|rho| neuron, candidate
    # within 0.05 -> still pass (the L19 n6334/n5233 situation)
    rho2 = np.zeros(9216)
    rho2[5233] = 0.8794
    rho2[candidate] = -0.8794
    for n, c in zip(rng_ctrls, ref["control_rhos"]):
        rho2[n] = c + 0.02  # within tolerance
    res2 = check_c2c_reproduction(rho2, ref, candidate)
    assert res2["pass"] is True
    assert res2["top_neuron"] == 5233
    assert res2["checks"]["top_strength"] is True

    # candidate drift beyond tolerance -> fail
    rho3 = np.zeros(9216)
    rho3[candidate] = -0.70
    rho3[5233] = 0.8794
    res3 = check_c2c_reproduction(rho3, ref, candidate)
    assert res3["pass"] is False
    assert res3["checks"]["candidate"] is False

    # layer-wide strength collapse -> fail top_strength
    rho4 = np.zeros(9216)
    rho4[candidate] = -0.80
    rho4[9000] = 0.50
    res4 = check_c2c_reproduction(rho4, ref, candidate)
    assert res4["checks"]["top_strength"] is False
    assert res4["checks"]["candidate"] is False
    assert res4["pass"] is False

    # one control beyond tolerance -> fail controls
    rho5 = np.zeros(9216)
    rho5[candidate] = -0.89
    for i, (n, c) in enumerate(zip(rng_ctrls, ref["control_rhos"])):
        rho5[n] = c + (0.06 if i == 0 else 0.0)
    res5 = check_c2c_reproduction(rho5, ref, candidate)
    assert res5["checks"]["controls"] is False
    assert res5["pass"] is False


def test_activation_scales_stacks_2d_chunks():
    import torch

    # regression: capture chunks are [seq, n_neurons] (2-D); stacking over
    # prompts must yield [n_prompts, seq, n_neurons], not a 2-D cat.
    chunks_19 = [torch.ones(5, 2) * (1.0 + 0.1 * i) for i in range(3)]
    chunks_15 = [torch.ones(5, 1) * 2.0 for _ in range(3)]
    coord_keys = [(15, 8490), (19, 6334), (19, 454)]
    s = activation_scales({19: chunks_19, 15: chunks_15}, coord_keys)
    # neuron 6334 is column 0: 15 sorted values = 5x1.0 + 5x1.1 + 5x1.2;
    # linear q90: index 0.9*(15-1)=12.6 -> 1.2
    assert s["L19_n6334"] == pytest.approx(1.2)
    assert s["L19_n454"] == pytest.approx(1.2)  # column 1, same values
    assert s["L15_n8490"] == pytest.approx(2.0)
