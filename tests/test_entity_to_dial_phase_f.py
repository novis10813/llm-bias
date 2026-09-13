"""Phase F tests (fake models, no checkpoint).

Covers the directional push transform (constant across positions, bit-exact
zero no-op), the low-level projected transplant (bit-exact parity with the
Phase E ``project_delta`` path, idempotence), the dual-hook composite
(all-no-op bit-exact clean, strict single fire, exception-safe hook
cleanup), ``load_pca_basis`` fail-closed paths, the F2 four-way
classification (including the jitter-band boundary), the Gate F1 logic
(R1 denominator rule, 0.85 boundary, secondary group medians, consistency),
and the monkeypatched ``run_phase_f`` smoke pipeline (F1 + F2 serial,
manifest complete, no-op discipline, dual-hook composition property,
anonymous identity, schema).
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.entity_to_dial import pipeline
from llm_bias.entity_to_dial.analysis import classify_f2_push, evaluate_gate_f
from llm_bias.entity_to_dial.dial_probe import (
    answer_token_ids,
    margin_from_log_probs,
    scoring_ids,
)
from llm_bias.entity_to_dial.joint_patch import (
    directional_push_transform,
    dual_hook_interventions,
    load_pca_basis,
    make_delta_transform,
    make_projected_transplant,
    project_delta,
)
from llm_bias.entity_to_dial.template import (
    DECISION_PREFIX,
    F2_JITTER_BAND,
    RATIO_MIN_FULL_DM,
)

from test_entity_to_dial_pipeline import (
    _CharTokenizer,
    _fake_upstream,
    _manifest,
    _patch_loaders,
    _qwen_fake,
    _read_jsonl,
    _write_jsonl,
)

FAKE_HIDDEN = 32  # _qwen_fake hidden_size
FAKE_DIAL = 5  # in-range fake dial channel
FROZEN_E1_RECORDS = 224  # 8 directions x 7 layers x 4 arms (e-01)
FROZEN_E2_RECORDS = 72  # 8 directions x 9 arms (e-01)


def _fake_basis() -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal 16 x 32 basis + strictly decreasing singular values."""
    basis = torch.eye(FAKE_HIDDEN)[:16]
    singular = torch.linspace(16.0, 1.0, 16)
    return basis, singular


def _live_toward(model, tokenizer, target_row, pure, source: str, target: str) -> float:
    """Toward-source delta of the clean target margin.

    On the fake 16-layer model L15 is the final layer, so every
    instruction-position patch at L15 is a structural zero and every F1
    arm reproduces this value bit-exactly.
    """
    tensor = torch.tensor([scoring_ids(tokenizer, target_row["formatted"])], dtype=torch.long)
    final_layer = int(model.n_layers) - 1
    residual = record_residuals(model, tensor, [final_layer])[final_layer]
    buy_id, sell_id = answer_token_ids(tokenizer, target_row["formatted"] + DECISION_PREFIX)
    patched = margin_from_log_probs(
        fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
    )
    return pipeline.toward_source_delta(patched, pure[source], pure[target])


def _fake_e01(tmp_path: Path, model, tokenizer, phase2a: Path, phase2b: Path) -> Path:
    """Fabricate an e-01 run with real fake-model L15 reference values
    (all arms structural-zero identical) + a valid 16 x 32 PCA basis."""
    e01 = tmp_path / "e01"
    (e01 / "forward_e1").mkdir(parents=True)
    (e01 / "forward_e2").mkdir(parents=True)
    (e01 / "analyze").mkdir(parents=True)
    directions = json.loads(
        (phase2b / "pairs" / "directions.json").read_text(encoding="utf-8")
    )["directions"]
    canon = pipeline._canonical_2a_rows(phase2a)
    pure = pipeline.pure_entity_margins(phase2a)
    basis, singular = _fake_basis()
    e1_recs, e2_recs = [], []
    for source, target in directions:
        value = _live_toward(model, tokenizer, canon[target], pure, source, target)
        direction = f"{source}->{target}"
        for layer in range(12, 19):
            for arm in ("joint", "full", "joint_noop", "full_noop"):
                e1_recs.append(
                    {
                        "phase": "e1",
                        "direction": direction,
                        "layer": layer,
                        "arm": arm,
                        "toward_source_delta_m": value if (layer == 15 and arm == "full") else 0.0,
                    }
                )
        for arm in (
            "pca_k1", "pca_k3", "pca_k8", "pca_k16", "pca_full",
            "pca_full_noop", "dial_transplant", "dial_noop", "footprint",
        ):
            e2_recs.append(
                {
                    "phase": "e2",
                    "direction": direction,
                    "layer": 15,
                    "arm": arm,
                    "toward_source_delta_m": value if arm in ("pca_k1", "pca_k8", "dial_transplant") else 0.0,
                }
            )
    assert len(e1_recs) == FROZEN_E1_RECORDS
    assert len(e2_recs) == FROZEN_E2_RECORDS
    _write_jsonl(e01 / "forward_e1" / "records.jsonl", e1_recs)
    _write_jsonl(e01 / "forward_e2" / "records.jsonl", e2_recs)
    (e01 / "analyze" / "summary.json").write_text(
        json.dumps(
            {
                "pca_basis_vectors": basis.tolist(),
                "pca_singular_values": [float(v) for v in singular.tolist()],
            }
        ),
        encoding="utf-8",
    )
    (e01 / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return e01


# ── directional push transform ───────────────────────────────────────────────


def test_directional_push_transform_constant_across_positions():
    torch.manual_seed(0)
    vector = torch.randn(FAKE_HIDDEN)
    tensor = torch.randn(1, 20, FAKE_HIDDEN, dtype=torch.bfloat16)
    out = directional_push_transform(vector, (3, 7, 15))(tensor)
    for pos in (3, 7, 15):
        expected = (tensor[0, pos, :].float() + vector).to(tensor.dtype)
        assert torch.equal(out[0, pos, :], expected)
    for pos in range(20):
        if pos not in (3, 7, 15):
            assert torch.equal(out[0, pos, :], tensor[0, pos, :])
    # Zero push is a bit-exact no-op.
    zero_out = directional_push_transform(torch.zeros(FAKE_HIDDEN), (3, 7))(tensor)
    assert torch.equal(zero_out, tensor)
    with pytest.raises(ValueError, match="non-finite"):
        directional_push_transform(torch.tensor([float("nan")] + [0.0] * (FAKE_HIDDEN - 1)), (1,))
    with pytest.raises(ValueError, match="nonempty"):
        directional_push_transform(vector, ())
    with pytest.raises(ValueError, match="1-D"):
        directional_push_transform(torch.randn(2, 4), (1,))


# ── projected transplant (low-level) ─────────────────────────────────────────


def test_make_projected_transplant_parity_with_project_delta():
    torch.manual_seed(1)
    p_dim, d, k = 6, FAKE_HIDDEN, 4
    delta = torch.randn(p_dim, d)
    q, _ = torch.linalg.qr(torch.randn(d, k))
    basis = q[:, :k]
    positions = list(range(10, 10 + p_dim))
    tensor = torch.randn(1, 20, d, dtype=torch.bfloat16)
    out = make_projected_transplant(delta, basis, positions)(tensor)
    projected = project_delta(delta, basis)
    for i, pos in enumerate(positions):
        expected = (tensor[0, pos, :].float() + projected[i]).to(tensor.dtype)
        assert torch.equal(out[0, pos, :], expected)
    for pos in range(20):
        if pos not in positions:
            assert torch.equal(out[0, pos, :], tensor[0, pos, :])
    # Zero delta is a bit-exact no-op.
    zero_out = make_projected_transplant(torch.zeros(p_dim, d), basis, positions)(tensor)
    assert torch.equal(zero_out, tensor)
    # Projection idempotence.
    assert torch.allclose(project_delta(projected, basis), projected, atol=1e-5)
    with pytest.raises(ValueError, match="matching the position count"):
        make_projected_transplant(delta, basis, (10, 11))
    with pytest.raises(ValueError, match="d matching"):
        make_projected_transplant(delta, torch.randn(d + 1, k), positions)


# ── dual-hook composite ──────────────────────────────────────────────────────


def test_dual_hook_interventions_noop_single_fire_cleanup():
    model = _qwen_fake()
    torch.manual_seed(2)
    tensor = torch.randint(2, 100, (1, 40))
    final_layer = int(model.n_layers) - 1
    clean = record_residuals(model, tensor, [final_layer])[final_layer]
    zero = make_delta_transform({pos: torch.zeros(FAKE_HIDDEN) for pos in (5, 6)})
    # All no-op: bit-exact clean margin input; fire counts reported.
    with dual_hook_interventions(model, final_layer, zero, FAKE_DIAL, {}) as fires:
        out = record_residuals(model, tensor, [final_layer])[final_layer]
    assert torch.equal(out, clean)
    assert fires == {"residual": 1, "dial": 0}
    # Strict single fire: a second forward inside the context raises.
    with pytest.raises(RuntimeError, match="fired twice"):
        with dual_hook_interventions(model, final_layer, zero, FAKE_DIAL, {8: 0.5}):
            model.forward(tensor)
            model.forward(tensor)
    # Exception-safe: no hooks leak after a mid-forward exception.
    layer_hooks_before = len(model.layers[final_layer]._forward_hooks)
    down_hooks_before = len(dense_down_projection(model.layers[final_layer])._forward_pre_hooks)
    with pytest.raises(RuntimeError, match="boom"):
        with dual_hook_interventions(model, final_layer, zero, FAKE_DIAL, {8: 0.5}):
            model.forward(tensor)
            raise RuntimeError("boom")
    assert len(model.layers[final_layer]._forward_hooks) == layer_hooks_before
    assert len(dense_down_projection(model.layers[final_layer])._forward_pre_hooks) == down_hooks_before


# ── load_pca_basis ───────────────────────────────────────────────────────────


def test_load_pca_basis_happy_and_fail_closed(tmp_path: Path):
    basis, singular = _fake_basis()
    path = tmp_path / "summary.json"

    def write(b, s):
        path.write_text(
            json.dumps(
                {
                    "pca_basis_vectors": b if isinstance(b, list) else b.tolist(),
                    "pca_singular_values": s if isinstance(s, list) else [float(v) for v in s.tolist()],
                }
            ),
            encoding="utf-8",
        )

    write(basis, singular)
    loaded_b, loaded_s = load_pca_basis(path, expected_k=16)
    # In-memory convention: [d, k] (right singular vectors as columns).
    assert loaded_b.shape == (FAKE_HIDDEN, 16)
    assert torch.allclose(loaded_b.T @ loaded_b, torch.eye(16), atol=1e-5)
    assert torch.equal(loaded_s, singular.float())

    bad_basis = [row[:] for row in basis.tolist()]
    bad_basis[0][0] += 0.5
    write(bad_basis, singular)
    with pytest.raises(ValueError, match="orthonormal"):
        load_pca_basis(path, expected_k=16)
    write(basis, [float(v) for v in singular.tolist()][::-1])
    with pytest.raises(ValueError, match="decreasing"):
        load_pca_basis(path, expected_k=16)
    write([[float("nan")] * FAKE_HIDDEN for _ in range(16)], singular)
    with pytest.raises(ValueError, match="non-finite"):
        load_pca_basis(path, expected_k=16)
    write(basis[:4], singular[:4])
    with pytest.raises(ValueError, match="must be"):
        load_pca_basis(path, expected_k=16)


# ── F2 four-way classification ───────────────────────────────────────────────


def test_classify_f2_push_classes():
    band = F2_JITTER_BAND
    assert (
        classify_f2_push({1.0: -0.2, 2.0: -0.4}, {1.0: 0.3, 2.0: 0.5}, jitter_band=band)
        == "signed_sell_axis"
    )
    assert (
        classify_f2_push({1.0: 0.2, 2.0: 0.4}, {1.0: -0.3, 2.0: -0.5}, jitter_band=band)
        == "signed_buy_axis"
    )
    assert (
        classify_f2_push({1.0: -0.2, 2.0: -0.4}, {1.0: -0.3, 2.0: -0.5}, jitter_band=band)
        == "attractor"
    )
    assert (
        classify_f2_push({1.0: 0.2, 2.0: 0.4}, {1.0: 0.3, 2.0: 0.5}, jitter_band=band)
        == "attractor"
    )
    assert (
        classify_f2_push({1.0: 0.0, 2.0: 0.0}, {1.0: 0.0, 2.0: 0.0}, jitter_band=band)
        == "context_dependent_or_null"
    )
    # Mixed signs → context-dependent.
    assert (
        classify_f2_push({1.0: 0.2, 2.0: -0.3}, {1.0: 0.3, 2.0: 0.5}, jitter_band=band)
        == "context_dependent_or_null"
    )
    # Boundary |ΔM| == band is jitter-band, not decisive.
    assert (
        classify_f2_push({1.0: -band, 2.0: -0.4}, {1.0: band, 2.0: 0.5}, jitter_band=band)
        == "context_dependent_or_null"
    )
    with pytest.raises(ValueError, match="verdict requires"):
        classify_f2_push({1.0: -0.2}, {1.0: 0.3, 2.0: 0.5}, jitter_band=band)


# ── Gate F1 logic ────────────────────────────────────────────────────────────


def _f1_records(dms: dict[str, tuple[float, ...]]) -> list[dict]:
    records = []
    for direction, (full, v1, k8, dial, combined) in dms.items():
        for arm, value in (
            ("full", full), ("v1", v1), ("k8", k8), ("dial", dial), ("combined", combined),
        ):
            records.append({"direction": direction, "arm": arm, "toward_source_delta_m": value})
    return records


def _f2_records_full() -> list[dict]:
    records = []
    for arm in ("v1", "dial_fp"):
        for sign in (1, -1):
            for alpha in (0.5, 1.0, 2.0):
                records.append(
                    {"arm": arm, "sign": sign, "alpha": alpha, "delta_m": 0.1 * sign * alpha}
                )
    return records


def _gate_kwargs(dms, top, bottom):
    return {
        "directions": list(dms),
        "top_group": top,
        "bottom_group": bottom,
        "e01_ref": {d: {"full": v[0], "v1": v[1], "k8": v[2], "dial": v[3]} for d, v in dms.items()},
        "min_full_dm": RATIO_MIN_FULL_DM,
        "f1_min": 0.85,
        "jitter_band": F2_JITTER_BAND,
        "consistency_tolerance": 0.01,
        "smoke": False,
    }


def test_evaluate_gate_f_gate_secondary_consistency():
    dms = {
        "A->B": (1.0, 0.7, 0.9, 0.4, 1.1),
        "C->D": (1.0, 0.75, 0.95, 0.45, 1.2),
        "E->F": (1.0, 0.8, 0.95, 0.5, 1.15),
        "G->H": (1.0, 0.6, 0.9, 0.4, 0.9),
        "I->J": (0.1, 0.0, 0.1, 0.0, 0.1),  # excluded (|full| < 0.2)
        "K->L": (0.8, 0.5, 0.8, 0.3, 0.85),
    }
    gate = evaluate_gate_f(_f1_records(dms), _f2_records_full(), **_gate_kwargs(dms, ["A", "E", "I"], ["C", "G", "K"]))
    f1 = gate["gate_f1"]
    assert f1["n_effective"] == 5
    assert f1["excluded_directions"] == ["I->J"]
    assert f1["median_ratio"] == pytest.approx(sorted([1.1, 1.2, 1.15, 0.9, 0.85 / 0.8])[2])
    assert f1["pass"] is True
    # Secondary group medians (excluded directions drop out).
    assert gate["f1_secondary"]["bottom_to_top"] == pytest.approx(statistics.median([1.2, 0.9, 0.85 / 0.8]))
    assert gate["f1_secondary"]["top_to_bottom"] == pytest.approx(statistics.median([1.1, 1.15]))
    # Consistency: in-run values equal the reference → max |Δ| = 0.
    assert all(
        v == 0.0
        for k, v in gate["f1_consistency"].items()
        if k != "tolerance" and v is not None
    )
    # F2 verdicts: plus > 0 and minus < 0 at both decisive doses → signed buy axis.
    assert gate["f2_verdict"] == {"v1": "signed_buy_axis", "dial_fp": "signed_buy_axis"}
    assert gate["gate"]["pass"] is True
    # Additive residual bookkeeping (summary key: interaction_delta_m; the
    # core serializer reserves the word "residual" in JSON keys).
    per = {p["direction"]: p for p in f1["per_direction"]}
    assert per["A->B"]["interaction_delta_m"] == pytest.approx(1.1 - 0.7 - 0.4)


def test_evaluate_gate_f_boundary_and_fail_closed():
    dms = {
        "A->B": (1.0, 0.6, 0.9, 0.3, 0.425),   # ratio 0.425
        "C->D": (1.0, 0.6, 0.9, 0.3, 0.425),   # ratio 0.425
        "E->F": (1.0, 0.6, 0.9, 0.3, 1.275),   # ratio 1.275
        "G->H": (1.0, 0.6, 0.9, 0.3, 1.275),   # ratio 1.275
        "I->J": (2.0, 1.0, 1.0, 0.5, 1.7),     # ratio 0.85 (boundary, included)
        "K->L": (2.0, 1.0, 1.0, 0.5, 1.7),     # ratio 0.85 (boundary, included)
    }
    gate = evaluate_gate_f(_f1_records(dms), _f2_records_full(), **_gate_kwargs(dms, ["A", "E", "I"], ["C", "G", "K"]))
    # Median exactly 0.85 → pass (≥).
    assert gate["gate_f1"]["pass"] is True
    assert gate["gate_f1"]["median_ratio"] == pytest.approx(0.85)
    # Missing arm record → fail closed.
    missing = [r for r in _f1_records(dms) if r["arm"] != "combined"]
    with pytest.raises(ValueError, match="expected exactly one combined"):
        evaluate_gate_f(missing, _f2_records_full(), **_gate_kwargs(dms, ["A", "E", "I"], ["C", "G", "K"]))
    # Incomplete F2 dose grid → fail closed.
    incomplete = [r for r in _f2_records_full() if not (r["arm"] == "v1" and r["alpha"] == 0.5)]
    with pytest.raises(ValueError, match="incomplete F2 dose grid"):
        evaluate_gate_f(_f1_records(dms), incomplete, **_gate_kwargs(dms, ["A", "E", "I"], ["C", "G", "K"]))
    # Smoke → not evaluated.
    smoke_gate = evaluate_gate_f(
        _f1_records(dms), _f2_records_full(), **{**_gate_kwargs(dms, ["A", "E", "I"], ["C", "G", "K"]), "smoke": True}
    )
    assert smoke_gate["status"] == "not_evaluated"


def test_evaluate_gate_f_formal_keys_pass_core_serializer_guard():
    # The formal summary keys must not trip the core raw-payload guard
    # ("residual" is a reserved key part; the additive-residual quantity is
    # persisted as interaction_delta_m).
    from llm_bias.core.artifacts.io import _RAW

    dms = {
        "A->B": (1.0, 0.7, 0.9, 0.4, 1.1),
        "C->D": (0.1, 0.0, 0.1, 0.0, 0.1),  # excluded direction included in per_direction
    }
    gate = evaluate_gate_f(
        _f1_records(dms), _f2_records_full(), **_gate_kwargs(dms, ["A"], ["C"])
    )

    def walk(obj, key=""):
        norm = str(key).lower().replace("-", "_")
        assert not any(part in norm.split("_") for part in _RAW), f"reserved key {key}"
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                walk(v, str(k))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)

    walk(gate)



# ── pipeline smoke (fake model) ──────────────────────────────────────────────


def _fake_m_anon(model, tokenizer, phase2a: Path) -> float:
    canon = pipeline._canonical_2a_rows(phase2a)
    anon = pipeline._anonymous_row(tokenizer, canon["NSC"])
    tensor = torch.tensor([scoring_ids(tokenizer, anon["formatted"])], dtype=torch.long)
    final_layer = int(model.n_layers) - 1
    residual = record_residuals(model, tensor, [final_layer])[final_layer]
    buy_id, sell_id = answer_token_ids(tokenizer, anon["formatted"] + DECISION_PREFIX)
    return margin_from_log_probs(
        fp32_next_token_log_probs(model, residual[:, -1, :]), buy_id, sell_id
    )


def _run_f_smoke(tmp_path: Path, monkeypatch) -> Path:
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    model = _qwen_fake()
    e01 = _fake_e01(tmp_path, model, tokenizer, phase2a, phase2b)
    _patch_loaders(monkeypatch, model, tokenizer)
    monkeypatch.setattr(pipeline, "DIAL_NEURON", FAKE_DIAL)
    # Fake-model anonymous margin anchor (the fake has no real-model scale).
    monkeypatch.setattr(pipeline, "F2_M_ANON_REF", _fake_m_anon(model, tokenizer, phase2a))
    return pipeline.run_phase_f(
        model_path="fake-model",
        run_id="smoke-fake-f",
        phase2a_run=phase2a,
        phase2b_run=phase2b,
        phasee_run=e01,
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )


def test_run_phase_f_smoke_with_fake_model(tmp_path, monkeypatch):
    run_root = _run_f_smoke(tmp_path, monkeypatch)
    manifest = _manifest(run_root)
    assert manifest["status"] == "complete"
    assert {stage for stage in manifest["stages"]} >= {"prepare", "forward_f1", "forward_f2", "analyze"}

    rows = _read_jsonl(run_root / "prepare" / "rows.jsonl")
    # 16 canonical rows + 1 anonymous row.
    assert len(rows) == 17
    anon = [r for r in rows if r.get("prompt_type") == "anon"]
    assert len(anon) == 1
    assert anon[0]["source_anchor"] == "NSC"
    assert anon[0]["instruction_span"][1] > anon[0]["instruction_span"][0]
    provenance = json.loads((run_root / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["protocol"] == pipeline.PROTOCOL_F
    assert provenance["protocol_rev"] == pipeline.PROTOCOL_F_REV
    assert provenance["anonymous_identity_check"] == "16/16 identical"
    assert len(provenance["anonymous_prompt_sha256"]) == 64
    assert provenance["pca_basis"]["d"] == FAKE_HIDDEN
    e01_ref = json.loads((run_root / "prepare" / "e01_ref.json").read_text(encoding="utf-8"))["reference"]
    assert len(e01_ref) == 8
    assert all(set(v) == {"full", "v1", "k8", "dial"} for v in e01_ref.values())

    f1 = _read_jsonl(run_root / "forward_f1" / "records.jsonl")
    # 1 direction x (5 arms + 5 no-op + 2 composition variants)
    assert len(f1) == 12
    assert {r["arm"] for r in f1} == {
        "full", "v1", "k8", "dial", "combined",
        "full_noop", "v1_noop", "k8_noop", "dial_noop", "combined_noop",
        "combined_dialzero", "combined_v1zero",
    }
    for r in f1:
        assert r["phase"] == "f1"
        if r["arm"].endswith("_noop"):
            assert r["noop_delta_m"] == 0.0  # bit-exact no-op discipline (incl. dual-hook no-op)
    by_arm = {}
    for r in f1:
        by_arm.setdefault(r["arm"], r)
    # Composition property: zero-side == single-arm patched margin, bit-exact.
    assert by_arm["combined_dialzero"]["patched_margin"] == by_arm["v1"]["patched_margin"]
    assert by_arm["combined_v1zero"]["patched_margin"] == by_arm["dial"]["patched_margin"]
    # Fake-model structural zero: every patched margin equals the live margin.
    assert all(r["patched_margin"] == r["live_target_margin"] for r in f1)
    # Consistency vs the fabricated e-01 reference is enforced inline in the
    # forward stage (fail-closed in smoke) and aggregated in the summary.

    f2 = _read_jsonl(run_root / "forward_f2" / "records.jsonl")
    # 2 arms x 2 signs x 1 alpha (smoke) + 1 zero-push no-op.
    assert len(f2) == 5
    assert {r["arm"] for r in f2} == {"v1", "dial_fp", "noop"}
    # Anonymous-prompt push records (layer/prompt provenance in metadata).
    f2_meta = json.loads((run_root / "forward_f2" / "metadata.json").read_text(encoding="utf-8"))
    assert f2_meta["layer"] == 15
    assert f2_meta["m_anon_band_ok"] is True
    assert all(r["jitter_band"] for r in f2 if r["arm"] != "noop")  # fake structural zeros
    assert [r for r in f2 if r["arm"] == "noop"][0]["noop_delta_m"] == 0.0
    assert all(abs(r["push_norm"]) >= 0.0 for r in f2)

    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    # smoke → gate not evaluated (no per-arm numbers, no verdicts).
    assert summary["gate_f"]["status"] == "not_evaluated"
    assert summary["gate_f1"] is None
    assert summary["f1_consistency"] is None
    assert summary["f2_verdict"] is None
    assert summary["m_anon_band_ok"] is True
    assert summary["push_base"] >= 0.0
    assert summary["smoke"] is True
    assert summary["raw_runtime_payloads"] is False
    assert "pca_basis_vectors" not in summary  # compact artifact


def test_run_phase_f_fails_closed_on_anonymous_identity(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    model = _qwen_fake()
    e01 = _fake_e01(tmp_path, model, tokenizer, phase2a, phase2b)
    _patch_loaders(monkeypatch, model, tokenizer)
    monkeypatch.setattr(pipeline, "DIAL_NEURON", FAKE_DIAL)
    # Corrupt one non-anchor 2A prompt so its anonymous header substitution
    # differs from the anchor's (fails the identity check fail-closed).
    prompts_path = phase2a / "prepare" / "prompts.jsonl"
    prompts = _read_jsonl(prompts_path)
    it = [r for r in prompts if r["ticker"] == "IT" and r.get("reverse") is False and r.get("order") == 0][0]
    it["prompt"] = it["prompt"].replace("Stock Name: [", "Stock Named: [", 1)
    _write_jsonl(prompts_path, prompts)
    with pytest.raises(ValueError, match="anonymous header"):
        pipeline.run_phase_f(
            model_path="fake-model",
            run_id="smoke-fake-f-bad",
            phase2a_run=phase2a,
            phase2b_run=phase2b,
            phasee_run=e01,
            artifact_root=tmp_path / "artifacts-bad",
            smoke=True,
        )
