"""Phase E tests (fake models, no checkpoint).

Covers the joint/full/projected transform arithmetic (fp32 → dtype cast,
bit-exact no-ops, block-delta identity), the PCA state-direction
machinery (right singular vectors, orthonormality, determinism,
projection completeness), the dial capture/transplant hooks
(position restriction, strict single fire, hook cleanup), the R1
denominator rule and gate E arithmetic, and the monkeypatched
run_phase_e smoke pipeline (E1 + E2 serial, manifest complete, 2B
full-reference acceptance, no-op discipline, schema).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import record_block_states, residual_interventions
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.entity_to_dial import pipeline
from llm_bias.entity_to_dial.analysis import evaluate_gate_e
from llm_bias.entity_to_dial.block_patch import make_position_transform, nearest_position_mapping
from llm_bias.entity_to_dial.dial_probe import (
    answer_token_ids,
    dial_value_capture,
    margin_from_log_probs,
    scoring_ids,
)
from llm_bias.entity_to_dial.joint_patch import (
    dial_channel_transplant,
    dial_footprint_direction,
    make_delta_transform,
    make_full_transform,
    make_joint_transform,
    make_projected_transform,
    pca_state_directions,
    project_delta,
    state_difference_rows,
    stack_state_difference,
)
from llm_bias.entity_to_dial.template import DECISION_PREFIX, RATIO_MIN_FULL_DM

from test_entity_to_dial_pipeline import (
    _CharTokenizer,
    _fake_upstream,
    _manifest,
    _patch_loaders,
    _qwen_fake,
    _read_jsonl,
)

FAKE_HIDDEN = 32  # _qwen_fake hidden_size
FAKE_INTERMEDIATE = 64  # _qwen_fake intermediate_size
FAKE_DIAL = 5  # in-range fake dial channel


# ── transform arithmetic ─────────────────────────────────────────────────────


def _fake_states(seed: int = 0, seq: int = 20, d: int = FAKE_HIDDEN, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    pre_s = torch.randn(1, seq, d, dtype=dtype)
    post_s = pre_s + 0.1 * torch.randn(1, seq, d, dtype=dtype)
    pre_t = torch.randn(1, seq, d, dtype=dtype)
    post_t = pre_t + 0.1 * torch.randn(1, seq, d, dtype=dtype)
    return pre_s, post_s, pre_t, post_t


def _mapping(seq: int, start: int = 10, length: int = 5) -> dict[int, int]:
    return {start + i: (start + i) % seq for i in range(length)}


def test_make_joint_transform_arithmetic_exact():
    pre_s, post_s, pre_t, post_t = _fake_states()
    mapping = _mapping(20)
    transform = make_joint_transform(pre_s, post_s, pre_t, post_t, mapping=mapping)
    out = transform(post_t)
    # Protocol formula, fp32 arithmetic, single cast back:
    # out[p] = bf16(fp32(post_t[p]) + (post_s[q] − pre_s[q]) − (post_t[p] − pre_t[p]))
    for p, q in mapping.items():
        row = (post_s[0, q].float() - pre_s[0, q].float()) - (post_t[0, p].float() - pre_t[0, p].float())
        expected = (post_t[0, p].float() + row).to(post_t.dtype)
        assert torch.equal(out[0, p, :], expected)
    # Unmapped positions untouched.
    touched = set(mapping)
    for pos in range(post_t.shape[1]):
        if pos not in touched:
            assert torch.equal(out[0, pos, :], post_t[0, pos, :])


def test_make_joint_transform_block_delta_identity():
    """joint(post_t) == pre_t + (post_s − pre_s) up to fp32 rounding."""
    pre_s, post_s, pre_t, post_t = _fake_states(dtype=torch.float32)
    mapping = _mapping(20)
    out = make_joint_transform(pre_s, post_s, pre_t, post_t, mapping=mapping)(post_t)
    for p, q in mapping.items():
        expected = pre_t[0, p] + (post_s[0, q] - pre_s[0, q])
        assert torch.allclose(out[0, p, :], expected, atol=1e-6)


def test_make_joint_transform_self_source_bit_exact():
    pre_s, post_s, pre_t, post_t = _fake_states()
    identity = {p: p for p in range(10, 15)}
    out = make_joint_transform(pre_t, post_t, pre_t, post_t, mapping=identity)(post_t)
    assert torch.equal(out, post_t)


def test_make_full_transform_bit_exact_and_noop():
    pre_s, post_s, pre_t, post_t = _fake_states()
    mapping = _mapping(20)
    out = make_full_transform(post_s, mapping=mapping)(post_t)
    reference = make_position_transform(post_s, mapping)(post_t)
    assert torch.equal(out, reference)  # 2B semantics, bit-exact
    identity = {p: p for p in mapping}
    assert torch.equal(make_full_transform(post_t, mapping=identity)(post_t), post_t)
    for pos in range(post_t.shape[1]):
        if pos not in mapping:
            assert torch.equal(out[0, pos, :], post_t[0, pos, :])


def test_make_delta_transform_validation():
    rows = {10: torch.randn(FAKE_HIDDEN)}
    with pytest.raises(ValueError, match="empty delta mapping"):
        make_delta_transform({})
    with pytest.raises(ValueError, match="non-finite"):
        make_delta_transform({10: torch.tensor([float("nan")])})
    transform = make_delta_transform(rows)
    tensor = torch.randn(1, 20, FAKE_HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="outside the tensor"):
        make_delta_transform({50: torch.randn(FAKE_HIDDEN)})(tensor)
    with pytest.raises(ValueError, match="width mismatch"):
        make_delta_transform({10: torch.randn(FAKE_HIDDEN + 1)})(tensor)


def test_state_difference_and_joint_relationship():
    """full − joint == pre_s − pre_t (fp32 chain, within one bf16 ulp)."""
    pre_s, post_s, pre_t, post_t = _fake_states()
    mapping = _mapping(20)
    joint = make_joint_transform(pre_s, post_s, pre_t, post_t, mapping=mapping)(post_t)
    full = make_full_transform(post_s, mapping=mapping)(post_t)
    for p, q in mapping.items():
        diff = full[0, p, :].float() - joint[0, p, :].float()
        expected = pre_s[0, q].float() - pre_t[0, p].float()
        assert torch.allclose(diff, expected, atol=0.02)  # bf16 ulp scale


def test_projected_transform_applies_projection():
    pre_s, post_s, pre_t, post_t = _fake_states()
    mapping = _mapping(20)
    rows = state_difference_rows(post_s, post_t, mapping=mapping)
    delta = stack_state_difference(rows)
    basis, _ = pca_state_directions([delta, -delta], 2)
    out = make_projected_transform(post_s, post_t, basis, mapping=mapping)(post_t)
    projected = project_delta(delta, basis)
    positions = sorted(mapping)
    for i, p in enumerate(positions):
        expected = (post_t[0, p].float() + projected[i]).to(post_t.dtype)
        assert torch.equal(out[0, p, :], expected)


# ── PCA machinery ────────────────────────────────────────────────────────────


def test_pca_state_directions_orthonormal_shape_deterministic():
    torch.manual_seed(3)
    z = [torch.randn(10, FAKE_HIDDEN) for _ in range(8)]
    basis, singular = pca_state_directions(z, 4)
    assert basis.shape == (FAKE_HIDDEN, 4)
    assert singular.shape == (4,)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-5)
    assert bool((singular[1:] <= singular[:-1] + 1e-6).all())
    basis2, singular2 = pca_state_directions(z, 4)
    assert torch.equal(basis, basis2) and torch.equal(singular, singular2)


def test_pca_state_directions_validation():
    z = [torch.randn(10, FAKE_HIDDEN)]
    with pytest.raises(ValueError, match="k must be"):
        pca_state_directions(z, 0)
    with pytest.raises(ValueError, match="rank bound"):
        pca_state_directions(z, FAKE_HIDDEN + 1)
    with pytest.raises(ValueError, match="nonempty"):
        pca_state_directions([], 1)
    with pytest.raises(ValueError, match="non-finite"):
        pca_state_directions([torch.tensor([float("nan")]).expand(2, FAKE_HIDDEN)], 1)


def test_project_delta_completeness():
    torch.manual_seed(4)
    # Rank-1 delta stack: the span of the basis contains every delta.
    direction = torch.randn(FAKE_HIDDEN)
    deltas = [direction * torch.randn(10, 1) for _ in range(8)]
    basis, _ = pca_state_directions(deltas, 4)
    for delta in deltas:
        projected = project_delta(delta, basis)
        assert torch.allclose(projected, delta, atol=1e-4)
    # Residual of a generic projection is orthogonal to the basis.
    generic = torch.randn(10, FAKE_HIDDEN)
    residual = generic - project_delta(generic, basis)
    assert torch.allclose(residual @ basis, torch.zeros(10, 4), atol=1e-4)
    # Empty basis projects to zero.
    assert torch.equal(project_delta(generic, torch.empty(FAKE_HIDDEN, 0)), torch.zeros(10, FAKE_HIDDEN))


# ── dial hooks ───────────────────────────────────────────────────────────────


def _fake_model() -> object:
    return _qwen_fake()


def test_dial_value_capture_complete_and_removed():
    model = _fake_model()
    torch.manual_seed(5)
    tensor = torch.randint(2, 100, (1, 40))
    positions = (5, 10, 20)
    with dial_value_capture(model, 15, FAKE_DIAL, positions) as box:
        model.forward(tensor)
    assert sorted(box) == list(positions)
    for value in box.values():
        assert torch.isfinite(torch.tensor(value))
    module = dense_down_projection(model.layers[15])
    assert not module._forward_pre_hooks  # hook removed


def test_dial_value_capture_fails_closed():
    model = _fake_model()
    tensor = torch.randint(2, 100, (1, 40))
    # No forward inside the context → incomplete capture.
    with pytest.raises(RuntimeError, match="dial capture incomplete"):
        with dial_value_capture(model, 15, FAKE_DIAL, (5, 10)):
            pass
    with pytest.raises(ValueError, match="out of range"):
        with dial_value_capture(model, 99, FAKE_DIAL, (5,)):
            model.forward(tensor)
    with pytest.raises(ValueError, match="out of range"):
        with dial_value_capture(model, 15, FAKE_INTERMEDIATE + 1, (5,)):
            model.forward(tensor)
    with pytest.raises(ValueError, match="nonempty"):
        with dial_value_capture(model, 15, FAKE_DIAL, ()):
            model.forward(tensor)
    with pytest.raises(ValueError, match="outside the sequence"):
        with dial_value_capture(model, 15, FAKE_DIAL, (5, 999)):
            model.forward(tensor)


def _transplant_observation(model) -> tuple:
    """Hook observing the down-projection input during one forward.

    Register INSIDE the transplant context to observe the post-transplant
    value (forward pre-hooks run in registration order).
    """
    seen: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        values = args[0]
        if "input" not in seen:
            seen["input"] = values.detach().clone()
        return None

    handle = dense_down_projection(model.layers[15]).register_forward_pre_hook(hook)
    return handle, seen


def test_dial_transplant_position_restriction():
    model = _fake_model()
    torch.manual_seed(6)
    tensor = torch.randint(2, 100, (1, 40))
    positions = {8: 0.5, 12: -0.25}
    # Clean observation (no transplant).
    clean_handle, clean_seen = _transplant_observation(model)
    try:
        model.forward(tensor)
    finally:
        clean_handle.remove()
    clean_input = clean_seen["input"]
    # Transplant observation: registered inside the context, so it sees
    # the post-transplant input.
    with dial_channel_transplant(model, 15, FAKE_DIAL, positions):
        obs_handle, obs_seen = _transplant_observation(model)
        try:
            record_residuals(model, tensor, [15])
        finally:
            obs_handle.remove()
    observed = obs_seen["input"]
    # Instruction positions: channel value changed by exactly δ (fp32 → bf16).
    for pos, delta in positions.items():
        expected = (clean_input[0, pos, FAKE_DIAL].float() + delta).to(clean_input.dtype)
        assert torch.equal(observed[0, pos, FAKE_DIAL], expected)
    # All other positions: the MLP input is bit-exact unchanged.
    for pos in range(observed.shape[1]):
        if pos not in positions:
            assert torch.equal(observed[0, pos, :], clean_input[0, pos, :])


def test_dial_transplant_strict_single_fire_and_noop():
    model = _fake_model()
    torch.manual_seed(7)
    tensor = torch.randint(2, 100, (1, 40))
    # Two forwards under one context → the second firing raises.
    with pytest.raises(RuntimeError, match="fired twice"):
        with dial_channel_transplant(model, 15, FAKE_DIAL, {8: 0.5}):
            model.forward(tensor)
            model.forward(tensor)
    # All-zero deltas: no hook, bit-exact clean output.
    clean = record_residuals(model, tensor, [15])[15]
    with dial_channel_transplant(model, 15, FAKE_DIAL, {8: 0.0, 12: 0.0}):
        zero = record_residuals(model, tensor, [15])[15]
    assert torch.equal(zero, clean)
    with pytest.raises(ValueError, match="non-finite"):
        with dial_channel_transplant(model, 15, FAKE_DIAL, {8: float("nan")}):
            model.forward(tensor)


def test_dial_footprint_direction_is_weight_column():
    model = _fake_model()
    vector = dial_footprint_direction(model, 15, FAKE_DIAL)
    weight = dense_down_projection(model.layers[15]).weight
    expected = weight[:, FAKE_DIAL].float()
    expected = expected / expected.norm()
    assert vector.shape == (FAKE_HIDDEN,)
    assert torch.allclose(vector, expected, atol=1e-6)
    assert torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-6)
    with pytest.raises(ValueError, match="out of range"):
        dial_footprint_direction(model, 15, FAKE_INTERMEDIATE)


# ── R1 denominator rule + gate E arithmetic ─────────────────────────────────


def _e1_records(full_dms: dict[str, float], joint_dms: dict[str, float]) -> list[dict]:
    records = []
    for d, dm in full_dms.items():
        records.append(
            {"arm": "full", "direction": d, "layer": 15, "toward_source_delta_m": dm}
        )
    for d, dm in joint_dms.items():
        records.append(
            {"arm": "joint", "direction": d, "layer": 15, "toward_source_delta_m": dm}
        )
    return records


def _e2_records(dial_dms: dict[str, float], k8_dms: dict[str, float]) -> list[dict]:
    records = []
    for d, dm in dial_dms.items():
        records.append({"arm": "dial_transplant", "direction": d, "layer": 15, "toward_source_delta_m": dm})
    for d, dm in k8_dms.items():
        records.append({"arm": "pca_k8", "direction": d, "layer": 15, "toward_source_delta_m": dm})
    return records


def test_gate_e1_median_and_denominator_rule():
    # 8 directions; two excluded by R1 (|ΔM_full| < 0.2), one at the
    # boundary (exactly 0.2 → included).
    full = {f"D{i}": v for i, v in enumerate([1.0, 0.5, -0.4, 0.2, -0.19, 0.2, 0.3, 0.8])}
    joint = {
        "D0": 0.6,   # ratio 0.6
        "D1": 0.2,   # ratio 0.4
        "D2": -0.2,  # ratio 0.5
        "D3": 0.1,   # ratio 0.5 (boundary direction, included)
        "D4": 0.0,   # excluded
        "D5": 0.0,   # ratio 0.0 (boundary direction, included)
        "D6": 0.2,   # ratio ~0.667
        "D7": 0.4,   # ratio 0.5
    }
    e2 = _e2_records(
        {"D0": 0.1, "D1": 0.05, "D2": 0.1, "D3": 0.05, "D4": 0.0, "D5": 0.0, "D6": 0.1, "D7": 0.2},
        {d: full[d] for d in full},
    )
    gate = evaluate_gate_e(
        _e1_records(full, joint), e2,
        layers=[15], gate_layer=15, k_values=[8],
        min_full_dm=RATIO_MIN_FULL_DM, e1_min=0.5, e2b_min=0.5,
        falsifier_max=0.05, smoke=False,
    )
    e1 = gate["gate_e1"]
    assert e1["n_effective"] == 7  # D4 excluded
    assert e1["excluded_directions"] == ["D4"]
    ratios = sorted([0.6, 0.4, 0.5, 0.5, 0.0, 0.2 / 0.3, 0.5])
    assert e1["median_ratio"] == pytest.approx(ratios[len(ratios) // 2])
    # e2b: dial ratios over effective directions: D0 0.1, D1 0.1, D2 −0.25
    # (negative full denominator), D3 0.25, D5 0.0, D6 0.333, D7 0.25 →
    # median 0.1 < 0.5 → fail; ≥ 0.05 → not falsified.
    e2b = gate["gate_e2b"]
    assert e2b["pass"] is False
    assert e2b["falsified"] is False
    assert e2b["median_ratio"] == pytest.approx(0.1)
    assert gate["gate"]["pass"] is False
    # e2a curve: k=8 ratios == 1.0 for all effective directions.
    assert gate["e2a_curve"]["8"]["median_ratio"] == pytest.approx(1.0)


def test_gate_e1_boundary_inclusion_and_pass():
    # Median exactly at the 0.5 threshold → pass (≥).
    full = {f"D{i}": 1.0 for i in range(4)}
    joint = {"D0": 0.5, "D1": 0.5, "D2": 0.4, "D3": 0.6}
    e2 = _e2_records({d: 0.6 for d in full}, {d: 1.0 for d in full})
    gate = evaluate_gate_e(
        _e1_records(full, joint), e2,
        layers=[15], gate_layer=15, k_values=[8],
        min_full_dm=RATIO_MIN_FULL_DM, e1_min=0.5, e2b_min=0.5,
        falsifier_max=0.05, smoke=False,
    )
    assert gate["gate_e1"]["pass"] is True
    assert gate["gate_e2b"]["pass"] is True
    assert gate["gate"]["pass"] is True


def test_gate_e2b_falsifier_and_validation():
    full = {f"D{i}": 1.0 for i in range(4)}
    joint = {d: 0.6 for d in full}
    e2 = _e2_records({d: 0.01 for d in full}, {d: 1.0 for d in full})
    gate = evaluate_gate_e(
        _e1_records(full, joint), e2,
        layers=[15], gate_layer=15, k_values=[8],
        min_full_dm=RATIO_MIN_FULL_DM, e1_min=0.5, e2b_min=0.5,
        falsifier_max=0.05, smoke=False,
    )
    assert gate["gate_e2b"]["falsified"] is True
    assert gate["gate_e2b"]["pass"] is False
    # Missing e2 arm record → fail closed.
    with pytest.raises(ValueError, match="expected exactly one"):
        evaluate_gate_e(
            _e1_records(full, joint), [r for r in e2 if r["arm"] != "dial_transplant"],
            layers=[15], gate_layer=15, k_values=[8],
            min_full_dm=RATIO_MIN_FULL_DM, e1_min=0.5, e2b_min=0.5,
            falsifier_max=0.05, smoke=False,
        )
    # Joint without matched full arm → fail closed.
    with pytest.raises(ValueError, match="matched full arm"):
        evaluate_gate_e(
            _e1_records({"D0": 1.0}, {"D0": 0.5, "D1": 0.5}), e2,
            layers=[15], gate_layer=15, k_values=[8],
            min_full_dm=RATIO_MIN_FULL_DM, e1_min=0.5, e2b_min=0.5,
            falsifier_max=0.05, smoke=False,
        )
    # Smoke → not evaluated.
    smoke_gate = evaluate_gate_e(
        _e1_records(full, joint), e2,
        layers=[15], gate_layer=15, k_values=[1, 8],
        min_full_dm=RATIO_MIN_FULL_DM, e1_min=0.5, e2b_min=0.5,
        falsifier_max=0.05, smoke=True,
    )
    assert smoke_gate["status"] == "not_evaluated"


# ── 2B full reference (smoke acceptance #2 fixture) ─────────────────────────


def _patch_2b_full_ref(phase2b: Path, model, tokenizer, phase2a: Path) -> None:
    """Replace the fabricated 2B instruction-span reference (L11–15) with
    real full-swap measurements on the fake model, so the in-run smoke
    acceptance compares in-run vs archive on bit-exact terms."""
    pure = pipeline.pure_entity_margins(phase2a)
    rows = pipeline._canonical_2a_rows(phase2a)
    directions = pipeline.frozen_directions(phase2b)
    final_layer = int(model.n_layers) - 1
    measured: dict[str, dict[int, tuple[float, float]]] = {}
    for source, target in directions:
        src_row, tgt_row = rows[source], rows[target]
        src_tensor = torch.tensor([scoring_ids(tokenizer, src_row["formatted"])], dtype=torch.long)
        tgt_tensor = torch.tensor([scoring_ids(tokenizer, tgt_row["formatted"])], dtype=torch.long)
        buy_id, sell_id = answer_token_ids(tokenizer, tgt_row["formatted"] + DECISION_PREFIX)
        mapping = nearest_position_mapping(
            tuple(int(p) for p in src_row["instruction_span"]),
            tuple(int(p) for p in tgt_row["instruction_span"]),
        )
        source_states = record_block_states(model, src_tensor, list(range(11, 16)))
        per_layer = {}
        for layer in range(11, 16):
            transform = make_full_transform(source_states[layer]["post"], mapping=mapping)
            with residual_interventions(model, {layer: transform}):
                residual = record_residuals(model, tgt_tensor, [final_layer])[final_layer]
            log_probs = fp32_next_token_log_probs(model, residual[:, -1, :])
            patched = margin_from_log_probs(log_probs, buy_id, sell_id)
            per_layer[layer] = (
                pipeline.toward_source_delta(patched, pure[source], pure[target]),
                pipeline.normalized_transfer(patched, pure[source], pure[target]),
            )
        measured[f"{source}->{target}"] = per_layer
    path = phase2b / "sweep" / "records.jsonl"
    records = _read_jsonl(path)
    for record in records:
        if record["span"] != "instruction":
            continue
        direction, layer = record["direction"], record["layer"]
        if direction in measured and layer in measured[direction]:
            record["toward_source_delta_m"] = measured[direction][layer][0]
            record["normalized_transfer"] = measured[direction][layer][1]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── pipeline smoke ───────────────────────────────────────────────────────────


def _run_e_smoke(tmp_path: Path, monkeypatch, *, corrupt_ref: bool = False) -> Path:
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    model = _qwen_fake()
    _patch_2b_full_ref(phase2b, model, tokenizer, phase2a)
    if corrupt_ref:
        path = phase2b / "sweep" / "records.jsonl"
        records = _read_jsonl(path)
        for record in records:
            if record["direction"] == "NSC->IT" and record["span"] == "instruction" and record["layer"] == 12:
                record["toward_source_delta_m"] += 0.3
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _patch_loaders(monkeypatch, model, tokenizer)
    # Fake model: 16 layers (final = 15, = E2_LAYER), intermediate 64.
    monkeypatch.setattr(pipeline, "DIAL_NEURON", FAKE_DIAL)
    return pipeline.run_phase_e(
        model_path="fake-model",
        run_id="smoke-fake-e",
        phase2a_run=phase2a,
        phase2b_run=phase2b,
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )


def test_run_phase_e_smoke_with_fake_model(tmp_path, monkeypatch):
    run_root = _run_e_smoke(tmp_path, monkeypatch)
    manifest = _manifest(run_root)
    assert manifest["status"] == "complete"
    assert {stage for stage in manifest["stages"]} >= {"prepare", "forward_e1", "forward_e2", "analyze"}

    rows = _read_jsonl(run_root / "prepare" / "rows.jsonl")
    assert len(rows) == 16
    for row in rows:
        assert row["instruction_last_position"] == row["instruction_span"][1] - 1
    provenance = json.loads((run_root / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["protocol"] == pipeline.PROTOCOL_E
    assert provenance["protocol_rev"] == pipeline.PROTOCOL_E_REV
    assert (run_root / "prepare" / "full_ref.json").is_file()

    e1 = _read_jsonl(run_root / "forward_e1" / "records.jsonl")
    # 2 directions x 2 layers x 4 arms
    assert len(e1) == 16
    assert {r["arm"] for r in e1} == {"joint", "full", "joint_noop", "full_noop"}
    noops = [r for r in e1 if r["arm"].endswith("_noop")]
    assert len(noops) == 8
    for row in noops:
        assert row["noop_delta_m"] == 0.0
    # L15 is the final layer of the 16-layer fake: the post-block patch at
    # the non-final instruction position is not read by the final-position
    # tail → the patched margin is bit-exact the live (unpatched) margin.
    for row in e1:
        if row["layer"] == 15:
            assert row["patched_margin"] == row["live_target_margin"]
    for row in e1:
        for key in ("patched_margin", "toward_source_delta_m", "normalized_transfer",
                    "m_source", "m_target", "live_target_margin"):
            assert torch.isfinite(torch.tensor(row[key], dtype=torch.float64))
    full_records = [r for r in e1 if r["arm"] == "full"]
    for row in full_records:
        assert "full_t_ref" in row
        assert isinstance(row["excluded"], bool)

    e2 = _read_jsonl(run_root / "forward_e2" / "records.jsonl")
    # 2 directions x (2 k + full + full_noop + dial + dial_noop)
    assert len(e2) == 12
    assert {r["arm"] for r in e2} == {
        "pca_k1", "pca_k8", "pca_full", "pca_full_noop", "dial_transplant", "dial_noop",
    }
    for row in e2:
        assert row["layer"] == 15
        assert torch.isfinite(torch.tensor(row["patched_margin"], dtype=torch.float64))
    for row in e2:
        if row["arm"] in ("pca_full_noop", "dial_noop"):
            assert row["noop_delta_m"] == 0.0
    # L15 structural zero (fake final layer): every E2 patch arm reproduces
    # the E1 L15 full arm's margin exactly.
    e1_full_l15 = {
        r["direction"]: r["patched_margin"] for r in e1 if r["arm"] == "full" and r["layer"] == 15
    }
    for row in e2:
        if row["arm"] not in ("pca_full_noop", "dial_noop"):
            assert row["patched_margin"] == e1_full_l15[row["direction"]]
    dial = [r for r in e2 if r["arm"] == "dial_transplant"]
    assert all(len(r["dial_delta_per_position"]) > 0 for r in dial)
    # pca_full reproduces the E1 L15 full arm (consistency enforced in-run).
    e2_meta = json.loads((run_root / "forward_e2" / "metadata.json").read_text(encoding="utf-8"))
    assert e2_meta["svd_deterministic"] is True
    assert len(e2_meta["singular_values"]) == pipeline.E2_PCA_DIM
    assert len(e2_meta["basis_vectors"]) == pipeline.E2_PCA_DIM
    assert all(len(v) == FAKE_HIDDEN for v in e2_meta["basis_vectors"])

    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["smoke"] is True
    assert summary["gate_e"] == {"status": "not_evaluated", "reason": "smoke grid (mechanism validation only)"}
    assert summary["pca_basis_vectors"] == e2_meta["basis_vectors"]
    assert summary["pca_singular_values"] == e2_meta["singular_values"]


def test_run_phase_e_smoke_full_ref_mismatch_fails_closed(tmp_path, monkeypatch):
    """A corrupted 2B full reference must fail the smoke acceptance (#2)."""
    with pytest.raises(ValueError, match="smoke full-arm mismatch"):
        _run_e_smoke(tmp_path, monkeypatch, corrupt_ref=True)


def test_run_phase_e_prepare_span_length_fails_closed(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    model = _qwen_fake()
    _patch_2b_full_ref(phase2b, model, tokenizer, phase2a)
    _patch_loaders(monkeypatch, model, tokenizer)
    monkeypatch.setattr(pipeline, "DIAL_NEURON", FAKE_DIAL)
    # Corrupt the canonical NSC row's instruction span length in 2A.
    prompts_path = phase2a / "prepare" / "prompts.jsonl"
    rows = _read_jsonl(prompts_path)
    corrupted = False
    for row in rows:
        if row["ticker"] == "NSC" and row["reverse"] is False and row["order"] == 0:
            span = list(row["instruction_span"])
            span[1] = span[1] - 3
            row["instruction_span"] = span
            corrupted = True
    assert corrupted
    with prompts_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="not 1:1 aligned"):
        pipeline.run_phase_e(
            model_path="fake-model",
            run_id="smoke-fake-e-corrupt",
            phase2a_run=phase2a,
            phase2b_run=phase2b,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )


def test_make_delta_transform_zero_rows_bit_exact():
    """Zero-row delta (the no-op path) is a bit-exact identity."""
    tensor = torch.randn(1, 20, FAKE_HIDDEN, dtype=torch.bfloat16)
    transform = make_delta_transform({10: torch.zeros(FAKE_HIDDEN), 12: torch.zeros(FAKE_HIDDEN)})
    assert torch.equal(transform(tensor), tensor)
