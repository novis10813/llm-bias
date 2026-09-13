"""Phase D attribution tests (fake models, no checkpoint).

Covers the per-position derivative hook (shape, position selection,
strict single-fire, exception-safe hook removal, frozen/requires-grad
paths), the top-channel statistics (Spearman/argmax/sector agreement/
matched controls), and the monkeypatched run_phase_d smoke pipeline
(D1 + D2 serial, manifest complete, no-op enforcement, schema).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_bias.balanced_evidence_gap.spans import resolve_row
from llm_bias.balanced_evidence_gap.template import SECTOR_OF
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.core.inference.mlp_addition import mlp_summed_derivatives
from llm_bias.entity_to_dial import pipeline
from llm_bias.entity_to_dial.attribution import (
    differentiable_margin,
    mlp_all_positions_derivative,
    mlp_position_derivative,
    top_channel_stats,
)

from test_entity_to_dial_pipeline import (
    BUY_ID,
    SELL_ID,
    _CharTokenizer,
    _fake_upstream,
    _identity_format,
    _manifest,
    _patch_loaders,
    _qwen_fake,
    _read_jsonl,
)

WIDTH_INTERMEDIATE = 64  # _qwen_fake intermediate_size


def _sequence(model: object, length: int = 32) -> torch.Tensor:
    torch.manual_seed(7)
    ids = torch.randint(2, 100, (1, length))
    return ids.to(model.input_device)


def _backward_once(model: object, layers: list[int], position: int, tensor: torch.Tensor):
    with mlp_position_derivative(model, layers, position) as captured:
        margin = differentiable_margin(model, tensor, BUY_ID, SELL_ID)
        margin.backward()
    return captured


# ── mlp_position_derivative ──────────────────────────────────────────────────


def test_mlp_position_derivative_shape_finite_and_position_selection():
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    captured = _backward_once(model, [0, 1, 3], 10, tensor)
    assert sorted(captured) == [0, 1, 3]
    for layer, vector in captured.items():
        assert vector.shape == (WIDTH_INTERMEDIATE,)
        assert vector.dtype == torch.float32
        assert vector.device.type == "cpu"
        assert torch.isfinite(vector).all()
    # Position selection: a different absolute position yields a different
    # signed gradient vector (deterministic fake model, fixed seed).
    other = _backward_once(model, [3], 20, tensor)
    assert not torch.allclose(captured[3], other[3])


def test_mlp_position_derivative_input_validation():
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    with pytest.raises(ValueError, match="nonnegative"):
        with mlp_position_derivative(model, [0], -1):
            pass
    with pytest.raises(ValueError, match="layer out of range"):
        with mlp_position_derivative(model, [8], 0):
            pass
    with pytest.raises(ValueError, match="unique nonempty layers"):
        with mlp_position_derivative(model, [0, 0], 0):
            pass
    with pytest.raises(ValueError, match="unique nonempty layers"):
        with mlp_position_derivative(model, [], 0):
            pass
    with pytest.raises(ValueError, match="position outside"):
        _backward_once(model, [0], tensor.shape[1], tensor)
    with pytest.raises(RuntimeError, match="grad mode"):
        with torch.no_grad(), mlp_position_derivative(model, [0], 5):
            differentiable_margin(model, tensor, BUY_ID, SELL_ID)


def test_mlp_position_derivative_double_fire_fails_closed():
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    with mlp_position_derivative(model, [1], 5) as captured:
        margin = differentiable_margin(model, tensor, BUY_ID, SELL_ID)
        margin.backward()
        assert 1 in captured
        # A second forward+backward in the same context fires the hooks
        # again and must fail closed (single teacher-forced forward only).
        with pytest.raises(RuntimeError, match="fired twice"):
            differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()


def test_mlp_position_derivative_hooks_removed_on_exception():
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    hooked_modules = [dense_down_projection(model.layers[layer]) for layer in (0, 1)]
    original_forward = model.forward

    def broken_forward(*args, **kwargs):
        raise RuntimeError("boom")

    model.forward = broken_forward
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with mlp_position_derivative(model, [0, 1], 5):
                differentiable_margin(model, tensor, BUY_ID, SELL_ID)
    finally:
        model.forward = original_forward
    for module in hooked_modules:
        assert not module._forward_pre_hooks
    # The model still forwards cleanly after the failed context.
    with torch.no_grad():
        model.forward(tensor)


def test_mlp_position_derivative_requires_grad_path_matches_frozen():
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    frozen = _backward_once(model, [2, 3], 9, tensor)
    # Re-enable parameter gradients on the first layer so the down-proj
    # input already carries grad: the hook must take the no-detach branch
    # and produce the identical signed gradient (same forward values).
    for layer in model.layers:
        for param in layer.parameters():
            param.requires_grad_(True)
    try:
        live = _backward_once(model, [2, 3], 9, tensor)
    finally:
        for layer in model.layers:
            for param in layer.parameters():
                param.requires_grad_(False)
    for layer in (2, 3):
        assert torch.allclose(frozen[layer], live[layer], rtol=1e-5, atol=1e-8)


def test_all_positions_derivative_shape_partition_and_slice():
    """All-position capture: shape, finiteness, partition identity vs the
    core summed derivative, and slice consistency with the position hook."""
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    position = 10
    with mlp_all_positions_derivative(model, 3) as captured:
        differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
        all_positions = captured[3]
    assert all_positions.shape == (tensor.shape[1], WIDTH_INTERMEDIATE)
    assert all_positions.dtype == torch.float32
    assert all_positions.device.type == "cpu"
    assert torch.isfinite(all_positions).all()
    with mlp_summed_derivatives(model, [3]) as summed:
        differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
        g_summed = summed[3]
    partition = all_positions.sum(dim=0)
    rel_err = float((partition - g_summed).norm() / max(partition.norm(), g_summed.norm()))
    assert rel_err <= 0.05  # bf16 accumulation tolerance
    # Slice consistency: the position hook must recover the same row of
    # the all-position capture (deterministic CPU fake model).
    with mlp_position_derivative(model, [3], position) as pos_captured:
        differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
        g_position = pos_captured[3]
    assert g_position.shape == (WIDTH_INTERMEDIATE,)
    assert torch.allclose(g_position, all_positions[position], rtol=1e-5, atol=1e-8)


def test_all_positions_derivative_strict_single_fire_and_grad_mode():
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    with mlp_all_positions_derivative(model, 2) as captured:
        differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
        assert 2 in captured
        with pytest.raises(RuntimeError, match="fired twice"):
            differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
    with pytest.raises(ValueError, match="layer out of range"):
        with mlp_all_positions_derivative(model, 8):
            pass
    with pytest.raises(RuntimeError, match="grad mode"):
        with torch.no_grad(), mlp_all_positions_derivative(model, 2):
            differentiable_margin(model, tensor, BUY_ID, SELL_ID)


def test_position_derivative_final_layer_structural_zero():
    """At the final layer, the derivative at a non-final position is exactly
    zero: no later computation reads that position's final-layer output.
    This invariant is what makes the smoke structural-zero check sound."""
    model = _qwen_fake(num_layers=8)
    tensor = _sequence(model)
    final_layer = int(model.n_layers) - 1
    position = 5  # not the final position (last = tensor.shape[1] - 1)
    captured = _backward_once(model, [final_layer], position, tensor)
    assert torch.equal(captured[final_layer], torch.zeros_like(captured[final_layer]))
    # And at the final position the derivative is nonzero (connectivity).
    at_final = _backward_once(model, [final_layer], tensor.shape[1] - 1, tensor)
    assert float(at_final[final_layer].abs().sum()) > 0.0


def test_position_derivative_sign_agreement_with_summed():
    """The smoke acceptance mechanic: position vs all-position-summed sign.

    On the deterministic fake model the agreement value is checked for
    determinism and range (the >= 0.8 acceptance threshold is enforced by
    the smoke pipeline on the real model).
    """
    model = _qwen_fake(num_layers=8)
    tokenizer = _CharTokenizer()
    row = resolve_row(
        tokenizer, "NSC", "Name NSC", SECTOR_OF["NSC"],
        reverse=False, order=0, format_fn=_identity_format,
    )
    tensor = torch.tensor([row["prompt_ids"]], dtype=torch.long)
    position = row["instruction_span"][1] - 1
    layer = 3

    def agreement() -> float:
        with mlp_position_derivative(model, [layer], position) as captured:
            differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
            g_position = captured[layer]
        with mlp_summed_derivatives(model, [layer]) as summed:
            differentiable_margin(model, tensor, BUY_ID, SELL_ID).backward()
            g_summed = summed[layer]
        assert g_position.shape == g_summed.shape == (WIDTH_INTERMEDIATE,)
        return float((torch.sign(g_position) == torch.sign(g_summed)).float().mean())

    first = agreement()
    assert 0.0 <= first <= 1.0
    # Deterministic: same model, same input, same objective.
    assert agreement() == first


# ── top_channel_stats ────────────────────────────────────────────────────────


def _synthetic_gradients() -> tuple[dict, dict, dict]:
    """8 companies / 2 sectors / 12 channels with two perfectly
    anti-correlated channels (index 3: rho=-1, index 7: rho=+1); every
    other channel is constant (degenerate, rho=0)."""
    margins = {f"c{i:02d}": float(i) for i in range(8)}
    sector_of = {f"c{i:02d}": ("S1" if i < 4 else "S2") for i in range(8)}
    layer = {f"c{i:02d}": [1.0] * 12 for i in range(8)}
    for i in range(8):
        layer[f"c{i:02d}"][3] = -float(i)
        layer[f"c{i:02d}"][7] = float(i)
    gradients = {2: layer}
    return gradients, margins, sector_of


def test_top_channel_stats_argmax_tie_lowest_index_and_agreement():
    gradients, margins, sector_of = _synthetic_gradients()
    stats = top_channel_stats(
        gradients, margins, sector_of, controls_n=10, controls_seed_base=42
    )
    entry = stats[2]
    # |rho| ties at 1.0 between channels 3 and 7 -> lowest index wins.
    assert entry["top_channel_idx"] == 3
    assert entry["top_channel_rho"] == pytest.approx(-1.0)
    # Top channel is negative in both sectors -> agreement 2/2.
    assert entry["top_channel_sector_agreement"] == 2
    assert entry["n_sectors"] == 2
    assert entry["n_companies"] == 8
    # Control sampling follows the frozen per-layer seed rule
    # (random.Random(42 + layer).sample(range(width), 10)); the expected
    # max |rho| is computed from the protocol rule, not the implementation.
    import random as _random

    control_idxs = _random.Random(42 + 2).sample(range(12), 10)
    expected_max = 1.0 if set(control_idxs) & {3, 7} else 0.0
    assert sorted(entry["control_channel_idxs"]) == sorted(control_idxs)
    assert entry["max_control_rho"] == expected_max
    # Deterministic control sampling under the frozen per-layer seed.
    assert top_channel_stats(
        gradients, margins, sector_of, controls_n=10, controls_seed_base=42
    )[2]["control_channel_idxs"] == entry["control_channel_idxs"]


def test_top_channel_stats_positive_top_channel_and_partial_agreement():
    gradients, margins, sector_of = _synthetic_gradients()
    # Flip channel 7 to dominate (|rho|=1 at index 7, channel 3 weakened).
    for ticker, vector in gradients[2].items():
        vector[3] = 1.0
        vector[7] = float(margins[ticker])
    # Make sector S2 disagree with the positive top channel.
    for i in (4, 5, 6, 7):
        gradients[2][f"c{i:02d}"][7] = 0.0 - float(i)
        margins[f"c{i:02d}"] = -float(i)
    stats = top_channel_stats(gradients, margins, sector_of)
    entry = stats[2]
    assert entry["top_channel_idx"] == 7
    # S1 values (0..3) are positive, S2 values (-4..-7) negative:
    # rho across companies is still negative (values track margins), so
    # sign = -1 and S2 agrees, S1 does not -> agreement 1/2.
    assert entry["top_channel_sector_agreement"] == 1


def test_top_channel_stats_degenerate_and_validation():
    gradients, margins, sector_of = _synthetic_gradients()
    # All margins equal -> every channel degenerate -> rho 0, agreement 0.
    flat = dict(margins)
    for ticker in flat:
        flat[ticker] = 1.0
    entry = top_channel_stats({2: gradients[2]}, flat, sector_of)[2]
    assert entry["top_channel_rho"] == 0.0
    assert entry["top_channel_sector_agreement"] == 0
    assert entry["top_channel_idx"] == 0

    bad_width = {2: {"c00": [1.0] * 12, "c01": [1.0] * 11}}
    with pytest.raises(ValueError, match="width mismatch"):
        top_channel_stats(bad_width, margins, sector_of)
    with pytest.raises(ValueError, match="metadata"):
        top_channel_stats({2: gradients[2]}, margins, {t: s for t, s in sector_of.items() if t == "c00"})
    with pytest.raises(ValueError, match="no company gradients"):
        top_channel_stats({2: {}}, margins, sector_of)


# ── run_phase_d smoke pipeline (monkeypatched) ──────────────────────────────


def _aligned_canonical_rows(monkeypatch):
    """Normalize the fake instruction spans to a common 1:1-aligned range.

    The char-level fake tokenizer makes ticker length shift the span
    start per company; the real 2A data already satisfies the equal-span
    invariant, so the fake rows are aligned the same way here.
    """
    real = pipeline._canonical_2a_rows

    def patched(phase2a_run):
        canon = real(phase2a_run)
        lo = max(int(row["instruction_span"][0]) for row in canon.values())
        hi = min(int(row["instruction_span"][1]) for row in canon.values())
        assert lo < hi
        for row in canon.values():
            row["instruction_span"] = [lo, hi]
        return canon

    monkeypatch.setattr(pipeline, "_canonical_2a_rows", patched)


def test_run_phase_d_smoke_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)
    # Fake model has 16 layers (final = 15); smoke grid (12, 15) stays in
    # range and includes the final layer.
    monkeypatch.setattr(pipeline, "SMOKE_D_LAYERS", (12, 15))

    run_root = pipeline.run_phase_d(
        model_path="fake-model",
        run_id="smoke-fake-d",
        phase2a_run=phase2a,
        phase2b_run=phase2b,
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )
    manifest = _manifest(run_root)
    assert manifest["status"] == "complete"
    assert {stage for stage in manifest["stages"]} >= {"prepare", "forward_d1", "forward_d2", "analyze"}

    rows = _read_jsonl(run_root / "prepare" / "rows.jsonl")
    # All 16 canonical companies (D2 population), not just the 2 smoke
    # direction tickers.
    from llm_bias.balanced_evidence_gap.template import ALL_TICKERS

    assert {r["ticker"] for r in rows} == set(ALL_TICKERS)
    for row in rows:
        assert row["instruction_last_position"] == row["instruction_span"][1] - 1
        assert row["instruction_span"][1] - row["instruction_span"][0] > 0
    provenance = json.loads((run_root / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["protocol"] == pipeline.PROTOCOL_D
    assert provenance["protocol_rev"] == pipeline.PROTOCOL_D_REV
    assert (
        provenance["instruction_span_length"]
        == rows[0]["instruction_span"][1] - rows[0]["instruction_span"][0]
    )

    d1 = _read_jsonl(run_root / "forward_d1" / "records.jsonl")
    # 2 directions x (2 layers x 2 components + 2 final + 2 layers x 2 no-ops)
    assert len(d1) == 20
    noop = [r for r in d1 if r["component"].startswith("self_noop_")]
    assert len(noop) == 8
    for row in noop:
        assert row["noop_delta_m"] == 0.0
    final = [r for r in d1 if r["component"] in ("final_mlp", "final_attn")]
    assert len(final) == 4
    assert {r["layer"] for r in final} == {15}
    assert {r["component"] for r in d1 if not r["component"].startswith("self_noop")} == {
        "mlp", "attn", "final_mlp", "final_attn"
    }
    for row in d1:
        for key in ("patched_margin", "toward_source_delta_m", "normalized_transfer",
                    "m_source", "m_target", "live_target_margin"):
            assert torch.isfinite(torch.tensor(row[key], dtype=torch.float64))

    d2 = _read_jsonl(run_root / "forward_d2" / "records.jsonl")
    assert [r["layer"] for r in d2] == [12, 15]
    for row in d2:
        assert row["phase"] == "d2"
        assert row["n_companies"] == 1  # smoke: SMOKE_D2_TICKER only
        assert 0 <= row["top_channel_idx"] < WIDTH_INTERMEDIATE
        assert row["vector_finite"] is True
        assert torch.isfinite(torch.tensor(row["top_channel_rho"], dtype=torch.float64))
        assert 0 <= row["top_channel_sector_agreement"] <= row["n_sectors"]
        assert len(row["control_channel_idxs"]) == 10
    d2_metadata = json.loads((run_root / "forward_d2" / "metadata.json").read_text(encoding="utf-8"))
    assert d2_metadata["companies"] == ["NSC"]
    acceptance = d2_metadata["derivative_acceptance"]
    assert acceptance["probe_layer"] == 12
    # L15 is the final layer of the 16-layer fake: the derivative at the
    # non-final instruction position is structurally zero.
    assert acceptance["structural_zero_final_layer"] is True
    assert acceptance["nonfinal_nonzero_norm"] > 0.0
    assert acceptance["partition_rel_err"] <= acceptance["partition_tolerance"]
    assert 0.0 <= acceptance["sign_agreement_descriptive"] <= 1.0
    assert acceptance["pass"] is True

    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_d"]["status"] == "not_evaluated"  # smoke grid
    assert set(summary["curves"]["mlp"]) == {"12", "15"}
    assert set(summary["curves"]["attn"]) == {"12", "15"}
    assert set(summary["final_position_sanity"]) == {"final_mlp", "final_attn"}
    assert set(summary["d2_descriptive"]) == {"12", "15"}
    assert summary["smoke"] is True
    # No raw payloads anywhere: no reserved words in any persisted key.
    for record in d1 + d2:
        for key in record:
            assert not any(
                part in {"activation", "gradient", "residual", "hidden_state"}
                for part in key.split("_")
            )


def test_run_phase_d_smoke_derivative_acceptance_fails_closed(tmp_path, monkeypatch):
    """A corrupted all-position-summed derivative must fail the smoke.

    The partition identity compares the pipeline's own all-position capture
    against the core mlp_summed_derivatives output; perturbing the core
    output breaks the identity deterministically.
    """
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)
    monkeypatch.setattr(pipeline, "SMOKE_D_LAYERS", (12, 15))

    import contextlib

    real_summed = pipeline.mlp_summed_derivatives

    class _CorruptedView:
        """Records proxy: entries are filled during the caller's backward()
        into the core's dict, so the perturbation applies at access time."""

        def __init__(self, source: dict) -> None:
            self._source = source

        def __getitem__(self, key):
            return self._source[key] * -3.0

    @contextlib.contextmanager
    def corrupted_summed(model, layers):
        with real_summed(model, layers) as records:
            yield _CorruptedView(records)

    monkeypatch.setattr(pipeline, "mlp_summed_derivatives", corrupted_summed)
    with pytest.raises(ValueError, match="derivative acceptance failed"):
        pipeline.run_phase_d(
            model_path="fake-model",
            run_id="smoke-fake-d-badcheck",
            phase2a_run=phase2a,
            phase2b_run=phase2b,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )


def test_run_phase_d_noop_enforcement_fails_closed(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)
    monkeypatch.setattr(pipeline, "SMOKE_D_LAYERS", (12, 15))
    monkeypatch.setattr(pipeline, "NOOP_TOLERANCE", -1.0)  # force violation

    with pytest.raises(ValueError, match="no-op violated"):
        pipeline.run_phase_d(
            model_path="fake-model",
            run_id="smoke-fake-d-badnoop",
            phase2a_run=phase2a,
            phase2b_run=phase2b,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )
    from llm_bias.core.artifact_paths import dataset_slug, model_slug

    manifest_path = (
        tmp_path / "artifacts" / model_slug("fake-model") / dataset_slug(pipeline.DATASET)
        / "runs" / "smoke-fake-d-badnoop" / "manifest.json"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_run_phase_d_upstream_sweep_count_fails_closed(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)
    sweep = phase2b / "sweep" / "records.jsonl"
    lines = sweep.read_text(encoding="utf-8").splitlines()
    sweep.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record count mismatch"):
        pipeline.run_phase_d(
            model_path="fake-model",
            run_id="smoke-fake-d-badsweep",
            phase2a_run=phase2a,
            phase2b_run=phase2b,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )


def test_run_phase_d_instruction_span_length_fails_closed(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)
    monkeypatch.setattr(pipeline, "SMOKE_D_LAYERS", (12, 15))
    # Corrupt one stored row's instruction span so the equal-span
    # invariant (protocol §4.1) is violated -> prepare fails closed.
    prompts_path = phase2a / "prepare" / "prompts.jsonl"
    rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    target = next(r for r in rows if r["ticker"] == "NSC" and r["reverse"] is False and r["order"] == 0)
    target["instruction_span"] = [target["instruction_span"][0], target["instruction_span"][1] - 5]
    with prompts_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="not 1:1 aligned"):
        pipeline.run_phase_d(
            model_path="fake-model",
            run_id="smoke-fake-d-badspan",
            phase2a_run=phase2a,
            phase2b_run=phase2b,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )
