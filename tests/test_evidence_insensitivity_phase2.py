"""Fake-model regression tests for evidence-insensitivity Phase 2 (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.evidence_insensitivity import phase2
from llm_bias.evidence_insensitivity.phase2 import (
    DEFAULT_CAPTURE_LAYER,
    PHASE2_CONDITIONS,
    derive_stance_axis,
    fit_offset_gain,
    layer_localization,
    select_capture_companies,
    select_step_a_subsample,
)

N_COMPANIES = 503
DISCOVERY_RESPONSIVE = 42
DISCOVERY_FIXED = 360
HOLDOUT_RESPONSIVE = 8
HOLDOUT_FIXED = 93
SLUG = "fake-model"


class CharTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) % 251 for char in text]}

    def decode(self, values, **_kwargs):
        return "".join(chr(v) for v in values)


class FakePhase2Model:
    """Deterministic residual-stream model with n_layers identity blocks.

    ``forward`` returns a [batch, seq, d] tensor (extract_logits path: raw
    tensor is returned as-is). Per-layer drift makes layers distinguishable
    for the Step A sweep.
    """

    def __init__(self, n_layers: int = 6, d_model: int = 16, out_dtype: torch.dtype = torch.float32):
        self.n_layers = n_layers
        self.d_model = d_model
        self.out_dtype = out_dtype
        self.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

    def forward(self, input_ids_tensor, attention_mask=None, **_kwargs):
        batch, seq = input_ids_tensor.shape
        x = torch.zeros(batch, seq, self.d_model, dtype=torch.float32)
        acc = torch.zeros(batch, self.d_model, dtype=torch.float32)
        for t in range(seq):
            acc += torch.arange(self.d_model, dtype=torch.float32).expand(batch, -1) * (
                1.0 + (input_ids_tensor[:, t] % 5).to(torch.float32).unsqueeze(-1)
            )
            x[:, t] = acc
        x = x.to(self.out_dtype)
        for i, block in enumerate(self.layers):
            x = x + i * 0.001
            x = block(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _fake_group_table(n_resp_discovery: int = DISCOVERY_RESPONSIVE) -> list[dict]:
    groups = []
    sectors = ["Technology", "Healthcare", "Financials", "Energy"]
    for i in range(N_COMPANIES):
        ticker = f"T{i:03d}"
        if i < n_resp_discovery:
            group, split = "evidence-responsive", "discovery"
        elif i < DISCOVERY_RESPONSIVE + DISCOVERY_FIXED:
            group, split = "fixed-sell", "discovery"
        elif i < DISCOVERY_RESPONSIVE + DISCOVERY_FIXED + HOLDOUT_RESPONSIVE:
            group, split = "evidence-responsive", "hold-out"
        else:
            group, split = "fixed-sell", "hold-out"
        contrast = 0.5 + (i % 100) / 100.0  # deterministic positive C_c
        groups.append(
            {
                "ticker": ticker,
                "group": group,
                "gics_sector": sectors[i % len(sectors)],
                "split": split,
                "contrast_c": contrast,
                "d0": "sell",
                "m0": -5.0,
                "prior_label": None,
            }
        )
    return groups


def _fake_phase1_run(artifact_root: Path, slug: str = SLUG, run_id: str = "phase1-gpu-bf16-01", tokenizer: CharTokenizer | None = None, n_resp_discovery: int = DISCOVERY_RESPONSIVE) -> Path:
    tokenizer = tokenizer or CharTokenizer()
    run_dir = artifact_root / slug / "evidence-insensitivity" / "runs" / run_id
    (run_dir / "prepare").mkdir(parents=True, exist_ok=True)
    (run_dir / "analyze").mkdir(parents=True, exist_ok=True)
    import hashlib

    prompt_rows = []
    for i in range(N_COMPANIES):
        ticker = f"T{i:03d}"
        for condition in ["zero", "N6", "N8", "N10", "N15", "P6", "P8", "P10", "P15"]:
            text = f"prompt {ticker} {condition}"
            ids = input_ids(tokenizer, text, add_special_tokens=True)
            prompt_rows.append(
                {
                    "prompt_id": hashlib.sha256(text.encode()).hexdigest()[:12],
                    "ticker": ticker,
                    "company_name": f"Company {ticker}",
                    "gics_sector": "Technology",
                    "split": "discovery",
                    "arm": "primary",
                    "condition": condition,
                    "polarity_score": 0,
                    "prompt_text": text,
                    "evidence_sha": "x",
                    "header_span": [0, 3],
                    "evidence_span": [3, len(ids) - 3],
                    "instruction_span": [len(ids) - 3, len(ids)],
                }
            )
    with open(run_dir / "prepare" / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for row in prompt_rows:
            fh.write(json.dumps(row) + "\n")
    summary = {
        "schema_version": "evidence-insensitivity-phase1-v1",
        "groups": _fake_group_table(n_resp_discovery=n_resp_discovery),
        "descriptive_stats": {},
        "gates": {},
    }
    (run_dir / "analyze" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_select_capture_companies_counts_and_failclosed():
    groups = _fake_group_table()
    selected = select_capture_companies(groups, "discovery")
    assert len(selected) == DISCOVERY_RESPONSIVE + DISCOVERY_FIXED
    discovery_tickers = {g["ticker"] for g in groups if g["split"] == "discovery"}
    assert all(s["ticker"] in discovery_tickers for s in selected)
    with pytest.raises(ValueError, match="503 unique"):
        select_capture_companies(groups[:400])
    broken = list(groups)
    for g in broken[:100]:
        g["group"] = None
    with pytest.raises(ValueError, match="unlabeled"):
        select_capture_companies(broken)


def test_select_step_a_subsample_stratified_and_fallback():
    groups = _fake_group_table()
    subsample = select_step_a_subsample(groups)
    assert len(subsample) == 16 and len(set(subsample)) == 16
    by_group = {}
    for g in groups:
        if g["split"] == "discovery":
            by_group.setdefault(g["group"], set()).add(g["ticker"])
    n_resp = sum(1 for t in subsample if t in by_group["evidence-responsive"])
    assert n_resp == 8  # both groups >= 8 -> 8/8
    # fallback: only 5 responsive in discovery -> 5 + 11 fixed-sell
    small = _fake_group_table(n_resp_discovery=5)
    subsample2 = select_step_a_subsample(small)
    assert len(subsample2) == 16 and len(set(subsample2)) == 16
    by_group2 = {g["group"] for g in small if g["split"] == "discovery" and g["group"] == "evidence-responsive"}
    assert by_group2 == {"evidence-responsive"}
    n_resp2 = sum(1 for t in subsample2 if t in {g["ticker"] for g in small if g["split"] == "discovery" and g["group"] == "evidence-responsive"})
    assert n_resp2 == 5  # smaller group taken fully


def test_derive_stance_axis_direction_and_zero_norm():
    rng = np.random.default_rng(0)
    n15 = rng.normal(size=(20, 8)).astype(np.float32)
    mean_dir = np.arange(1.0, 9.0, dtype=np.float32)
    p15 = n15 + mean_dir + rng.normal(scale=0.01, size=(20, 8)).astype(np.float32)
    d = derive_stance_axis(p15, n15)
    assert d.shape == (8,)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5
    assert np.corrcoef(d, mean_dir)[0, 1] > 0.99
    with pytest.raises(ValueError, match="zero norm"):
        derive_stance_axis(n15, n15)


def test_fit_offset_gain_linear_and_nonlinear():
    # exact linear: r = 2 + 3 * (v/4)
    r_n15, r_zero, r_p15 = -1.0, 2.0, 5.0
    offset, gain, nonlin = fit_offset_gain(r_zero, r_n15, r_p15)
    assert offset == pytest.approx(2.0, abs=1e-9)
    assert gain == pytest.approx(3.0, abs=1e-9)
    assert nonlin == pytest.approx(0.0, abs=1e-9)
    # V-shape: strong nonlinearity
    offset, gain, nonlin = fit_offset_gain(0.0, -5.0, 0.0)
    assert nonlin > 1.0


def test_layer_localization_picks_correlated_layer_and_tie_breaks_shallow():
    n, d = 16, 8
    tickers = [f"C{i:02d}" for i in range(n)]
    c_c = np.array([float(i) for i in range(n)], dtype=np.float64)
    rng = np.random.default_rng(1)
    states_by_layer: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    target = np.arange(d, dtype=np.float32)
    for L in range(4):
        states_by_layer[L] = {"P15": {}, "N15": {}}
        for i, t in enumerate(tickers):
            base = rng.normal(scale=1.0, size=d).astype(np.float32)
            if L == 2:
                signal = float(i) * target * 0.5
            else:
                signal = rng.normal(scale=0.01, size=d).astype(np.float32)
            states_by_layer[L]["P15"][t] = base + signal
            states_by_layer[L]["N15"][t] = base
    l_star, corrs = layer_localization(states_by_layer, dict(zip(tickers, c_c)))
    assert l_star == 2
    assert set(corrs) == {0, 1, 2, 3}
    # tie: two layers equally correlated -> shallower wins
    tie_states = {
        L: {
            "P15": {t: np.full(d, float(i), dtype=np.float32) for i, t in enumerate(tickers)},
            "N15": {t: np.zeros(d, dtype=np.float32) for t in tickers},
        }
        for L in (1, 3)
    }
    l_tie, _ = layer_localization(tie_states, dict(zip(tickers, c_c)))
    assert l_tie == 1


def test_full_lifecycle_bf16_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Real models (bf16) emit bfloat16 residuals; forward_stage must convert
    # to float32 before numpy (numpy has no bf16 scalar type).
    summary = _run_full_lifecycle(tmp_path, monkeypatch, anchor_layer=3, model_dtype=torch.bfloat16)
    assert summary["gates"]["G-2A"]["pass"] is True
    entry = summary["per_company"]["T000"]
    assert all(np.isfinite([entry[k] for k in ("r_zero", "r_n15", "r_p15", "offset", "gain")]))


def test_capture_position_bounds_check():
    from llm_bias.core.inference.forward import capture_position_residuals, encode_batch

    model = FakePhase2Model(n_layers=6, d_model=16)
    rows = [[10, 11, 12, 13], [10, 11, 12]]  # lengths 4 and 3 (right-padded to 4)
    encoded = encode_batch(rows, "cpu")
    # in-bounds positions work
    res = capture_position_residuals(model, encoded, torch.tensor([2, 1]), layers=[3])
    assert res[3].shape == (2, 16)
    # position beyond the unpadded length fails closed
    with pytest.raises(ValueError, match="outside the row's unpadded length"):
        capture_position_residuals(model, encoded, torch.tensor([3, 3]), layers=[3])
    # negative position fails closed
    with pytest.raises(ValueError, match="outside the row's unpadded length"):
        capture_position_residuals(model, encoded, torch.tensor([-1, 0]), layers=[3])


def _run_full_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, anchor_layer: int | None, run_id: str = "phase2-test", n_resp_discovery: int = DISCOVERY_RESPONSIVE, model_dtype: torch.dtype = torch.float32) -> dict:
    tokenizer = CharTokenizer()
    _fake_phase1_run(tmp_path, SLUG, "phase1-gpu-bf16-01", tokenizer, n_resp_discovery=n_resp_discovery)
    if anchor_layer is None:
        monkeypatch.delitem(phase2.DEFAULT_CAPTURE_LAYER, SLUG, raising=False)
    else:
        monkeypatch.setitem(phase2.DEFAULT_CAPTURE_LAYER, SLUG, anchor_layer)
    phase2.run_phase2_prepare(run_id, artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug=SLUG)
    model = FakePhase2Model(n_layers=6, d_model=16, out_dtype=model_dtype)
    phase2.run_phase2_forward(run_id, artifact_root=tmp_path, model_path="fake", model=model, tokenizer=tokenizer, device="cpu", model_slug=SLUG)
    result = phase2.run_phase2_analyze(run_id, artifact_root=tmp_path, model_slug=SLUG)
    return json.loads(Path(result).read_text(encoding="utf-8"))


def test_full_lifecycle_frozen_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_full_lifecycle(tmp_path, monkeypatch, anchor_layer=3)
    run_dir = tmp_path / SLUG / "evidence-insensitivity" / "runs" / "phase2-test"
    records = [json.loads(line) for line in (run_dir / "forward" / "records.jsonl").read_text().splitlines() if line]
    assert len(records) == (DISCOVERY_RESPONSIVE + DISCOVERY_FIXED) * 3
    assert summary["n_companies"] == DISCOVERY_RESPONSIVE + DISCOVERY_FIXED
    assert summary["capture_layer"] == 3
    assert summary["gates"]["G-2A"]["pass"] is True
    assert summary["gates"]["G-2C"]["pass"] is True
    assert "welch" in summary["contrast"]["offset"]
    assert "layer_sweep" not in summary
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    # axis: unit vector + finite
    axis = json.loads((run_dir / "forward" / "axis.json").read_text())
    d = np.array(axis["d_stance"])
    assert d.shape == (16,) and abs(float(np.linalg.norm(d)) - 1.0) < 1e-4
    for key in ("r2_stance", "pca_explained_top5"):
        assert key in axis
    # per_company has offset/gain
    entry = summary["per_company"]["T000"]
    for key in ("offset", "gain", "nonlinearity", "r_zero", "r_n15", "r_p15", "c_c"):
        assert key in entry


def test_full_lifecycle_step_a_localization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_full_lifecycle(tmp_path, monkeypatch, anchor_layer=None)
    run_dir = tmp_path / SLUG / "evidence-insensitivity" / "runs" / "phase2-test"
    assert summary["capture_layer_basis"].startswith("Step A")
    sweep = json.loads((run_dir / "forward" / "layer_sweep.json").read_text())
    assert 0 <= summary["capture_layer"] < 6
    assert set(int(k) for k in sweep["corrs"]) == set(range(6))
    assert sweep["layer_star"] == summary["capture_layer"]
    metadata = json.loads((run_dir / "forward" / "metadata.json").read_text())
    assert metadata["determinism_check"]["n_prompts"] == 20


def test_capture_position_invariant_violation(tmp_path: Path):
    tokenizer = CharTokenizer()
    _fake_phase1_run(tmp_path, SLUG, "phase1-gpu-bf16-01", tokenizer)
    run_dir = tmp_path / SLUG / "evidence-insensitivity" / "runs" / "phase1-gpu-bf16-01"
    rows = [json.loads(line) for line in (run_dir / "prepare" / "prompts.jsonl").read_text().splitlines() if line]
    rows[0]["instruction_span"] = [0, 1000]  # wrong end
    with open(run_dir / "prepare" / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="capture position out of bounds"):
        phase2.run_phase2_prepare("phase2-bad", artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug=SLUG)


def test_gate_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # G-2C: responsive group shrunk below 10 -> gate fails, contrast skipped
    summary = _run_full_lifecycle(tmp_path, monkeypatch, anchor_layer=3, run_id="phase2-lowpower", n_resp_discovery=5)
    assert summary["gates"]["G-2C"]["pass"] is False
    assert summary["gates"]["G-2C"]["value"]["evidence-responsive"] == 5
    assert summary["gates_passed"] is False
    assert "welch" not in summary["contrast"]["offset"]
    full = _run_full_lifecycle(tmp_path / "b", monkeypatch, anchor_layer=3, run_id="phase2-test")
    assert "welch" in full["contrast"]["offset"]
    # G-2B boundary consistency: pass flag matches the threshold
    assert full["gates"]["G-2B"]["pass"] == (full["gates"]["G-2B"]["value"] >= phase2.STANCE_R2_MIN)


def test_summary_serializes_through_core_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_full_lifecycle(tmp_path, monkeypatch, anchor_layer=3)
    from llm_bias.core.artifacts.io import write_json

    # core write_json enforces the no-raw-tensor guard; must not raise
    write_json(tmp_path / "summary-check.json", summary, overwrite=True)
