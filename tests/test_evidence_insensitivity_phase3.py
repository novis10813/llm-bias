"""Fake-model regression tests for evidence-insensitivity Phase 3 (no GPU).

The fake model is a vectorized residual-stream toy: dimension 0 carries a
running sum of per-token features ``v(t) = ((tok % 3) - 1) * (1000 - t)``,
so each position state accumulates all upstream tokens (like a causal
residual stream). With ``_lm_head`` wired as buy=+x0 / sell=-x0, the
decision-position margin is exactly ``2 * x0`` (the log-sum-exp cancels in
the difference). The N15/P15 prompts differ only in the evidence digit
positions, so a polarity-transfer swap at any coordinate at or after the
first differing digit shifts the final-position margin by a constant
``2 * (x0_P15 - x0_N15) > 0`` at every layer, while a swap at the entity
header position (identical partial sum) leaves it bit-identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from llm_bias.core.prompt_input.encoding import input_ids, token_span
from llm_bias.evidence_insensitivity import phase3
from llm_bias.evidence_insensitivity.phase3 import (
    LAYER_GRIDS,
    PHASE1_DEFAULT_RUN,
    active_combos,
    _prompt_positions,
    _swap_transform,
    select_sample,
)
from llm_bias.evidence_insensitivity.template import DECISION_PREFIX, build_prompt, prompt_char_spans

SLUG = "fake-model"
FAKE_GRID = (0, 2, 4, 5, 7)
N_LAYERS = 8
D_MODEL = 16
VOCAB = 2048
N_COMPANIES = 503
DISCOVERY_RESPONSIVE = 42
DISCOVERY_FIXED = 360
HOLDOUT_RESPONSIVE = 8
BUY_ID = 10
SELL_ID = 11


class WordCharTokenizer:
    """Character-level tokenizer with pre-registered multi-char tokens
    ``buy`` (id 10) and ``sell`` (id 11); greedy left-to-right matching so
    appending ``buy``/``sell`` to any text is prefix-stable."""

    def __init__(self):
        self.eos_token_id = 1
        self.pad_token_id = 0

    def _encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        i, n = 0, len(text)
        while i < n:
            if text.startswith("buy", i):
                ids.append(BUY_ID)
                offsets.append((i, i + 3))
                i += 3
            elif text.startswith("sell", i):
                ids.append(SELL_ID)
                offsets.append((i, i + 4))
                i += 4
            else:
                c = text[i]
                ids.append(1000 + ord(c))
                offsets.append((i, i + 1))
                i += 1
        return ids, offsets

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, return_special_tokens_mask=False):
        ids, offsets = self._encode(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_special_tokens_mask:
            out["special_tokens_mask"] = [False] * len(ids)
        return out

    def decode(self, values, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        chars = []
        for v in values:
            if v == BUY_ID:
                chars.append("buy")
            elif v == SELL_ID:
                chars.append("sell")
            elif v >= 1000:
                chars.append(chr(v - 1000))
        return "".join(chars)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, x):
        self.post_attention_layernorm(x)
        return x


def _suffix_ids(decision: str) -> list[int]:
    return WordCharTokenizer()._encode(f'{{"decision": "{decision}", "reason": "ok"}}')[0]


class FakePhase3Model:
    """Deterministic intervention-test model (see module docstring)."""

    def __init__(self, n_layers: int = N_LAYERS, d_model: int = D_MODEL, vocab: int = VOCAB, out_dtype: torch.dtype = torch.float32, noise: float = 0.0, lm_scale: float = 1.0):
        self.n_layers = n_layers
        self.d_model = d_model
        self.out_dtype = out_dtype
        self.noise = noise
        self.hf_model = self
        self.layers = nn.ModuleList([FakeBlock() for _ in range(n_layers)])
        self._final_norm = nn.Identity()
        self._lm_head = nn.Linear(d_model, vocab, bias=False)
        w = torch.zeros(vocab, d_model)
        w[BUY_ID, 0] = lm_scale
        w[SELL_ID, 0] = -lm_scale
        with torch.no_grad():
            self._lm_head.weight.copy_(w)

    def forward(self, input_ids_tensor, attention_mask=None, **_kwargs):
        batch, seq = input_ids_tensor.shape
        x = torch.zeros(batch, seq, self.d_model, dtype=torch.float32, device=input_ids_tensor.device)
        t = torch.arange(seq, dtype=torch.float32, device=input_ids_tensor.device)
        v = ((input_ids_tensor % 3) - 1).to(torch.float32) * (1000.0 - t)
        if self.noise > 0:
            v = v + torch.randn(batch, seq, dtype=torch.float32, device=input_ids_tensor.device) * self.noise
        x[:, :, 0] = v.cumsum(dim=1)
        x = x.to(self.out_dtype)
        for i, block in enumerate(self.layers):
            x = x + i * 0.001
            x = block(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate(self, prompt_ids, **_kwargs):
        with torch.no_grad():
            out = self.forward(prompt_ids)
            # one decode step, mimicking the HF generation loop: the block
            # receives a [1, 1] hidden state and active position transforms
            # must pass it through (prefill-only guard)
            self.forward(torch.tensor([[_suffix_ids("buy")[0]]], dtype=torch.long, device=prompt_ids.device))
        score = float(out[0, -1, 0].float())
        decision = "buy" if score > 0.0 else "sell"
        suffix = _suffix_ids(decision)
        return torch.cat([prompt_ids, torch.tensor([suffix], dtype=torch.long, device=prompt_ids.device)], dim=1)


def _group_table(n_resp_discovery: int = DISCOVERY_RESPONSIVE) -> list[dict]:
    sectors = ["Technology", "Healthcare", "Financials", "Energy"]
    groups = []
    for i in range(N_COMPANIES):
        if i < n_resp_discovery:
            group, split = "evidence-responsive", "discovery"
        elif i < n_resp_discovery + DISCOVERY_FIXED:
            group, split = "fixed-sell", "discovery"
        elif i < n_resp_discovery + DISCOVERY_FIXED + HOLDOUT_RESPONSIVE:
            group, split = "evidence-responsive", "hold-out"
        else:
            group, split = "fixed-sell", "hold-out"
        groups.append({"ticker": f"T{i:03d}", "group": group, "gics_sector": sectors[i % len(sectors)], "split": split})
    return groups


def _fake_phase1_run(artifact_root: Path, tokenizer: WordCharTokenizer, *, slug: str = SLUG, run_id: str = PHASE1_DEFAULT_RUN, n_resp_discovery: int = DISCOVERY_RESPONSIVE) -> Path:
    run_dir = artifact_root / slug / "evidence-insensitivity" / "runs" / run_id
    (run_dir / "prepare").mkdir(parents=True, exist_ok=True)
    (run_dir / "analyze").mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    for i in range(N_COMPANIES):
        ticker = f"T{i:03d}"
        name = f"Company {ticker}"
        for condition in ("N15", "P15"):
            raw = build_prompt(ticker, name, condition)
            char_spans = prompt_char_spans(raw)
            prompt_rows.append(
                {
                    "prompt_id": f"{ticker}-{condition}",
                    "ticker": ticker,
                    "company_name": name,
                    "gics_sector": "Technology",
                    "split": "discovery",
                    "arm": "primary",
                    "condition": condition,
                    "polarity_score": -4 if condition == "N15" else 4,
                    "prompt_text": raw,
                    "evidence_span": list(token_span(tokenizer, raw, char_spans["evidence"][0], char_spans["evidence"][1])),
                    "instruction_span": list(token_span(tokenizer, raw, char_spans["instruction"][0], char_spans["instruction"][1])),
                }
            )
    with open(run_dir / "prepare" / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for row in prompt_rows:
            fh.write(json.dumps(row) + "\n")
    (run_dir / "analyze" / "summary.json").write_text(
        json.dumps({"schema_version": "evidence-insensitivity-phase1-v1", "groups": _group_table(n_resp_discovery), "descriptive_stats": {}, "gates": {}}),
        encoding="utf-8",
    )
    # Phase 2 anchor metadata consumed by prepare_stage
    phase2_forward = run_dir.parent / "phase2-gpu-bf16-01" / "forward"
    phase2_forward.mkdir(parents=True, exist_ok=True)
    (phase2_forward / "metadata.json").write_text(json.dumps({"capture_layer": 5, "model": {"name": "fake"}}), encoding="utf-8")
    return run_dir


def _run_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, run_id: str = "phase3-test", n_resp_discovery: int = DISCOVERY_RESPONSIVE, sample_per_group: int = phase3.SAMPLE_PER_GROUP, **model_kwargs) -> dict:
    monkeypatch.setitem(LAYER_GRIDS, SLUG, FAKE_GRID)
    monkeypatch.setattr(phase3, "SAMPLE_PER_GROUP", sample_per_group)
    tokenizer = WordCharTokenizer()
    _fake_phase1_run(tmp_path, tokenizer, n_resp_discovery=n_resp_discovery)
    phase3.run_phase3_prepare(run_id, artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug=SLUG)
    model = FakePhase3Model(**model_kwargs)
    phase3.run_phase3_forward(run_id, artifact_root=tmp_path, model_path="fake", model=model, tokenizer=tokenizer, device="cpu", model_slug=SLUG)
    result = phase3.run_phase3_analyze(run_id, artifact_root=tmp_path, model_slug=SLUG)
    return json.loads(Path(result).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_select_sample_sizes_seeded_and_fail_closed():
    groups = _group_table()
    sample = select_sample(groups)
    assert len(sample) == 2 * 42
    by_group = {}
    for row in sample:
        by_group.setdefault(row["group"], []).append(row["ticker"])
    assert len(by_group["evidence-responsive"]) == 42
    assert len(by_group["fixed-sell"]) == 42
    # the 42 responsive discovery companies are exactly the whole group
    all_r = {g["ticker"] for g in groups if g["group"] == "evidence-responsive" and g["split"] == "discovery"}
    assert set(by_group["evidence-responsive"]) == all_r
    # seeded subsample is deterministic
    assert sample == select_sample(groups)
    # fail closed when a group is too small
    with pytest.raises(ValueError, match="has 5 <"):
        select_sample(_group_table(n_resp_discovery=5))


def test_swap_transform_replaces_single_position():
    x = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    src = torch.full((1, 3), 9.0)
    out = _swap_transform(2, src, 4)(x)
    assert out.shape == x.shape
    assert torch.equal(out[:, 2, :], src)
    assert torch.equal(out[:, :2, :], x[:, :2, :])
    assert torch.equal(out[:, 3:, :], x[:, 3:, :])
    assert out is not x
    # decode-step shapes (single token) pass through untouched
    decode = torch.zeros(1, 1, 3)
    assert _swap_transform(2, src, 4)(decode) is decode
    # a different full length is also untouched (prefill guard)
    other = torch.zeros(1, 5, 3)
    assert _swap_transform(2, src, 4)(other) is other


def _fake_scan_rows() -> list[dict]:
    rows = []
    for L in (1, 2, 3, 4):
        for p in ("entity", "evidence", "instruction", "prompt_end"):
            base = {"entity": 0.1, "evidence": 0.5, "instruction": 0.2, "prompt_end": 2.0}[p]
            for t in ("A", "B"):
                rows.append({"ticker": t, "group": "g", "layer": L, "position": p, "direction": "T1", "delta_m": base})
                rows.append({"ticker": t, "group": "g", "layer": L, "position": p, "direction": "T2", "delta_m": -base})
    return rows


def test_active_combos_topk_anchor_and_dedup():
    rows = _fake_scan_rows()
    combos = active_combos(rows, anchor_combo=(3, "instruction"))
    # top-3 are the three prompt_end layers (tied median 2.0 -> shallowest first)
    assert combos[:3] == [(1, "prompt_end"), (2, "prompt_end"), (3, "prompt_end")]
    assert (3, "instruction") in combos
    assert len(combos) <= 4
    # anchor already in top-k -> no duplicate
    combos2 = active_combos(rows, anchor_combo=(1, "prompt_end"))
    assert len(combos2) == 3
    assert len(set(combos2)) == 3


def test_prompt_positions_ordered_and_bounded():
    tokenizer = WordCharTokenizer()
    raw = build_prompt("T000", "Company T000", "N15")
    char_spans = prompt_char_spans(raw)
    spans = {
        "evidence": list(token_span(tokenizer, raw, char_spans["evidence"][0], char_spans["evidence"][1])),
        "instruction": list(token_span(tokenizer, raw, char_spans["instruction"][0], char_spans["instruction"][1])),
    }
    pos_raw, offset_raw = _prompt_positions(tokenizer, raw, raw, spans)
    pos_prefix, offset_prefix = _prompt_positions(tokenizer, raw + DECISION_PREFIX, raw, spans)
    assert offset_raw == offset_prefix == 0  # fake tokenizer: no BOS
    for pos in (pos_raw, pos_prefix):
        assert pos["entity"] < pos["evidence"] < pos["instruction"] <= pos["prompt_end"]
    from llm_bias.core.prompt_input.encoding import input_ids

    assert pos_raw["prompt_end"] == len(input_ids(tokenizer, raw)) - 1
    assert pos_prefix["prompt_end"] == len(input_ids(tokenizer, raw + DECISION_PREFIX)) - 1
    assert pos_prefix["prompt_end"] > pos_raw["prompt_end"]
    # span positions are prefix-stable for this tokenizer
    assert pos_prefix["entity"] == pos_raw["entity"]
    assert pos_prefix["evidence"] == pos_raw["evidence"]
    assert pos_prefix["instruction"] == pos_raw["instruction"]


def test_prompt_positions_bos_offset_and_mismatch():
    """Gemma BOS incident: stored Phase 1 spans from a BOS-less tokenizer
    must cross-check as a uniform +1 offset; non-uniform drift fails closed."""
    tokenizer = WordCharTokenizer()
    raw = build_prompt("T000", "Company T000", "N15")
    char_spans = prompt_char_spans(raw)
    spans = {
        "evidence": list(token_span(tokenizer, raw, char_spans["evidence"][0], char_spans["evidence"][1])),
        "instruction": list(token_span(tokenizer, raw, char_spans["instruction"][0], char_spans["instruction"][1])),
    }
    # stored spans recorded one token earlier (as if the prepare tokenizer
    # lacked the BOS that the inference tokenizer forces) -> offset +1
    stored_earlier = {name: [s, e - 1] for name, (s, e) in spans.items()}
    _, offset = _prompt_positions(tokenizer, raw, raw, stored_earlier)
    assert offset == 1
    # non-uniform drift fails closed
    bad = {name: list(v) for name, v in spans.items()}
    bad["instruction"] = [bad["instruction"][0], bad["instruction"][1] - 2]
    with pytest.raises(ValueError, match="cross-check failed"):
        _prompt_positions(tokenizer, raw, raw, bad)


# ---------------------------------------------------------------------------
# Intervention canary (hook plumbing)
# ---------------------------------------------------------------------------


def _margin(model, tokenizer, text: str, transforms=None) -> float:
    from llm_bias.core.continuation_scoring import continuation_token_ids

    from llm_bias.evidence_insensitivity.phase3 import _margin_at_end

    _, buy_ids = continuation_token_ids(tokenizer, text + DECISION_PREFIX, "buy")
    _, sell_ids = continuation_token_ids(tokenizer, text + DECISION_PREFIX, "sell")
    ids = torch.tensor([input_ids(tokenizer, text + DECISION_PREFIX)], dtype=torch.long)
    return _margin_at_end(model, ids, buy_ids[0], sell_ids[0], transforms)


def test_patch_only_affects_target_position():
    from llm_bias.core.inference.interventions import record_block_states

    tokenizer = WordCharTokenizer()
    model = FakePhase3Model()
    n15 = build_prompt("T000", "Company T000", "N15") + DECISION_PREFIX
    p15 = build_prompt("T000", "Company T000", "P15") + DECISION_PREFIX
    ids_n15 = torch.tensor([input_ids(tokenizer, n15)])
    ids_p15 = torch.tensor([input_ids(tokenizer, p15)])
    with torch.no_grad():
        states_p15 = record_block_states(model, ids_p15, layers=[N_LAYERS - 1, 2])
    base = _margin(model, tokenizer, build_prompt("T000", "Company T000", "N15"))
    final_pos = len(input_ids(tokenizer, n15)) - 1
    n15_len = ids_n15.shape[1]
    # T1 at the final position: margin must rise (P15 state carries more buy)
    d_final = _margin(model, tokenizer, build_prompt("T000", "Company T000", "N15"), {N_LAYERS - 1: _swap_transform(final_pos, states_p15[N_LAYERS - 1]["post"][0, final_pos, :], n15_len)}) - base
    assert d_final > 0.0
    # the same swap at an intermediate layer shifts the margin by the same amount
    # (per-layer drift cancels between baseline and patched runs)
    d_mid = _margin(model, tokenizer, build_prompt("T000", "Company T000", "N15"), {2: _swap_transform(final_pos, states_p15[2]["post"][0, final_pos, :], n15_len)}) - base
    assert d_mid == pytest.approx(d_final, abs=1e-3)
    # a swap at the entity header position (identical state across conditions)
    # leaves the final-position margin bit-identical; the fake has no attention,
    # so a swap at any non-final row is also invisible to the final margin
    d_entity = _margin(model, tokenizer, build_prompt("T000", "Company T000", "N15"), {N_LAYERS - 1: _swap_transform(5, states_p15[N_LAYERS - 1]["post"][0, 5, :], n15_len)}) - base
    assert d_entity == 0.0


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle_all_gates_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_lifecycle(tmp_path, monkeypatch)
    assert summary["n_companies"] == 84
    assert summary["group_sizes"] == {"evidence-responsive": 42, "fixed-sell": 42}
    assert summary["gates_passed"] is True
    for gate in ("G-3A", "G-3B", "G-3C"):
        assert summary["gates"][gate]["pass"] is True, summary["gates"][gate]
    assert summary["gates"]["G-3B"]["value"] > 0.0
    assert summary["gates"]["G-3C"]["value"]["parse_rate"] == 1.0
    # effect map: every (layer, position) x direction x group present
    for direction in ("T1", "T2"):
        for group in ("evidence-responsive", "fixed-sell"):
            entry = summary["effect_map_median_delta_m"][direction][group]
            assert len(entry) == len(FAKE_GRID) * 4
    # S-vs-R entries for each selected combo
    assert set(summary["s_vs_r"]) == {f"T1@{L}:{p}" for L, p in summary["active_combos"]}
    for entry in summary["s_vs_r"].values():
        fisher = entry["t1_flip_fisher"]
        # The behavior endpoint is based on paired generated decisions, not margin sign.
        assert all(
            entry["t1_flip_rate"][group]["valid_pair_count"] == 42
            for group in ("evidence-responsive", "fixed-sell")
        )
        # fake model produces zero flips -> guarded undefined-odds path
        assert fisher["p"] is None and fisher["flip"] == {"evidence-responsive": 0, "fixed-sell": 0}
        # fake model has identical per-company deltas -> zero variance -> Welch guarded out
        assert "t1_margin_welch" not in entry or all(
            np.isfinite(entry["t1_margin_welch"][k]) for k in ("t", "cohen_d")
        )
    # scan records compact and finite
    run_dir = tmp_path / SLUG / "evidence-insensitivity" / "runs" / "phase3-test"
    scan = [json.loads(line) for line in (run_dir / "forward" / "scan_records.jsonl").read_text().splitlines() if line]
    assert len(scan) == 84 * 2 * len(FAKE_GRID) * 4
    for row in scan:
        for key in ("baseline_m", "patched_m", "delta_m"):
            assert np.isfinite(row[key])
    # generation records: decisions parse, flip flag consistent
    gen = [json.loads(line) for line in (run_dir / "forward" / "generation_records.jsonl").read_text().splitlines() if line]
    combos = summary["active_combos"]
    assert len(gen) == 84 * 2 * len(combos)
    for row in gen:
        assert row["baseline_decision"] in ("buy", "sell")
        assert row["patched_decision"] in ("buy", "sell")
        expected_flip = row["patched_decision"] != row["baseline_decision"]
        assert row["flip"] is expected_flip
    # manifest complete
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    metadata = json.loads((run_dir / "forward" / "metadata.json").read_text())
    assert metadata["determinism_check"]["max_abs_delta_m"] == 0.0
    assert metadata["determinism_check"]["margin_mismatch_count"] == 0
    assert metadata["determinism_check"]["generation_mismatch_count"] == 0


def test_full_lifecycle_bf16_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_lifecycle(tmp_path, monkeypatch, run_id="phase3-bf16", out_dtype=torch.bfloat16)
    assert summary["gates_passed"] is True
    run_dir = tmp_path / SLUG / "evidence-insensitivity" / "runs" / "phase3-bf16"
    scan = [json.loads(line) for line in (run_dir / "forward" / "scan_records.jsonl").read_text().splitlines() if line]
    assert all(np.isfinite(r["delta_m"]) for r in scan)


def test_g3b_intervention_inefficacy_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(LAYER_GRIDS, SLUG, FAKE_GRID)
    tokenizer = WordCharTokenizer()
    _fake_phase1_run(tmp_path, tokenizer)
    phase3.run_phase3_prepare("phase3-g3b", artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug=SLUG)
    model = FakePhase3Model(lm_scale=0.0)  # zero unembedding -> margin always 0
    with pytest.raises(ValueError, match="G-3B intervention efficacy failed"):
        phase3.run_phase3_forward("phase3-g3b", artifact_root=tmp_path, model_path="fake", model=model, tokenizer=tokenizer, device="cpu", model_slug=SLUG)


def test_g3a_determinism_gate_fails_with_noise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_lifecycle(tmp_path, monkeypatch, run_id="phase3-noisy", noise=0.5)
    assert summary["gates"]["G-3A"]["pass"] is False
    assert summary["gates_passed"] is False
    run_dir = tmp_path / SLUG / "evidence-insensitivity" / "runs" / "phase3-noisy"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"  # gate is recorded, not raised


def test_forward_fails_closed_on_coordinate_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Gemma BOS incident regression: if the stored position table does not
    index the forward-time encodings (prepare/forward tokenizer drift),
    forward must fail closed before any intervention forward."""
    import json as _json
    from pathlib import Path as _Path

    monkeypatch.setitem(LAYER_GRIDS, SLUG, FAKE_GRID)
    tokenizer = WordCharTokenizer()
    _fake_phase1_run(tmp_path, tokenizer)
    phase3.run_phase3_prepare("phase3-drift", artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug=SLUG)
    sel_path = _Path(tmp_path) / SLUG / "evidence-insensitivity" / "runs" / "phase3-drift" / "prepare" / "selection.json"
    sel = _json.loads(sel_path.read_text())
    first_t = sel["sample"][0]["ticker"]
    sel["positions"][first_t]["N15"]["prefix"]["prompt_end"] -= 1  # simulate forced-BOS drift
    sel_path.write_text(_json.dumps(sel))
    model = FakePhase3Model()
    with pytest.raises(ValueError, match="coordinate drift"):
        phase3.run_phase3_forward("phase3-drift", artifact_root=tmp_path, model_path="fake", model=model, tokenizer=tokenizer, device="cpu", model_slug=SLUG)


def test_load_tokenizer_for_inference_applies_force_bos(monkeypatch: pytest.MonkeyPatch):
    from llm_bias.core import model as core_model

    class _Stub:
        def __init__(self, bos_id):
            self.bos_token_id = bos_id
            self.add_bos_token = False

    monkeypatch.setattr(core_model, "resolve_model_name", lambda model: model)
    monkeypatch.setattr(core_model.transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: _Stub(2 if name == "with-bos" else None)))
    with_bos = core_model.load_tokenizer_for_inference("with-bos")
    assert with_bos.add_bos_token is True
    without = core_model.load_tokenizer_for_inference("no-bos")
    assert without.add_bos_token is False


def test_prepare_position_out_of_bounds_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(LAYER_GRIDS, SLUG, FAKE_GRID)
    tokenizer = WordCharTokenizer()
    run_dir = _fake_phase1_run(tmp_path, tokenizer)
    rows = [json.loads(line) for line in (run_dir / "prepare" / "prompts.jsonl").read_text().splitlines() if line]
    rows[0]["instruction_span"] = [0, 100000]
    with open(run_dir / "prepare" / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="cross-check failed"):
        phase3.run_phase3_prepare("phase3-badpos", artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug=SLUG)


def test_prepare_missing_layer_grid_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tokenizer = WordCharTokenizer()
    _fake_phase1_run(tmp_path, tokenizer)
    with pytest.raises(ValueError, match="no frozen layer grid"):
        phase3.run_phase3_prepare("phase3-nogrid", artifact_root=tmp_path, model_path="fake", tokenizer=tokenizer, model_slug="unknown-model")


def test_summary_serializes_through_core_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summary = _run_lifecycle(tmp_path, monkeypatch, run_id="phase3-json")
    from llm_bias.core.artifacts.io import write_json

    write_json(tmp_path / "summary-check.json", summary, overwrite=True)
    reloaded = json.loads((tmp_path / "summary-check.json").read_text())
    assert reloaded["gates_passed"] is True
