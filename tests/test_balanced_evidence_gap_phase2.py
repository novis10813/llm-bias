"""Regression tests for the Phase 2 (cross-entity probe / patching) package.

All model-dependent tests use deterministic CPU fake models and a
character-level fake tokenizer; no checkpoint is loaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_bias.balanced_evidence_gap import patch_pipeline
from llm_bias.balanced_evidence_gap import pipeline as phase2a_pipeline
from llm_bias.balanced_evidence_gap.analysis import (
    bootstrap_ci,
    detect_handoff,
    exact_sign_flip_p,
    evaluate_gate_2a,
    evaluate_gate_2a_rev2,
    evaluate_gate_2c,
    holm_adjusted,
    iqr,
    normalized_transfer,
    select_margin_groups,
    spearman,
    toward_source_delta,
)
from llm_bias.balanced_evidence_gap.rev2 import run_rev2_gate
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.balanced_evidence_gap.intervention import (
    AttentionEdgeZeroing,
    answer_token_ids,
    capture_mlp_channel,
    capture_residuals,
    make_span_transform,
    margin_from_log_probs,
    mlp_margin_attribution,
    nearest_position_mapping,
    patched_final_margin,
    random_match_positions,
)
from llm_bias.balanced_evidence_gap.spans import prompt_char_spans, resolve_row
from llm_bias.balanced_evidence_gap.template import (
    ALL_TICKERS,
    DIAL_LAYER,
    EVIDENCE_N1,
    EVIDENCE_N2,
    EVIDENCE_P1,
    EVIDENCE_P2,
    build_prompt,
    variant_id,
)

BUY_ID = 240
SELL_ID = 241
# Must cover ord values of the frozen template's unicode dashes (— = 8212).
VOCAB = 9000
WIDTH = 1
N_LAYERS_16 = 16


# ── fakes ────────────────────────────────────────────────────────────────────


class _CharTokenizer:
    """Character tokenizer with single-token buy/sell continuations."""

    chat_template = "fake"

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_special_tokens_mask: bool = False,
        **_kwargs,
    ) -> SimpleNamespace:
        del add_special_tokens
        text = str(text)
        suffix = None
        for candidate, token_id in (
            ("buy", BUY_ID),
            ("sell", SELL_ID),
        ):
            if text.endswith(candidate):
                suffix = (candidate, token_id)
                break
        if suffix is None:
            values = [ord(char) for char in text]
        else:
            candidate, token_id = suffix
            prefix = text[: -len(candidate)]
            values = [ord(char) for char in prefix] + [token_id]
        if return_offsets_mapping:
            prefix_len = len(text) - (len(suffix[0]) if suffix else 0)
            offsets = [(i, i + 1) for i in range(prefix_len)]
            if suffix:
                offsets.append((prefix_len, len(text)))
            return SimpleNamespace(
                input_ids=values,
                offset_mapping=offsets,
                special_tokens_mask=[False] * len(values),
            )
        return SimpleNamespace(input_ids=values)

    def apply_chat_template(self, messages, **_kwargs):
        return "«" + messages[0]["content"] + "»"

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(i) for i in token_ids if i < 128)


class _FakeRMSNorm(nn.Module):
    variance_epsilon = 1e-6

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(d_model))

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight


class _SumLayer(nn.Module):
    """Bounded position mixing: stays finite over 16 stacked layers."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.seen = hidden.detach().clone()
        return hidden * 0.5 + hidden.sum(dim=1, keepdim=True) * 0.001


class _PatchingModel(nn.Module):
    """CPU-only decoder whose final margin follows the residual sum."""

    def __init__(self, n_layers: int = N_LAYERS_16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
        self.layers = nn.ModuleList([_SumLayer() for _ in range(n_layers)])
        self.n_layers = n_layers
        self.input_device = "cpu"
        self._final_norm = _FakeRMSNorm(WIDTH)
        self._lm_head = nn.Linear(WIDTH, VOCAB, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[ord("a"), 0] = 1.0
            self.embedding.weight[ord("z"), 0] = -1.0
            self._lm_head.weight.zero_()
            self._lm_head.weight[BUY_ID, 0] = 1.0
            self._lm_head.weight[SELL_ID, 0] = -1.0

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Mlp(nn.Module):
    def __init__(self, width: int, mlp_width: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(width, mlp_width, bias=False)
        self.down_proj = nn.Linear(mlp_width, width, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.relu(self.up_proj(hidden)))


class _MlpBlock(nn.Module):
    def __init__(self, width: int, mlp_width: int) -> None:
        super().__init__()
        self.mlp = _Mlp(width, mlp_width)

    def forward(self, hidden):
        return hidden + self.mlp(hidden)


class _MlpModel(nn.Module):
    def __init__(self, n_layers: int = 2, width: int = 4, mlp_width: int = 32) -> None:
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(VOCAB, width)
        self.layers = nn.ModuleList(
            [_MlpBlock(width, mlp_width) for _ in range(n_layers)]
        )
        self.n_layers = n_layers
        self.input_device = "cpu"
        self._final_norm = _FakeRMSNorm(width)
        self._lm_head = nn.Linear(width, VOCAB, bias=False)
        torch.manual_seed(11)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeNorm2(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = 1e-6

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)


class _FakeGQA(nn.Module):
    """Qwen3.5-style gated GQA attention (fp32-exact reconstruction)."""

    def __init__(self, broken: bool = False) -> None:
        super().__init__()
        self.broken = broken
        self.head_dim = 2
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.scaling = 2**-0.5
        self.q_proj = nn.Linear(5, 16, bias=False)
        self.k_proj = nn.Linear(5, 4, bias=False)
        self.v_proj = nn.Linear(5, 4, bias=False)
        self.o_proj = nn.Linear(8, 5, bias=False)
        self.q_norm = _FakeNorm2(2)
        self.k_norm = _FakeNorm2(2)
        torch.manual_seed(4)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        batch, sequence, _ = hidden_states.shape
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch, sequence, 4, 4), 2, dim=-1
        )
        key = self.k_norm(self.k_proj(hidden_states).view(batch, sequence, 2, 2)).transpose(1, 2)
        val = self.v_proj(hidden_states).view(batch, sequence, 2, 2).transpose(1, 2)
        query = self.q_norm(query).transpose(1, 2)
        key = key.repeat_interleave(2, dim=1)
        val = val.repeat_interleave(2, dim=1)
        scores = query @ key.transpose(-1, -2) * self.scaling
        causal = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], -float("inf"))
        weights = scores.softmax(-1)
        output = (weights @ val).transpose(1, 2).reshape(batch, sequence, 8)
        output = output * gate.sigmoid().reshape(batch, sequence, 8)
        if self.broken:
            output = output + 10.0
        return self.o_proj(output), weights


class _FullAttnBlock(nn.Module):
    block_type = "full_attention"

    def __init__(self, broken: bool = False) -> None:
        super().__init__()
        self.self_attn = _FakeGQA(broken=broken)
        self.in_proj = nn.Linear(1, 5, bias=False)
        self.out_proj = nn.Linear(5, 1, bias=False)
        torch.manual_seed(5)

    def forward(self, hidden):
        attention_out, _ = self.self_attn(self.in_proj(hidden))
        return hidden + self.out_proj(attention_out)


class _LinearBlock(nn.Module):
    block_type = "linear_attention"

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(6)

    def forward(self, hidden):
        return hidden


class _AttnModel(nn.Module):
    def __init__(self, broken: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 1)
        self.layers = nn.ModuleList(
            [_LinearBlock(), _LinearBlock(), _LinearBlock(), _FullAttnBlock(broken)]
        )
        self.n_layers = 4
        self.input_device = "cpu"
        self._final_norm = _FakeRMSNorm(1)
        self._lm_head = nn.Linear(1, VOCAB, bias=False)
        with torch.no_grad():
            self._lm_head.weight.zero_()
            self._lm_head.weight[BUY_ID, 0] = 1.0
            self._lm_head.weight[SELL_ID, 0] = -1.0

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_load_model(model: nn.Module, tokenizer: _CharTokenizer):
    def _load(model_path, *, dtype=None):
        return model, tokenizer, torch.device("cpu")

    return _load


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _identity_format(tokenizer, prompt, **_kwargs):
    return "«" + prompt + "»"


def _probe_rows(tokenizer: _CharTokenizer) -> tuple[dict, dict]:
    source = resolve_row(
        tokenizer, "AMAT", "Applied Materials", "Industrials",
        reverse=False, order=0, format_fn=_identity_format,
    )
    target = resolve_row(
        tokenizer, "DE", "Deere & Co", "Industrials",
        reverse=False, order=0, format_fn=_identity_format,
    )
    return source, target


# ── template ─────────────────────────────────────────────────────────────────


def test_frozen_evidence_sentences_appear_verbatim():
    prompt = build_prompt("C", "Citigroup", 0, False)
    for sentence in (EVIDENCE_P1, EVIDENCE_P2, EVIDENCE_N1, EVIDENCE_N2):
        assert sentence in prompt
    assert "Stock Ticker: [C]" in prompt
    assert "Stock Name: [Citigroup]" in prompt
    assert '"buy" or "sell"' in prompt


def test_variant_grid_is_16_x_2_x_2():
    orders = (0, 1)
    reverses = (False, True)
    seen = set()
    for ticker in ALL_TICKERS:
        for order in orders:
            for reverse in reverses:
                seen.add(variant_id(ticker, reverse, order))
    assert len(ALL_TICKERS) == 16
    assert len(seen) == 64
    ordered = build_prompt("C", "Citigroup", 0, False)
    reversed_order = build_prompt("C", "Citigroup", 1, False)
    assert ordered.index(EVIDENCE_P1) < ordered.index(EVIDENCE_N2)
    assert reversed_order.index(EVIDENCE_N2) < reversed_order.index(EVIDENCE_P1)
    reversed_options = build_prompt("C", "Citigroup", 0, True)
    assert '"sell" or "buy"' in reversed_options


# ── spans ────────────────────────────────────────────────────────────────────


def test_prompt_char_spans_ordered_and_covered():
    prompt = build_prompt("AMAT", "Applied Materials", 0, False)
    spans = prompt_char_spans(prompt)
    entity, evidence, instruction = spans["entity"], spans["evidence"], spans["instruction"]
    assert entity[0] < entity[1] < evidence[0] < evidence[1] < instruction[0] < instruction[1]
    assert prompt[entity[0] : entity[1]].startswith("Stock Ticker: [AMAT]")
    assert EVIDENCE_P1 in prompt[evidence[0] : evidence[1]]
    assert prompt[instruction[0] :].startswith("Your final response")


def test_resolve_row_maps_token_spans_and_positions():
    tokenizer = _CharTokenizer()
    source, target = _probe_rows(tokenizer)
    for row in (source, target):
        assert row["final_position"] == len(row["prompt_ids"]) - 1
        assert row["entity_position"] == row["entity_span"][1] - 1
        assert row["entity_span"][0] < row["evidence_span"][0] < row["instruction_span"][0]
    # "DE" / "Deere & Co" are shorter than "AMAT" / "Applied Materials".
    assert (
        target["entity_span"][1] - target["entity_span"][0]
        < source["entity_span"][1] - source["entity_span"][0]
    )
    # Evidence text is shared, so evidence spans have equal token length.
    assert target["evidence_span"][1] - target["evidence_span"][0] == (
        source["evidence_span"][1] - source["evidence_span"][0]
    )


# ── analysis pure functions ─────────────────────────────────────────────────


def test_iqr_and_bootstrap_ci_known_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert iqr(values) == pytest.approx(3.5)
    lower, upper = bootstrap_ci(values, n=500, seed=42)
    assert 2.5 < lower < upper < 6.5


def test_spearman_perfect_and_anticorrelated():
    assert spearman([1, 2, 3, 4], [4, 9, 16, 25]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [25, 16, 9, 4]) == pytest.approx(-1.0)


def test_exact_sign_flip_p_extremes():
    assert exact_sign_flip_p([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1 / 16)
    assert exact_sign_flip_p([1.0, -1.0, 1.0, -1.0]) == pytest.approx(11 / 16)
    with pytest.raises(ValueError, match="nonzero"):
        exact_sign_flip_p([0.0, 0.0])


def test_holm_adjusted_is_monotone_and_capped():
    adjusted = holm_adjusted([0.01, 0.02, 0.5])
    assert adjusted == [pytest.approx(0.03), pytest.approx(0.04), pytest.approx(0.5)]
    assert max(holm_adjusted([0.9, 0.9, 0.9])) <= 1.0


def test_toward_source_delta_and_normalized_transfer_signs():
    # source above target: patched above target is positive.
    assert toward_source_delta(0.5, 2.0, -1.0) > 0
    assert toward_source_delta(-1.5, 2.0, -1.0) < 0
    assert normalized_transfer(-0.5, 2.0, -1.0) == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="identical"):
        toward_source_delta(0.0, 1.0, 1.0)


def _gate_2a_inputs(**overrides):
    tickers = [f"T{i:02d}" for i in range(12)]
    pure = {t: float(i) for i, t in enumerate(tickers)}
    phase1 = {t: float(i) + 0.01 for i, t in enumerate(tickers)}
    inputs = dict(
        pure_entity_margins=pure,
        phase1_named_margins=phase1,
        framing_pair_deltas=[0.2] * 8,
        valid_rate=1.0,
    )
    inputs.update(overrides)
    return inputs


def test_evaluate_gate_2a_pass_and_failure_modes():
    gate = evaluate_gate_2a(**_gate_2a_inputs())
    assert gate["pass"] and gate["phase2b_authorized"]

    tight = evaluate_gate_2a(**_gate_2a_inputs(pure_entity_margins={f"T{i:02d}": 0.05 * i for i in range(12)}))
    assert tight["criteria"]["iqr"]["pass"] is False and tight["pass"] is False

    noisy = {f"T{i:02d}": (10.0 if i % 2 else -10.0) for i in range(12)}
    anticorrelated = evaluate_gate_2a(**_gate_2a_inputs(pure_entity_margins=noisy))
    assert anticorrelated["criteria"]["spearman_vs_phase1"]["pass"] is False

    framed = evaluate_gate_2a(**_gate_2a_inputs(framing_pair_deltas=[2.0] * 8))
    assert framed["criteria"]["framing_stability"]["pass"] is False


def _gate_2a_rev2_inputs(**overrides):
    tickers = [f"T{i:02d}" for i in range(16)]
    pure = {t: float(i) for i, t in enumerate(tickers)}
    gaps = {t: float(i) + 0.1 for i, t in enumerate(tickers)}
    inputs = dict(
        pure_entity_margins=pure,
        phase1_gaps=gaps,
        framing_pair_deltas=[0.2] * 8,
        valid_rate=1.0,
    )
    inputs.update(overrides)
    return inputs


def test_select_margin_groups_frozen_rule_and_fail_closed():
    pure = {f"T{i:02d}": float(i) for i in range(5)}
    bottom, top = select_margin_groups(pure)
    assert bottom == ["T00", "T01"] and top == ["T03", "T04"]
    with pytest.raises(ValueError):
        select_margin_groups({f"T{i:02d}": float(i) for i in range(4)})


def test_evaluate_gate_2a_rev2_pass_and_failure_modes():
    gate = evaluate_gate_2a_rev2(**_gate_2a_rev2_inputs())
    assert gate["gate"] == "2A-rev2"
    assert gate["pass"] and gate["phase2b_authorized"]
    group = gate["criteria"]["group_construct_check"]
    assert group["top"] == ["T14", "T15"] and group["bottom"] == ["T00", "T01"]
    assert group["n_positive"] == group["n_pairs"] == 4

    tight = evaluate_gate_2a_rev2(**_gate_2a_rev2_inputs(
        pure_entity_margins={f"T{i:02d}": 0.05 * i for i in range(16)}))
    assert tight["criteria"]["iqr"]["pass"] is False and tight["pass"] is False

    inverted = evaluate_gate_2a_rev2(**_gate_2a_rev2_inputs(
        phase1_gaps={f"T{i:02d}": 15.0 - i for i in range(16)}))
    assert inverted["criteria"]["spearman_vs_phase1_gap"]["pass"] is False
    assert inverted["criteria"]["group_construct_check"]["pass"] is False
    assert inverted["pass"] is False

    # one inverted pair inside the groups, full ranking still correlated
    gaps = {f"T{i:02d}": float(i) + 0.1 for i in range(16)}
    gaps["T01"], gaps["T14"] = gaps["T14"], gaps["T01"]
    group_only = evaluate_gate_2a_rev2(**_gate_2a_rev2_inputs(phase1_gaps=gaps))
    assert group_only["criteria"]["spearman_vs_phase1_gap"]["pass"] is True
    assert group_only["criteria"]["group_construct_check"]["pass"] is False
    assert group_only["pass"] is False

    framed = evaluate_gate_2a_rev2(**_gate_2a_rev2_inputs(framing_pair_deltas=[2.0] * 8))
    assert framed["criteria"]["framing_stability"]["pass"] is False

    invalid = evaluate_gate_2a_rev2(**_gate_2a_rev2_inputs(valid_rate=0.99))
    assert invalid["criteria"]["schema_valid_rate"]["pass"] is False


def test_run_rev2_gate_reanalysis(tmp_path):
    import hashlib

    base = {t: -2.0 + 0.1 * i for i, t in enumerate(ALL_TICKERS)}
    rows = []
    for t in ALL_TICKERS:
        for reverse in (False, True):
            for order in (0, 1):
                margin = base[t] + (0.05 if reverse else 0.0) + (0.02 if order else 0.0)
                rows.append({
                    "id": f"{t}-r{int(reverse)}-o{order}",
                    "ticker": t,
                    "margin": margin,
                    "reverse": reverse,
                    "order": order,
                    "decision": "sell",
                })
    phase2a_run = tmp_path / "phase2a-gpu-bf16-01"
    (phase2a_run / "forward").mkdir(parents=True)
    results_path = phase2a_run / "forward" / "results.jsonl"
    results_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    phase1_summary = tmp_path / "phase1-summary.json"
    phase1_summary.write_text(json.dumps({
        "per_company": {
            t: {"gap_mean": base[t], "named_margin_median": base[t] - 0.5}
            for t in ALL_TICKERS
        }
    }), encoding="utf-8")

    run_root = run_rev2_gate(
        model_name="fake-model",
        phase2a_run=phase2a_run,
        phase1_summary=phase1_summary,
        run_id="rev2-gate-test-01",
        artifact_root=tmp_path / "artifacts",
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert {s: v["status"] for s, v in manifest["stages"].items()} == {
        "prepare": "complete", "analyze": "complete",
    }
    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_2a_rev2"]["pass"] is True
    assert summary["gate_2a_rev2"]["phase2b_authorized"] is True
    assert summary["n_prompts"] == 64
    prov = json.loads((run_root / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["phase2a_run"]["results_sha256"] == hashlib.sha256(results_path.read_bytes()).hexdigest()
    assert prov["phase2a_run"]["n_records"] == 64
    assert prov["reanalysis_only"] is True


def test_detect_handoff_crossover_band_and_fallback():
    layers = list(range(16))
    entity_t = {l: 0.9 - 0.03 * l for l in layers}
    context_t = {l: 0.5 + 0.04 * l for l in layers}
    result = detect_handoff(layers, entity_t, context_t)
    assert result["crossover"]
    assert result["crossover_band"] == list(range(6, 16))
    assert result["interval"] == list(range(5, 16))
    assert result["attention_layers"] == [7, 11, 15]
    assert result["mlp_layers"] == list(range(5, 16))

    flat = detect_handoff(
        layers, {l: 1.0 - 0.01 * l for l in layers}, {l: 0.1 for l in layers}
    )
    assert flat["crossover"] is False
    assert flat["fallback"] == "maximal_entity_transfer"
    assert flat["peaks"] == [0, 1, 2]
    assert flat["interval"] == [0, 1, 2, 3]
    assert flat["mlp_layers"] == [0, 1, 2, 3]
    assert flat["attention_layers"] == [3]


def test_evaluate_gate_2c_attention_and_mlp_semantics():
    # MLP arm: per-layer existence test; a structurally-zero layer must not
    # contaminate the verdict (regression: cross-layer min/max aggregation).
    mlp_arm = {
        "per_layer": {
            "12": {"top_neuron": 1, "top_spearman": 0.9, "abs_top_spearman": 0.9,
                   "control_max_abs_rho": 0.6, "sector_agreement": 1.0,
                   "sign_flip_p": 0.002, "sign_flip_p_adjusted": 0.04},
            "31": {"top_neuron": 0, "top_spearman": 0.0, "abs_top_spearman": 0.0,
                   "control_max_abs_rho": 0.0, "sector_agreement": 0.0,
                   "sign_flip_p": 1.0, "sign_flip_p_adjusted": 1.0},
        }
    }
    gate = evaluate_gate_2c(attention_arm=None, mlp_arm=mlp_arm)
    assert gate["attention_arm"]["status"] == "not_applicable"
    assert gate["mlp_arm"]["passing_layers"] == [12]
    assert gate["mlp_arm"]["pass"] is True
    assert gate["pass"] is True

    # each per-layer criterion is necessary
    bad_sector = {"12": dict(mlp_arm["per_layer"]["12"], sector_agreement=0.5)}
    assert evaluate_gate_2c(attention_arm=None, mlp_arm={"per_layer": bad_sector})["pass"] is False
    bad_p = {"12": dict(mlp_arm["per_layer"]["12"], sign_flip_p_adjusted=0.2)}
    assert evaluate_gate_2c(attention_arm=None, mlp_arm={"per_layer": bad_p})["pass"] is False
    bad_top = {"12": dict(mlp_arm["per_layer"]["12"], abs_top_spearman=0.5)}
    assert evaluate_gate_2c(attention_arm=None, mlp_arm={"per_layer": bad_top})["pass"] is False
    assert evaluate_gate_2c(attention_arm=None, mlp_arm={"per_layer": {}})["pass"] is False

    # attention arm: existence — a non-top head may pass while the top head fails
    attention = {
        "status": "run",
        "head_effects": {
            "L3H0": {"mean_delta": 0.5, "direction_deltas": [0.5, 0.5], "control_mean_delta": 0.0},
            "L3H1": {"mean_delta": 0.4, "direction_deltas": [0.4, 0.4], "control_mean_delta": 0.0},
        },
        "holm_adjusted_p": {"L3H0": 0.25, "L3H1": 0.01},
        "top_head": "L3H0",
    }
    gate2 = evaluate_gate_2c(attention_arm=attention, mlp_arm={"per_layer": {}})
    assert gate2["attention_arm"]["top_head"] == "L3H0"
    assert gate2["attention_arm"]["passing_heads"] == ["L3H1"]
    assert gate2["attention_arm"]["pass"] is True
    assert gate2["pass"] is True

    attention_all_fail = dict(attention, holm_adjusted_p={"L3H0": 0.25, "L3H1": 1.0})
    gate3 = evaluate_gate_2c(attention_arm=attention_all_fail, mlp_arm={"per_layer": {}})
    assert gate3["attention_arm"]["pass"] is False
    assert gate3["pass"] is False


# ── intervention primitives ──────────────────────────────────────────────────


def test_nearest_position_mapping_identity_by_offset_and_nearest():
    assert nearest_position_mapping((2, 6), (5, 9)) == {5: 2, 6: 3, 7: 4, 8: 5}
    mapping = nearest_position_mapping((0, 4), (10, 12))
    assert set(mapping) == {10, 11}
    assert set(mapping.values()) <= {0, 3}


def test_make_span_transform_self_source_is_exact_noop():
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    source, target = _probe_rows(tokenizer)
    scoring = target["formatted"] + '{"decision": "'
    prompt_ids, buy_id, sell_id = answer_token_ids(tokenizer, scoring)
    tensor = torch.tensor([prompt_ids], dtype=torch.long)

    residuals = capture_residuals(model, tensor, [3])
    mapping = {pos: pos for pos in range(target["entity_span"][0], target["entity_span"][1])}
    transform = make_span_transform(residuals[3], mapping)
    log_probs = patched_final_margin(model, tensor, {3: transform})
    patched = margin_from_log_probs(log_probs, buy_id, sell_id)

    clean_log_probs = patched_final_margin(model, tensor, {})
    clean = margin_from_log_probs(clean_log_probs, buy_id, sell_id)
    assert patched == pytest.approx(clean, abs=1e-12)
    assert not model.layers[3]._forward_hooks


def test_span_transfer_moves_margin_toward_source():
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    source, target = _probe_rows(tokenizer)
    scoring_s = source["formatted"] + '{"decision": "'
    scoring_t = target["formatted"] + '{"decision": "'
    tensor_s = torch.tensor([answer_token_ids(tokenizer, scoring_s)[0]], dtype=torch.long)
    tensor_t = torch.tensor([answer_token_ids(tokenizer, scoring_t)[0]], dtype=torch.long)
    _, buy_id, sell_id = answer_token_ids(tokenizer, scoring_t)

    m_source = margin_from_log_probs(patched_final_margin(model, tensor_s, {}), buy_id, sell_id)
    m_target = margin_from_log_probs(patched_final_margin(model, tensor_t, {}), buy_id, sell_id)
    if m_source == m_target:
        pytest.skip("fake margins coincide; no contrast")

    residuals = capture_residuals(model, tensor_s, [7])
    mapping = nearest_position_mapping(
        tuple(source["entity_span"]), tuple(target["entity_span"])
    )
    log_probs = patched_final_margin(model, tensor_t, {7: make_span_transform(residuals[7], mapping)})
    patched = margin_from_log_probs(log_probs, buy_id, sell_id)
    assert toward_source_delta(patched, m_source, m_target) > 0
    assert not model.layers[7]._forward_hooks


def test_attention_edge_zeroing_reconstructs_and_zeroes_one_head():
    model = _AttnModel()
    o_probe: dict = {}
    tensor = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("d"), ord("a")]], dtype=torch.long)
    zero_positions = (0, 1)
    with AttentionEdgeZeroing(model, 3, head=1, zero_positions=zero_positions, query_position=4) as session:
        with model.layers[3].self_attn.o_proj.register_forward_pre_hook(
            lambda module, args: o_probe.update({"zeroed": args[0].clone()})
        ):
            model(tensor)
    with model.layers[3].self_attn.o_proj.register_forward_pre_hook(
        lambda module, args: o_probe.update({"clean": args[0].clone()})
    ):
        model(tensor)
    assert session.reconstruction_error is not None and session.reconstruction_error < 1e-3
    clean, zeroed = o_probe["clean"], o_probe["zeroed"]
    head_slice = slice(1 * 2, 2 * 2)
    assert not torch.allclose(zeroed[0, 4, head_slice], clean[0, 4, head_slice])
    assert torch.allclose(zeroed[0, 4, slice(0, 2)], clean[0, 4, slice(0, 2)])
    assert torch.allclose(zeroed[:, :4], clean[:, :4])
    assert not model.layers[3].self_attn.o_proj._forward_pre_hooks


def test_attention_edge_zeroing_fails_closed_on_structural_mismatch():
    model = _AttnModel(broken=True)
    tensor = torch.tensor([[ord("a"), ord("b"), ord("c")]], dtype=torch.long)
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        with AttentionEdgeZeroing(model, 3, head=0, zero_positions=(0,), query_position=2):
            model(tensor)


def test_attention_edge_zeroing_rejects_invalid_configuration():
    model = _AttnModel()
    with pytest.raises(ValueError, match="full-attention"):
        AttentionEdgeZeroing(model, 0, head=0, zero_positions=(0,), query_position=2)
    with pytest.raises(ValueError, match="strictly before"):
        AttentionEdgeZeroing(model, 3, head=0, zero_positions=(2,), query_position=2)
    with pytest.raises(ValueError, match="head out of range"):
        with AttentionEdgeZeroing(model, 3, head=99, zero_positions=(0,), query_position=2):
            pass


def test_random_match_positions_matched_and_deterministic():
    sets = random_match_positions(10, (2, 3, 4), n_samples=5, seed=42)
    assert len(sets) == 5
    for sample in sets:
        assert len(sample) == 3
        assert all(0 <= p < 10 for p in sample)
        assert not set(sample) & {2, 3, 4}
    # regression: the query position itself must never be a zero position
    for seed in range(20):
        for sample in random_match_positions(10, (2, 3, 4), n_samples=10, seed=seed):
            assert 10 not in sample
    assert random_match_positions(10, (2, 3, 4), n_samples=5, seed=42) == sets
    assert random_match_positions(10, (2, 3, 4), n_samples=5, seed=43) != sets
    with pytest.raises(ValueError, match="not enough"):
        random_match_positions(3, (0, 1, 2, 3), n_samples=1, seed=42)


def test_capture_mlp_channel_records_one_coordinate():
    model = _MlpModel()
    tensor = torch.tensor([[ord("a"), ord("b")]], dtype=torch.long)
    with torch.no_grad():
        value = capture_mlp_channel(model, tensor, 1, 5, position=1)
    assert isinstance(value, float) and torch.isfinite(torch.tensor(value))
    with pytest.raises(ValueError, match="out of range"):
        capture_mlp_channel(model, tensor, 1, 999, position=1)


def test_mlp_margin_attribution_matches_numeric_gradient():
    torch.manual_seed(0)
    model = _MlpModel(n_layers=2, width=4, mlp_width=32)
    position = 1
    tensor = torch.tensor([[ord("a"), ord("b")]], dtype=torch.long)
    result = mlp_margin_attribution(
        model, tensor, layer=1, position=position, buy_id=BUY_ID, sell_id=SELL_ID
    )
    assert result["attribution"].shape == (32,)
    assert (result["attribution"] >= 0).all()
    assert torch.isfinite(torch.tensor(result["margin"]))

    # Independent gradient: rerun the model with the down-proj input at the
    # position replaced by a leaf, then backprop the FP32 tail margin.
    final_box: dict = {}

    def leaf_margin(leaf: torch.Tensor) -> None:
        def down_hook(_module, args):
            replaced = args[0].clone()
            replaced[0, position] = leaf
            return (replaced, *args[1:])

        handle_down = model.layers[1].mlp.down_proj.register_forward_pre_hook(down_hook)
        handle_final = model.layers[1].register_forward_hook(
            lambda _m, _i, output: final_box.update(residual=output)
        )
        try:
            with torch.enable_grad():
                model(tensor)
                hidden = final_box["residual"][0, -1].float()
                normalized = model._final_norm(hidden)
                contrast = (
                    model._lm_head.weight[BUY_ID].float()
                    - model._lm_head.weight[SELL_ID].float()
                )
                torch.dot(normalized, contrast).backward()
        finally:
            handle_down.remove()
            handle_final.remove()

    # Activation from the actual forward at the down-proj input.
    captured: dict = {}
    capture_handle = model.layers[1].mlp.down_proj.register_forward_pre_hook(
        lambda _m, args: captured.update(values=args[0])
    )
    with torch.no_grad():
        model(tensor)
    capture_handle.remove()
    activation = captured["values"][0, position].float()

    leaf = activation.clone().requires_grad_(True)
    leaf_margin(leaf)
    expected_attribution = (leaf.grad.float() * activation).abs()

    assert result["margin"] == pytest.approx(
        float(
            torch.dot(
                model._final_norm(final_box["residual"][0, -1].float()),
                (
                    model._lm_head.weight[BUY_ID].float()
                    - model._lm_head.weight[SELL_ID].float()
                ),
            )
        ),
        abs=1e-5,
    )
    assert torch.allclose(result["attribution"], expected_attribution, atol=1e-4)


# ── 2A pipeline (fake model) ─────────────────────────────────────────────────


def _write_fake_input(tmp_path: Path) -> Path:
    companies = [
        {"ticker": t, "name": f"Name {t}", "split": "test"} for t in ALL_TICKERS
    ]
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"companies": companies}), encoding="utf-8")
    return path


def _write_fake_phase1_summary(tmp_path: Path) -> Path:
    per_company = {
        t: {"named_margin_median": float(i)} for i, t in enumerate(ALL_TICKERS)
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"per_company": per_company}), encoding="utf-8")
    return path


def test_run_phase2a_smoke_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    model = _PatchingModel()
    monkeypatch.setattr(phase2a_pipeline, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(phase2a_pipeline, "load_model", _fake_load_model(model, tokenizer))

    run_root = phase2a_pipeline.run_phase2a(
        model_path="fake-model",
        run_id="smoke-fake",
        artifact_root=tmp_path / "artifacts",
        input_data=_write_fake_input(tmp_path),
        phase1_summary=_write_fake_phase1_summary(tmp_path),
        smoke=True,
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    prompts = (run_root / "prepare" / "prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    results = (run_root / "forward" / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(prompts) == 4 and len(results) == 4  # 4 tickers x 1 variant
    row = json.loads(results[0])
    assert isinstance(row["margin"], float)
    assert row["dial_channel_entity"] is None  # dial skipped in smoke
    assert not (run_root / "analyze").exists()


class _DialMlpBlock(_MlpBlock):
    """Mlp block plus global average mixing so positions couple (fake attention)."""

    def forward(self, hidden):
        hidden = hidden + hidden.mean(dim=1, keepdim=True) * 0.1
        return super().forward(hidden)


class _DialModel(nn.Module):
    """16-layer fake whose block-15 MLP input is dial-wide (neuron 8490)."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 4)
        self.layers = nn.ModuleList([_DialMlpBlock(4, 9216) for _ in range(DIAL_LAYER + 1)])
        self.n_layers = DIAL_LAYER + 1
        self.input_device = "cpu"
        self._final_norm = _FakeRMSNorm(4)
        self._lm_head = nn.Linear(4, VOCAB, bias=False)
        torch.manual_seed(3)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_run_phase2a_formal_analysis_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    model = _DialModel()
    monkeypatch.setattr(phase2a_pipeline, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(phase2a_pipeline, "load_model", _fake_load_model(model, tokenizer))

    run_root = phase2a_pipeline.run_phase2a(
        model_path="fake-model",
        run_id="formal-fake",
        artifact_root=tmp_path / "artifacts",
        input_data=_write_fake_input(tmp_path),
        phase1_summary=_write_fake_phase1_summary(tmp_path),
        smoke=False,
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_prompts"] == 64
    assert set(summary["pure_entity_margin_median"]) == set(ALL_TICKERS)
    assert "gate_2a" in summary and "h4_dial" in summary
    assert summary["h4_dial"]["coordinate"] == [15, 8490]
    assert isinstance(summary["h4_dial"]["entity_position_pearson"], float)


# ── 2B pipeline (fake model) ─────────────────────────────────────────────────


def _fake_phase2a_run(tmp_path: Path, tokenizer: _CharTokenizer) -> Path:
    source, target = _probe_rows(tokenizer)
    run_dir = tmp_path / "phase2a"
    _write_jsonl(run_dir / "prepare" / "prompts.jsonl", [source, target])
    _write_jsonl(
        run_dir / "forward" / "results.jsonl",
        [
            {"ticker": "AMAT", "margin": 2.0},
            {"ticker": "DE", "margin": -1.5},
        ],
    )
    return run_dir


def test_run_phase2b_smoke_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    model = _PatchingModel(n_layers=N_LAYERS_16)
    monkeypatch.setattr(patch_pipeline, "load_model", _fake_load_model(model, tokenizer))

    run_root = patch_pipeline.run_phase2b(
        model_path="fake-model",
        run_id="smoke-fake",
        phase2a_run=_fake_phase2a_run(tmp_path, tokenizer),
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    records = (run_root / "sweep" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    # 2 directions x 3 layers x 4 spans
    assert len(records) == 24
    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["phase2c_arms"]["attention_layers"]) <= {3, 7, 15}
    assert set(summary["handoff"]["interval"]) <= {3, 7, 15}
    for span, curve in summary["curves"].items():
        assert set(curve) == {"3", "7", "15"}  # JSON keys are strings
        for stats in curve.values():
            assert stats["n_directions"] == 2
            assert stats["delta_m_ci_95"] is None  # <4 directions: no CI


def test_select_directions_pairs_top2_bottom2_both_orders():
    clean = {f"T{i:02d}": float(i) for i in range(16)}
    directions = patch_pipeline.select_directions(clean)
    assert len(directions) == 8
    assert len(set(directions)) == 8
    tops = {"T15", "T14"}
    bottoms = {"T00", "T01"}
    for source, target in directions:
        assert {source, target} & tops and {source, target} & bottoms
    pairs = {frozenset(d) for d in directions}
    assert len(pairs) == 4


def test_analyze_2b_records_requires_all_directions():
    records = [
        {"layer": 3, "span": span, "toward_source_delta_m": 0.5, "normalized_transfer": 0.5}
        for span in ("entity", "evidence", "instruction", "final")
    ]
    with pytest.raises(ValueError, match="missing directions"):
        patch_pipeline.analyze_2b_records(records, layers=[3])


# ── 2C pipeline (fake model) ─────────────────────────────────────────────────


def _fake_phase2b_run(tmp_path: Path, *, attention: list[int], mlp: list[int]) -> Path:
    run_dir = tmp_path / "phase2b"
    summary = {
        "handoff": {"crossover": False, "interval": mlp},
        "phase2c_arms": {"attention_layers": attention, "mlp_layers": mlp},
    }
    (run_dir / "analyze").mkdir(parents=True, exist_ok=True)
    (run_dir / "analyze" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_compute_sector_agreement_semantics():
    sectors = {f"T{i:02d}": ("A" if i < 8 else "B") for i in range(16)}
    margins = {f"T{i:02d}": float(i) for i in range(16)}
    # both sectors: neuron tracks margin (positive, top_rho positive) -> 1.0
    agree = patch_pipeline.compute_sector_agreement(
        neuron_values={f"T{i:02d}": float(i) for i in range(16)},
        top_rho=1.0, sectors=sectors, margins=margins,
    )
    assert agree == 1.0
    # one sector inverted -> 0.5
    vals = {f"T{i:02d}": (float(7 - i) if i < 8 else float(i)) for i in range(16)}
    assert patch_pipeline.compute_sector_agreement(
        neuron_values=vals, top_rho=1.0, sectors=sectors, margins=margins,
    ) == 0.5
    # degenerate sector (constant neuron values) fails closed as non-agreement
    vals = {f"T{i:02d}": (0.0 if i < 8 else float(i)) for i in range(16)}
    assert patch_pipeline.compute_sector_agreement(
        neuron_values=vals, top_rho=1.0, sectors=sectors, margins=margins,
    ) == 0.5
    # constant margins in a sector also fail closed
    margins2 = dict(margins)
    for i in range(8):
        margins2[f"T{i:02d}"] = 1.0
    agree = patch_pipeline.compute_sector_agreement(
        neuron_values={f"T{i:02d}": float(i) for i in range(16)},
        top_rho=1.0, sectors=sectors, margins=margins2,
    )
    assert agree == 0.5
    # sectors with <3 tickers are skipped entirely
    sectors_small = {f"T{i:02d}": ("S" if i < 2 else ("A" if i < 8 else "B")) for i in range(16)}
    agree = patch_pipeline.compute_sector_agreement(
        neuron_values={f"T{i:02d}": float(i) for i in range(16)},
        top_rho=1.0, sectors=sectors_small, margins=margins,
    )
    assert agree == 1.0


def test_run_phase2c_smoke_mlp_only_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    model = _MlpModel(n_layers=2, width=4, mlp_width=32)
    monkeypatch.setattr(patch_pipeline, "load_model", _fake_load_model(model, tokenizer))

    run_root = patch_pipeline.run_phase2c(
        model_path="fake-model",
        run_id="smoke-fake",
        phase2a_run=_fake_phase2a_run(tmp_path, tokenizer),
        phase2b_run=_fake_phase2b_run(tmp_path, attention=[], mlp=[1]),
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_2c"]["attention_arm"]["status"] == "not_applicable"
    assert "per_layer" in summary["gate_2c"]["mlp_arm"]
    assert (run_root / "mlp" / "records.jsonl").exists()
    layers = json.loads((run_root / "mlp" / "layer_summaries.json").read_text(encoding="utf-8"))["layers"]
    assert layers[0]["layer"] == 1
    assert len(layers[0]["control_rhos"]) >= 1


def test_analyze_2c_records_attention_paired_difference_semantics():
    records = []
    for head in (0, 1):
        for direction in range(8):
            entity = 0.5 + 0.01 * head
            controls = [0.1, 0.2]
            records.append(
                {
                    "layer": 3,
                    "head": head,
                    "direction": f"d{direction}",
                    "entity_toward_source_delta_m": entity,
                    "control_toward_source_delta_ms": controls,
                }
            )
    summaries = [
        {
            "layer": 1,
            "top_neuron": 0,
            "top_spearman": 0.9,
            "abs_top_spearman": 0.9,
            "control_max_abs_rho": 0.1,
            "sector_agreement": 1.0,
            "sign_flip_p": 0.001,
        }
    ]
    summary = patch_pipeline.analyze_2c_records(
        records, summaries, attention_layers=[3], mlp_layers=[1]
    )
    gate = summary["gate_2c"]
    assert gate["pass"] is True
    top = gate["attention_arm"]
    assert top["status"] == "run"
    # head 1: 0.51 − 0.15 = 0.36 > head 0: 0.35
    assert top["top_head"] == "L3H1"
    assert top["top_effect"] == pytest.approx(0.36)
    assert top["sign_flip_p_adjusted"] < 0.05
    # both heads show 8/8 direction consistency -> both pass
    assert set(top["passing_heads"]) == {"L3H0", "L3H1"}
    # mlp per-layer carries the adjusted p used by the existence test
    mlp_layer = gate["mlp_arm"]["per_layer"]["1"]
    assert mlp_layer["sign_flip_p_adjusted"] == pytest.approx(0.001)
    assert gate["mlp_arm"]["passing_layers"] == [1]


def test_run_2c_gate_reanalysis(tmp_path):
    import hashlib

    from llm_bias.balanced_evidence_gap.gate_reanalysis import run_2c_gate_reanalysis

    src = tmp_path / "phase2c"
    (src / "attention").mkdir(parents=True)
    (src / "mlp").mkdir(parents=True)
    (src / "analyze").mkdir(parents=True)
    records = []
    for head in (0, 1):
        for direction in range(8):
            records.append({
                "layer": 15, "head": head, "direction": f"d{direction}",
                "entity_toward_source_delta_m": 0.5 + 0.01 * head,
                "control_toward_source_delta_ms": [0.1, 0.2],
            })
    att_path = src / "attention" / "records.jsonl"
    att_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    layer_summaries = {
        "schema_version": "1", "layers": [
            {"layer": 12, "top_neuron": 7, "top_spearman": 0.9, "abs_top_spearman": 0.9,
             "control_max_abs_rho": 0.6, "control_mean_rho": 0.05, "sector_agreement": 1.0,
             "sign_flip_p": 0.002, "control_rhos": []},
            {"layer": 31, "top_neuron": 0, "top_spearman": 0.0, "abs_top_spearman": 0.0,
             "control_max_abs_rho": 0.0, "control_mean_rho": 0.0, "sector_agreement": 0.0,
             "sign_flip_p": 1.0, "control_rhos": []},
        ]
    }
    mlp_path = src / "mlp" / "layer_summaries.json"
    mlp_path.write_text(json.dumps(layer_summaries), encoding="utf-8")
    (src / "analyze" / "summary.json").write_text(json.dumps({
        "attention_layers": [15], "mlp_layers": [12, 31],
    }), encoding="utf-8")

    run_root = run_2c_gate_reanalysis(
        model_name="fake-model", phase2c_run=src, run_id="gate-reanalysis-test",
        artifact_root=tmp_path / "artifacts",
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_2c"]
    # L31's structural zero must not contaminate the verdict
    assert gate["mlp_arm"]["passing_layers"] == [12]
    assert gate["mlp_arm"]["pass"] is True
    assert gate["pass"] is True
    assert gate["attention_arm"]["passing_heads"] == ["L15H0", "L15H1"]
    prov = json.loads((run_root / "prepare" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["source_run"]["attention_records_sha256"] == hashlib.sha256(att_path.read_bytes()).hexdigest()
    assert prov["source_run"]["attention_n_records"] == 16


def test_package_does_not_import_other_experiment_packages():
    import ast

    root = Path(__file__).resolve().parents[1] / "llm_bias" / "balanced_evidence_gap"
    forbidden_prefixes = (
        "llm_bias.entity_cell",
        "llm_bias.jspace_intervention",
        "llm_bias.investment_dial",
        "llm_bias.baseline_trial",
        "llm_bias.span_sensitivity",
        "llm_bias.prompt_analysis",
        "llm_bias.financial_soundness",
        "llm_bias.sector_context",
    )
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not any(name == p or name.startswith(p + ".") for p in forbidden_prefixes), (
                    f"{path.name} imports {name}"
                )
