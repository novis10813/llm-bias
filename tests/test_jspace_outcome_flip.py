"""Regression tests for the V2 outcome-conditioned decision-flip workflow."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.continuation_scoring import (
    fp32_next_token_log_probs,
    score_single_token_margin_fp32,
)
from llm_bias.jspace_intervention.outcome_flip import (
    Combination,
    _mcnemar_exact,
    analyze_outcome_flip,
    evidence_item_char_spans,
    evidence_item_end_positions,
    fit_outcome_directions,
    fit_prompt_layer_gradients,
    label_permutation_signs,
    parse_decision,
    run_outcome_flip_combination,
    run_outcome_flip_pipeline,
)
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig

_TOKEN_RE = re.compile(r"buy|sell|.")
BUY_ID = 1000
SELL_ID = 1001
# Rare control characters used to steer the fake model's clean decision:
# 'Z' pushes the decision-position margin buy-ward, 'z' sell-ward.
BUY_CTRL_ID = 1 + (ord("Z") % 900)
SELL_CTRL_ID = 1 + (ord("z") % 900)


def _token_id(token: str) -> int:
    if token == "buy":
        return BUY_ID
    if token == "sell":
        return SELL_ID
    return 1 + (ord(token) % 900)


def _encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    tokens, offsets = [], []
    for match in _TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return [_token_id(token) for token in tokens], offsets


class _FlipTokenizer:
    chat_template = "fake"
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        **_kwargs,
    ):
        ids, offsets = _encode(str(text))
        if return_offsets_mapping:
            return SimpleNamespace(
                input_ids=ids,
                offset_mapping=offsets,
                special_tokens_mask=[0] * len(ids),
            )
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, messages, **_kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def decode(self, ids, **_kwargs):
        parts = []
        for token_id in ids:
            if int(token_id) == BUY_ID:
                parts.append("buy")
            elif int(token_id) == SELL_ID:
                parts.append("sell")
            else:
                parts.append(chr(int(token_id) - 1))
        return "".join(parts)


class _MixingLayer(torch.nn.Module):
    """Linear global-average mixing (unused; kept for reference)."""

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.weight * hidden.mean(dim=1, keepdim=True)


class _FocusLayer(torch.nn.Module):
    """Asymmetric copy: the last position receives the final non-zero
    earlier position's residual.

    Symmetric mixing plus LayerNorm's mean subtraction would cancel any
    non-final-position intervention, so the fake instead gives the decision
    position a direct readout of the evidence position carrying the control
    token.  The focus index is data-derived (last non-zero position before
    the final one) so the same model works for every prompt layout.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        seq = hidden.shape[1]
        nonzero = hidden.abs().sum(dim=-1) > 1e-12
        nonzero = nonzero.clone()
        nonzero[:, -1] = False
        indices = nonzero[0].nonzero().squeeze(-1)
        focus = int(indices.max()) if indices.numel() else seq - 2
        one_hot = torch.zeros(1, seq, 1, dtype=hidden.dtype, device=hidden.device)
        one_hot[0, seq - 1, 0] = 1.0
        return hidden + self.weight * hidden[:, focus, :].unsqueeze(1) * one_hot


class _FlipModel(torch.nn.Module):
    """Deterministic differentiable fake with a buy/sell-oriented head.

    All embeddings are zero except the control characters ('Z'/'z', placed
    at the last evidence item's final token by the test prompts) and the
    final prompt character ('"'), which carries a fixed base vector.  The
    clean decision margin is therefore a small, prompt-controlled value,
    and the focus layer makes evidence-position interventions reach the
    decision position without 1/seq_len dilution.
    """

    _control_scale = 0.3
    _base_vector = (0.25, 0.5, 0.2, -0.1)

    def __init__(self, n_layers=4, d_model=4, vocab=2000, seed=0, head_scale=1.0):
        super().__init__()
        self._embed_tokens = torch.nn.Embedding(vocab, d_model)
        with torch.no_grad():
            self._embed_tokens.weight.zero_()
            unit = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self._embed_tokens.weight[BUY_CTRL_ID] = self._control_scale * unit
            self._embed_tokens.weight[SELL_CTRL_ID] = -self._control_scale * unit
        self._base = torch.tensor(self._base_vector)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(n_layers - 1)]
            + [_FocusLayer()]
        )
        self.n_layers = n_layers
        self.d_model = d_model
        self._final_norm = torch.nn.LayerNorm(d_model)
        self._lm_head = torch.nn.Linear(d_model, vocab, bias=False)
        with torch.no_grad():
            self._lm_head.weight.zero_()
            self._lm_head.weight[BUY_ID] = head_scale * torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            )
            self._lm_head.weight[SELL_ID] = -head_scale * torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            )
            self._lm_head.weight[:8] = torch.nn.init.normal_(
                torch.empty(8, d_model), generator=torch.Generator().manual_seed(seed)
            ) * 0.01
        # Mimic the HFLensModel wrapper, which freezes every parameter:
        # gradient fitting must root the autograd graph at the first fitted
        # layer instead of relying on trainable leaves, so a fake with
        # trainable params would hide that failure mode.
        for param in self.parameters():
            param.requires_grad_(False)
        self.config = SimpleNamespace(eos_token_id=None)
        self._hf_model = self
        self._buy_ctrl = BUY_CTRL_ID
        self._sell_ctrl = SELL_CTRL_ID
        self._generate_text = 'buy", "reason": "fine"}'

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self._embed_tokens(input_ids)
        offset = torch.zeros_like(hidden)
        offset[:, -1, :] = self._base.to(hidden.dtype)
        hidden = hidden + offset
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(self, prompt_ids, **_kwargs):
        continuation, _ = _encode(self._generate_text)
        return torch.cat(
            [prompt_ids, torch.tensor([continuation], dtype=torch.long)], dim=1
        )


def _prompt(ticker: str = "A1", control: str = "ZZ") -> str:
    # The control characters are the final characters of the last evidence
    # item, so the item-end position rule lands exactly on the fake model's
    # focus position.  'ZZ' biases the clean margin buy-ward, 'zz' sell-ward.
    return (
        f"Decide for [{ticker}].\n"
        "--- Evidence ---\n"
        "1. item a.\n"
        f"2. bias {control}\n"
        "---\n"
        'Respond with one valid JSON object containing only the keys "decision" '
        "(buy or sell) and \"reason\"."
    )


def _config_payload(**overrides) -> dict:
    payload = {
        "model": "fake-model",
        "source_sector": "Technology",
        "fitted_layers": [1, 2],
        "candidate_bands": [[1, 2]],
        "position_rules": ["evidence_item_end"],
        "dose_grid": [0.5],
        "split_manifest_sha256": "ab" * 32,
        "bootstrap_samples": 60,
        "max_new_tokens": 32,
    }
    payload.update(overrides)
    return payload


def _config(**overrides) -> OutcomeFlipConfig:
    return OutcomeFlipConfig.from_dict(_config_payload(**overrides))


def _materialize(
    ticker: str,
    *,
    tokenizer=None,
    config=None,
    prompt_column="prompt_with_context_attribute_0",
    split="discovery",
):
    from llm_bias.jspace_intervention.outcome_flip import (
        _materialize_records,
    )

    tokenizer = tokenizer or _FlipTokenizer()
    config = config or _config()
    records = [
        {
            "ticker": ticker,
            "name": f"{ticker} Corp",
            "sector": "Technology",
            "marketcap": "1000",
            "prompt_column": prompt_column,
            "prompt": _prompt(ticker),
            "record_id": f"record_{abs(hash((ticker, prompt_column, split))) % 16**8:08x}",
        }
    ]
    return _materialize_records(records, tokenizer, config)[0]


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


class _PositionVaryingModel(torch.nn.Module):
    """Hidden state varies by position so scorer positions are distinguishable."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.n_layers = 1
        self._embed_tokens = torch.nn.Embedding(2000, 4)
        self._final_norm = torch.nn.LayerNorm(4)
        self._lm_head = torch.nn.Linear(4, 2000, bias=False)
        torch.manual_seed(7)
        self._vectors = [
            torch.nn.init.normal_(torch.empty(4)) for _ in range(16)
        ]

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = torch.stack(
            [self._vectors[index % 16] for index in range(input_ids.shape[1])]
        ).unsqueeze(0)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_fp32_next_token_log_probs_is_differentiable() -> None:
    model = _PositionVaryingModel()
    vector = torch.nn.init.normal_(torch.empty(1, 4)).requires_grad_(True)
    log_probs = fp32_next_token_log_probs(model, vector)
    target = log_probs[0, 3] - log_probs[0, 5]
    grad = torch.autograd.grad(target, vector)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_single_token_margin_scores_prompt_final_position() -> None:
    """The V2 decision margin is P(token | prompt's final position).

    The scorer tokenizes the prompt alone and reads the next-token
    distribution at the prompt's final position, not the position after a
    candidate token.  With a position-varying fake the two positions give
    different values, so this pins the decision-position semantics.
    """
    model = _PositionVaryingModel()
    tokenizer = _FlipTokenizer()
    prompt_ids, _ = _encode("P Q R")
    prompt = tokenizer.decode(prompt_ids)
    margin = score_single_token_margin_fp32(
        model, tokenizer, prompt, "buy", "sell", device="cpu"
    )
    with torch.no_grad():
        expected_last = fp32_next_token_log_probs(
            model, model._vectors[len(prompt_ids) - 1]
        )
        expected_after = fp32_next_token_log_probs(
            model, model._vectors[len(prompt_ids)]
        )
    expected_last_value = float(
        expected_last[BUY_ID] - expected_last[SELL_ID]
    )
    expected_after_value = float(expected_after[BUY_ID] - expected_after[SELL_ID])
    assert margin.value == pytest.approx(expected_last_value, rel=1e-5)
    assert margin.value != pytest.approx(expected_after_value, abs=1e-3)


# ---------------------------------------------------------------------------
# Decision parsing and evidence positions
# ---------------------------------------------------------------------------


def test_parse_decision_variants() -> None:
    assert parse_decision('{\n  "decision": "buy", "reason": "x"}') == "buy"
    assert parse_decision('  "decision": "SELL"  ') == "sell"
    assert parse_decision('{"reason": "x"}') is None
    assert parse_decision("no json at all") is None


def test_evidence_item_char_spans_trim_trailing_newlines() -> None:
    spans = evidence_item_char_spans(_prompt())
    assert len(spans) == 2
    region = _prompt()
    assert region[spans[0][0] : spans[0][1]].endswith("item a.")
    assert region[spans[1][0] : spans[1][1]].endswith("bias ZZ")
    assert region[spans[0][1]] == "\n"
    assert evidence_item_char_spans("no evidence here") == []


def test_evidence_item_end_positions_are_last_tokens_of_items() -> None:
    tokenizer = _FlipTokenizer()
    raw = _prompt()
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": raw}])
    scoring_prompt = formatted + '{\n  "decision": "'
    positions = evidence_item_end_positions(tokenizer, scoring_prompt, raw)
    raw_offset = scoring_prompt.find(raw)
    expected = []
    for start, end in evidence_item_char_spans(raw):
        _, offsets = _encode(scoring_prompt)
        inside = [
            index
            for index, (s, e) in enumerate(offsets)
            if raw_offset + start <= s and e <= raw_offset + end
        ]
        expected.append(max(inside))
    assert positions == expected
    with pytest.raises(ValueError, match="evidence"):
        evidence_item_end_positions(
            tokenizer, scoring_prompt, "Decide for [A1]."
        )


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_outcome_flip_config_defaults_and_roundtrip() -> None:
    config = _config()
    assert config.safety_bound == 0.50
    assert config.scale_floor == 1.0
    assert config.tie_rule == "exclude_exact_zero"
    assert config.clean_margin_edges == (0.5, 1.5)
    assert config.min_flip_rate == 0.10
    assert config.parse_success_gate == 0.90
    assert config.agreement_gate == 0.70
    assert OutcomeFlipConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_bands": [[1, 5]]},  # band beyond fitted layers
        {"candidate_bands": [[2, 1]]},  # inverted band
        {"dose_grid": [0.6]},  # above safety bound
        {"dose_grid": [0.1, 0.1]},  # duplicate
        {"position_rules": ["evidence_item_end", "bogus"]},
        {"position_rules": []},
        {"clean_margin_edges": [1.5, 0.5]},
        {"clean_margin_edges": [0.0]},
        {"tie_rule": "break_randomly"},
        {"safety_bound": 0.0},
        {"scale_floor": 0.0},
        {"min_flip_rate": 1.5},
        {"max_new_tokens": 0},
        {"split_manifest_sha256": "zz" * 32},
        {"fitted_layers": [1, 1]},
        {"candidate_bands": []},
    ],
)
def test_outcome_flip_config_rejects_invalid_payload(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


# ---------------------------------------------------------------------------
# Direction fitting
# ---------------------------------------------------------------------------


def test_fit_prompt_layer_gradients_is_deterministic_and_increasing() -> None:
    model = _FlipModel()
    record = _materialize("G1")
    tensor = torch.tensor([record["prompt_ids"]])
    first = fit_prompt_layer_gradients(
        model,
        tensor,
        fitted_layers=[1, 2],
        positive_id=BUY_ID,
        negative_id=SELL_ID,
    )
    second = fit_prompt_layer_gradients(
        model,
        tensor,
        fitted_layers=[1, 2],
        positive_id=BUY_ID,
        negative_id=SELL_ID,
    )
    for layer in (1, 2):
        assert torch.allclose(first[layer], second[layer])
        assert first[layer].abs().sum() > 0
    # The gradient must point uphill for the target: a small analytic step in
    # the last-position gradient direction increases the decision-position
    # logit difference.
    with torch.no_grad():
        last = model.forward(tensor).last_hidden_state[:, -1, :]
        grad_last = (
            fit_prompt_layer_gradients(
                model,
                tensor,
                fitted_layers=[model.n_layers - 1],
                positive_id=BUY_ID,
                negative_id=SELL_ID,
            )[model.n_layers - 1]
        )[-1]
        step = 0.05 * grad_last / grad_last.norm()

    def target_at(perturbed: torch.Tensor) -> float:
        log_probs = fp32_next_token_log_probs(model, perturbed)
        return float((log_probs[0, BUY_ID] - log_probs[0, 1001]).detach())

    assert target_at(last + step) > target_at(last)
    # Hooks must be removed even though the call succeeded.
    assert not model.layers[1]._forward_hooks


def test_enable_deterministic_gpu_sets_cublas_workspace_and_flag(monkeypatch) -> None:
    import os

    from llm_bias.jspace_intervention import outcome_flip

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda flag: calls.append(flag)
    )
    try:
        outcome_flip._enable_deterministic_gpu()
    finally:
        assert torch.are_deterministic_algorithms_enabled() is False
    assert calls == [True]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_fit_prompt_layer_gradients_removes_hooks_on_failure() -> None:
    class BrokenModel(_FlipModel):
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            raise RuntimeError("boom")

    broken = BrokenModel()
    tensor = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="boom"):
        fit_prompt_layer_gradients(
            broken,
            tensor,
            fitted_layers=[1],
            positive_id=BUY_ID,
            negative_id=SELL_ID,
        )
    assert not broken.layers[1]._forward_hooks


def test_fit_outcome_directions_aggregation_and_label_permutation() -> None:
    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    config = _config()
    records = [
        _materialize("A1", tokenizer=tokenizer, config=config),
        _materialize("A2", tokenizer=tokenizer, config=config),
        _materialize("B1", tokenizer=tokenizer, config=config),
    ]
    fit = fit_outcome_directions(
        model=model,
        tokenizer=tokenizer,
        records=records,
        config=config,
        device=torch.device("cpu"),
    )
    assert fit.record_count == 3
    assert fit.ticker_count == 3
    rule = config.position_rules[0]
    for layer in config.fitted_layers:
        direction = fit.directions[rule][layer]
        assert float(direction.norm()) == pytest.approx(1.0, rel=1e-5)
        permutation = fit.permutation_directions[rule][layer]
        assert float(permutation.norm()) == pytest.approx(1.0, rel=1e-5)
        cosine = float((direction * permutation).sum())
        assert cosine < 0.99
    assert set(fit.signs) == {"A1", "A2", "B1"}
    assert all(value in {-1, 1} for value in fit.signs.values())
    assert fit.signs == label_permutation_signs(
        ["A1", "A2", "B1"], config.fitting_seed
    )


def test_label_permutation_signs_are_deterministic_and_seeded() -> None:
    signs_a = label_permutation_signs(["B", "A"], seed=123)
    signs_b = label_permutation_signs(["A", "B"], seed=123)
    assert signs_a == signs_b
    signs_c = label_permutation_signs(["B", "A"], seed=124)
    assert signs_a != signs_c


# ---------------------------------------------------------------------------
# Record runner
# ---------------------------------------------------------------------------


def _runner_directions(config) -> tuple[dict, dict]:
    d = torch.tensor([1.0, 0.0, 0.0, 0.0])
    p = torch.tensor([0.0, 1.0, 0.0, 0.0])
    return (
        {layer: d.clone() for layer in config.fitted_layers},
        {layer: p.clone() for layer in config.fitted_layers},
    )


def test_run_outcome_flip_combination_dose_matched_and_flip_roles() -> None:
    from llm_bias.jspace_intervention.outcome_flip import prepare_clean_record

    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    config = _config()
    record = _materialize("R1", tokenizer=tokenizer, config=config)
    directions, permutation_directions = _runner_directions(config)
    clean = prepare_clean_record(
        model=model,
        tokenizer=tokenizer,
        record=record,
        config=config,
        device=torch.device("cpu"),
    )
    combination = Combination(1, 2, "evidence_item_end", 0.5)
    rows = run_outcome_flip_combination(
        model=model,
        tokenizer=tokenizer,
        record=record,
        config=config,
        combination=combination,
        clean=clean,
        device=torch.device("cpu"),
        directions=directions,
        permutation_directions=permutation_directions,
        control_seed=42,
        generate_clean=True,
        generate_outcome_all=True,
    )
    by_arm_sign = {(row["arm"], row["sign"]): row for row in rows}
    # 5 arms x 2 signs + noop + clean = 12 rows
    assert len(rows) == 12
    noop = by_arm_sign[("noop", 0)]
    assert noop["intervened_margin"] == pytest.approx(noop["clean_margin"])
    assert noop["flipped"] is False
    clean_row = next(row for row in rows if row["arm"] == "clean")
    assert clean_row["generation"]["parse_success"] is True
    assert clean_row["generation"]["parsed_decision"] == "buy"
    # Dose matching: every steering arm delivers the same relative
    # perturbation as the outcome arm (same positions, scales, and dose).
    outcome_rel = {
        (row["arm"], row["sign"]): row["delivered_dose"][
            "relative_perturbation_max"
        ]
        for row in rows
        if row["arm"] in ("outcome", "matched_random", "label_permutation")
    }
    reference = outcome_rel[("outcome", 1)]
    assert reference == pytest.approx(0.5, rel=1e-5)
    for value in outcome_rel.values():
        assert value == pytest.approx(reference, rel=1e-5)
    assert all(
        row["delivered_dose"]["safety_ok"] for row in rows if row["arm"] != "clean"
    )
    # Position controls deliver at the same dose but different positions.
    shuffled = by_arm_sign[("shuffled_evidence", 1)]
    final = by_arm_sign[("final_position", 1)]
    assert shuffled["intervention_positions"] != record["positions"][
        "evidence_item_end"
    ]
    assert final["intervention_positions"] == [
        len(record["prompt_ids"]) - 1
    ]
    assert final["delivered_dose"]["relative_perturbation_max"] == pytest.approx(
        reference, rel=1e-5
    )
    # Outcome steering moves the margin in the expected direction for the two
    # signs.
    plus = by_arm_sign[("outcome", 1)]
    minus = by_arm_sign[("outcome", -1)]
    assert plus["delta_margin"] > 0.0
    assert minus["delta_margin"] < 0.0
    # Determinism: a repeated run produces identical margins.
    repeated = run_outcome_flip_combination(
        model=model,
        tokenizer=tokenizer,
        record=record,
        config=config,
        combination=combination,
        clean=prepare_clean_record(
            model=model,
            tokenizer=tokenizer,
            record=record,
            config=config,
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
        directions=directions,
        permutation_directions=permutation_directions,
        control_seed=42,
    )
    assert [row["intervened_margin"] for row in repeated] == [
        row["intervened_margin"] for row in rows if row["arm"] != "clean"
    ][: len(repeated)]


def test_run_outcome_flip_combination_flip_roles_on_boundary_flip() -> None:
    from llm_bias.jspace_intervention.outcome_flip import (
        fit_outcome_directions,
        prepare_clean_record,
    )

    model = _FlipModel(head_scale=0.05)  # weak head -> margin near zero
    tokenizer = _FlipTokenizer()
    config = _config()
    record = _materialize("FL1", tokenizer=tokenizer, config=config)
    fit = fit_outcome_directions(
        model=model,
        tokenizer=tokenizer,
        records=[record],
        config=config,
        device=torch.device("cpu"),
    )
    rule = config.position_rules[0]
    directions = fit.directions[rule]
    permutation_directions = fit.permutation_directions[rule]
    clean = prepare_clean_record(
        model=model,
        tokenizer=tokenizer,
        record=record,
        config=config,
        device=torch.device("cpu"),
    )
    if clean["clean_decision"] == "tie":
        pytest.skip("fake model produced an exact tie")
    combination = Combination(1, 2, "evidence_item_end", 50.0)
    rows = run_outcome_flip_combination(
        model=model,
        tokenizer=tokenizer,
        record=record,
        config=config,
        combination=combination,
        clean=clean,
        device=torch.device("cpu"),
        directions=directions,
        permutation_directions=permutation_directions,
        control_seed=7,
    )
    by_sign = {r["sign"]: r for r in rows if r["arm"] == "outcome"}
    plus, minus = by_sign[1], by_sign[-1]
    # The fitted axis points buy-uphill: the push against the clean decision
    # must cross the boundary, and the flip is a target flip to the pushed
    # decision.
    if clean["clean_decision"] == "buy":
        assert minus["flipped"] is True
        assert minus["flip_role"] == "target"
        assert minus["intervened_decision"] == "sell"
        assert plus["flipped"] is False or plus["flip_role"] == "reverse"
    else:
        assert plus["flipped"] is True
        assert plus["flip_role"] == "target"
        assert plus["intervened_decision"] == "buy"
        assert minus["flipped"] is False or minus["flip_role"] == "reverse"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analysis_row(
    *,
    ticker: str,
    arm: str,
    sign: int,
    clean: str,
    flipped: bool = False,
    role: str | None = None,
    margin: float = -0.5,
    generation: dict | None = None,
) -> dict:
    return {
        "arm": arm,
        "sign": sign,
        "ticker": ticker,
        "combination": {
            "band": [1, 2],
            "position_rule": "evidence_item_end",
            "relative_dose": 0.5,
        },
        "layers": [1, 2],
        "intervention_positions": [3],
        "clean_margin": margin,
        "clean_decision": clean,
        "intervened_margin": margin + (0.5 if flipped else 0.0),
        "delta_margin": 0.5 if flipped else 0.0,
        "intervened_decision": (
            "buy" if clean == "sell" else "sell"
        )
        if flipped
        else clean,
        "flipped": flipped,
        "flip_role": role,
        "position_scale_min": 1.0,
        "position_scale_mean": 1.0,
        "position_scale_max": 1.0,
        "delivered_dose": {
            "relative_perturbation_min": 0.5,
            "relative_perturbation_mean": 0.5,
            "relative_perturbation_max": 0.5,
            "safety_bound": 0.5,
            "safety_ok": True,
        },
        "score": {},
        "generation": generation,
    }


def _all_arms(rows: list, ticker: str, sign: int, clean: str, margin: float) -> list:
    for arm in (
        "outcome",
        "matched_random",
        "label_permutation",
        "shuffled_evidence",
        "final_position",
    ):
        rows.append(_analysis_row(ticker=ticker, arm=arm, sign=sign, clean=clean, margin=margin))
    return rows


def _buy_sell_rows(
    pairs: list[tuple[str, bool, bool]],
) -> list:
    """Build one row per arm for each (ticker, outcome_flip, random_flip)."""
    rows = []
    for ticker, outcome_flip, random_flip in pairs:
        for arm, flipped in (
            ("outcome", outcome_flip),
            ("matched_random", random_flip),
            ("label_permutation", False),
            ("shuffled_evidence", False),
            ("final_position", False),
        ):
            rows.append(
                _analysis_row(
                    ticker=ticker,
                    arm=arm,
                    sign=1,
                    clean="sell",
                    flipped=flipped,
                    role="target" if flipped else None,
                )
            )
        for arm in (
            "outcome",
            "matched_random",
            "label_permutation",
            "shuffled_evidence",
            "final_position",
        ):
            rows.append(_analysis_row(ticker=ticker, arm=arm, sign=-1, clean="buy"))
    return rows


def test_analyze_outcome_flip_estimands_and_exact_paired() -> None:
    config = _config()
    rows = _buy_sell_rows(
        [("T1", True, False), ("T2", True, False)]
    )
    summary = analyze_outcome_flip(rows, config=config, split="test")
    gates = summary["gates"]
    combo = summary["combinations"][0]
    buy = combo["directions"]["buy"]
    assert buy["arms"]["outcome"]["target_flip_rate"] == pytest.approx(1.0)
    assert buy["arms"]["matched_random"]["target_flip_rate"] == pytest.approx(0.0)
    assert buy["specificity"]["point"] == pytest.approx(1.0)
    # Two discordant pairs, both favoring outcome:
    # p = 2 * C(2, 0) / 2^2 = 0.5
    assert buy["specificity"]["exact_paired_p"] == pytest.approx(0.5)
    assert buy["tie_excluded_count"] == 0
    # Buy differences are all 1.0 (CI [1, 1]), but the sell direction has no
    # flips at all (CI [0, 0]), so the both-directions gate stays False.
    assert buy["specificity"]["ci95"] == [1.0, 1.0]
    assert gates["specificity_ci_lower_positive"] is False
    assert gates["both_directions_nonzero_target_flips"] is False
    assert gates["net_specificity_positive"] is False
    assert gates["success"] is False
    sell = combo["directions"]["sell"]
    assert sell["arms"]["outcome"]["target_flip_rate"] == 0.0
    assert sell["specificity"]["point"] == 0.0


def test_analyze_outcome_flip_concordant_flips_carry_no_specificity() -> None:
    config = _config()
    rows = _buy_sell_rows(
        [("T1", True, True), ("T2", False, False)]
    )
    summary = analyze_outcome_flip(rows, config=config, split="test")
    buy = summary["combinations"][0]["directions"]["buy"]
    assert buy["arms"]["outcome"]["target_flip_rate"] == pytest.approx(0.5)
    assert buy["arms"]["matched_random"]["target_flip_rate"] == pytest.approx(0.5)
    assert buy["specificity"]["point"] == pytest.approx(0.0)
    # No discordant pairs -> the exact paired test is undefined.
    assert buy["specificity"]["exact_paired_p"] is None


def test_analyze_outcome_flip_ties_excluded_and_strata() -> None:
    config = _config()
    rows = [
        _analysis_row(ticker="T1", arm="outcome", sign=1, clean="tie", margin=0.0),
        _analysis_row(ticker="T1", arm="matched_random", sign=1, clean="tie", margin=0.0),
        _analysis_row(
            ticker="T2", arm="outcome", sign=1, clean="sell", margin=0.3,
            flipped=True, role="target",
        ),
        _analysis_row(
            ticker="T2", arm="matched_random", sign=1, clean="sell", margin=0.3
        ),
        _analysis_row(ticker="T3", arm="outcome", sign=1, clean="sell", margin=2.0),
        _analysis_row(
            ticker="T3", arm="matched_random", sign=1, clean="sell", margin=2.0
        ),
    ]
    summary = analyze_outcome_flip(rows, config=config, split="test")
    buy = summary["combinations"][0]["directions"]["buy"]
    # T1 is a tie: excluded once per direction (one outcome row).
    assert buy["tie_excluded_count"] == 1
    assert buy["arms"]["outcome"]["eligible_ticker_count"] == 2
    strata = {row["stratum"]: row for row in buy["margin_strata"]}
    assert strata["|M| <= 0.5"]["record_count"] == 1
    assert strata["|M| > 1.5"]["record_count"] == 1


def test_test_gates_pass_on_strong_specificity() -> None:
    config = _config(min_flip_rate=0.5, parse_success_gate=1.0, agreement_gate=1.0)
    rows = []
    generation = {
        "max_new_tokens": 32,
        "finish_reason": "model_stop",
        "generated_text": 'buy", "reason": "x"}',
        "parsed_decision": "buy",
        "parse_success": True,
    }
    for ticker in ("T1", "T2", "T3"):
        rows.append(
            _analysis_row(
                ticker=ticker, arm="outcome", sign=1, clean="sell",
                flipped=True, role="target", generation=generation,
            )
        )
        for arm in (
            "matched_random",
            "label_permutation",
            "shuffled_evidence",
            "final_position",
        ):
            rows.append(_analysis_row(ticker=ticker, arm=arm, sign=1, clean="sell"))
        sell_generation = {
            **generation,
            "generated_text": 'sell", "reason": "y"}',
            "parsed_decision": "sell",
        }
        rows.append(
            _analysis_row(
                ticker=ticker, arm="outcome", sign=-1, clean="buy", margin=0.4,
                flipped=True, role="target", generation=sell_generation,
            )
        )
        for arm in (
            "matched_random",
            "label_permutation",
            "shuffled_evidence",
            "final_position",
        ):
            rows.append(
                _analysis_row(ticker=ticker, arm=arm, sign=-1, clean="buy", margin=0.4)
            )
    summary = analyze_outcome_flip(rows, config=config, split="test")
    gates = summary["gates"]
    assert gates["both_directions_nonzero_target_flips"] is True
    assert gates["min_flip_rate"] is True
    assert gates["controls_below_min_rate"] is True
    assert gates["generation_parse_success"] is True
    assert gates["generation_agreement"] is True
    assert gates["safety_bound_respected"] is True
    assert gates["net_specificity_positive"] is True
    assert gates["reverse_not_excess"] is True
    assert gates["holm_adjusted_specificity"] in {True, False}


def test_mcnemar_exact_values() -> None:
    assert _mcnemar_exact([]) is None
    assert _mcnemar_exact([(True, False), (True, False)]) == pytest.approx(0.5)
    assert _mcnemar_exact(
        [(True, False), (False, True), (True, False), (True, False)]
    ) == pytest.approx(2 * (1 + 4) / 16)
    assert _mcnemar_exact([(True, True), (False, False)]) is None


def test_select_calibration_fail_closed_without_dose_gate() -> None:
    config = _config(parse_success_gate=1.0)
    rows = []
    for ticker in ("T1", "T2"):
        for arm in (
            "outcome",
            "matched_random",
            "label_permutation",
            "shuffled_evidence",
            "final_position",
        ):
            rows.append(
                _analysis_row(
                    ticker=ticker, arm=arm, sign=1, clean="sell",
                    flipped=arm == "outcome",
                    role="target" if arm == "outcome" else None,
                )
            )
            rows.append(
                _analysis_row(
                    ticker=ticker, arm=arm, sign=-1, clean="buy", margin=0.5,
                    flipped=arm == "outcome",
                    role="target" if arm == "outcome" else None,
                )
            )
    # No generations -> parse success None -> no eligible dose -> fail closed.
    with pytest.raises(ValueError, match="parse-success gate"):
        analyze_outcome_flip(rows, config=config, split="calibration")


def test_select_calibration_picks_selected_combination_and_binds() -> None:
    config = _config()
    generation = {
        "max_new_tokens": 32,
        "finish_reason": "model_stop",
        "generated_text": 'buy", "reason": "x"}',
        "parsed_decision": "buy",
        "parse_success": True,
    }
    rows = []
    for combo in ([1, 2], [1, 1]):
        for ticker in ("T1", "T2"):
            for arm in (
                "outcome",
                "matched_random",
                "label_permutation",
                "shuffled_evidence",
                "final_position",
            ):
                buy = _analysis_row(
                    ticker=ticker, arm=arm, sign=1, clean="sell",
                    flipped=arm == "outcome", role="target" if arm == "outcome" else None,
                )
                buy["combination"]["band"] = combo
                if arm == "outcome":
                    buy["generation"] = generation
                rows.append(buy)
                sell = _analysis_row(
                    ticker=ticker, arm=arm, sign=-1, clean="buy", margin=0.5,
                    flipped=arm == "outcome", role="target" if arm == "outcome" else None,
                )
                sell["combination"]["band"] = combo
                if arm == "outcome":
                    sell["generation"] = {
                        **generation,
                        "generated_text": 'sell", "reason": "y"}',
                        "parsed_decision": "sell",
                    }
                rows.append(sell)
    summary = analyze_outcome_flip(rows, config=config, split="calibration")
    selection = summary["selection"]
    assert selection["artifact_type"] == "outcome_flip_selection"
    assert selection["selected"]["band"] in ([1, 2], [1, 1])
    assert selection["selected"]["position_rule"] == "evidence_item_end"
    assert selection["selected"]["relative_dose"] == 0.5
    assert len(selection["candidates"]) == 2
    # The pipeline binds config/identity hashes when writing the artifact;
    # the bare analysis selection carries neither.
    assert "config_sha256" not in selection
    assert "direction_identity_sha256" not in selection


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


def _write_pipeline_inputs(tmp_path: Path) -> dict[str, Path]:
    input_path = tmp_path / "trial_plan_prompts.csv"
    # Control characters make every split contain both decision directions:
    # odd tickers are clean-buy, even tickers clean-sell.
    tickers = {
        "D1": ("discovery", "ZZ"),
        "D2": ("discovery", "zz"),
        "C1": ("calibration", "ZZ"),
        "C2": ("calibration", "zz"),
        "E1": ("test", "ZZ"),
        "E2": ("test", "zz"),
    }
    lines = ["Date,ticker,name,sector,marketcap,prompt_with_context_attribute_0"]
    for ticker, (split, control) in tickers.items():
        prompt = _prompt(ticker, control=control).replace('"', '""')
        lines.append(
            f"2026-01-01,{ticker},{ticker} Corp,Technology,1000,\"{prompt}\""
        )
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "artifact_type": "jspace_intervention_splits",
                "schema_version": 1,
                "assignments": {ticker: split for ticker, (split, _) in tickers.items()},
            }
        ),
        encoding="utf-8",
    )
    return {"input": input_path, "split": split_path}


def _write_outcome_config(tmp_path: Path, split_path: Path) -> Path:
    from llm_bias.core.artifact_paths import sha256_file
    from llm_bias.core.artifacts.io import write_json

    config = _config(
        split_manifest_sha256=sha256_file(split_path)
    )
    config_path = tmp_path / "outcome_flip_config.json"
    write_json(
        config_path,
        {
            **config.to_dict(),
            "artifact_type": "outcome_flip_config",
            "schema_version": 1,
        },
        overwrite=True,
    )
    return config_path


def _patch_outcome_pipeline(monkeypatch) -> None:
    from llm_bias.jspace_intervention import outcome_flip

    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    monkeypatch.setattr(
        outcome_flip, "load_tokenizer", lambda model_name: tokenizer
    )
    monkeypatch.setattr(
        outcome_flip,
        "load_model",
        lambda model_name: (model, tokenizer, torch.device("cpu")),
    )


def _manifest(run_root: Path) -> dict:
    return json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))


def test_run_outcome_flip_pipeline_full_lifecycle(tmp_path: Path, monkeypatch) -> None:
    _patch_outcome_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"

    discovery_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="v2-discovery-test",
        artifact_root=artifact_root,
        split_name="discovery",
    )
    manifest = _manifest(discovery_root)
    assert manifest["status"] == "complete"
    for stage in ("prepare", "forward", "analyze"):
        assert manifest["stages"][stage]["status"] == "complete"
    identity_path = discovery_root / "forward" / "direction_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["artifact_type"] == "outcome_flip_direction_identity"
    assert identity["record_count"] == 2
    assert identity["ticker_count"] == 2
    assert set(identity["fitted_layers"]) == {1, 2}
    for rule in identity["position_rules"]:
        assert set(identity["directions"][rule]) == {"1", "2"}
        assert set(identity["label_permutation_directions"][rule]) == {"1", "2"}
    digest = identity["directions"]["evidence_item_end"]["1"]["sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)

    calibration_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="v2-calibration-test",
        artifact_root=artifact_root,
        split_name="calibration",
        direction_identity_path=identity_path,
    )
    calibration_manifest = _manifest(calibration_root)
    assert calibration_manifest["status"] == "complete"
    selection_path = calibration_root / "analyze" / "outcome_flip_selection.json"
    assert selection_path.exists()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["artifact_type"] == "outcome_flip_selection"
    assert selection["selected"]["position_rule"] == "evidence_item_end"
    assert selection["config_sha256"]
    assert selection["direction_identity_sha256"]

    test_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="v2-test-run",
        artifact_root=artifact_root,
        split_name="test",
        direction_identity_path=identity_path,
        calibration_selection_path=selection_path,
    )
    test_manifest = _manifest(test_root)
    assert test_manifest["status"] == "complete"
    analysis = json.loads(
        (test_root / "analyze" / "outcome_flip_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert "gates" in analysis
    assert "success" in analysis["gates"]
    results = [
        json.loads(line)
        for line in (
            test_root / "forward" / "outcome_flip_results.jsonl"
        ).open(encoding="utf-8")
        if line.strip()
    ]
    assert all(row["artifact_type"] == "outcome_flip_result" for row in results)
    # The test run must only contain the frozen combination.
    assert {
        tuple(row["combination"]["band"]) for row in results
    } == {tuple(selection["selected"]["band"])}


def test_run_outcome_flip_pipeline_fail_closed_on_tampered_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_outcome_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="v2-discovery-tamper",
        artifact_root=artifact_root,
        split_name="discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["directions"]["evidence_item_end"]["1"]["sha256"] = "00" * 32
    tampered_path = tmp_path / "tampered_identity.json"
    tampered_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        run_outcome_flip_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="v2-calibration-tamper",
            artifact_root=artifact_root,
            split_name="calibration",
            direction_identity_path=tampered_path,
        )
    tamper_root = (
        artifact_root
        / "fake-model"
        / "jspace-outcome-direction-flip"
        / "runs"
        / "v2-calibration-tamper"
    )
    assert _manifest(tamper_root)["status"] == "failed"


def test_run_outcome_flip_pipeline_rejects_missing_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_outcome_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    with pytest.raises(ValueError, match="require --direction-identity"):
        run_outcome_flip_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="v2-no-identity",
            artifact_root=tmp_path / "artifacts",
            split_name="calibration",
        )
    assert not (
        tmp_path
        / "artifacts"
        / "fake-model"
        / "jspace-outcome-direction-flip"
        / "runs"
        / "v2-no-identity"
    ).exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prepare_outcome_flip_config(tmp_path, monkeypatch, capsys) -> None:
    from llm_bias.jspace_intervention import cli

    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {"artifact_type": "jspace_intervention_splits", "assignments": {}}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "config.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "prepare-outcome-flip-config",
            "--model",
            "fake-model",
            "--source-sector",
            "Technology",
            "--split-manifest",
            str(split_path),
            "--bands",
            "2-3,1-4",
            "--output",
            str(output),
        ],
    )
    cli.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "outcome_flip_config"
    # Default fitted layers are the union of the bands.
    assert payload["fitted_layers"] == [1, 2, 3, 4]
    assert payload["candidate_bands"] == [[2, 3], [1, 4]]
    assert payload["position_rules"] == [
        "evidence_item_end",
        "evidence_span_all",
    ]
    assert payload["dose_grid"] == [0.05, 0.1, 0.2, 0.4]


def test_cli_validate_config_dispatches_outcome_flip(
    tmp_path, monkeypatch, capsys
) -> None:
    from llm_bias.jspace_intervention import cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                **_config().to_dict(),
                "artifact_type": "outcome_flip_config",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "validate-config",
            "--config",
            str(config_path),
        ],
    )
    cli.main()
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["fitted_layers"] == [1, 2]
    assert emitted["safety_bound"] == 0.5


def test_cli_run_outcome_flip_dispatch(monkeypatch, tmp_path) -> None:
    from llm_bias.jspace_intervention import cli, outcome_flip

    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(
        outcome_flip,
        "run_outcome_flip_pipeline",
        fake_pipeline,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                **_config().to_dict(),
                "artifact_type": "outcome_flip_config",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {"artifact_type": "jspace_intervention_splits", "assignments": {}}
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "trial_plan_prompts.csv"
    input_path.write_text("Date,ticker,prompt_with_context_attribute_0\n", encoding="utf-8")
    identity_path = tmp_path / "identity.json"
    identity_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "run-outcome-flip",
            "--input",
            str(input_path),
            "--split-manifest",
            str(split_path),
            "--config",
            str(config_path),
            "--model",
            "fake-model",
            "--run-id",
            "v2-dispatch",
            "--split",
            "calibration",
            "--direction-identity",
            str(identity_path),
        ],
    )
    cli.main()
    assert captured["split_name"] == "calibration"
    assert captured["direction_identity_path"] == identity_path
    assert captured["model_name"] == "fake-model"
