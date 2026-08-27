"""Regression tests for the J-space valence vocabulary readout workflow."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llm_bias.core.artifact_paths import run_root, sha256_file
from llm_bias.jspace_intervention import valence_readout
from llm_bias.jspace_intervention.valence import (
    build_valence_pair,
    evidence_item_hash,
    render_valence_prompt,
    resolve_valence_layers,
    select_valence_trials,
    validate_raw_trial_row,
)
from llm_bias.jspace_intervention.valence_readout import (
    CANDIDATE_LABEL,
    MIN_MEAN_PROBABILITY,
    analyze_valence_contrast,
    build_frozen_candidates,
    rank_valence_tokens,
    run_valence_readout_pipeline,
)


class _Tokenized:
    def __init__(self, ids, *, offsets=None, specials=None):
        self.input_ids = ids
        if offsets is not None:
            self.offset_mapping = offsets
        if specials is not None:
            self.special_tokens_mask = specials


class _FakeTokenizer:
    """Handcrafted single-token coverage for eligibility rules."""

    chat_template = "fake"
    all_special_tokens = []
    tokens = {
        " alpha": 10,
        " beta": 11,
        " gamma": 12,
        " buy": 13,
        " sell": 14,
        " risk": 15,
        " {": 16,
        " decision": 17,
    }

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
    ):
        if text in self.tokens:
            return _Tokenized([self.tokens[text]])
        ids = [ord(char) % 97 + 1 for char in text]
        if return_offsets_mapping:
            offsets = [(index, index + 1) for index in range(len(text))]
            return _Tokenized(ids, offsets=offsets, specials=[0] * len(ids))
        return _Tokenized(ids)

    def apply_chat_template(self, messages, **_kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def decode(self, ids, **_kwargs):
        reverse = {value: key for key, value in self.tokens.items()}
        return reverse.get(ids[0], f"tok{ids[0]}")


def _raw_row(
    ticker="T1",
    sector="Technology",
    trial_key="key",
    trial_index=0,
    set_index=0,
    condition="attribute",
    evidence=None,
):
    if evidence is None:
        evidence = [
            {"kind": "qual", "side": "buy", "text": f"positive qual {ticker}"},
            {"kind": "qual", "side": "sell", "text": f"negative qual {ticker}"},
            {"kind": "quant", "side": "buy", "text": f"positive quant {ticker}"},
            {"kind": "quant", "side": "sell", "text": f"negative quant {ticker}"},
        ]
    return {
        "condition": condition,
        "evidence": evidence,
        "marketcap": 1.0e9,
        "model": "fake",
        "name": f"Name {ticker}",
        "prompt": "unused-original-four-item-body",
        "sector": sector,
        "seed": 1,
        "set_index": set_index,
        "ticker": ticker,
        "trial_index": trial_index,
        "trial_key": trial_key,
    }


def test_validate_raw_trial_row_requires_attribute_condition_and_full_evidence() -> None:
    identity = validate_raw_trial_row(_raw_row())
    assert identity["ticker"] == "T1"
    assert identity["evidence"][("buy", "qual")] == "positive qual T1"
    assert identity["trial_key"] == "key"

    with pytest.raises(ValueError, match="requires 'attribute'"):
        validate_raw_trial_row(_raw_row(condition="intensity"))
    with pytest.raises(ValueError, match="missing field"):
        validate_raw_trial_row({**_raw_row(), "ticker": "  "})
    with pytest.raises(ValueError, match="trial identity"):
        validate_raw_trial_row({**_raw_row(), "trial_index": "0"})
    broken = {
        **_raw_row(),
        "evidence": [
            {"kind": "qual", "side": "buy", "text": "x"},
            {"kind": "quant", "side": "buy", "text": "y"},
            {"kind": "qual", "side": "sell", "text": "z"},
        ],
    }
    with pytest.raises(ValueError, match="missing items"):
        validate_raw_trial_row(broken)
    duplicated = {
        **_raw_row(),
        "evidence": [
            {"kind": "qual", "side": "buy", "text": "x"},
            {"kind": "qual", "side": "buy", "text": "y"},
            {"kind": "quant", "side": "buy", "text": "z"},
            {"kind": "qual", "side": "sell", "text": "w"},
            {"kind": "quant", "side": "sell", "text": "v"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_trial_row(duplicated)


def test_evidence_item_hash_is_stable_and_distinct() -> None:
    first = evidence_item_hash("buy", "qual", "text")
    assert first == evidence_item_hash("buy", "qual", "text")
    assert first != evidence_item_hash("sell", "qual", "text")
    assert first != evidence_item_hash("buy", "quant", "text")


def test_select_valence_trials_is_seeded_hash_ordered_and_deterministic() -> None:
    assignments = {"T1": "discovery", "T2": "discovery", "T3": "discovery",
                   "T4": "test", "S1": "discovery"}
    rows = []
    for ticker in ("T1", "T2", "T3"):
        for trial in range(4):
            rows.append(_raw_row(
                ticker=ticker,
                trial_index=trial,
                set_index=trial,
                trial_key=f"{ticker}-k{trial}",
            ))
    # Out-of-scope rows: other sector, other split, other condition (broken
    # evidence on the intensity row must be skipped, not an error).
    rows.append(_raw_row(ticker="S1", sector="Utilities", trial_key="S1-k0"))
    rows.append(_raw_row(ticker="T4", trial_key="T4-k0"))
    rows.append(_raw_row(ticker="T1", trial_key="T1-intensity", condition="intensity"))
    # A valid attribute trial with two buy-qual and two sell-quant items is
    # ineligible for matched qual/quant pairing and must be skipped, not fail.
    rows.append(_raw_row(
        ticker="T1",
        trial_key="T1-unbalanced",
        evidence=[
            {"kind": "qual", "side": "buy", "text": "buy q1"},
            {"kind": "qual", "side": "buy", "text": "buy q2"},
            {"kind": "quant", "side": "sell", "text": "sell n1"},
            {"kind": "quant", "side": "sell", "text": "sell n2"},
        ],
    ))

    selected = select_valence_trials(
        rows,
        sector="Technology",
        split_assignments=assignments,
        split_name="discovery",
        trials_per_ticker=2,
        seed=7,
    )

    expected = []
    for ticker in ("T1", "T2", "T3"):
        keys = sorted(
            (f"{ticker}-k{trial}" for trial in range(4)),
            key=lambda key: hashlib.sha256(f"7:{ticker}:{key}".encode()).hexdigest(),
        )
        expected.extend(keys[:2])
    assert [row["trial_key"] for row in selected] == expected
    assert len(selected) == 6
    assert selected == select_valence_trials(
        rows,
        sector="Technology",
        split_assignments=assignments,
        split_name="discovery",
        trials_per_ticker=2,
        seed=7,
    )
    with pytest.raises(ValueError, match="no .* attribute rows match"):
        select_valence_trials(
            rows,
            sector="Technology",
            split_assignments=assignments,
            split_name="calibration",
            trials_per_ticker=2,
            seed=7,
        )


def test_build_valence_pair_renders_canonical_prompts_and_preserves_hashes() -> None:
    row = _raw_row(ticker="V9")
    pair = build_valence_pair(row)

    assert pair["source_trial_key"] == "key"
    assert pair["record_id"].startswith("record_")
    assert set(pair["evidence_item_hashes"]) == {
        "buy_qual", "buy_quant", "sell_qual", "sell_quant",
    }
    positive = pair["prompts"]["positive"]
    negative = pair["prompts"]["negative"]
    header = "Refer to the evidence below to make a final investment decision.\n"
    assert positive.startswith(header + "Stock Ticker: [V9]\nStock Name: [Name V9]\n--- Evidence ---\n")
    assert positive.split("--- Evidence ---\n")[0] == negative.split("--- Evidence ---\n")[0]
    assert "positive qual V9" in positive and "positive quant V9" in positive
    assert "negative qual V9" not in positive and "negative quant V9" not in positive
    assert "negative qual V9" in negative and "negative quant V9" in negative
    assert "positive qual V9" not in negative and "positive quant V9" not in negative
    instruction = (
        'Respond with one valid JSON object containing only the keys "decision" '
        '(buy | sell) and "reason" (brief justification). Do not choose hold.'
    )
    assert positive.endswith("---\n" + instruction)
    assert negative.endswith("---\n" + instruction)
    # The original four-item prompt body is never reused.
    assert "unused-original-four-item-body" not in positive + negative
    # Rendering is canonical from the structured evidence only.
    assert positive == render_valence_prompt(
        ticker="V9",
        name="Name V9",
        qual_text="positive qual V9",
        quant_text="positive quant V9",
    )
    # Stable across rebuilds.
    assert build_valence_pair(row) == pair


def test_resolve_valence_layers_adds_final_layer_and_validates_range() -> None:
    assert resolve_valence_layers([15, 14, 15], 63) == [14, 15, 63]
    assert resolve_valence_layers([63], 63) == [63]
    with pytest.raises(ValueError, match="out of range"):
        resolve_valence_layers([64], 63)
    with pytest.raises(ValueError, match="out of range"):
        resolve_valence_layers([-1], 63)


def test_rank_valence_tokens_scores_match_closed_form() -> None:
    positive = torch.tensor([0.6, 0.4, 0.0])
    negative = torch.tensor([0.2, 0.4, 0.4])
    rows = rank_valence_tokens(positive, negative, side="positive", top_k=3)
    assert [row["token_id"] for row in rows] == [0, 1, 2]
    top = rows[0]
    assert top["probability_diff"] == pytest.approx(0.4)
    assert top["smoothed_log_ratio"] == pytest.approx(math.log(3.0))
    assert top["js_contribution"] == pytest.approx(
        0.5 * (0.6 * math.log(0.6 / 0.4) + 0.2 * math.log(0.2 / 0.4)), rel=1e-6
    )
    middle = rows[1]
    assert middle["probability_diff"] == pytest.approx(0.0)
    assert middle["smoothed_log_ratio"] == pytest.approx(0.0)
    assert middle["js_contribution"] == pytest.approx(0.0, abs=1e-9)
    zero_mass = rows[2]
    assert zero_mass["js_contribution"] == pytest.approx(
        0.5 * 0.4 * math.log(0.4 / 0.2), rel=1e-6
    )
    negative_side = rank_valence_tokens(positive, negative, side="negative", top_k=3)
    assert [row["token_id"] for row in negative_side] == [2, 1, 0]
    with pytest.raises(ValueError, match="side must be one of"):
        rank_valence_tokens(positive, negative, side="buy")


def test_valence_readout_averages_full_softmax_before_topk() -> None:
    # Per-prompt positive distributions: token 0 is top-1 in BOTH prompts and
    # token 1 is only ever top-2, so a per-prompt top-k (average-top-k) scheme
    # could never rank token 1 first. The averaged full-softmax means put
    # token 1 first.
    p1 = torch.tensor([0.40, 0.35, 0.15, 0.10])
    p2 = torch.tensor([0.39, 0.34, 0.17, 0.10])
    q = torch.tensor([0.30, 0.20, 0.25, 0.25])
    assert int(torch.argmax(p1)) == 0 and int(torch.argmax(p2)) == 0
    assert int(p1.argsort(descending=True)[1]) == 1
    assert int(p2.argsort(descending=True)[1]) == 1

    condition_sums = {("positive", 14): p1 + p2, ("negative", 14): q + q}
    condition_counts = {("positive", 14): 2, ("negative", 14): 2}
    contrast = analyze_valence_contrast(
        condition_sums=condition_sums,
        condition_counts=condition_counts,
        layers=[14],
        band_layers=[14],
    )
    positive_rows = contrast["layer_14"]["positive"]
    assert [row["token_id"] for row in positive_rows[:2]] == [1, 0]
    assert positive_rows[0]["probability_diff"] == pytest.approx(0.345 - 0.20)
    assert positive_rows[0]["mean_positive"] == pytest.approx(0.345)
    assert positive_rows[0]["mean_negative"] == pytest.approx(0.20)
    # Band scope (single layer here) agrees with the layer scope.
    assert contrast["band_14-14"]["positive"][0]["token_id"] == 1
    negative_rows = contrast["layer_14"]["negative"]
    assert [row["token_id"] for row in negative_rows[:2]] == [3, 2]


def _row(token, token_id, side_mean, own_mean, diff, log_ratio=2.0, js=0.01):
    row = {
        "rank": 1,
        "token": token,
        "token_id": token_id,
        "mean_positive": 0.001,
        "mean_negative": 0.001,
        "probability_diff": diff,
        "smoothed_log_ratio": log_ratio,
        "js_contribution": js,
    }
    row["mean_positive" if side_mean == "positive" else "mean_negative"] = own_mean
    return row


def test_frozen_candidates_apply_eligibility_and_sign_consistency() -> None:
    tokenizer = _FakeTokenizer()
    ranked = {
        "positive": [
            _row(" alpha", 10, "positive", 0.02, 0.019),
            _row(" buy", 13, "positive", 0.5, 0.4),          # answer word: excluded
            _row(" {", 16, "positive", 0.3, 0.2),            # punctuation: excluded
            _row(" decision", 17, "positive", 0.3, 0.2),     # format boilerplate: excluded
            _row(" fragmentary", 99, "positive", 0.3, 0.2),  # multi-token: excluded
            _row(" beta", 11, "positive", 0.015, 0.014),     # sign-inconsistent ticker
            _row(" risk", 15, "positive", 1e-7, 1e-7),       # below min mean probability
        ],
        "negative": [
            _row(" gamma", 12, "negative", 0.03, -0.03),
        ],
    }
    # (ticker x band layer) difference vectors (positive - negative):
    # alpha always +, beta flips sign, gamma always -, risk always +.
    diff_vectors = [
        torch.tensor([0.0] * 10 + [0.01, 0.01, -0.02, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0] * 10 + [0.02, -0.05, -0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    candidates = build_frozen_candidates(
        ranked=ranked,
        tokenizer=tokenizer,
        diff_vectors=diff_vectors,
        n_tickers=1,
        n_layers=2,
    )
    assert [candidate["concept"] for candidate in candidates] == ["gamma", "alpha"]
    assert candidates[0]["side"] == "negative"
    assert candidates[1]["side"] == "positive"
    assert candidates[1]["token_id"] == 10
    assert all(candidate["label"] == CANDIDATE_LABEL for candidate in candidates)
    assert all(candidate["sign_consistent_tickers"] == 1 for candidate in candidates)
    assert all(candidate["sign_consistent_layers"] == 2 for candidate in candidates)
    assert all(candidate["leave_one_ticker_out_sign_stable"] for candidate in candidates)
    with pytest.raises(ValueError, match="difference vector"):
        build_frozen_candidates(
            ranked=ranked, tokenizer=tokenizer, diff_vectors=[],
            n_tickers=1, n_layers=1,
        )


class _ManyTokenTokenizer:
    chat_template = "fake"
    all_special_tokens = []

    def __init__(self, count=14):
        self.tokens = {f" c{i}": 100 + i for i in range(count)}

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        if text in self.tokens:
            return _Tokenized([self.tokens[text]])
        return _Tokenized([ord(char) % 97 + 1 for char in text])

    def decode(self, ids, **_kwargs):
        reverse = {value: key for key, value in self.tokens.items()}
        return reverse.get(ids[0], f"tok{ids[0]}")


def test_frozen_candidates_are_capped_and_ordered_by_band_difference() -> None:
    tokenizer = _ManyTokenTokenizer(count=14)
    vector = torch.zeros(114)
    vector[100:114] = 1.0
    ranked = {
        "positive": [
            _row(f" c{i}", 100 + i, "positive", 0.01, 0.14 - 0.01 * i)
            for i in range(14)
        ],
        "negative": [],
    }
    candidates = build_frozen_candidates(
        ranked=ranked,
        tokenizer=tokenizer,
        diff_vectors=[vector],
        n_tickers=1,
        n_layers=1,
        min_mean_probability=MIN_MEAN_PROBABILITY,
    )
    assert [candidate["concept"] for candidate in candidates] == [f"c{i}" for i in range(12)]
    assert candidates[0]["band_probability_diff"] == pytest.approx(0.14)


def test_run_valence_readout_pipeline_with_fake_model(tmp_path, monkeypatch) -> None:
    import jlens

    class _FakeValenceModel:
        n_layers = 5
        d_model = 8
        device = torch.device("cpu")

        def __init__(self, vocab=30, seed=0):
            generator = torch.Generator().manual_seed(seed)
            self._embed_tokens = torch.nn.Embedding(vocab, 8)
            torch.nn.init.normal_(self._embed_tokens.weight, generator=generator)
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(5)])
            self._final_norm = torch.nn.LayerNorm(8)
            self._lm_head = torch.nn.Linear(8, vocab, bias=False)
            torch.nn.init.normal_(self._lm_head.weight, generator=generator)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = self._embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return SimpleNamespace(last_hidden_state=self._final_norm(hidden))

        def unembed(self, residual):
            weight_dtype = self._lm_head.weight.dtype
            return self._lm_head(self._final_norm(residual.to(weight_dtype)))

    class _WorkflowTokenizer:
        chat_template = "fake"
        all_special_tokens = []
        pad_token_id = 0

        def __init__(self, vocab=30):
            self.vocab = vocab

        def __call__(
            self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs
        ):
            ids = [ord(char) % (self.vocab - 1) + 1 for char in text]
            if return_offsets_mapping:
                offsets = [(index, index + 1) for index in range(len(text))]
                return _Tokenized(ids, offsets=offsets, specials=[0] * len(ids))
            return _Tokenized(ids)

        def apply_chat_template(self, messages, **_kwargs):
            return "<user>" + messages[0]["content"] + "<assistant>"

        def decode(self, ids, **_kwargs):
            return "".join(chr(97 + (int(token) - 1) % 26) for token in ids)

    model = _FakeValenceModel()
    tokenizer = _WorkflowTokenizer()
    lens = jlens.JacobianLens({2: torch.eye(8), 3: torch.eye(8)}, n_prompts=1, d_model=8)
    lens_file = tmp_path / "jacobian_lens.pt"
    lens_file.write_bytes(b"fake-lens")
    monkeypatch.setattr(valence_readout, "load_tokenizer", lambda model_name: tokenizer)
    monkeypatch.setattr(
        valence_readout, "load_model",
        lambda model_name: (model, tokenizer, torch.device("cpu")),
    )
    monkeypatch.setattr(
        valence_readout, "load_validated_lens",
        lambda **_kwargs: SimpleNamespace(lens=lens, path=lens_file, source="fake"),
    )

    rows = []
    for ticker in ("V1", "V2"):
        for trial in range(3):
            rows.append(_raw_row(
                ticker=ticker, trial_index=trial, set_index=trial,
                trial_key=f"{ticker}-key-{trial}",
            ))
    raw_input = tmp_path / "trial_plan.jsonl"
    raw_input.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(json.dumps({
        "artifact_type": "jspace_intervention_splits",
        "schema_version": 1,
        "input": "fake-source",
        "input_sha256": "ab" * 32,
        "seed": 0,
        "ratios": {"discovery": 3, "calibration": 1, "test": 1},
        "assignments": {"V1": "discovery", "V2": "discovery", "V3": "test"},
    }), encoding="utf-8")

    run_root = run_valence_readout_pipeline(
        input_path=raw_input,
        split_manifest=split_manifest,
        model_name="fake-model",
        run_id="valence-test",
        artifact_root=tmp_path / "artifacts",
        trials_per_ticker=2,
        layers=[2, 3],
        top_k=5,
        max_seq_len=1024,
        seed=7,
    )

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for stage in ("prepare", "forward", "analyze"):
        assert manifest["stages"][stage]["status"] == "complete"
    input_types = {ref["artifact_type"] for ref in manifest["input_refs"]}
    assert input_types == {"raw_trial_plan", "jspace_intervention_splits"}
    assert [ref["artifact_type"] for ref in manifest["lens_refs"]] == ["jacobian_lens"]

    pairs = [
        json.loads(line)
        for line in (run_root / "prepare" / "valence_pairs.jsonl").open()
    ]
    # 2 tickers x trials_per_ticker=2 = 4 source-trial pairs.
    assert len(pairs) == 4
    assert {pair["ticker"] for pair in pairs} == {"V1", "V2"}
    for pair in pairs:
        assert pair["artifact_type"] == "valence_pairs"
        assert set(pair["evidence_item_hashes"]) == {
            "buy_qual", "buy_quant", "sell_qual", "sell_quant",
        }
        assert f"positive qual {pair['ticker']}" in pair["prompts"]["positive"]
        assert f"negative qual {pair['ticker']}" not in pair["prompts"]["positive"]
        assert f"negative qual {pair['ticker']}" in pair["prompts"]["negative"]
    prepare_metadata = json.loads(
        (run_root / "prepare" / "metadata.json").read_text(encoding="utf-8")
    )
    assert prepare_metadata["raw_input_sha256"] == sha256_file(raw_input)
    assert prepare_metadata["split_manifest_sha256"] == sha256_file(split_manifest)
    assert prepare_metadata["split_input_sha256"] == "ab" * 32
    assert prepare_metadata["pair_count"] == 4
    assert prepare_metadata["ticker_count"] == 2

    readout = [
        json.loads(line)
        for line in (run_root / "forward" / "valence_readout.jsonl").open()
    ]
    # 4 pairs x 2 conditions = 8 prompt readouts.
    assert len(readout) == 8
    assert {record["condition"] for record in readout} == {"positive", "negative"}
    for record in readout:
        assert record["artifact_type"] == "valence_readout"
        assert [layer["layer"] for layer in record["layers"]] == [2, 3, 4]
        assert [layer["is_output"] for layer in record["layers"]] == [False, False, True]
        assert record["readout_position"] == "mean_of_qual_and_quant_evidence_item_end_tokens"
        for layer in record["layers"]:
            assert [item["kind"] for item in layer["readout_positions"]] == ["qual", "quant"]
            assert 1 <= len(layer["top_tokens"]) <= 5
            assert [token["rank"] for token in layer["top_tokens"]] == list(
                range(1, len(layer["top_tokens"]) + 1)
            )
            assert layer["entropy_nats"] > 0
            assert layer["effective_temperature"] > 0
    forward_metadata = json.loads(
        (run_root / "forward" / "metadata.json").read_text(encoding="utf-8")
    )
    assert forward_metadata["layers"] == [2, 3, 4]
    assert forward_metadata["band_layers"] == [2, 3]
    assert forward_metadata["final_layer"] == 4
    assert forward_metadata["readout_position"] == "mean_of_qual_and_quant_evidence_item_end_tokens"

    contrast = [
        json.loads(line)
        for line in (run_root / "analyze" / "valence_token_contrast.jsonl").open()
    ]
    scopes = {row["scope"] for row in contrast}
    assert scopes == {"layer_2", "layer_3", "layer_4", "band_2-3"}
    # top_k_per_side (50) is capped by the fake vocabulary (30).
    assert len(contrast) == 4 * 2 * 30
    for row in contrast:
        assert row["side"] in {"positive", "negative"}
        assert {"probability_diff", "smoothed_log_ratio", "js_contribution"} <= set(row)

    candidates_document = json.loads(
        (run_root / "analyze" / "frozen_candidate_suggestions.json").read_text(encoding="utf-8")
    )
    # The fake tokenizer decodes no leading-space single tokens, so no
    # candidate can satisfy single_leading_space_token eligibility.
    assert candidates_document["candidates"] == []
    assert candidates_document["criteria"]["min_mean_probability"] == MIN_MEAN_PROBABILITY


def test_preflight_length_failure_creates_no_run(tmp_path, monkeypatch) -> None:
    class _LongTokenizer:
        chat_template = "fake"
        all_special_tokens = []
        pad_token_id = 0

        def __call__(
            self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs
        ):
            ids = list(range(len(text)))
            if return_offsets_mapping:
                offsets = [(index, index + 1) for index in range(len(text))]
                return _Tokenized(ids, offsets=offsets, specials=[0] * len(ids))
            return _Tokenized(ids)

        def apply_chat_template(self, messages, **_kwargs):
            return "<user>" + messages[0]["content"] + "<assistant>"

        def decode(self, ids, **_kwargs):
            return "tok"

    raw_input = tmp_path / "trial_plan.jsonl"
    raw_input.write_text(json.dumps(_raw_row()) + "\n", encoding="utf-8")
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(json.dumps({
        "artifact_type": "jspace_intervention_splits",
        "input_sha256": "cd" * 32,
        "assignments": {"T1": "discovery"},
    }), encoding="utf-8")
    monkeypatch.setattr(valence_readout, "load_tokenizer", lambda model_name: _LongTokenizer())

    with pytest.raises(ValueError, match="limit 50"):
        run_valence_readout_pipeline(
            input_path=raw_input,
            split_manifest=split_manifest,
            model_name="fake-model",
            run_id="preflight",
            artifact_root=tmp_path / "artifacts",
            max_seq_len=50,
        )
    expected_root = run_root(
        "fake-model", "jspace-valence-readout", "preflight",
        artifact_root=tmp_path / "artifacts",
    )
    assert not expected_root.exists()


def test_run_valence_readout_cli_defaults() -> None:
    from llm_bias.jspace_intervention.cli import build_parser

    args = build_parser().parse_args([
        "run-valence-readout",
        "--input", "raw.jsonl",
        "--split-manifest", "splits.json",
        "--model", "fake",
        "--run-id", "valence",
    ])
    assert args.sector == "Technology"
    assert args.split == "discovery"
    assert args.trials_per_ticker == 3
    assert args.layers == ",".join(str(layer) for layer in range(14, 27))
    assert args.top_k == 30
    assert args.max_seq_len == 1024
    assert args.seed == 0
    assert args.dataset == "jspace-valence-readout"
    assert args.artifact_root == "artifacts"
    assert args.lens is None


def test_run_valence_readout_cli_dispatch(monkeypatch, tmp_path) -> None:
    from llm_bias.jspace_intervention import cli

    calls: list[dict] = []
    module = ModuleType("llm_bias.jspace_intervention.valence_readout")

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return tmp_path / "run"

    module.run_valence_readout_pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "llm_bias.jspace_intervention.valence_readout", module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "run-valence-readout",
            "--input", "raw.jsonl",
            "--split-manifest", "splits.json",
            "--model", "fake",
            "--run-id", "valence",
            "--layers", "4,5",
            "--trials-per-ticker", "2",
            "--sector", "Healthcare",
        ],
    )
    cli.main()
    assert len(calls) == 1
    assert calls[0]["layers"] == [4, 5]
    assert calls[0]["trials_per_ticker"] == 2
    assert calls[0]["sector"] == "Healthcare"
    assert calls[0]["split_name"] == "discovery"
    assert calls[0]["seed"] == 0
