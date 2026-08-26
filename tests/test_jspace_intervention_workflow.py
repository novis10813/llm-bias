from __future__ import annotations

import torch
import pytest

from llm_bias.jspace_intervention.analysis import ticker_clustered_effect
from llm_bias.jspace_intervention.candidates import select_prototype
from llm_bias.jspace_intervention.pipeline import run_swap_pipeline
from llm_bias.jspace_intervention.prompting import prepare_scoring_prompt
from llm_bias.jspace_intervention.runner import layer_prototypes
from llm_bias.jspace_intervention.schemas import InterventionConfig, PrototypeSpec
from llm_bias.jspace_intervention.splits import (
    assign_balanced_ticker_splits,
    assign_ticker_splits,
    split_counts,
)


class _Tokenized:
    def __init__(self, ids):
        self.input_ids = ids


class _FakeTokenizer:
    chat_template = "fake"
    tokens = {" alpha": 10, " beta": 11, " risk": 12, " sell": 13}

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
    ):
        if text in self.tokens:
            return _Tokenized([self.tokens[text]])
        result = _Tokenized([ord(char) for char in text])
        if return_offsets_mapping:
            result.offset_mapping = [(index, index + 1) for index in range(len(text))]
            result.special_tokens_mask = [0] * len(text)
        return result

    def apply_chat_template(self, messages, **_kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def decode(self, ids, **_kwargs):
        reverse = {value: key for key, value in self.tokens.items()}
        return reverse[ids[0]]


def test_ticker_splits_are_grouped_sector_stratified_and_deterministic() -> None:
    mapping = {
        **{f"T{i}": "Technology" for i in range(10)},
        **{f"F{i}": "Financial Services" for i in range(10)},
    }
    first = assign_ticker_splits(mapping, seed=7)
    second = assign_ticker_splits(mapping, seed=7)

    assert first == second
    counts = split_counts(first, mapping)
    assert counts["Technology"] == {"discovery": 6, "calibration": 2, "test": 2}
    assert counts["Financial Services"] == {
        "discovery": 6,
        "calibration": 2,
        "test": 2,
    }


def test_balanced_split_handles_incomplete_five_company_block() -> None:
    mapping = {f"T{i}": "Technology" for i in range(58)}
    rates = {ticker: index / 58 for index, ticker in enumerate(mapping)}
    marketcaps = {ticker: float(index + 1) for index, ticker in enumerate(mapping)}

    assignments = assign_balanced_ticker_splits(
        mapping, rates, marketcaps, seed=7
    )

    assert len(assignments) == 58
    assert set(assignments.values()) == {"discovery", "calibration", "test"}


def test_candidate_selection_requires_complete_single_tokens() -> None:
    rows = [
        {"document": "Technology", "token": "alpha", "logodds_z": 4.0},
        {"document": "Technology", "token": "fragment", "logodds_z": 3.5},
        {"document": "Technology", "token": "beta", "logodds_z": 3.0},
        {"document": "Technology", "token": "sell", "logodds_z": 9.0},
    ]
    spec = select_prototype(
        rows,
        tokenizer=_FakeTokenizer(),
        sector="Technology",
        score_type="logodds_z",
        top_n=2,
    )

    assert [token.token for token in spec.tokens] == ["alpha", "beta"]
    assert sum(token.weight for token in spec.tokens) == pytest.approx(1.0)


def test_contrastive_tfidf_uses_positive_difference_and_disjoint_words() -> None:
    rows = [
        {"document": "Technology", "token": "alpha", "tfidf": 0.8},
        {"document": "Technology", "token": "risk", "tfidf": 0.2},
        {"document": "Financial Services", "token": "alpha", "tfidf": 0.3},
        {"document": "Financial Services", "token": "risk", "tfidf": 0.9},
    ]
    technology = select_prototype(
        rows,
        tokenizer=_FakeTokenizer(),
        sector="Technology",
        score_type="tfidf",
        contrast_sector="Financial Services",
        top_n=2,
    )
    financial = select_prototype(
        rows,
        tokenizer=_FakeTokenizer(),
        sector="Financial Services",
        score_type="tfidf",
        contrast_sector="Technology",
        top_n=2,
    )

    assert technology.score_type == "contrastive_tfidf"
    assert [token.token for token in technology.tokens] == ["alpha"]
    assert technology.tokens[0].selection_score == pytest.approx(0.5)
    assert [token.token for token in financial.tokens] == ["risk"]


def test_intervention_config_requires_noop_and_cross_sector() -> None:
    prototype = {
        "name": "Technology:logodds_z",
        "sector": "Technology",
        "score_type": "logodds_z",
        "tokens": [{"token": "alpha", "token_id": 10, "weight": 1.0}],
    }
    other = {
        **prototype,
        "name": "Financial Services:logodds_z",
        "sector": "Financial Services",
    }
    config = InterventionConfig.from_dict(
        {"source": prototype, "target": other, "layers": [14, 15], "alphas": [0, 1]}
    )
    assert config.layers == (14, 15)

    with pytest.raises(ValueError, match="no-op"):
        InterventionConfig.from_dict(
            {"source": prototype, "target": other, "layers": [14], "alphas": [1]}
        )

    answer_prototype = {
        **prototype,
        "tokens": [{"token": "buy", "token_id": 10, "weight": 1.0}],
    }
    with pytest.raises(ValueError, match="answer candidates"):
        InterventionConfig.from_dict(
            {
                "source": answer_prototype,
                "target": other,
                "layers": [14],
                "alphas": [0, 1],
            }
        )


def test_pipeline_rejects_split_manifest_not_frozen_in_config(tmp_path) -> None:
    split = tmp_path / "splits.json"
    split.write_text('{"assignments":{"T":"test"}}')
    config = tmp_path / "config.json"
    config.write_text('{"split_manifest_sha256":"' + "0" * 64 + '"}')

    with pytest.raises(ValueError, match="does not match"):
        run_swap_pipeline(
            input_path=tmp_path / "missing.csv",
            split_manifest=split,
            config_path=config,
            model_name="unused",
            run_id="unused",
            artifact_root=tmp_path,
        )


def test_ticker_clustered_effect_weights_tickers_equally() -> None:
    rows = [
        {"ticker": "A", "delta_margin": 1.0},
        {"ticker": "A", "delta_margin": 1.0},
        {"ticker": "A", "delta_margin": 1.0},
        {"ticker": "B", "delta_margin": -1.0},
    ]
    summary = ticker_clustered_effect(rows, seed=3, bootstrap_samples=100)

    assert summary["mean_delta_margin"] == pytest.approx(0.0)
    assert summary["ticker_count"] == 2
    assert summary["record_count"] == 4


def test_prompt_preparation_maps_only_the_evidence_span() -> None:
    raw = (
        "Header\n--- Evidence ---\nfirst fact\nsecond fact"
        "\n---\nRespond with one JSON object"
    )
    tokenizer = _FakeTokenizer()
    scoring, span = prepare_scoring_prompt(
        tokenizer, raw, decision_prefix='{\"decision\":\"'
    )

    decoded_span = scoring[span[0] : span[1]]
    assert "first fact" in decoded_span
    assert "second fact" in decoded_span
    assert "Header" not in decoded_span
    assert "Respond" not in decoded_span


def test_layer_prototypes_derive_vectors_from_canonical_jacobians() -> None:
    class Model:
        _lm_head = torch.nn.Linear(2, 3, bias=False)

    model = Model()
    with torch.no_grad():
        model._lm_head.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )

    class Lens:
        jacobians = {14: torch.eye(2)}

    spec = PrototypeSpec.from_dict(
        {
            "name": "test",
            "sector": "Technology",
            "score_type": "logodds_z",
            "tokens": [
                {"token": "alpha", "token_id": 0, "weight": 1.0},
                {"token": "beta", "token_id": 1, "weight": 1.0},
            ],
        }
    )
    prototype = layer_prototypes(model, Lens(), spec, [14])[14]

    assert torch.allclose(prototype, torch.tensor([2**-0.5, 2**-0.5]))
