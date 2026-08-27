"""Regression tests for the minimal J-space token causal screen."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llm_bias.core.artifact_paths import run_root, sha256_file
from llm_bias.jspace_intervention.analysis import analyze_token_screen
from llm_bias.jspace_intervention.runner import (
    run_token_screen_record,
    token_screen_directions,
)
from llm_bias.jspace_intervention.schemas import TokenScreenCandidate, TokenScreenConfig


def _candidate(token_id: int = 1, side: str = "negative", token: str = " alpha") -> dict:
    return {
        "token": token,
        "token_id": token_id,
        "side": side,
        "mean_positive": 0.01,
        "mean_negative": 0.02,
        "band_probability_diff": -0.01,
        "band_smoothed_log_ratio": -1.0,
        "band_js_contribution": 0.01,
    }


def _config_payload(**overrides) -> dict:
    payload = {
        "candidates": [_candidate()],
        "model": "fake-model",
        "source_sector": "Technology",
        "layers": [0],
        "candidate_artifact_path": "candidates.json",
        "candidate_artifact_sha256": "ab" * 32,
        "split_manifest_sha256": "cd" * 32,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_token_screen_config_defaults_and_roundtrip() -> None:
    config = TokenScreenConfig.from_dict(_config_payload())
    assert config.alphas == (-1.0, 0.0, 1.0)
    assert config.controls == ("token", "matched_random")
    assert config.top_positions == 3
    assert config.loading_threshold == 0.0
    assert config.positive_candidate == "buy"
    assert config.negative_candidate == "sell"
    assert config.positive_dose == 1.0
    assert config.candidates[0].representation_side == "negative"
    assert TokenScreenConfig.from_dict(config.to_dict()) == config


def test_token_screen_config_accepts_reordered_symmetric_doses() -> None:
    config = TokenScreenConfig.from_dict(_config_payload(alphas=[0.5, -0.5, 0.0]))
    assert config.positive_dose == 0.5


@pytest.mark.parametrize(
    "alphas",
    [
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, -0.5],
        [float("inf"), 0.0, -float("inf")],
        [-2.0, -1.0, 1.0],
    ],
)
def test_token_screen_config_rejects_non_symmetric_doses(alphas: list) -> None:
    with pytest.raises(ValueError, match="symmetric"):
        TokenScreenConfig.from_dict(_config_payload(alphas=alphas))


def test_token_screen_config_rejects_duplicates_answer_words_and_bad_provenance() -> None:
    with pytest.raises(ValueError, match="unique"):
        TokenScreenConfig.from_dict(
            _config_payload(
                candidates=[_candidate(1), _candidate(1, side="positive", token=" beta")]
            )
        )
    with pytest.raises(ValueError, match="answer candidates"):
        TokenScreenConfig.from_dict(_config_payload(candidates=[_candidate(1, token=" buy")]))
    with pytest.raises(ValueError, match="side"):
        TokenScreenConfig.from_dict(_config_payload(candidates=[_candidate(side="buy")]))
    with pytest.raises(ValueError, match="finite"):
        TokenScreenConfig.from_dict(
            _config_payload(candidates=[{**_candidate(), "mean_positive": float("nan")}])
        )
    with pytest.raises(ValueError, match="SHA-256"):
        TokenScreenConfig.from_dict(_config_payload(split_manifest_sha256="zz" * 32))
    with pytest.raises(ValueError, match="control"):
        TokenScreenConfig.from_dict(_config_payload(controls=["token", "shuffled"]))
    with pytest.raises(ValueError, match="at least one candidate"):
        TokenScreenConfig.from_dict(_config_payload(candidates=[]))
    with pytest.raises(ValueError, match="layers"):
        TokenScreenConfig.from_dict(_config_payload(layers=[1, 1]))
    with pytest.raises(ValueError, match="top_positions"):
        TokenScreenConfig.from_dict(_config_payload(top_positions=0))


def test_token_screen_candidate_reads_side_alias() -> None:
    candidate = TokenScreenCandidate.from_dict(_candidate(side="positive"))
    assert candidate.representation_side == "positive"
    assert candidate.to_dict()["representation_side"] == "positive"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _TupleBlock(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class _ScreenModel(torch.nn.Module):
    """One identity layer; token 1 = buy, token 2 = sell; score at position 0."""

    def __init__(self, prompt_embedding=(1.0, 2.0)) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TupleBlock()])
        self.n_layers = 1
        self.embedding = torch.nn.Embedding(3, 2)
        self.register_buffer(
            "unembedding",
            torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        )
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[0] = torch.tensor(prompt_embedding)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        value = self.embedding(input_ids)
        for layer in self.layers:
            value = layer(value)[0]
        return value

    def unembed(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.unembedding.T


class _ScreenTokenizer:
    mapping = {"P": [0], "Pbuy": [0, 1], "Psell": [0, 2]}

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        return {"input_ids": self.mapping[text]}


def _screen_config(**overrides) -> TokenScreenConfig:
    return TokenScreenConfig.from_dict(_config_payload(**overrides))


def _run_screen(model, config, directions, control_seed=0):
    return run_token_screen_record(
        model=model,
        tokenizer=_ScreenTokenizer(),
        scoring_prompt="P",
        evidence_span=(0, 1),
        config=config,
        device="cpu",
        directions=directions,
        control_seed=control_seed,
    )


def test_token_screen_directions_are_unembedding_rows_of_jacobians() -> None:
    class HeadModel:
        _lm_head = torch.nn.Linear(2, 3, bias=False)

    model = HeadModel()
    with torch.no_grad():
        model._lm_head.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )

    class Lens:
        jacobians = {14: torch.tensor([[2.0, 0.0], [0.0, 3.0]])}

    candidates = [
        TokenScreenCandidate.from_dict(_candidate(1, side="positive")),
        TokenScreenCandidate.from_dict(_candidate(2, side="negative", token=" beta")),
    ]
    directions = token_screen_directions(model, Lens(), candidates, [14])
    assert torch.allclose(directions[1][14], torch.tensor([0.0, 3.0]))
    assert torch.allclose(directions[2][14], torch.tensor([2.0, 3.0]))
    with pytest.raises(ValueError, match="does not contain layer"):
        token_screen_directions(model, Lens(), candidates, [15])


def test_token_screen_runner_symmetric_dose_math_and_scales() -> None:
    model = _ScreenModel()
    config = _screen_config(top_positions=1)
    directions = {1: {0: torch.tensor([0.0, 1.0])}}
    rows = _run_screen(model, config, directions)

    assert len(rows) == 6
    by_arm_alpha = {(row["arm"], row["alpha"]): row for row in rows}
    token_plus = by_arm_alpha[("token", 1.0)]
    token_minus = by_arm_alpha[("token", -1.0)]
    token_zero = by_arm_alpha[("token", 0.0)]

    # Clean margin at position 0: logits [0, 2, 1] -> M = 1.
    assert token_plus["clean_margin"] == pytest.approx(1.0, abs=1e-5)
    # +1: scale = median |coordinate| = 2.0, delta = +2 -> [1, 4] -> M = 3.
    assert token_plus["intervened_margin"] == pytest.approx(3.0, abs=1e-5)
    assert token_plus["delta_margin"] == pytest.approx(2.0, abs=1e-5)
    # -1 subtracts the direction: [1, 0] -> M = -1 (side is provenance only).
    assert token_minus["intervened_margin"] == pytest.approx(-1.0, abs=1e-5)
    assert token_minus["delta_margin"] == pytest.approx(-2.0, abs=1e-5)
    assert token_zero["delta_margin"] == 0.0
    assert token_zero["intervened_margin"] == pytest.approx(token_zero["clean_margin"])
    assert token_plus["delivered_alpha"] == 1.0
    assert token_zero["delivered_alpha"] == 0.0

    dose = token_plus["delivered_dose"]
    assert token_plus["coordinate_scale_min"] == pytest.approx(2.0)
    assert token_plus["coordinate_scale_mean"] == pytest.approx(2.0)
    assert token_plus["coordinate_scale_max"] == pytest.approx(2.0)
    assert dose["coordinate_before_mean"] == pytest.approx(2.0, abs=1e-5)
    assert dose["coordinate_after_mean"] == pytest.approx(4.0, abs=1e-5)
    assert dose["perturbation_norm"] == pytest.approx(2.0, rel=1e-5)
    assert dose["state_norm"] == pytest.approx(math.sqrt(5.0), rel=1e-5)
    assert dose["relative_perturbation_norm"] == pytest.approx(2.0 / math.sqrt(5.0), rel=1e-5)
    assert dose["direction_norm_min"] == pytest.approx(1.0)
    assert dose["direction_norm_max"] == pytest.approx(1.0)
    assert token_plus["loaded_positions"]["loaded"] is True
    assert token_plus["intervention_positions"] == [0]
    assert not model.layers[0]._forward_hooks


def test_token_screen_random_arm_matches_norm_positions_and_scale() -> None:
    model = _ScreenModel()
    config = _screen_config()
    directions = {1: {0: torch.tensor([0.0, 1.0])}}
    rows = _run_screen(model, config, directions)
    by_arm_alpha = {(row["arm"], row["alpha"]): row for row in rows}

    for alpha in (-1.0, 1.0):
        token = by_arm_alpha[("token", alpha)]
        random = by_arm_alpha[("matched_random", alpha)]
        # Same PRIMARY positions and local scale...
        assert random["intervention_positions"] == token["intervention_positions"]
        assert random["coordinate_scale_min"] == pytest.approx(
            token["coordinate_scale_min"]
        )
        assert random["coordinate_scale_max"] == pytest.approx(
            token["coordinate_scale_max"]
        )
        # ...and a same-norm direction, so the delivered norm matches exactly.
        assert random["delivered_dose"]["direction_norm_min"] == pytest.approx(
            token["delivered_dose"]["direction_norm_min"]
        )
        assert random["delivered_dose"]["perturbation_norm"] == pytest.approx(
            token["delivered_dose"]["perturbation_norm"], rel=1e-5
        )
    # The random arm is deterministic under the same control seed.
    again = _run_screen(model, config, directions)
    assert [
        row["delivered_dose"]["perturbation_norm"]
        for row in again
        if row["arm"] == "matched_random" and row["alpha"] != 0.0
    ] == [
        row["delivered_dose"]["perturbation_norm"]
        for row in rows
        if row["arm"] == "matched_random" and row["alpha"] != 0.0
    ]
    assert not model.layers[0]._forward_hooks


def test_token_screen_exact_zero_scale_is_floored_to_1e_6() -> None:
    # Clean residual [2, 0] is orthogonal to the direction [0, 1].
    model = _ScreenModel(prompt_embedding=(2.0, 0.0))
    config = _screen_config(top_positions=1)
    directions = {1: {0: torch.tensor([0.0, 1.0])}}
    rows = _run_screen(model, config, directions)
    token_plus = next(row for row in rows if row["arm"] == "token" and row["alpha"] == 1.0)

    assert token_plus["coordinate_scale_min"] == pytest.approx(1e-6)
    assert token_plus["delivered_dose"]["coordinate_before_mean"] == pytest.approx(0.0)
    assert token_plus["delivered_dose"]["coordinate_after_mean"] == pytest.approx(1e-6, abs=1e-9)
    assert token_plus["delivered_dose"]["perturbation_norm"] == pytest.approx(1e-6, rel=1e-3)
    assert token_plus["delta_margin"] == pytest.approx(0.0, abs=1e-5)


def test_token_screen_side_is_provenance_only() -> None:
    directions = {1: {0: torch.tensor([0.0, 1.0])}}
    negative = _run_screen(
        _ScreenModel(), _screen_config(), directions
    )
    positive = _run_screen(
        _ScreenModel(), _screen_config(candidates=[_candidate(side="positive")]), directions
    )
    assert [row["representation_side"] for row in negative] == ["negative"] * 6
    assert [row["representation_side"] for row in positive] == ["positive"] * 6
    assert [row["intervened_margin"] for row in negative] == [
        row["intervened_margin"] for row in positive
    ]
    assert [row["delivered_dose"]["perturbation_norm"] for row in negative] == [
        row["delivered_dose"]["perturbation_norm"] for row in positive
    ]


def test_token_screen_runner_rejects_missing_directions() -> None:
    with pytest.raises(ValueError, match="miss candidates"):
        _run_screen(_ScreenModel(), _screen_config(), {})


def test_token_screen_runner_removes_hooks_when_forward_raises() -> None:
    model = _ScreenModel()
    original_forward = model.forward
    state = {"calls": 0}

    def flaky(input_ids, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 4:  # first transformed scoring forward
            raise RuntimeError("stop")
        return original_forward(input_ids, *args, **kwargs)

    model.forward = flaky
    with pytest.raises(RuntimeError, match="stop"):
        _run_screen(model, _screen_config(), {1: {0: torch.tensor([0.0, 1.0])}})
    assert not model.layers[0]._forward_hooks


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

RECORDS = {"r1": "T1", "r2": "T1", "r3": "T2", "r4": "T3"}


def _row(token_id: int, arm: str, alpha: float, delta: float, ticker: str, record_id: str) -> dict:
    return {
        "candidate": f"tok{token_id}",
        "token_id": token_id,
        "representation_side": "positive",
        "mean_positive": 0.01,
        "mean_negative": 0.005,
        "band_probability_diff": 0.005,
        "band_smoothed_log_ratio": 0.7,
        "band_js_contribution": 0.001,
        "arm": arm,
        "alpha": alpha,
        "delta_margin": delta,
        "ticker": ticker,
        "record_id": record_id,
    }


def _dose_rows(token_id: int, arm: str, slopes: dict[str, float], dose: float = 1.0) -> list[dict]:
    rows = []
    for record_id, slope in slopes.items():
        ticker = RECORDS[record_id]
        rows.append(_row(token_id, arm, -dose, -dose * slope, ticker, record_id))
        rows.append(_row(token_id, arm, 0.0, 0.0, ticker, record_id))
        rows.append(_row(token_id, arm, dose, dose * slope, ticker, record_id))
    return rows


def test_analyze_token_screen_symmetric_slope_specificity_and_shortlist() -> None:
    zero = {record_id: 0.0 for record_id in RECORDS}
    rows = (
        _dose_rows(1, "token", {record_id: 1.0 for record_id in RECORDS})
        + _dose_rows(1, "matched_random", zero)
        + _dose_rows(2, "token", zero)
        + _dose_rows(2, "matched_random", {record_id: 1.0 for record_id in RECORDS})
        + _dose_rows(3, "token", zero)
        + _dose_rows(
            3, "matched_random",
            {"r1": 1.0, "r2": -1.0, "r3": 1.0, "r4": -1.0},
        )
        + _dose_rows(5, "token", {record_id: 0.5 for record_id in RECORDS})
        + _dose_rows(5, "matched_random", zero)
        + _dose_rows(6, "token", {record_id: 0.3 for record_id in RECORDS})
        + _dose_rows(6, "matched_random", zero)
    )
    summary = analyze_token_screen(rows, positive_dose=1.0, bootstrap_samples=100)

    assert summary["estimand"] == "symmetric_slope"
    assert summary["interpretation"] == "exploratory_discovery"
    assert summary["unit"] == "ticker"
    by_id = {candidate["token_id"]: candidate for candidate in summary["candidates"]}

    # Candidate 1: constant slopes -> equal-ticker mean 1.0 with 3 tickers.
    assert by_id[1]["token_arm"]["mean"] == pytest.approx(1.0)
    assert by_id[1]["token_arm"]["ticker_count"] == 3
    assert by_id[1]["token_arm"]["record_count"] == 4
    assert by_id[1]["token_arm"]["sign_consistent_tickers"] == 3
    assert by_id[1]["matched_random_arm"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert by_id[1]["specificity"]["mean"] == pytest.approx(1.0)
    assert by_id[1]["specificity"]["ci95"][0] > 0
    assert by_id[1]["screen_direction"] == "buy_shifting"
    # Candidate 2: specificity is the negative of the random slopes.
    assert by_id[2]["specificity"]["mean"] == pytest.approx(-1.0)
    assert by_id[2]["specificity"]["ci95"][1] < 0
    assert by_id[2]["specificity"]["sign_consistent_tickers"] == 3
    assert by_id[2]["screen_direction"] == "sell_shifting"
    # Candidate 3: zero ticker means -> p = 1 exactly, CI at zero.
    assert by_id[3]["specificity"]["mean"] == pytest.approx(0.0)
    assert by_id[3]["specificity"]["sign_flip_p"] == pytest.approx(1.0)
    assert by_id[3]["specificity"]["sign_flip_p_holm"] == pytest.approx(1.0)
    assert by_id[3]["specificity"]["sign_consistent_tickers"] == 0
    assert by_id[3]["screen_direction"] == "not_specific"
    assert by_id[5]["screen_direction"] == "buy_shifting"
    assert by_id[6]["screen_direction"] == "buy_shifting"

    # Shortlist: at most two per direction, ranked by |mean specificity|.
    assert summary["shortlist"]["buy_shifting"] == [1, 5]
    assert summary["shortlist"]["sell_shifting"] == [2]
    # Holm never decreases a raw p-value.
    for candidate in summary["candidates"]:
        specificity = candidate["specificity"]
        if specificity is not None:
            assert specificity["sign_flip_p_holm"] >= specificity["sign_flip_p"]


def test_analyze_token_screen_holm_adjusts_specificity_p() -> None:
    zero = {record_id: 0.0 for record_id in RECORDS}
    rows = (
        # A: constant slopes -> sign-flip p near 2/8 of resamples (only the two
        # all-signs-flipped patterns reach the observed |mean|).
        _dose_rows(1, "token", {record_id: 1.0 for record_id in RECORDS})
        + _dose_rows(1, "matched_random", zero)
        # B: zero mean -> p exactly 1.
        + _dose_rows(2, "token", zero)
        + _dose_rows(
            2, "matched_random",
            {"r1": 1.0, "r2": -1.0, "r3": 1.0, "r4": -1.0},
        )
        # C: buy-shifting with a smaller magnitude.
        + _dose_rows(3, "token", {record_id: 0.4 for record_id in RECORDS})
        + _dose_rows(3, "matched_random", zero)
    )
    summary = analyze_token_screen(rows, positive_dose=1.0, bootstrap_samples=100)
    by_id = {candidate["token_id"]: candidate for candidate in summary["candidates"]}

    raw_a = by_id[1]["specificity"]["sign_flip_p"]
    adjusted_a = by_id[1]["specificity"]["sign_flip_p_holm"]
    assert 0.15 < raw_a < 0.40
    # Holm rank 0: min(1, 3 * p) stays below 1 and dominates the raw p.
    assert 0.4 < adjusted_a < 1.0
    assert adjusted_a >= raw_a
    assert by_id[2]["specificity"]["sign_flip_p"] == pytest.approx(1.0)
    assert by_id[2]["specificity"]["sign_flip_p_holm"] == pytest.approx(1.0)
    assert by_id[3]["specificity"]["sign_flip_p_holm"] >= by_id[3]["specificity"]["sign_flip_p"]


def test_analyze_token_screen_uses_symmetric_dose_denominator() -> None:
    # Dose 0.5: slope = (0.6 - (-0.2)) / (2 * 0.5) = 0.8, not 0.4.
    rows = _dose_rows(1, "token", {"r1": 0.8, "r2": 0.8, "r3": 0.8}, dose=0.5)
    summary = analyze_token_screen(rows, positive_dose=0.5, bootstrap_samples=50)
    assert summary["positive_dose"] == 0.5
    assert summary["candidates"][0]["token_arm"]["mean"] == pytest.approx(0.8)
    assert summary["candidates"][0]["specificity"] is None
    assert summary["candidates"][0]["screen_direction"] == "not_evaluated"


def test_analyze_token_screen_requires_both_symmetric_doses() -> None:
    rows = [
        _row(1, "token", 0.5, 0.3, "T1", "r1"),
        _row(1, "token", -0.5, -0.1, "T1", "r1"),
        _row(1, "token", 0.0, 0.0, "T1", "r1"),
        _row(1, "token", 0.5, 0.3, "T1", "r2"),  # missing -0.5 dose
    ]
    with pytest.raises(ValueError, match="missing symmetric doses"):
        analyze_token_screen(rows, positive_dose=0.5)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _pipeline_prompt(ticker: str) -> str:
    return (
        "Refer to the evidence below to make a final investment decision.\n"
        f"Stock Ticker: [{ticker}]\n"
        f"Stock Name: [Name {ticker}]\n"
        "--- Evidence ---\n"
        f"1. fact {ticker} one\n"
        f"2. fact {ticker} two\n"
        "---\n"
        'Respond with one valid JSON object containing only the keys "decision" '
        '(buy | sell) and "reason" (brief justification). Do not choose hold.'
    )


def _write_pipeline_inputs(tmp_path: Path, *, candidate_sha: str | None = None) -> dict:
    import csv

    candidates_path = tmp_path / "frozen_candidate_suggestions.json"
    candidates_path.write_text(
        json.dumps(
            {
                "artifact_type": "frozen_candidate_suggestions",
                "schema_version": 2,
                "candidates": [
                    {
                        "token": " alpha",
                        "token_id": 5,
                        "side": "positive",
                        "mean_positive": 0.01,
                        "mean_negative": 0.005,
                        "band_probability_diff": 0.005,
                        "band_smoothed_log_ratio": 0.7,
                        "band_js_contribution": 0.001,
                    },
                    {
                        "token": " beta",
                        "token_id": 6,
                        "side": "negative",
                        "mean_positive": 0.004,
                        "mean_negative": 0.01,
                        "band_probability_diff": -0.006,
                        "band_smoothed_log_ratio": -0.9,
                        "band_js_contribution": 0.001,
                    },
                ],
                "label": "transported-representation candidate; not causal evidence",
            }
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "artifact_type": "jspace_intervention_splits",
                "input_sha256": "ab" * 32,
                "assignments": {"T1": "discovery", "T2": "discovery", "T3": "test"},
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "trial_plan_prompts.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ticker", "name", "sector", "marketcap", "prompt_with_context_attribute_0"],
        )
        writer.writeheader()
        for ticker in ("T1", "T2", "T3"):
            writer.writerow(
                {
                    "ticker": ticker,
                    "name": f"Name {ticker}",
                    "sector": "Technology",
                    "marketcap": "1000000000",
                    "prompt_with_context_attribute_0": _pipeline_prompt(ticker),
                }
            )
    return {
        "candidates": candidates_path,
        "split": split_path,
        "input": input_path,
        "candidate_sha": candidate_sha if candidate_sha is not None else sha256_file(candidates_path),
    }


def _write_token_screen_config(tmp_path: Path, paths: dict, *, run_name: str) -> Path:
    from llm_bias.jspace_intervention.schemas import TokenScreenConfig
    from llm_bias.core.artifacts.io import write_json

    payload = json.loads(paths["candidates"].read_text(encoding="utf-8"))
    config = TokenScreenConfig.from_dict(
        {
            "candidates": payload["candidates"],
            "model": "fake-model",
            "source_sector": "Technology",
            "layers": [2, 3],
            "top_positions": 2,
            "candidate_artifact_path": str(paths["candidates"]),
            "candidate_artifact_sha256": paths["candidate_sha"],
            "split_manifest_sha256": sha256_file(paths["split"]),
        }
    )
    config_path = tmp_path / f"token_screen_config_{run_name}.json"
    write_json(
        config_path,
        {**config.to_dict(), "artifact_type": "jspace_token_screen_config", "schema_version": 1},
        overwrite=True,
    )
    return config_path


def _fake_screen_model():
    class FakeModel:
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

    class Tokenizer:
        chat_template = "fake"
        all_special_tokens = []
        pad_token_id = 0

        def __init__(self, vocab=30):
            self.vocab = vocab

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
            ids = [ord(char) % (self.vocab - 1) + 1 for char in text]
            if return_offsets_mapping:
                return SimpleNamespace(
                    input_ids=ids,
                    offset_mapping=[(index, index + 1) for index in range(len(text))],
                    special_tokens_mask=[0] * len(ids),
                )
            return SimpleNamespace(input_ids=ids)

        def apply_chat_template(self, messages, **_kwargs):
            return "<user>" + messages[0]["content"] + "<assistant>"

        def decode(self, ids, **_kwargs):
            return "".join(chr(97 + (int(token) - 1) % 26) for token in ids)

    return FakeModel(), Tokenizer()


def _patch_pipeline(monkeypatch, tmp_path: Path):
    import jlens

    from llm_bias.jspace_intervention import pipeline

    model, tokenizer = _fake_screen_model()
    lens = jlens.JacobianLens({2: torch.eye(8), 3: torch.eye(8)}, n_prompts=1, d_model=8)
    lens_file = tmp_path / "jacobian_lens.pt"
    lens_file.write_bytes(b"fake-lens")
    monkeypatch.setattr(pipeline, "load_tokenizer", lambda model_name: tokenizer)
    monkeypatch.setattr(
        pipeline, "load_model", lambda model_name: (model, tokenizer, torch.device("cpu"))
    )
    monkeypatch.setattr(
        pipeline,
        "load_validated_lens",
        lambda **_kwargs: SimpleNamespace(lens=lens, path=lens_file, source="fake"),
    )


def test_run_token_screen_pipeline_full_lifecycle(tmp_path, monkeypatch) -> None:
    from llm_bias.jspace_intervention.pipeline import run_token_screen_pipeline

    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_token_screen_config(tmp_path, paths, run_name="lifecycle")
    _patch_pipeline(monkeypatch, tmp_path)

    run_root = run_token_screen_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="screen-test",
        artifact_root=tmp_path / "artifacts",
    )

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for stage in ("prepare", "forward", "analyze"):
        assert manifest["stages"][stage]["status"] == "complete"
    input_types = {ref["artifact_type"] for ref in manifest["input_refs"]}
    assert input_types == {
        "trial_plan_prompts",
        "jspace_intervention_splits",
        "jspace_token_screen_config",
        "frozen_candidate_suggestions",
    }
    assert [ref["artifact_type"] for ref in manifest["lens_refs"]] == ["jacobian_lens"]

    prepare_metadata = json.loads(
        (run_root / "prepare" / "metadata.json").read_text(encoding="utf-8")
    )
    assert prepare_metadata["candidate_artifact_sha256"] == sha256_file(paths["candidates"])
    assert prepare_metadata["split_manifest_sha256"] == sha256_file(paths["split"])
    assert prepare_metadata["positive_dose"] == 1.0
    assert prepare_metadata["record_count"] == 2
    assert prepare_metadata["candidate_count"] == 2
    prompt_records = [
        json.loads(line)
        for line in (run_root / "prepare" / "prompt_records.jsonl").open()
    ]
    assert {record["ticker"] for record in prompt_records} == {"T1", "T2"}
    assert all(record["prompt_column"] == "prompt_with_context_attribute_0" for record in prompt_records)

    rows = [json.loads(line) for line in (run_root / "forward" / "token_screen_results.jsonl").open()]
    # 2 prompts x 2 candidates x 2 arms x 3 alphas.
    assert len(rows) == 24
    for row in rows:
        assert row["artifact_type"] == "token_screen_result"
        assert row["token_id"] in {5, 6}
        assert row["arm"] in {"token", "matched_random"}
        assert row["alpha"] in {-1.0, 0.0, 1.0}
        assert row["representation_side"] in {"positive", "negative"}
        assert row["split"] == "discovery"
        assert {"band_probability_diff", "band_smoothed_log_ratio", "band_js_contribution"} <= set(row)
    for row in rows:
        if row["alpha"] == 0.0:
            assert row["delta_margin"] == 0.0
            assert row["intervened_margin"] == pytest.approx(row["clean_margin"])
    # Clean margin is computed once per prompt and shared across arms/candidates.
    clean_by_record = {}
    for row in rows:
        clean_by_record.setdefault(row["record_id"], set()).add(row["clean_margin"])
    assert all(len(values) == 1 for values in clean_by_record.values())
    # Matched-random arm: same positions/scales and same-norm direction.
    token_rows = {
        (row["record_id"], row["token_id"], row["alpha"]): row
        for row in rows
        if row["arm"] == "token"
    }
    for row in rows:
        if row["arm"] != "matched_random" or row["alpha"] == 0.0:
            continue
        token_row = token_rows[(row["record_id"], row["token_id"], row["alpha"])]
        assert row["intervention_positions"] == token_row["intervention_positions"]
        assert row["coordinate_scale_max"] == pytest.approx(token_row["coordinate_scale_max"])
        assert row["delivered_dose"]["perturbation_norm"] == pytest.approx(
            token_row["delivered_dose"]["perturbation_norm"], rel=1e-5
        )

    forward_metadata = json.loads((run_root / "forward" / "metadata.json").read_text(encoding="utf-8"))
    assert forward_metadata["layers"] == [2, 3]
    assert forward_metadata["controls"] == ["token", "matched_random"]
    assert forward_metadata["estimand"] == "symmetric_slope"

    summary = json.loads((run_root / "analyze" / "token_screen_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_type"] == "token_screen_analysis"
    assert summary["interpretation"] == "exploratory_discovery"
    assert summary["positive_dose"] == 1.0
    assert summary["forward_sha256"] == sha256_file(run_root / "forward" / "token_screen_results.jsonl")
    assert {candidate["token_id"] for candidate in summary["candidates"]} == {5, 6}
    for candidate in summary["candidates"]:
        assert candidate["token_arm"] is not None
        assert candidate["matched_random_arm"] is not None
        assert candidate["specificity"] is not None
        assert set(candidate["specificity"]) >= {"mean", "ci95", "sign_flip_p", "sign_flip_p_holm", "sign_consistent_tickers", "ticker_count", "record_count"}
    assert set(summary["shortlist"]) == {"buy_shifting", "sell_shifting"}
    analyze_metadata = json.loads((run_root / "analyze" / "metadata.json").read_text(encoding="utf-8"))
    assert analyze_metadata["interpretation"] == "exploratory_discovery"
    assert sum(analyze_metadata["shortlist_counts"].values()) <= 4

    # Deterministic: a second run produces identical forward rows.
    repeat_root = run_token_screen_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="screen-test-repeat",
        artifact_root=tmp_path / "artifacts",
    )
    repeat_rows = [
        json.loads(line)
        for line in (repeat_root / "forward" / "token_screen_results.jsonl").open()
    ]
    assert repeat_rows == rows


def test_run_token_screen_pipeline_rejects_tampered_candidate_sha(tmp_path, monkeypatch) -> None:
    from llm_bias.jspace_intervention.pipeline import run_token_screen_pipeline

    paths = _write_pipeline_inputs(tmp_path, candidate_sha="0" * 64)
    config_path = _write_token_screen_config(tmp_path, paths, run_name="tampered")
    _patch_pipeline(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="does not match the SHA bound in config"):
        run_token_screen_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="tampered",
            artifact_root=tmp_path / "artifacts",
        )
    assert not run_root(
        "fake-model", "jspace-token-screen", "tampered", artifact_root=tmp_path / "artifacts"
    ).exists()


def test_run_token_screen_pipeline_rejects_wrong_split_manifest(tmp_path, monkeypatch) -> None:
    from llm_bias.jspace_intervention.pipeline import run_token_screen_pipeline

    paths = _write_pipeline_inputs(tmp_path)
    config = json.loads(_write_token_screen_config(tmp_path, paths, run_name="split").read_text())
    config["split_manifest_sha256"] = "0" * 64
    config_path = tmp_path / "token_screen_config_split.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="discovery manifest frozen in config"):
        run_token_screen_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="split",
            artifact_root=tmp_path / "artifacts",
        )
    assert not run_root(
        "fake-model", "jspace-token-screen", "split", artifact_root=tmp_path / "artifacts"
    ).exists()


def test_run_token_screen_pipeline_preflight_creates_no_run(tmp_path, monkeypatch) -> None:
    from llm_bias.jspace_intervention.pipeline import run_token_screen_pipeline

    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_token_screen_config(tmp_path, paths, run_name="preflight")
    _patch_pipeline(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="limit 50"):
        run_token_screen_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="preflight",
            artifact_root=tmp_path / "artifacts",
            max_seq_len=50,
        )
    assert not run_root(
        "fake-model", "jspace-token-screen", "preflight", artifact_root=tmp_path / "artifacts"
    ).exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_prepare_token_screen_config_binds_shas(tmp_path, monkeypatch, capsys) -> None:
    from llm_bias.jspace_intervention import cli

    paths = _write_pipeline_inputs(tmp_path)
    output = tmp_path / "token_screen_config.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "prepare-token-screen-config",
            "--candidates", str(paths["candidates"]),
            "--model", "fake-model",
            "--source-sector", "Technology",
            "--split-manifest", str(paths["split"]),
            "--alphas=-1,0,1",
            "--layers", "14,15",
            "--output", str(output),
        ],
    )
    cli.main()
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "jspace_token_screen_config"
    assert payload["model"] == "fake-model"
    assert payload["candidate_artifact_sha256"] == sha256_file(paths["candidates"])
    assert payload["split_manifest_sha256"] == sha256_file(paths["split"])
    assert payload["layers"] == [14, 15]
    assert payload["controls"] == ["token", "matched_random"]
    assert [candidate["token_id"] for candidate in payload["candidates"]] == [5, 6]
    assert payload["candidates"][0]["representation_side"] == "positive"

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"artifact_type": "not-candidates", "candidates": []}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "prepare-token-screen-config",
            "--candidates", str(broken),
            "--model", "fake-model",
            "--source-sector", "Technology",
            "--split-manifest", str(paths["split"]),
            "--output", str(output),
        ],
    )
    with pytest.raises(ValueError, match="frozen_candidate_suggestions"):
        cli.main()


def test_validate_config_dispatches_token_screen(tmp_path, monkeypatch, capsys) -> None:
    from llm_bias.jspace_intervention import cli

    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_token_screen_config(tmp_path, paths, run_name="validate")
    monkeypatch.setattr(
        "sys.argv",
        ["jspace-intervention", "validate-config", "--config", str(config_path)],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["candidates"]) == 2
    assert payload["alphas"] == [-1.0, 0.0, 1.0]


def test_run_token_screen_cli_defaults() -> None:
    from llm_bias.jspace_intervention.cli import build_parser

    args = build_parser().parse_args([
        "run-token-screen",
        "--input", "prompts.csv",
        "--split-manifest", "splits.json",
        "--config", "config.json",
        "--model", "fake",
        "--run-id", "screen",
    ])
    assert args.dataset == "jspace-token-screen"
    assert args.split == "discovery"
    assert args.prompt_columns is None
    assert args.max_seq_len == 1024
    assert args.lens is None
    assert args.artifact_root == "artifacts"


def test_run_token_screen_cli_dispatch(monkeypatch, tmp_path) -> None:
    from llm_bias.jspace_intervention import cli

    calls: list[dict] = []
    module = ModuleType("llm_bias.jspace_intervention.pipeline")

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return tmp_path / "run"

    module.run_token_screen_pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "llm_bias.jspace_intervention.pipeline", module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "run-token-screen",
            "--input", "prompts.csv",
            "--split-manifest", "splits.json",
            "--config", "config.json",
            "--model", "fake",
            "--run-id", "screen",
            "--split", "calibration",
            "--max-records", "5",
            "--prompt-column", "prompt_with_context_attribute_1",
        ],
    )
    cli.main()
    assert len(calls) == 1
    assert calls[0]["split_name"] == "calibration"
    assert calls[0]["max_records"] == 5
    assert calls[0]["prompt_columns"] == {"prompt_with_context_attribute_1"}
    assert calls[0]["dataset"] == "jspace-token-screen"
