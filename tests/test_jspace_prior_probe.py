"""Regression tests for the V2 zero-evidence header-only prior probe."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.artifacts.io import write_json
from llm_bias.core.continuation_scoring import score_single_token_margin_fp32
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.prompt_input.encoding import input_ids
from llm_bias.jspace_intervention import cli
from llm_bias.jspace_intervention import outcome_flip, prior_probe
from llm_bias.jspace_intervention.outcome_flip import (
    _discovery_verification_records,
    _materialize_records,
    fit_outcome_directions,
    run_outcome_flip_pipeline,
)
from llm_bias.jspace_intervention.prior_probe import (
    NEUTRAL_EVIDENCE_ITEM,
    PROBE_TEMPLATE,
    analyze_prior_probe,
    build_probe_records,
    project_positions,
    render_probe_prompt,
    run_prior_probe_pipeline,
)
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig, PriorProbeConfig

_TOKEN_RE = re.compile(r"buy|sell|.")
BUY_ID = 1000
SELL_ID = 1001
BUY_CTRL_ID = 1 + (ord("Z") % 900)
SELL_CTRL_ID = 1 + (ord("z") % 900)


def _token_id(token: str) -> int:
    if token == "buy":
        return BUY_ID
    if token == "sell":
        return SELL_ID
    return 1 + (ord(token) % 900)


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
        tokens, offsets = [], []
        for match in _TOKEN_RE.finditer(str(text)):
            tokens.append(match.group(0))
            offsets.append((match.start(), match.end()))
        if return_offsets_mapping:
            return SimpleNamespace(
                input_ids=[_token_id(token) for token in tokens],
                offset_mapping=offsets,
                special_tokens_mask=[0] * len(tokens),
            )
        return SimpleNamespace(input_ids=[_token_id(token) for token in tokens])

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


class _FocusLayer(torch.nn.Module):
    """The last position receives the final non-zero earlier residual."""

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

    Parameters are frozen to mirror the HFLensModel wrapper, so the
    gradient-fitting re-root path is exercised.
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
        for param in self.parameters():
            param.requires_grad_(False)
        self.config = SimpleNamespace(eos_token_id=None)
        self._hf_model = self

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self._embed_tokens(input_ids)
        offset = torch.zeros_like(hidden)
        offset[:, -1, :] = self._base.to(hidden.dtype)
        hidden = hidden + offset
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _discovery_prompt(ticker: str, control: str = "ZZ") -> str:
    """A V2 discovery prompt for the fake model (item-end lands on `control`)."""
    return (
        f"Decide for [{ticker}].\n"
        "--- Evidence ---\n"
        "1. item a.\n"
        f"2. bias {control}\n"
        "---\n"
        'Respond with one valid JSON object containing only the keys "decision" '
        "(buy or sell) and \"reason\"."
    )


def _v2_config_payload(**overrides) -> dict:
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


def _v2_config(**overrides) -> OutcomeFlipConfig:
    return OutcomeFlipConfig.from_dict(_v2_config_payload(**overrides))


# ---------------------------------------------------------------------------
# Template and config schema
# ---------------------------------------------------------------------------


def test_render_probe_prompt_freezes_version_1_template() -> None:
    prompt = render_probe_prompt(ticker="NVDA", sector="Technology")
    assert prompt == (
        "Refer to the evidence below to make a final investment decision.\n"
        "Stock Ticker: [NVDA]\n"
        "Sector: [Technology]\n"
        "--- Evidence ---\n"
        "1. No evidence provided.\n"
        "---\n"
        'Respond with one valid JSON object containing only the keys "decision" '
        '(buy | sell) and "reason" (brief justification). Do not choose hold.'
    )
    other = render_probe_prompt(ticker="JPM", sector="Financial Services")
    assert prompt != other
    # Only the two header identity lines may differ; the neutral body and the
    # instruction block are byte-identical across conditions.
    assert prompt.split("--- Evidence ---\n", 1)[1] == other.split(
        "--- Evidence ---\n", 1
    )[1]
    header = prompt.split("--- Evidence ---\n", 1)[0].splitlines()
    assert header[1:3] == ["Stock Ticker: [NVDA]", "Sector: [Technology]"]
    other_header = other.split("--- Evidence ---\n", 1)[0].splitlines()
    assert other_header[1:3] == ["Stock Ticker: [JPM]", "Sector: [Financial Services]"]
    assert NEUTRAL_EVIDENCE_ITEM == "No evidence provided."
    assert "{ticker}" in PROBE_TEMPLATE and "{sector}" in PROBE_TEMPLATE


def _probe_config_payload(**overrides) -> dict:
    payload = {
        "model": "fake-model",
        "input": "data/trial.csv",
        "input_sha256": "ab" * 32,
        "split_manifest": "splits.json",
        "split_manifest_sha256": "cd" * 32,
        "outcome_flip_config": "outcome_flip_config.json",
        "outcome_flip_config_sha256": "ef" * 32,
        "direction_identity": "direction_identity.json",
        "direction_identity_sha256": "01" * 32,
        "position_rule": "evidence_item_end",
        "conditions": [
            {"ticker": "D1", "sector": "Technology"},
            {"ticker": "D1", "sector": "Financial Services"},
            {"ticker": "D2", "sector": "Technology"},
            {"ticker": "D2", "sector": "Financial Services"},
        ],
        "contrast_tickers": ["D1", "D2"],
    }
    payload.update(overrides)
    return payload


def test_prior_probe_config_defaults_and_roundtrip() -> None:
    config = PriorProbeConfig.from_dict(_probe_config_payload())
    assert config.contrast_sectors == ("Technology", "Financial Services")
    assert config.neutral_evidence_item == "No evidence provided."
    assert config.scale_floor == 1.0
    assert config.max_seq_len == 1024
    assert config.decision_prefix == '{\n  "decision": "'
    assert PriorProbeConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "overrides",
    [
        {"position_rule": "bogus"},
        {"conditions": []},
        # one ticker
        {"conditions": [
            {"ticker": "A", "sector": "Technology"},
            {"ticker": "A", "sector": "Financial Services"},
        ]},
        # one sector label
        {"conditions": [
            {"ticker": "A", "sector": "Technology"},
            {"ticker": "B", "sector": "Technology"},
        ]},
        # duplicate (ticker, sector)
        {"conditions": [
            {"ticker": "A", "sector": "Technology"},
            {"ticker": "A", "sector": "Technology"},
        ]},
        {"neutral_evidence_item": "buy signal"},
        {"neutral_evidence_item": "line one\nline two"},
        {"neutral_evidence_item": ""},
        {"scale_floor": 0.0},
        {"max_seq_len": 0},
        {"bootstrap_samples": 1},
        {"input_sha256": "zz" * 32},
        {"contrast_tickers": ["A", "A"]},
        {"contrast_tickers": ["A"]},
        {"contrast_sectors": ["Technology", "Utilities"]},
        {"input": ""},
    ],
)
def test_prior_probe_config_rejects_invalid_payload(overrides: dict) -> None:
    with pytest.raises(ValueError):
        PriorProbeConfig.from_dict(_probe_config_payload(**overrides))


# ---------------------------------------------------------------------------
# Projection normalization
# ---------------------------------------------------------------------------


def test_project_positions_normalization() -> None:
    direction = torch.nn.functional.normalize(torch.tensor([1.0, 2.0, 0.0, 0.0]), dim=0)
    orthogonal = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=0)
    state = 2.0 * direction + 3.0 * orthogonal
    result = project_positions(
        {1: state.unsqueeze(0).unsqueeze(0)},
        {1: direction},
        position=0,
        scale_floor=1.0,
    )
    expected_norm = (2.0**2 + 3.0**2) ** 0.5
    assert result["1"]["projection"] == pytest.approx(2.0, rel=1e-6)
    assert result["1"]["norm"] == pytest.approx(expected_norm, rel=1e-6)
    assert result["1"]["projection_relative"] == pytest.approx(
        2.0 / expected_norm, rel=1e-6
    )
    # Sub-floor norms use the frozen scale floor in the relative value.
    small = project_positions(
        {1: (0.3 * direction).unsqueeze(0).unsqueeze(0)},
        {1: direction},
        position=0,
        scale_floor=1.0,
    )
    assert small["1"]["projection"] == pytest.approx(0.3, rel=1e-6)
    assert small["1"]["projection_relative"] == pytest.approx(0.3, rel=1e-6)


def test_project_positions_rejects_non_unit_direction() -> None:
    direction = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
    with pytest.raises(ValueError, match="unit norm"):
        project_positions(
            {1: direction.unsqueeze(0).unsqueeze(0)},
            {1: 2.0 * direction},
            position=0,
            scale_floor=1.0,
        )


# ---------------------------------------------------------------------------
# Condition resolution
# ---------------------------------------------------------------------------


def _probe_config_for_records(**overrides) -> PriorProbeConfig:
    return PriorProbeConfig.from_dict(_probe_config_payload(**overrides))


def test_build_probe_records_resolution_and_failures() -> None:
    config = _probe_config_for_records()
    index = {
        "D1": {"name": "D1 Corp", "sector": "Technology", "marketcap": "1000"},
        "D2": {"name": "D2 Corp", "sector": "Technology", "marketcap": "1000"},
        "F1": {"name": "F1 Bank", "sector": "Financial Services", "marketcap": "2000"},
    }
    records = build_probe_records(
        config,
        ticker_index=index,
        assignments={"D1": "discovery", "D2": "test", "F1": "test"},
    )
    assert [(record["ticker"], record["sector"]) for record in records] == [
        ("D1", "Technology"),
        ("D1", "Financial Services"),
        ("D2", "Technology"),
        ("D2", "Financial Services"),
    ]
    assert [record["own_sector"] for record in records] == [True, False, True, False]
    assert records[0]["canonical_sector"] == "Technology"
    assert records[0]["split"] == "discovery"
    assert records[3]["split"] == "test"
    assert records[0]["prompt"] == render_probe_prompt(
        ticker="D1", sector="Technology"
    )
    with pytest.raises(ValueError, match="not in the input CSV"):
        build_probe_records(
            _probe_config_for_records(
                conditions=[
                    {"ticker": "X9", "sector": "Technology"},
                    {"ticker": "D1", "sector": "Financial Services"},
                ],
                contrast_tickers=["X9", "D1"],
            ),
            ticker_index=index,
            assignments={},
        )
    with pytest.raises(ValueError, match="canonical sector"):
        build_probe_records(
            _probe_config_for_records(
                conditions=[
                    {"ticker": "D1", "sector": "Technology"},
                    {"ticker": "D2", "sector": "Utilities"},
                ],
                contrast_sectors=["Technology", "Utilities"],
            ),
            ticker_index=index,
            assignments={},
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analysis_rows():
    rows = []
    values = {
        ("D1", "Technology"): (0.5, 1.0),
        ("D1", "Financial Services"): (-0.5, 0.0),
        ("D2", "Technology"): (1.5, 2.0),
        ("D2", "Financial Services"): (0.5, 1.0),
    }
    for (ticker, sector), (margin, projection) in values.items():
        canonical = "Technology" if ticker == "D1" else "Financial Services"
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "canonical_sector": canonical,
                "margin": margin,
                "layers": {
                    "1": {
                        "projection": projection,
                        "projection_relative": projection / 2.0,
                        "norm": 2.0,
                    },
                    "2": {
                        "projection": -projection,
                        "projection_relative": -projection / 2.0,
                        "norm": 2.0,
                    },
                },
            }
        )
    return rows


def test_analyze_prior_probe_group_means_and_contrasts() -> None:
    config = _probe_config_for_records()
    summary = analyze_prior_probe(_analysis_rows(), config=config, layers=[1, 2])
    assert summary["condition_count"] == 4
    assert summary["groups"]["by_sector_label"]["Technology"]["mean_margin"] == pytest.approx(1.0)
    assert summary["groups"]["by_ticker"]["D1"]["mean_margin"] == pytest.approx(0.0)
    # Sector-label contrast: per-ticker (Technology - Financial Services),
    # averaged over tickers: D1: 1.0, D2: 1.0 -> 1.0.
    sector = summary["contrasts"]["sector_label"]
    assert sector["margin"]["point"] == pytest.approx(1.0)
    assert "Technology label - Financial Services label" in sector["definition"]
    ci = sector["margin"]["ci95"]
    assert isinstance(ci, list) and len(ci) == 2
    assert sector["layers"]["1"]["projection"]["point"] == pytest.approx(1.0)
    # Pair contrast D1 - D2 averaged over the two sector labels: (1.0 + -3.0)/2.
    pair = summary["contrasts"]["d1_minus_d2"]
    assert pair["margin"]["point"] == pytest.approx(-1.0)
    # Ticker-group contrast: Technology tickers minus Financial Services tickers.
    group = summary["contrasts"]["ticker_group"]
    assert group["margin"]["point"] == pytest.approx(0.0 - 1.0)
    assert "association" in summary["interpretation"]
    with pytest.raises(ValueError, match="contrast tickers"):
        analyze_prior_probe(
            _analysis_rows(),
            config=_probe_config_for_records(contrast_tickers=["D1", "X9"]),
            layers=[1, 2],
        )


# ---------------------------------------------------------------------------
# Shared pipeline fixtures (fake V2 discovery + probe inputs)
# ---------------------------------------------------------------------------


def _write_probe_pipeline_inputs(tmp_path: Path) -> dict[str, Path]:
    input_path = tmp_path / "trial_plan_prompts.csv"
    tickers = {
        "D1": ("discovery", "Technology", "ZZ"),
        "D2": ("discovery", "Technology", "zz"),
        "F1": ("discovery", "Financial Services", "ZZ"),
        "F2": ("calibration", "Financial Services", "ZZ"),
    }
    lines = ["Date,ticker,name,sector,marketcap,prompt_with_context_attribute_0"]
    for ticker, (split, sector, control) in tickers.items():
        prompt = _discovery_prompt(ticker, control=control).replace('"', '""')
        lines.append(f"2026-01-01,{ticker},{ticker} Corp,{sector},1000,\"{prompt}\"")
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "artifact_type": "jspace_intervention_splits",
                "schema_version": 1,
                "assignments": {ticker: split for ticker, (split, _, _) in tickers.items()},
            }
        ),
        encoding="utf-8",
    )
    return {"input": input_path, "split": split_path}


def _write_outcome_config(tmp_path: Path, split_path: Path) -> Path:
    config = _v2_config(split_manifest_sha256=sha256_file(split_path))
    config_path = tmp_path / "outcome_flip_config.json"
    write_json(
        config_path,
        {**config.to_dict(), "artifact_type": "outcome_flip_config", "schema_version": 1},
        overwrite=True,
    )
    return config_path


def _write_probe_config(
    tmp_path: Path,
    *,
    input_path: Path,
    split_path: Path,
    v2_config_path: Path,
    identity_path: Path,
    suffix: str = "",
    **overrides,
) -> Path:
    payload = _probe_config_payload(
        input=str(input_path),
        input_sha256=sha256_file(input_path),
        split_manifest=str(split_path),
        split_manifest_sha256=sha256_file(split_path),
        outcome_flip_config=str(v2_config_path),
        outcome_flip_config_sha256=sha256_file(v2_config_path),
        direction_identity=str(identity_path),
        direction_identity_sha256=sha256_file(identity_path),
        conditions=[
            {"ticker": "D1", "sector": "Technology"},
            {"ticker": "D1", "sector": "Financial Services"},
            {"ticker": "F2", "sector": "Technology"},
            {"ticker": "F2", "sector": "Financial Services"},
        ],
        contrast_tickers=["D1", "F2"],
        **overrides,
    )
    config = PriorProbeConfig.from_dict(payload)
    path = tmp_path / f"prior_probe_config{suffix}.json"
    write_json(
        path,
        {
            **config.to_dict(),
            "artifact_type": prior_probe.PROBE_CONFIG_ARTIFACT_TYPE,
            "schema_version": 1,
        },
        overwrite=True,
    )
    return path


def _patch_model_loading(monkeypatch, module, model, tokenizer) -> None:
    monkeypatch.setattr(module, "load_tokenizer", lambda model_name: tokenizer)
    monkeypatch.setattr(
        module, "load_model", lambda model_name: (model, tokenizer, torch.device("cpu"))
    )


def _manifest(run_root: Path) -> dict:
    return json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))


def _run_fake_discovery(
    *,
    model,
    tokenizer,
    paths: dict[str, Path],
    config_path: Path,
    artifact_root: Path,
    run_id: str,
) -> Path:
    return run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id=run_id,
        artifact_root=artifact_root,
        split_name="discovery",
    )


# ---------------------------------------------------------------------------
# Pipeline lifecycle (fake model, no GPU)
# ---------------------------------------------------------------------------


def test_run_prior_probe_pipeline_full_lifecycle(tmp_path: Path, monkeypatch) -> None:
    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    _patch_model_loading(monkeypatch, outcome_flip, model, tokenizer)
    _patch_model_loading(monkeypatch, prior_probe, model, tokenizer)
    paths = _write_probe_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_fake_discovery(
        model=model,
        tokenizer=tokenizer,
        paths=paths,
        config_path=config_path,
        artifact_root=artifact_root,
        run_id="probe-v2-discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    probe_config_path = _write_probe_config(
        tmp_path,
        input_path=paths["input"],
        split_path=paths["split"],
        v2_config_path=config_path,
        identity_path=identity_path,
    )
    probe_root = run_prior_probe_pipeline(
        config_path=probe_config_path,
        model_name="fake-model",
        run_id="probe-lifecycle",
        artifact_root=artifact_root,
    )
    assert probe_root == artifact_root / "fake-model" / "jspace-outcome-direction-flip" / "runs" / "probe-lifecycle"
    manifest = _manifest(probe_root)
    assert manifest["status"] == "complete"
    for stage in ("prepare", "forward", "analyze"):
        assert manifest["stages"][stage]["status"] == "complete"

    records = [
        json.loads(line)
        for line in (probe_root / "prepare" / "prior_probe_records.jsonl").open()
        if line.strip()
    ]
    assert len(records) == 4
    assert all(record["artifact_type"] == "outcome_prior_probe_record" for record in records)
    by_condition = {(record["ticker"], record["sector"]): record for record in records}
    assert by_condition[("D1", "Technology")]["own_sector"] is True
    assert by_condition[("D1", "Financial Services")]["own_sector"] is False
    assert by_condition[("F2", "Technology")]["own_sector"] is False
    assert by_condition[("F2", "Financial Services")]["own_sector"] is True
    assert by_condition[("F2", "Financial Services")]["split"] == "calibration"
    assert by_condition[("D1", "Technology")]["prompt"] == render_probe_prompt(
        ticker="D1", sector="Technology"
    )

    rows = [
        json.loads(line)
        for line in (probe_root / "forward" / "prior_probe_results.jsonl").open()
        if line.strip()
    ]
    assert len(rows) == 4
    assert all(row["artifact_type"] == "outcome_prior_probe_result" for row in rows)
    for row in rows:
        assert set(row["layers"]) == {"1", "2"}
        for layer in row["layers"].values():
            assert math.isfinite(layer["projection"])
            assert math.isfinite(layer["projection_relative"])
            assert math.isfinite(layer["norm"])
        assert row["margin_decision"] in {"buy", "sell", "tie"}
        assert row["measurement_position"] == row["scoring_prompt_token_count"] - 1

    # The reported margin must equal the V2 scorer applied to the same prompt.
    v2_config = OutcomeFlipConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    first = next(row for row in rows if (row["ticker"], row["sector"]) == ("D1", "Technology"))
    scoring_prompt = prior_probe.prepare_probe_scoring_prompt(
        tokenizer,
        render_probe_prompt(ticker="D1", sector="Technology"),
        decision_prefix=v2_config.decision_prefix,
    )
    expected_margin = score_single_token_margin_fp32(
        model, tokenizer, scoring_prompt, "buy", "sell", device="cpu"
    ).value
    assert first["margin"] == pytest.approx(expected_margin, rel=1e-6, abs=1e-8)

    # The reported projections must equal the dot product with the frozen
    # (recomputed, hash-verified) directions.
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assignments = json.loads(paths["split"].read_text(encoding="utf-8"))["assignments"]
    verification_records = _discovery_verification_records(
        paths["input"], assignments=assignments, config=v2_config, identity=identity
    )
    fit = fit_outcome_directions(
        model=model,
        tokenizer=tokenizer,
        records=_materialize_records(verification_records, tokenizer, v2_config),
        config=v2_config,
        device=torch.device("cpu"),
    )
    directions = fit.directions["evidence_item_end"]
    prompt_ids = input_ids(tokenizer, scoring_prompt, add_special_tokens=True)
    residuals = record_residuals(
        model, torch.tensor([prompt_ids]), v2_config.fitted_layers
    )
    for layer in v2_config.fitted_layers:
        state = residuals[layer][0, -1, :].float()
        expected_projection = float(torch.dot(state, directions[layer]))
        norm = float(state.norm())
        reported = first["layers"][str(layer)]
        assert reported["projection"] == pytest.approx(expected_projection, rel=1e-5, abs=1e-7)
        assert reported["projection_relative"] == pytest.approx(
            expected_projection / max(norm, 1.0), rel=1e-5, abs=1e-7
        )
        assert reported["norm"] == pytest.approx(norm, rel=1e-5, abs=1e-7)

    forward_metadata = json.loads(
        (probe_root / "forward" / "metadata.json").read_text(encoding="utf-8")
    )
    assert forward_metadata["direction_verified"] is True
    assert forward_metadata["position_rule"] == "evidence_item_end"

    analysis = json.loads(
        (probe_root / "analyze" / "prior_probe_analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["artifact_type"] == "outcome_prior_probe_analysis"
    assert analysis["condition_count"] == 4
    assert set(analysis["groups"]) == {"by_sector_label", "by_ticker", "by_ticker_group"}
    assert "sector_label" in analysis["contrasts"]
    assert "d1_minus_f2" in analysis["contrasts"]
    assert "ticker_group" in analysis["contrasts"]
    assert "association" in analysis["interpretation"]


def test_run_prior_probe_pipeline_fail_closed_on_tampered_identity(
    tmp_path: Path, monkeypatch
) -> None:
    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    _patch_model_loading(monkeypatch, outcome_flip, model, tokenizer)
    _patch_model_loading(monkeypatch, prior_probe, model, tokenizer)
    paths = _write_probe_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_fake_discovery(
        model=model,
        tokenizer=tokenizer,
        paths=paths,
        config_path=config_path,
        artifact_root=artifact_root,
        run_id="probe-v2-discovery-tamper",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["directions"]["evidence_item_end"]["1"]["sha256"] = "00" * 32
    tampered_path = tmp_path / "tampered_identity.json"
    write_json(tampered_path, identity, overwrite=True)
    probe_config_path = _write_probe_config(
        tmp_path,
        input_path=paths["input"],
        split_path=paths["split"],
        v2_config_path=config_path,
        identity_path=tampered_path,
        suffix="-tamper",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        run_prior_probe_pipeline(
            config_path=probe_config_path,
            model_name="fake-model",
            run_id="probe-tamper",
            artifact_root=artifact_root,
        )
    assert (
        _manifest(
            artifact_root
            / "fake-model"
            / "jspace-outcome-direction-flip"
            / "runs"
            / "probe-tamper"
        )["status"]
        == "failed"
    )


def test_run_prior_probe_rejects_model_mismatch_and_missing_config(
    tmp_path: Path, monkeypatch
) -> None:
    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    _patch_model_loading(monkeypatch, prior_probe, model, tokenizer)
    _patch_model_loading(monkeypatch, outcome_flip, model, tokenizer)
    paths = _write_probe_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_fake_discovery(
        model=model,
        tokenizer=tokenizer,
        paths=paths,
        config_path=config_path,
        artifact_root=artifact_root,
        run_id="probe-v2-discovery-mismatch",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    probe_config_path = _write_probe_config(
        tmp_path,
        input_path=paths["input"],
        split_path=paths["split"],
        v2_config_path=config_path,
        identity_path=identity_path,
        suffix="-mismatch",
        model="other-model",
    )
    with pytest.raises(ValueError, match="does not match model"):
        run_prior_probe_pipeline(
            config_path=probe_config_path,
            model_name="fake-model",
            run_id="probe-model-mismatch",
            artifact_root=artifact_root,
        )
    assert not (
        artifact_root / "fake-model" / "jspace-outcome-direction-flip" / "runs" / "probe-model-mismatch"
    ).exists()
    with pytest.raises(FileNotFoundError):
        run_prior_probe_pipeline(
            config_path=tmp_path / "missing_probe_config.json",
            model_name="fake-model",
            run_id="probe-missing-config",
            artifact_root=artifact_root,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prepare_prior_probe_config(tmp_path: Path, monkeypatch, capsys) -> None:
    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    _patch_model_loading(monkeypatch, outcome_flip, model, tokenizer)
    paths = _write_probe_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    discovery_root = _run_fake_discovery(
        model=model,
        tokenizer=tokenizer,
        paths=paths,
        config_path=config_path,
        artifact_root=tmp_path / "artifacts",
        run_id="probe-cli-discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    output = tmp_path / "probe_config.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "prepare-prior-probe-config",
            "--model",
            "fake-model",
            "--input",
            str(paths["input"]),
            "--split-manifest",
            str(paths["split"]),
            "--outcome-flip-config",
            str(config_path),
            "--direction-identity",
            str(identity_path),
            "--condition",
            "D1:Technology",
            "--condition",
            "D1:Financial Services",
            "--condition",
            "F2:Technology",
            "--condition",
            "F2:Financial Services",
            "--contrast-tickers",
            "D1:F2",
            "--output",
            str(output),
        ],
    )
    cli.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "outcome_prior_probe_config"
    assert payload["input_sha256"] == sha256_file(paths["input"])
    assert payload["split_manifest_sha256"] == sha256_file(paths["split"])
    assert payload["outcome_flip_config_sha256"] == sha256_file(config_path)
    assert payload["direction_identity_sha256"] == sha256_file(identity_path)
    assert payload["position_rule"] == "evidence_item_end"
    assert payload["contrast_sectors"] == ["Technology", "Financial Services"]
    assert payload["neutral_evidence_item"] == "No evidence provided."
    assert len(payload["conditions"]) == 4
    PriorProbeConfig.from_dict(payload)  # round-trips through the schema


def test_cli_prepare_prior_probe_config_rejects_bad_conditions(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _write_probe_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    identity_path = tmp_path / "identity.json"
    write_json(
        identity_path,
        {
            "artifact_type": "outcome_flip_direction_identity",
            "split": "discovery",
            "model": "fake-model",
            "input_sha256": sha256_file(paths["input"]),
            "split_manifest_sha256": sha256_file(paths["split"]),
            "config_sha256": sha256_file(config_path),
            "position_rules": ["evidence_item_end"],
            "fitted_layers": [1, 2],
        },
        overwrite=True,
    )
    argv = [
        "jspace-intervention",
        "prepare-prior-probe-config",
        "--model",
        "fake-model",
        "--input",
        str(paths["input"]),
        "--split-manifest",
        str(paths["split"]),
        "--outcome-flip-config",
        str(config_path),
        "--direction-identity",
        str(identity_path),
        "--condition",
        "X9:Technology",
        "--condition",
        "D1:Financial Services",
        "--contrast-tickers",
        "X9:D1",
        "--output",
        str(tmp_path / "probe_config.json"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="not in the input CSV"):
        cli.main()


def test_cli_prepare_prior_probe_config_rejects_identity_config_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _write_probe_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    identity_path = tmp_path / "identity.json"
    write_json(
        identity_path,
        {
            "artifact_type": "outcome_flip_direction_identity",
            "split": "discovery",
            "model": "fake-model",
            "input_sha256": sha256_file(paths["input"]),
            "split_manifest_sha256": sha256_file(paths["split"]),
            "config_sha256": "00" * 32,  # does not bind the V2 config
            "position_rules": ["evidence_item_end"],
            "fitted_layers": [1, 2],
        },
        overwrite=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "prepare-prior-probe-config",
            "--model",
            "fake-model",
            "--input",
            str(paths["input"]),
            "--split-manifest",
            str(paths["split"]),
            "--outcome-flip-config",
            str(config_path),
            "--direction-identity",
            str(identity_path),
            "--condition",
            "D1:Technology",
            "--condition",
            "D1:Financial Services",
            "--contrast-tickers",
            "D1:D1x",
            "--output",
            str(tmp_path / "probe_config.json"),
        ],
    )
    with pytest.raises(ValueError, match="different outcome flip config"):
        cli.main()


def test_cli_run_prior_probe_dispatch(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(prior_probe, "run_prior_probe_pipeline", fake_pipeline)
    config_path = tmp_path / "probe_config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "run-prior-probe",
            "--config",
            str(config_path),
            "--model",
            "fake-model",
            "--run-id",
            "probe-dispatch",
            "--dataset",
            "jspace-outcome-direction-flip",
        ],
    )
    cli.main()
    assert captured["config_path"] == config_path
    assert captured["model_name"] == "fake-model"
    assert captured["run_id"] == "probe-dispatch"
    assert captured["dataset"] == "jspace-outcome-direction-flip"
    assert captured["artifact_root"] == "artifacts"
