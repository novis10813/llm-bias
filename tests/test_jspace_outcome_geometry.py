"""Regression tests for the V2 outcome direction geometric projection workflow."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.core.analysis.statistics import direction_hash
from llm_bias.core.lens_loader import LoadedLens
from llm_bias.jspace_intervention.outcome_flip import run_outcome_flip_pipeline
from llm_bias.jspace_intervention.outcome_geometry import (
    _materialize_records,
    _sector_state_rows,
    accumulate_sector_states,
    analyze_outcome_geometry,
    project_onto_direction,
    run_outcome_geometry_pipeline,
)
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig

BUY_ID = 1000
SELL_ID = 1001

_TOKEN_RE = re.compile(r"buy|sell|.")


def _token_id(token: str) -> int:
    if token == "buy":
        return BUY_ID
    if token == "sell":
        return SELL_ID
    return 1 + (ord(token) % 900)


class _GeometryTokenizer:
    """Character tokenizer with whole-word buy/sell (mirrors the V2 fake)."""

    chat_template = "fake"
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, **_):
        ids, offsets = [], []
        for match in _TOKEN_RE.finditer(str(text)):
            token = match.group(0)
            ids.append(_token_id(token))
            offsets.append((match.start(), match.end()))
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
        return "".join(
            "buy" if i == BUY_ID else "sell" if i == SELL_ID else chr(int(i) - 1)
            for i in ids
        )


class _MeanMixingLayer(torch.nn.Module):
    """Add a fraction of the global mean to every position.

    One copy at the first fitted layer makes the captured state depend on
    every token of the prompt (sector-distinct sector states, including at
    the final position); one copy downstream of the fitted layers makes the
    decision-position readout depend on every position, so the outcome
    gradient is nonzero at every position (the V2 focus-layer fake only
    steers its single focus position).
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.weight * hidden.mean(dim=1, keepdim=True)


class _GeometryModel(torch.nn.Module):
    """Deterministic fake with distinct seeded embeddings per token.

    Layers 1 and 2 (the fitted layers) capture ``embedding + 0.5 * mean``,
    a closed form the tests verify directly.  All parameters are frozen to
    mirror the HFLensModel wrapper used for direction fitting.
    """

    def __init__(self, n_layers=4, d_model=8, vocab=2000, seed=0):
        super().__init__()
        self._embed_tokens = torch.nn.Embedding(vocab, d_model)
        with torch.no_grad():
            self._embed_tokens.weight.copy_(
                torch.nn.init.normal_(
                    torch.empty(vocab, d_model),
                    generator=torch.Generator().manual_seed(seed),
                )
                * 0.05
            )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Identity(),
                _MeanMixingLayer(),
                torch.nn.Identity(),
                _MeanMixingLayer(),
            ]
        )
        self.n_layers = n_layers
        self.d_model = d_model
        self._final_norm = torch.nn.LayerNorm(d_model)
        self._lm_head = torch.nn.Linear(d_model, vocab, bias=False)
        with torch.no_grad():
            self._lm_head.weight.zero_
            unit = torch.zeros(d_model)
            unit[0] = 1.0
            self._lm_head.weight[BUY_ID] = unit
            self._lm_head.weight[SELL_ID] = -unit
        for param in self.parameters():
            param.requires_grad_(False)
        self.config = SimpleNamespace(eos_token_id=None)
        self._hf_model = self

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self._embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _prompt(ticker: str) -> str:
    return (
        f"Decide for [{ticker}].\n"
        "--- Evidence ---\n"
        "1. first item here.\n"
        f"2. second item {ticker}.\n"
        "---\n"
        'Respond with one valid JSON object containing only the keys "decision" '
        "(buy or sell) and \"reason\"."
    )


def _config(**overrides) -> OutcomeFlipConfig:
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
    return OutcomeFlipConfig.from_dict(payload)


def _write_pipeline_inputs(tmp_path: Path) -> dict[str, Path]:
    """Two sectors, four discovery tickers each, one prompt column."""
    tickers = {
        "D1": "Technology",
        "D2": "Technology",
        "D3": "Technology",
        "D4": "Technology",
        "F1": "Financial Services",
        "F2": "Financial Services",
        "F3": "Financial Services",
        "F4": "Financial Services",
    }
    lines = ["Date,ticker,name,sector,marketcap,prompt_with_context_attribute_0"]
    for ticker, sector in tickers.items():
        prompt = _prompt(ticker).replace('"', '""')
        lines.append(
            f"2026-01-01,{ticker},{ticker} Corp,{sector},1000,\"{prompt}\""
        )
    input_path = tmp_path / "trial_plan_prompts.csv"
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "artifact_type": "jspace_intervention_splits",
                "schema_version": 1,
                "assignments": {ticker: "discovery" for ticker in tickers},
            }
        ),
        encoding="utf-8",
    )
    return {"input": input_path, "split": split_path}


def _write_v2_config(tmp_path: Path, split_path: Path) -> Path:
    from llm_bias.core.artifact_paths import sha256_file
    from llm_bias.core.artifacts.io import write_json

    config = _config(split_manifest_sha256=sha256_file(split_path))
    config_path = tmp_path / "outcome_flip_config.json"
    write_json(
        config_path,
        {**config.to_dict(), "artifact_type": "outcome_flip_config", "schema_version": 1},
        overwrite=True,
    )
    return config_path


def _patch_pipeline(monkeypatch) -> None:
    from llm_bias.jspace_intervention import outcome_flip, outcome_geometry

    model = _GeometryModel()
    tokenizer = _GeometryTokenizer()
    monkeypatch.setattr(outcome_flip, "load_tokenizer", lambda model_name: tokenizer)
    monkeypatch.setattr(
        outcome_flip,
        "load_model",
        lambda model_name: (model, tokenizer, torch.device("cpu")),
    )
    monkeypatch.setattr(outcome_geometry, "load_tokenizer", lambda model_name: tokenizer)
    monkeypatch.setattr(
        outcome_geometry,
        "load_model",
        lambda model_name: (model, tokenizer, torch.device("cpu")),
    )


def _run_discovery(paths: dict[str, Path], config_path: Path, artifact_root: Path) -> Path:
    return run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="geo-discovery",
        artifact_root=artifact_root,
        split_name="discovery",
    )


def _manifest(run_root: Path) -> dict:
    return json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Projection math
# ---------------------------------------------------------------------------


def test_project_onto_direction_unit_direction() -> None:
    result = project_onto_direction([3.0, 4.0], [1.0, 0.0])
    assert result["coefficient"] == pytest.approx(3.0)
    assert result["direction_norm"] == pytest.approx(1.0)
    assert result["vector_norm"] == pytest.approx(5.0)
    assert result["parallel_norm"] == pytest.approx(3.0)
    assert result["perpendicular_norm"] == pytest.approx(4.0)
    assert result["cosine"] == pytest.approx(0.6)
    assert result["angle_deg"] == pytest.approx(math.degrees(math.acos(0.6)))
    assert result["parallel_fraction"] == pytest.approx(9.0 / 25.0)
    assert result["pythagoras_error"] < 1e-12


def test_project_onto_direction_is_normalization_invariant() -> None:
    # The general coefficient divides by ||d||^2, so scaling the direction
    # changes the coefficient but not the parallel/perpendicular split.
    unit = project_onto_direction([3.0, 4.0], [1.0, 0.0])
    scaled = project_onto_direction([3.0, 4.0], [3.0, 0.0])
    assert scaled["direction_norm"] == pytest.approx(3.0)
    assert scaled["coefficient"] == pytest.approx(1.0)
    for key in (
        "parallel_norm",
        "perpendicular_norm",
        "cosine",
        "angle_deg",
        "parallel_fraction",
    ):
        assert scaled[key] == pytest.approx(unit[key])


def test_project_onto_direction_orthogonal_and_antiparallel() -> None:
    orthogonal = project_onto_direction([0.0, 1.0], [1.0, 0.0])
    assert orthogonal["coefficient"] == pytest.approx(0.0)
    assert orthogonal["parallel_norm"] == pytest.approx(0.0)
    assert orthogonal["perpendicular_norm"] == pytest.approx(1.0)
    assert orthogonal["angle_deg"] == pytest.approx(90.0)
    antiparallel = project_onto_direction([1.0, 0.0], [-1.0, 0.0])
    assert antiparallel["coefficient"] == pytest.approx(-1.0)
    assert antiparallel["angle_deg"] == pytest.approx(180.0)
    assert antiparallel["parallel_fraction"] == pytest.approx(1.0)


def test_project_onto_direction_zero_vector_and_invalid_inputs() -> None:
    zero = project_onto_direction([0.0, 0.0], [1.0, 0.0])
    assert zero["coefficient"] == pytest.approx(0.0)
    assert zero["cosine"] is None
    assert zero["angle_deg"] is None
    assert zero["parallel_fraction"] is None
    with pytest.raises(ValueError, match="widths differ"):
        project_onto_direction([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="positive and finite"):
        project_onto_direction([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        project_onto_direction([1.0, float("nan")], [1.0, 0.0])


# ---------------------------------------------------------------------------
# Sector state accumulation (fake model, closed-form expectation)
# ---------------------------------------------------------------------------


def _materialize_sectors(model: _GeometryModel, tokenizer, config) -> list[dict]:
    records = []
    for ticker, sector in (
        ("D1", "Technology"),
        ("D2", "Technology"),
        ("F1", "Financial Services"),
        ("F2", "Financial Services"),
    ):
        records.append(
            {
                "ticker": ticker,
                "name": f"{ticker} Corp",
                "sector": sector,
                "marketcap": "1000",
                "prompt_column": "prompt_with_context_attribute_0",
                "prompt": _prompt(ticker),
                "record_id": f"record_{ticker.lower()}",
            }
        )
    return _materialize_records(records, tokenizer, config)


def test_accumulate_sector_states_matches_closed_form() -> None:
    model = _GeometryModel()
    tokenizer = _GeometryTokenizer()
    config = _config()
    records = _materialize_sectors(model, tokenizer, config)
    layers = list(config.fitted_layers)
    state = accumulate_sector_states(model, records, layers=layers, device="cpu")

    assert set(state) == {"Technology", "Financial Services"}
    for sector in state:
        for position_set in state[sector]:
            for layer in state[sector][position_set]:
                entry = state[sector][position_set][layer]
                assert entry["count"] == 2
    # Fitted layers 1/2 capture ``embedding + 0.5 * global mean``, so the
    # per-sector mean state has a closed form independent of the capture path.
    expected = {}
    for sector in ("Technology", "Financial Services"):
        sector_records = [r for r in records if r["sector"] == sector]
        for position_set in records[0]["position_sets"]:
            for layer in layers:
                sums = []
                for record in sector_records:
                    embeddings = model._embed_tokens(
                        torch.tensor([record["prompt_ids"]])
                    ).float()
                    mixed = embeddings + 0.5 * embeddings.mean(dim=1, keepdim=True)
                    positions = torch.as_tensor(
                        sorted(set(record["position_sets"][position_set]))
                    )
                    sums.append(mixed[0, positions].mean(0))
                expected[(sector, position_set, layer)] = torch.stack(sums).mean(0)
    for (sector, position_set, layer), mean in expected.items():
        actual = state[sector][position_set][layer]["sum"] / 2
        assert torch.allclose(actual, mean, atol=1e-6), (sector, position_set, layer)

    rows = _sector_state_rows(state)
    assert len(rows) == 2 * len(records[0]["position_sets"]) * len(layers)
    by_key = {
        (row["sector"], row["position_set"], row["layer"]): row for row in rows
    }
    row = by_key[("Technology", "final_position", layers[0])]
    mean = state["Technology"]["final_position"][layers[0]]["sum"] / 2
    assert row["state_mean_norm"] == pytest.approx(float(mean.norm()))
    assert row["state_mean_sha256"] == direction_hash(mean)
    assert row["record_count"] == 2


# ---------------------------------------------------------------------------
# Analysis metrics
# ---------------------------------------------------------------------------


def _synthetic_state_and_directions() -> tuple[dict, dict]:
    torch.manual_seed(3)
    source = {
        posset: {layer: torch.randn(8) for layer in (1, 2)}
        for posset in ("evidence_item_end", "final_position")
    }
    contrast = {
        posset: {layer: torch.randn(8) for layer in (1, 2)}
        for posset in ("evidence_item_end", "final_position")
    }
    state = {
        "Technology": {
            posset: {
                layer: {"sum": vector, "count": 1, "record_norm_sum": 0.0}
                for layer, vector in source[posset].items()
            }
            for posset in source
        },
        "Financial Services": {
            posset: {
                layer: {"sum": vector, "count": 1, "record_norm_sum": 0.0}
                for layer, vector in contrast[posset].items()
            }
            for posset in contrast
        },
    }
    directions = {
        "evidence_item_end": {layer: torch.randn(8) / 3.0 for layer in (1, 2)}
    }
    return state, directions


def test_analyze_outcome_geometry_metrics_and_tfidf_section() -> None:
    state, directions = _synthetic_state_and_directions()
    tfidf = {
        "source": {1: torch.randn(8), 2: torch.randn(8)},
        "contrast": {1: torch.randn(8), 2: torch.randn(8)},
    }
    result = analyze_outcome_geometry(
        directions=directions,
        state=state,
        source_sector="Technology",
        contrast_sector="Financial Services",
        tfidf_prototypes=tfidf,
    )
    assert result["position_sets"] == ["evidence_item_end", "final_position"]
    assert result["rules"] == ["evidence_item_end"]
    for layer_key, layer in (("1", 1), ("2", 2)):
        entry = result["layers"][layer_key]
        for position_set in result["position_sets"]:
            delta = (
                state["Technology"][position_set][layer]["sum"]
                - state["Financial Services"][position_set][layer]["sum"]
            )
            sector_state = entry["sector_state"][position_set]
            assert sector_state["delta_state_norm"] == pytest.approx(float(delta.norm()))
            assert sector_state["delta_state_sha256"] == direction_hash(delta)
            projection = sector_state["projection"]["evidence_item_end"]
            expected = project_onto_direction(delta, directions["evidence_item_end"][layer])
            for key, value in expected.items():
                if value is None:
                    assert projection[key] is None
                else:
                    assert projection[key] == pytest.approx(value)
        # Non-unit direction: the reported norm must expose that fact.
        assert entry["direction"]["evidence_item_end"][layer_key]["norm"] == pytest.approx(
            float(directions["evidence_item_end"][layer].norm())
        )
        tfidf_entry = entry["tfidf_prototype"]
        for label in ("source", "contrast"):
            vector = tfidf[label][layer]
            assert tfidf_entry[label]["sha256"] == direction_hash(vector)
            projected = tfidf_entry[label]["projection"]["evidence_item_end"]
            # Split-norm reconstruction of the (float32) vector norm.
            split = projected["parallel_norm"] ** 2 + projected["perpendicular_norm"] ** 2
            assert split == pytest.approx(float(vector.norm()) ** 2, rel=1e-6)
        assert -1.0 <= tfidf_entry["cosine_source_contrast"] <= 1.0
        assert set(tfidf_entry["cosine_to_delta_state"]) == set(result["position_sets"])


# ---------------------------------------------------------------------------
# Pipeline lifecycle (fake model, no GPU)
# ---------------------------------------------------------------------------


def test_run_outcome_geometry_pipeline_full_lifecycle(tmp_path: Path, monkeypatch) -> None:
    _patch_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_v2_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"

    discovery_root = _run_discovery(paths, config_path, artifact_root)
    identity_path = discovery_root / "forward" / "direction_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))

    run_root = run_outcome_geometry_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        direction_identity_path=identity_path,
        model_name="fake-model",
        run_id="geo-run",
        artifact_root=artifact_root,
        contrast_sector="Financial Services",
    )
    manifest = _manifest(run_root)
    assert manifest["status"] == "complete"
    for stage in ("prepare", "forward", "analyze"):
        assert manifest["stages"][stage]["status"] == "complete"

    prepare_metadata = json.loads(
        (run_root / "prepare" / "metadata.json").read_text(encoding="utf-8")
    )
    assert prepare_metadata["source_record_count"] == 4
    assert prepare_metadata["contrast_record_count"] == 4
    prompt_rows = [
        json.loads(line)
        for line in (run_root / "prepare" / "prompt_records.jsonl").open()
        if line.strip()
    ]
    assert {row["sector"] for row in prompt_rows} == {
        "Technology",
        "Financial Services",
    }
    assert all(row["artifact_type"] == "outcome_geometry_prompt_record" for row in prompt_rows)

    stats_rows = [
        json.loads(line)
        for line in (run_root / "forward" / "sector_state_statistics.jsonl").open()
        if line.strip()
    ]
    # 2 sectors x 2 position sets (evidence_item_end + final_position) x 2 layers
    assert len(stats_rows) == 2 * 2 * 2
    assert {row["artifact_type"] for row in stats_rows} == {"outcome_geometry_sector_state"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["state_mean_sha256"]) for row in stats_rows)

    forward_metadata = json.loads(
        (run_root / "forward" / "metadata.json").read_text(encoding="utf-8")
    )
    assert forward_metadata["direction_verified"] is True
    assert forward_metadata["direction_identity_sha256"]
    for rule, per_layer in forward_metadata["direction"].items():
        for layer_key, entry in per_layer.items():
            assert entry["sha256"] == identity["directions"][rule][layer_key]["sha256"]
            assert entry["norm"] == pytest.approx(1.0)

    analysis = json.loads(
        (run_root / "analyze" / "outcome_geometry_analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["artifact_type"] == "outcome_geometry_analysis"
    assert analysis["source_sector"] == "Technology"
    assert analysis["contrast_sector"] == "Financial Services"
    assert analysis["record_counts"] == {
        "source": 4,
        "contrast": 4,
        "source_tickers": 4,
        "contrast_tickers": 4,
    }
    assert set(analysis["layers"]) == {"1", "2"}
    assert analysis["position_sets"] == ["evidence_item_end", "final_position"]
    for layer_key, entry in analysis["layers"].items():
        for position_set in analysis["position_sets"]:
            sector_state = entry["sector_state"][position_set]
            delta_norm = sector_state["delta_state_norm"]
            projection = sector_state["projection"]["evidence_item_end"]
            # Pythagoras: parallel^2 + perpendicular^2 == delta^2 (float64 math).
            assert projection["parallel_norm"] ** 2 + projection["perpendicular_norm"] ** 2 == pytest.approx(
                delta_norm**2, rel=1e-9, abs=1e-12
            )
            assert 0.0 <= projection["parallel_fraction"] <= 1.0
            assert 0.0 <= projection["angle_deg"] <= 180.0
            assert projection["direction_norm"] == pytest.approx(1.0)
            assert re.fullmatch(r"[0-9a-f]{64}", sector_state["delta_state_sha256"])
        # The sector difference must be nonzero for this fake (distinct tickers).
        assert all(
            entry["sector_state"][position_set]["delta_state_norm"] > 0.0
            for position_set in analysis["position_sets"]
        )
    assert "not causal" in " ".join(analysis["interpretation_limits"])


def test_run_outcome_geometry_pipeline_fail_closed_on_tampered_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_v2_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_discovery(paths, config_path, artifact_root)
    identity_path = discovery_root / "forward" / "direction_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["directions"]["evidence_item_end"]["1"]["sha256"] = "00" * 32
    tampered_path = tmp_path / "tampered_identity.json"
    tampered_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        run_outcome_geometry_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            direction_identity_path=tampered_path,
            model_name="fake-model",
            run_id="geo-tamper",
            artifact_root=artifact_root,
        )
    assert (
        _manifest(
            artifact_root / "fake-model" / "jspace-outcome-direction-geometry" / "runs" / "geo-tamper"
        )["status"]
        == "failed"
    )


def test_run_outcome_geometry_pipeline_rejects_bad_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_v2_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_discovery(paths, config_path, artifact_root)
    identity_path = discovery_root / "forward" / "direction_identity.json"
    with pytest.raises(FileNotFoundError, match="direction identity"):
        run_outcome_geometry_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            direction_identity_path=tmp_path / "missing.json",
            model_name="fake-model",
            run_id="geo-missing-identity",
            artifact_root=artifact_root,
        )
    assert not (
        artifact_root / "fake-model" / "jspace-outcome-direction-geometry" / "runs" / "geo-missing-identity"
    ).exists()
    with pytest.raises(ValueError, match="no discovery prompt records"):
        run_outcome_geometry_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            direction_identity_path=identity_path,
            model_name="fake-model",
            run_id="geo-bad-sector",
            artifact_root=artifact_root,
            contrast_sector="Healthcare",
        )
    with pytest.raises(ValueError, match="must differ from the source sector"):
        run_outcome_geometry_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            direction_identity_path=identity_path,
            model_name="fake-model",
            run_id="geo-same-sector",
            artifact_root=artifact_root,
            contrast_sector="Technology",
        )


def _fake_tfidf_config(
    tmp_path: Path, *, model_name: str = "fake-model", score_type: str = "contrastive_tfidf"
) -> Path:
    def spec(sector: str, token_ids: tuple[int, ...]) -> dict:
        return {
            "name": f"{sector}:{score_type}",
            "sector": sector,
            "score_type": score_type,
            "tokens": [
                {
                    "token": f"{sector.lower()}-token-{index}",
                    "token_id": token_id,
                    "weight": 0.5,
                    "selection_score": 0.1,
                }
                for index, token_id in enumerate(token_ids)
            ],
        }

    path = tmp_path / "tfidf_config.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "jspace_intervention_config",
                "model": model_name,
                "source": spec("Technology", (20, 21)),
                "target": spec("Financial Services", (22, 23)),
                "layers": [1, 2],
            }
        ),
        encoding="utf-8",
    )
    return path


def _patch_fake_lens(monkeypatch, tmp_path: Path, d_model: int = 8) -> Path:
    from llm_bias.jspace_intervention import outcome_geometry

    lens_path = tmp_path / "lens.pt"
    lens_path.write_bytes(b"fake-lens")

    def fake_load(**_kwargs):
        identity = torch.eye(d_model)
        return LoadedLens(
            lens=SimpleNamespace(jacobians={1: identity, 2: identity}),
            path=lens_path,
            metadata={"provenance": {"source": "fake"}},
        )

    monkeypatch.setattr(outcome_geometry, "load_validated_lens", fake_load)
    return lens_path


def test_run_outcome_geometry_pipeline_with_tfidf(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_v2_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_discovery(paths, config_path, artifact_root)
    lens_path = _patch_fake_lens(monkeypatch, tmp_path)
    tfidf_config_path = _fake_tfidf_config(tmp_path)

    run_root = run_outcome_geometry_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        direction_identity_path=discovery_root / "forward" / "direction_identity.json",
        model_name="fake-model",
        run_id="geo-tfidf-run",
        artifact_root=artifact_root,
        contrast_sector="Financial Services",
        lens_path=lens_path,
        tfidf_config_path=tfidf_config_path,
    )
    assert _manifest(run_root)["status"] == "complete"
    forward_metadata = json.loads(
        (run_root / "forward" / "metadata.json").read_text(encoding="utf-8")
    )
    assert forward_metadata["tfidf_layers"] == [1, 2]
    assert set(forward_metadata["tfidf_prototypes"]) == {"source", "contrast"}
    analysis = json.loads(
        (run_root / "analyze" / "outcome_geometry_analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["tfidf_config"] == str(tfidf_config_path)
    for layer_key, entry in analysis["layers"].items():
        tfidf_entry = entry["tfidf_prototype"]
        for label in ("source", "contrast"):
            assert tfidf_entry[label]["norm"] == pytest.approx(1.0, rel=1e-5)
            projected = tfidf_entry[label]["projection"]["evidence_item_end"]
            assert (
                projected["parallel_norm"] ** 2 + projected["perpendicular_norm"] ** 2
                == pytest.approx(tfidf_entry[label]["norm"] ** 2, rel=1e-6)
            )
        assert -1.0 <= tfidf_entry["cosine_source_contrast"] <= 1.0


def test_run_outcome_geometry_pipeline_rejects_non_tfidf_prototype(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_v2_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = _run_discovery(paths, config_path, artifact_root)
    lens_path = _patch_fake_lens(monkeypatch, tmp_path)
    logodds_path = _fake_tfidf_config(tmp_path, score_type="logodds_z")
    with pytest.raises(ValueError, match="contrastive_tfidf"):
        run_outcome_geometry_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            direction_identity_path=discovery_root / "forward" / "direction_identity.json",
            model_name="fake-model",
            run_id="geo-logodds",
            artifact_root=artifact_root,
            lens_path=lens_path,
            tfidf_config_path=logodds_path,
        )
    assert not (
        artifact_root
        / "fake-model"
        / "jspace-outcome-direction-geometry"
        / "runs"
        / "geo-logodds"
    ).exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_run_outcome_geometry_dispatch(monkeypatch, tmp_path) -> None:
    from llm_bias.jspace_intervention import cli, outcome_geometry

    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(outcome_geometry, "run_outcome_geometry_pipeline", fake_pipeline)
    for name in (
        "input.csv",
        "splits.json",
        "config.json",
        "identity.json",
        "tfidf.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "run-outcome-geometry",
            "--input",
            str(tmp_path / "input.csv"),
            "--split-manifest",
            str(tmp_path / "splits.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--direction-identity",
            str(tmp_path / "identity.json"),
            "--model",
            "fake-model",
            "--run-id",
            "geo-cli",
            "--contrast-sector",
            "Healthcare",
            "--tfidf-config",
            str(tmp_path / "tfidf.json"),
        ],
    )
    cli.main()
    assert captured["contrast_sector"] == "Healthcare"
    assert captured["tfidf_config_path"] == tmp_path / "tfidf.json"
    assert captured["dataset"] == "jspace-outcome-direction-geometry"
