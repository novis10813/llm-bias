"""Regression tests for the V2 direction decode (Jacobian-lens readout of d_l)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llm_bias.core.continuation_scoring import (
    fp32_next_token_log_probs,
    fp32_next_token_logits,
)
from llm_bias.core.lens_loader import LoadedLens
from llm_bias.jspace_intervention.outcome_decode import (
    ANTISYMMETRY_REL_TOL,
    build_decode_summary,
    decode_direction_layers,
    resolve_answer_token_ids,
    run_outcome_decode_pipeline,
    transported_direction_logits,
)
from llm_bias.jspace_intervention.outcome_flip import run_outcome_flip_pipeline
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig

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
                parts.append(chr(max(int(token_id) - 1, 0)))
        return "".join(parts)


class _FocusLayer(torch.nn.Module):
    """Asymmetric copy: the last position receives the final non-zero
    earlier position's residual (see test_jspace_outcome_flip.py)."""

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
    """Deterministic differentiable fake with a buy/sell-oriented head."""

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


class _FakeLens:
    """JacobianLens operator contract: transport(residual, layer) == residual @ J_l.T."""

    def __init__(self, jacobians: dict[int, torch.Tensor]):
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = 1
        self.d_model = next(iter(self.jacobians.values())).shape[0]

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.jacobians[layer].to(residual.device)
        return residual @ J.T


def _fake_jacobians(layers=(1, 2), d_model=4, seed=0) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        layer: torch.nn.init.orthogonal_(
            torch.empty(d_model, d_model), generator=generator
        )
        for layer in layers
    }


def _prompt(ticker: str = "A1", control: str = "ZZ") -> str:
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
    from llm_bias.jspace_intervention.outcome_flip import _materialize_records

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


def _fit_directions(config=None) -> dict[int, torch.Tensor]:
    from llm_bias.jspace_intervention.outcome_flip import fit_outcome_directions

    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    config = config or _config()
    fit = fit_outcome_directions(
        model=model,
        tokenizer=tokenizer,
        records=[
            _materialize("A1", tokenizer=tokenizer, config=config),
            _materialize("A2", tokenizer=tokenizer, config=config),
            _materialize("B1", tokenizer=tokenizer, config=config),
        ],
        config=config,
        device=torch.device("cpu"),
    )
    return fit.directions[config.position_rules[0]]


# ---------------------------------------------------------------------------
# fp32 tail
# ---------------------------------------------------------------------------


class _TailModel(torch.nn.Module):
    def __init__(self, d_model=4, vocab=32):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.n_layers = 1
        self._final_norm = torch.nn.LayerNorm(d_model)
        self._lm_head = torch.nn.Linear(d_model, vocab, bias=False)
        torch.manual_seed(11)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = torch.nn.functional.one_hot(
            input_ids, num_classes=self.d_model
        ).float()
        return SimpleNamespace(last_hidden_state=hidden)


def test_fp32_next_token_logits_is_the_log_probs_tail_without_softmax() -> None:
    model = _TailModel()
    residual = torch.nn.init.normal_(torch.empty(2, 4))
    logits = fp32_next_token_logits(model, residual)
    log_probs = fp32_next_token_log_probs(model, residual)
    assert logits.shape == (2, 32)
    assert torch.allclose(
        F.log_softmax(logits, dim=-1), log_probs, atol=1e-6
    )


# ---------------------------------------------------------------------------
# Decode math
# ---------------------------------------------------------------------------


def test_transport_antisymmetry_and_norm_change() -> None:
    model = _FlipModel()
    lens = _FakeLens({1: 2.0 * torch.eye(4)})
    direction = torch.nn.init.normal_(torch.empty(4)).reshape(-1)
    direction = direction / direction.norm()
    z_plus = transported_direction_logits(model, lens, direction, 1)
    z_minus = transported_direction_logits(model, lens, -direction, 1)
    assert torch.allclose(z_plus, -z_minus, atol=1e-5)
    # The transport is linear but not norm-preserving: J = 2I doubles the
    # unit direction, so the decoded logit magnitude is set by ||J d||, not
    # by ||d||.
    transported = lens.transport(direction.unsqueeze(0), 1)
    assert float(transported.norm()) == pytest.approx(2.0, rel=1e-5)
    with pytest.raises(ValueError, match="source layer"):
        transported_direction_logits(model, lens, direction, 9)


def test_decode_direction_layers_records_and_antisymmetry() -> None:
    model = _FlipModel()
    lens = _FakeLens(_fake_jacobians())
    directions = _fit_directions()
    tokenizer = _FlipTokenizer()
    records, probability_vectors, max_deviation = decode_direction_layers(
        model=model,
        lens=lens,
        directions=directions,
        layers=[1, 2],
        top_k=5,
        answer_ids={"positive": BUY_ID, "negative": SELL_ID},
        tokenizer=tokenizer,
    )
    assert max_deviation <= ANTISYMMETRY_REL_TOL
    assert [(record["layer"], record["sign"]) for record in records] == [
        (1, 1), (1, -1), (2, 1), (2, -1)
    ]
    by_sign = {}
    for record in records:
        assert len(record["top_tokens"]) == 5
        logits = [row["logit"] for row in record["top_tokens"]]
        assert logits == sorted(logits, reverse=True)
        assert all(0.0 <= row["probability"] <= 1.0 for row in record["top_tokens"])
        assert record["positive"]["token_id"] == BUY_ID
        assert record["negative"]["token_id"] == SELL_ID
        assert record["direction_hash"] == re.fullmatch(
            r"[0-9a-f]{64}", record["direction_hash"]
        ).group(0)
        assert record["direction_norm"] == pytest.approx(1.0, rel=1e-5)
        by_sign.setdefault(record["sign"], {})[record["layer"]] = record
    for layer in (1, 2):
        plus, minus = by_sign[1][layer], by_sign[-1][layer]
        assert plus["answer_logit_margin"] == pytest.approx(
            -minus["answer_logit_margin"], abs=1e-5
        )
        assert plus["answer_probability_margin"] == pytest.approx(
            -minus["answer_probability_margin"], abs=1e-5
        )
    assert probability_vectors[1][0].dtype == torch.float64
    assert probability_vectors[1][0].shape == (2000,)
    assert float(probability_vectors[1][0].sum()) == pytest.approx(1.0, rel=1e-4)


def test_decode_direction_layers_antisymmetry_self_check_fails_closed(
    monkeypatch,
) -> None:
    from llm_bias.jspace_intervention import outcome_decode

    monkeypatch.setattr(outcome_decode, "ANTISYMMETRY_REL_TOL", -1.0)
    model = _FlipModel()
    lens = _FakeLens(_fake_jacobians())
    directions = _fit_directions()
    with pytest.raises(ValueError, match="antisymmetry self-check failed"):
        decode_direction_layers(
            model=model,
            lens=lens,
            directions=directions,
            layers=[1],
            top_k=5,
            answer_ids={"positive": BUY_ID, "negative": SELL_ID},
            tokenizer=_FlipTokenizer(),
        )


def test_resolve_answer_token_ids_happy_path_and_fail_closed(
    monkeypatch,
) -> None:
    from llm_bias.jspace_intervention import outcome_decode

    tokenizer = _FlipTokenizer()
    config = _config()
    records = [
        _materialize("A1", tokenizer=tokenizer, config=config),
        _materialize("A2", tokenizer=tokenizer, config=config),
    ]
    resolved = resolve_answer_token_ids(tokenizer, records, config)
    assert resolved == {"buy": BUY_ID, "sell": SELL_ID}
    with pytest.raises(ValueError, match="single continuation token"):
        resolve_answer_token_ids(
            tokenizer, records, _config(positive_candidate="buy sell")
        )
    calls = {"n": 0}

    def _unstable_continuation(tokenizer_, prompt, candidate):
        calls["n"] += 1
        return ([1, 2], [7 if calls["n"] == 1 else 8])

    monkeypatch.setattr(
        outcome_decode, "continuation_token_ids", _unstable_continuation
    )
    with pytest.raises(ValueError, match="not unique"):
        resolve_answer_token_ids(tokenizer, records, config)
    with pytest.raises(ValueError, match="requires discovery records"):
        resolve_answer_token_ids(tokenizer, [], config)


def test_band_scope_averages_before_topk() -> None:
    tokenizer = _FlipTokenizer()
    vocab = 8
    # Layer 1 puts all mass on token 3; layer 2 splits between 3 and 5.
    p1 = torch.zeros(vocab, dtype=torch.float64)
    p1[3] = 1.0
    p2 = torch.zeros(vocab, dtype=torch.float64)
    p2[3] = 0.5
    p2[5] = 0.5
    probability_vectors = {
        1: [p1, p2],
        -1: [1.0 - p1, 1.0 - p2],
    }
    plus_record = {
        "layer": 1, "sign": 1,
        "answer_logit_margin": 0.5, "answer_probability_margin": 0.25,
        "positive": {"token_id": BUY_ID, "rank": 7},
        "negative": {"token_id": SELL_ID, "rank": 8},
    }
    minus_record = {
        "layer": 1, "sign": -1,
        "answer_logit_margin": -0.5, "answer_probability_margin": -0.25,
        "positive": {"token_id": BUY_ID, "rank": 8},
        "negative": {"token_id": SELL_ID, "rank": 7},
    }
    plus2 = dict(plus_record, layer=2)
    minus2 = dict(minus_record, layer=2)
    summary = build_decode_summary(
        band=[1, 2],
        position_rule="evidence_item_end",
        layers=[1, 2],
        top_k=3,
        records=[plus_record, minus_record, plus2, minus2],
        probability_vectors=probability_vectors,
        answer_ids={"positive": 4, "negative": 6},
        tokenizer=tokenizer,
    )
    band = summary["band_scope"]["plus_d"]
    # Mean probability: token 3 = 0.75, token 5 = 0.25, all else 0 -> the
    # averaged vector, not either per-layer vector, defines the band order
    # (ties break by ascending token id).
    assert [row["token_id"] for row in band["top_tokens"]] == [3, 5, 0]
    assert band["top_tokens"][0]["probability"] == pytest.approx(0.75)
    assert band["top_tokens"][1]["probability"] == pytest.approx(0.25)
    # The answer tokens (4/6) have zero band mass: the margin is 0, not the
    # per-layer margin.
    assert band["probability_margin"] == pytest.approx(0.0)
    margins = {row["layer"]: row for row in summary["answer_margins"]}
    assert margins[1]["plus_d"]["logit_margin"] == pytest.approx(0.5)
    assert margins[2]["minus_d"]["probability_margin"] == pytest.approx(-0.25)
    assert "not standalone causal" in summary["interpretation"]


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


def _write_pipeline_inputs(tmp_path: Path) -> dict[str, Path]:
    input_path = tmp_path / "trial_plan_prompts.csv"
    tickers = {
        "D1": ("discovery", "ZZ"),
        "D2": ("discovery", "zz"),
    }
    lines = ["Date,ticker,name,sector,marketcap,prompt_with_context_attribute_0"]
    for ticker, (split, control) in tickers.items():
        prompt = _prompt(ticker, control=control).replace('"', '""')
        lines.append(f"2026-01-01,{ticker},{ticker} Corp,Technology,1000,\"{prompt}\"")
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

    config = _config(split_manifest_sha256=sha256_file(split_path))
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


def _write_selection(
    tmp_path: Path, config_path: Path, identity_path: Path, **selected_overrides
) -> Path:
    from llm_bias.core.artifact_paths import sha256_file

    selected = {
        "band": [1, 2],
        "position_rule": "evidence_item_end",
        "relative_dose": 0.5,
    }
    selected.update(selected_overrides)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "artifact_type": "outcome_flip_selection",
                "schema_version": 1,
                "split": "calibration",
                "selected": selected,
                "config_sha256": sha256_file(config_path),
                "direction_identity_sha256": sha256_file(identity_path),
            }
        ),
        encoding="utf-8",
    )
    return selection_path


def _patch_decode_pipeline(monkeypatch, tmp_path: Path) -> None:
    from llm_bias.jspace_intervention import outcome_decode

    model = _FlipModel()
    tokenizer = _FlipTokenizer()
    lens = _FakeLens(_fake_jacobians())
    lens_path = tmp_path / "fake_lens.pt"
    torch.save({"J": {}, "n_prompts": 1, "source_layers": [1, 2], "d_model": 4}, lens_path)
    monkeypatch.setattr(
        outcome_decode, "load_tokenizer", lambda model_name: tokenizer
    )
    monkeypatch.setattr(
        outcome_decode,
        "load_model",
        lambda model_name: (model, tokenizer, torch.device("cpu")),
    )
    monkeypatch.setattr(
        outcome_decode,
        "load_validated_lens",
        lambda **_kwargs: LoadedLens(
            lens=lens, path=lens_path, metadata={"provenance": {"source": "test"}}
        ),
    )


def _patch_flip_pipeline(monkeypatch) -> None:
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


@pytest.fixture(autouse=True)
def _restore_deterministic_flag():
    """Pipeline entry points enable deterministic algorithms; keep the flag
    from leaking between tests."""
    previous = torch.are_deterministic_algorithms_enabled()
    yield
    if torch.are_deterministic_algorithms_enabled() and not previous:
        torch.use_deterministic_algorithms(False)


def test_run_outcome_decode_pipeline_full_lifecycle(tmp_path: Path, monkeypatch) -> None:
    _patch_flip_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"

    discovery_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="decode-discovery",
        artifact_root=artifact_root,
        split_name="discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    selection_path = _write_selection(tmp_path, config_path, identity_path)

    _patch_decode_pipeline(monkeypatch, tmp_path)
    run_root = run_outcome_decode_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="decode-run",
        direction_identity_path=identity_path,
        calibration_selection_path=selection_path,
        artifact_root=artifact_root,
        top_k=4,
    )
    manifest = _manifest(run_root)
    assert manifest["status"] == "complete"
    for stage in ("prepare", "forward", "analyze"):
        assert manifest["stages"][stage]["status"] == "complete"
    roles = {
        (ref["artifact_type"], ref["role"]) for ref in manifest["artifacts"]
    }
    assert ("jacobian_lens", "lens") in roles
    assert ("outcome_flip_direction_identity", "input") in roles
    assert ("outcome_flip_selection", "input") in roles

    source = json.loads(
        (run_root / "prepare" / "direction_source.json").read_text(encoding="utf-8")
    )
    assert source["artifact_type"] == "outcome_direction_decode_source"
    assert source["selected"]["band"] == [1, 2]
    assert source["selected"]["position_rule"] == "evidence_item_end"
    assert source["selected"]["relative_dose"] == 0.5
    assert source["decode_layers"] == [1, 2]
    assert source["answer_tokens"] == {
        "buy": {"token_id": BUY_ID, "token": "buy"},
        "sell": {"token_id": SELL_ID, "token": "sell"},
    }
    assert source["discovery_record_count"] == 2
    assert source["direction_identity_sha256"]

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    decode_rows = [
        json.loads(line)
        for line in (run_root / "forward" / "direction_decode.jsonl").open(
            encoding="utf-8"
        )
        if line.strip()
    ]
    assert [(row["layer"], row["sign"]) for row in decode_rows] == [
        (1, 1), (1, -1), (2, 1), (2, -1)
    ]
    for row in decode_rows:
        assert row["artifact_type"] == "outcome_direction_decode"
        expected_layer = identity["directions"]["evidence_item_end"][str(row["layer"])]
        assert row["direction_hash"] == expected_layer["sha256"]
    plus = next(row for row in decode_rows if row["sign"] == 1)
    minus = next(row for row in decode_rows if row["sign"] == -1 and row["layer"] == plus["layer"])
    assert plus["answer_logit_margin"] == pytest.approx(-minus["answer_logit_margin"], abs=1e-5)

    forward_metadata = json.loads(
        (run_root / "forward" / "metadata.json").read_text(encoding="utf-8")
    )
    assert forward_metadata["direction_recompute"]["record_count"] == 2
    assert forward_metadata["self_check"]["passed"] is True
    assert forward_metadata["self_check"]["max_abs_deviation"] <= ANTISYMMETRY_REL_TOL

    summary = json.loads(
        (run_root / "analyze" / "direction_decode_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["artifact_type"] == "outcome_direction_decode_summary"
    assert summary["band"] == [1, 2]
    assert set(summary["band_scope"]) == {"plus_d", "minus_d"}
    assert len(summary["band_scope"]["plus_d"]["top_tokens"]) == 4
    assert len(summary["answer_margins"]) == 2
    assert "not standalone causal" in summary["interpretation"]

    # No direction vector, gradient, Jacobian, activation, or residual payload
    # may appear in any artifact.
    forbidden = re.compile(r"(gradient|residual|activation|jacobian)", re.IGNORECASE)
    for path in sorted(run_root.rglob("*.json*")):
        if path.name == "manifest.json":
            continue
        text = path.read_text(encoding="utf-8")
        keys = re.findall(r'"([A-Za-z0-9_.-]+)"\s*:', text)
        assert not any(forbidden.search(key) for key in keys), (path, keys)


def test_run_outcome_decode_pipeline_fail_closed_on_tampered_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_flip_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="decode-discovery-tamper",
        artifact_root=artifact_root,
        split_name="discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["directions"]["evidence_item_end"]["1"]["sha256"] = "00" * 32
    tampered_path = tmp_path / "tampered_identity.json"
    tampered_path.write_text(json.dumps(identity), encoding="utf-8")
    selection_path = _write_selection(tmp_path, config_path, tampered_path)

    _patch_decode_pipeline(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_outcome_decode_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="decode-tamper",
            direction_identity_path=tampered_path,
            calibration_selection_path=selection_path,
            artifact_root=artifact_root,
        )
    tamper_root = (
        artifact_root / "fake-model" / "jspace-outcome-direction-decode"
        / "runs" / "decode-tamper"
    )
    assert _manifest(tamper_root)["status"] == "failed"


def test_run_outcome_decode_pipeline_rejects_unbound_selection(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_flip_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="decode-discovery-unbound",
        artifact_root=artifact_root,
        split_name="discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    other_identity_path = tmp_path / "other_identity.json"
    other_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    other_identity_path.write_text(json.dumps(other_identity), encoding="utf-8")
    # Selection binds the other identity, not the one being passed.
    selection_path = _write_selection(tmp_path, config_path, other_identity_path)
    _patch_decode_pipeline(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="does not bind"):
        run_outcome_decode_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="decode-unbound",
            direction_identity_path=identity_path,
            calibration_selection_path=selection_path,
            artifact_root=artifact_root,
        )
    assert not (
        artifact_root / "fake-model" / "jspace-outcome-direction-decode"
        / "runs" / "decode-unbound"
    ).exists()


def test_run_outcome_decode_pipeline_rejects_band_outside_candidate_set(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_flip_pipeline(monkeypatch)
    paths = _write_pipeline_inputs(tmp_path)
    config_path = _write_outcome_config(tmp_path, paths["split"])
    artifact_root = tmp_path / "artifacts"
    discovery_root = run_outcome_flip_pipeline(
        input_path=paths["input"],
        split_manifest=paths["split"],
        config_path=config_path,
        model_name="fake-model",
        run_id="decode-discovery-band",
        artifact_root=artifact_root,
        split_name="discovery",
    )
    identity_path = discovery_root / "forward" / "direction_identity.json"
    selection_path = _write_selection(
        tmp_path, config_path, identity_path, band=[1, 1]
    )
    _patch_decode_pipeline(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="not in the frozen candidate set"):
        run_outcome_decode_pipeline(
            input_path=paths["input"],
            split_manifest=paths["split"],
            config_path=config_path,
            model_name="fake-model",
            run_id="decode-bad-band",
            direction_identity_path=identity_path,
            calibration_selection_path=selection_path,
            artifact_root=artifact_root,
        )
    assert not (
        artifact_root / "fake-model" / "jspace-outcome-direction-decode"
        / "runs" / "decode-bad-band"
    ).exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_decode_outcome_direction_dispatch(monkeypatch, tmp_path) -> None:
    from llm_bias.jspace_intervention import cli, outcome_decode

    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(
        outcome_decode, "run_outcome_decode_pipeline", fake_pipeline
    )
    paths = _write_pipeline_inputs(tmp_path)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text("{}", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jspace-intervention",
            "decode-outcome-direction",
            "--input", str(paths["input"]),
            "--split-manifest", str(paths["split"]),
            "--config", str(config_path),
            "--direction-identity", str(identity_path),
            "--calibration-selection", str(selection_path),
            "--model", "fake-model",
            "--lens", "some/lens.pt",
            "--run-id", "decode-dispatch",
            "--top-k", "7",
        ],
    )
    cli.main()
    assert captured["run_id"] == "decode-dispatch"
    assert captured["dataset"] == "jspace-outcome-direction-decode"
    assert captured["top_k"] == 7
    assert captured["lens_path"] == "some/lens.pt"
    assert captured["direction_identity_path"] == identity_path
