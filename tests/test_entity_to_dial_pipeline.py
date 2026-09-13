"""End-to-end smoke tests for the entity-to-dial pipelines (fake models).

Monkeypatched load_model/load_tokenizer + deterministic CPU fake models +
fabricated upstream runs in temporary directories; no checkpoint is loaded.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_bias.balanced_evidence_gap.spans import resolve_row
from llm_bias.balanced_evidence_gap.template import ALL_TICKERS, SECTOR_OF
from llm_bias.core.artifact_paths import dataset_slug, model_slug
from llm_bias.entity_to_dial import pipeline

BUY_ID = 240
SELL_ID = 241
VOCAB = 9000
WIDTH = 1


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
        for candidate, token_id in (("buy", BUY_ID), ("sell", SELL_ID)):
            if text.endswith(candidate):
                suffix = (candidate, token_id)
                break
        if suffix is None:
            values = [ord(char) for char in text]
        else:
            prefix = text[: -len(suffix[0])]
            values = [ord(char) for char in prefix] + [suffix[1]]
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


def _identity_format(tokenizer, prompt, **_kwargs):
    return "«" + prompt + "»"


class _FakeRMSNorm(nn.Module):
    variance_epsilon = 1e-6

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(d_model))

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight


class _SumLayer(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 0.5 + hidden.sum(dim=1, keepdim=True) * 0.001


class _PatchingModel(nn.Module):
    """CPU-only decoder (Phase A fake): final margin follows the residual sum."""

    def __init__(self, n_layers: int = 16) -> None:
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


def _qwen_fake(num_layers: int = 16, seed: int = 0) -> object:
    """Small real Qwen3.5 fake model (Phase B/C fake)."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=[
            "full_attention" if (i % 4 == 3 or i == num_layers - 1) else "linear_attention"
            for i in range(num_layers)
        ],
        pad_token_id=0, eos_token_id=1,
    )
    raw = Qwen3_5ForCausalLM(config)
    raw.eval()

    class _Model:
        def __init__(self) -> None:
            self.layers = raw.model.layers
            self.n_layers = num_layers
            self.input_device = "cpu"
            self._final_norm = raw.model.norm
            self._lm_head = raw.lm_head

        def forward(self, input_ids, attention_mask=None):
            return raw(input_ids, attention_mask=attention_mask)

    return _Model()


def _fake_load_model(model: nn.Module, tokenizer: _CharTokenizer):
    def _load(model_path, *, dtype=None):
        return model, tokenizer, torch.device("cpu")

    return _load


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── fabricated upstream runs ─────────────────────────────────────────────────

TOP_GROUP = ("NSC", "BLK")
BOTTOM_GROUP = ("IT", "BDX")


def _base_margin(ticker: str, index: int) -> float:
    if ticker == "NSC":
        return 2.0
    if ticker == "BLK":
        return 1.8
    if ticker == "IT":
        return -2.0
    if ticker == "BDX":
        return -1.9
    return -0.5 + 0.1 * index


def _variant_margin(ticker: str, index: int, reverse: bool, order: int) -> float:
    return _base_margin(ticker, index) + 0.05 * reverse + 0.02 * order


def _frozen_directions() -> list[list[str]]:
    pairs = [[s, t] for s in TOP_GROUP for t in BOTTOM_GROUP]
    return pairs + [[t, s] for s, t in pairs]


def _fake_upstream(
    tmp_path: Path, tokenizer: _CharTokenizer, *, margins: dict[str, float] | None = None
) -> tuple[Path, Path, Path]:
    """Fabricate a complete 2A run + rev2 gate run + 2B run in temp dirs."""
    # 2A run: 64 prompt rows, 64 result rows, analyze summary, manifest.
    phase2a = tmp_path / "phase2a"
    prompt_rows, result_rows, medians = [], [], {}
    for index, ticker in enumerate(ALL_TICKERS):
        values = []
        for reverse in (False, True):
            for order in (0, 1):
                row = resolve_row(
                    tokenizer, ticker, f"Name {ticker}", SECTOR_OF[ticker],
                    reverse=reverse, order=order, format_fn=_identity_format,
                )
                prompt_rows.append(row)
                margin = (
                    margins[ticker] + 0.05 * reverse + 0.02 * order
                    if margins is not None
                    else _variant_margin(ticker, index, reverse, order)
                )
                values.append(margin)
                result_rows.append(
                    {
                        "id": row["id"],
                        "ticker": ticker,
                        "margin": margin,
                        "reverse": reverse,
                        "order": order,
                    }
                )
        medians[ticker] = statistics.median(values)
    _write_jsonl(phase2a / "prepare" / "prompts.jsonl", prompt_rows)
    _write_jsonl(phase2a / "forward" / "results.jsonl", result_rows)
    (phase2a / "analyze").mkdir(parents=True, exist_ok=True)
    (phase2a / "analyze" / "summary.json").write_text(
        json.dumps({"pure_entity_margin_median": medians}), encoding="utf-8"
    )
    (phase2a / "manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )

    # rev2 gate run: frozen groups + pass.
    rev2 = tmp_path / "rev2"
    (rev2 / "analyze").mkdir(parents=True, exist_ok=True)
    (rev2 / "analyze" / "summary.json").write_text(
        json.dumps({
            "gate_2a_rev2": {"pass": True, "phase2b_authorized": True},
            "descriptive": {
                "groups": {"top": list(TOP_GROUP), "bottom": list(BOTTOM_GROUP)}
            },
        }),
        encoding="utf-8",
    )
    (rev2 / "manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )

    # 2B run: frozen 8 directions + 1024 sweep records.
    phase2b = tmp_path / "phase2b"
    (phase2b / "pairs").mkdir(parents=True, exist_ok=True)
    (phase2b / "pairs" / "directions.json").write_text(
        json.dumps({"directions": _frozen_directions()}), encoding="utf-8"
    )
    sweep = []
    for source, target in _frozen_directions():
        for layer in range(32):
            for span in ("entity", "evidence", "instruction", "final"):
                sweep.append(
                    {
                        "direction": f"{source}->{target}",
                        "layer": layer,
                        "span": span,
                        "normalized_transfer": 0.1 + 0.001 * layer,
                        "toward_source_delta_m": 0.5,
                    }
                )
    _write_jsonl(phase2b / "sweep" / "records.jsonl", sweep)
    (phase2b / "manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    return phase2a, phase2b, rev2


def _patch_loaders(monkeypatch, model, tokenizer) -> None:
    monkeypatch.setattr(pipeline, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(pipeline, "load_model", _fake_load_model(model, tokenizer))


def _manifest(run_root: Path) -> dict:
    return json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))


def _common_kwargs(tmp_path: Path, upstream) -> dict:
    phase2a, phase2b, rev2 = upstream
    return dict(
        model_path="fake-model",
        phase2a_run=phase2a,
        phase2b_run=phase2b,
        phase2a_rev2_run=rev2,
        artifact_root=tmp_path / "artifacts",
    )


# ── Phase A smoke ────────────────────────────────────────────────────────────


def test_run_phase_a_smoke_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    upstream = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _PatchingModel(), tokenizer)

    run_root = pipeline.run_phase_a(
        run_id="smoke-fake-a", smoke=True, **_common_kwargs(tmp_path, upstream)
    )
    assert _manifest(run_root)["status"] == "complete"

    rows = _read_jsonl(run_root / "prepare" / "rows.jsonl")
    assert {r["ticker"] for r in rows} == {"NSC", "IT"}
    expected_margins = {"NSC": 2.035, "IT": -1.965}
    for row in rows:
        assert row["ticker_span"][1] > row["ticker_span"][0]
        assert row["name_span"][1] > row["name_span"][0]
        assert row["entity_span"][0] <= row["ticker_span"][0]
        assert row["name_span"][1] <= row["entity_span"][1]
        assert row["pure_entity_margin"] == pytest.approx(expected_margins[row["ticker"]])

    records = _read_jsonl(run_root / "forward" / "records.jsonl")
    # 2 directions x 4 layers x (ticker + name + self_noop)
    assert len(records) == 24
    noop = [r for r in records if r["token_group"] == "self_noop"]
    assert len(noop) == 8
    for row in noop:
        assert row["noop_delta_m"] == 0.0

    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_a"]["status"] == "not_evaluated"  # smoke grid
    assert set(summary["curves"]["ticker"]) == {"0", "3", "5", "9"}
    provenance = json.loads(
        (run_root / "prepare" / "provenance.json").read_text(encoding="utf-8")
    )
    # gap = min(2.035, 1.835) - max(-1.965, -1.865)
    assert provenance["group_gap"] == pytest.approx(3.7)
    assert provenance["phase2a_run"]["files"]["forward/results.jsonl"]["n_records"] == 64


# ── Phase B smoke ────────────────────────────────────────────────────────────


def test_run_phase_b_smoke_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    upstream = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)

    run_root = pipeline.run_phase_b(
        run_id="smoke-fake-b", smoke=True, **_common_kwargs(tmp_path, upstream)
    )
    assert _manifest(run_root)["status"] == "complete"

    records = _read_jsonl(run_root / "forward" / "records.jsonl")
    # 2 directions x 2 layers x (mlp + attn + 2 no-ops)
    assert len(records) == 16
    noop = [r for r in records if r["component"] == "self_noop"]
    assert len(noop) == 8
    for row in noop:
        assert row["noop_delta_m"] == 0.0
    assert {r["layer"] for r in records} == {12, 15}
    assert {r["noop_for"] for r in noop} == {"mlp", "attn"}

    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_b"]["status"] == "not_evaluated"  # smoke grid
    assert set(summary["curves"]["mlp"]) == {"12", "15"}


# ── Phase C smoke ────────────────────────────────────────────────────────────


def test_run_phase_c_smoke_with_fake_model(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    phase2a, _phase2b, _rev2 = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _qwen_fake(), tokenizer)

    # The fake model has 64 intermediate channels; the frozen dial neuron
    # (8490) does not exist, so pin the probe to an in-range coordinate.
    monkeypatch.setattr(pipeline, "DIAL_NEURON", 5)
    real_probe = pipeline.probe_forward

    def _probe(model, tensor, **kwargs):
        kwargs.setdefault("dial_layer", 15)
        kwargs.setdefault("dial_neuron", 5)
        return real_probe(model, tensor, **kwargs)

    monkeypatch.setattr(pipeline, "probe_forward", _probe)

    run_root = pipeline.run_phase_c(
        model_path="fake-model",
        run_id="smoke-fake-c",
        phase2a_run=phase2a,
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )
    assert _manifest(run_root)["status"] == "complete"

    rows = _read_jsonl(run_root / "prepare" / "rows.jsonl")
    named = [r for r in rows if r["prompt_type"] == "named"]
    anon = [r for r in rows if r["prompt_type"] == "anon"]
    assert {r["ticker"] for r in named} == {"NSC", "IT"}
    assert {r["ticker"] for r in anon} == {"NSC", "IT"}
    for row in anon:
        assert "Stock Ticker: [TICKER]" in row["prompt"]
        assert "Stock Name: [Company X]" in row["prompt"]
        assert row["prompt"].index("Stock Ticker:") == row["prompt"].index("Stock Ticker: [TICKER]")

    results = _read_jsonl(run_root / "forward" / "results.jsonl")
    clean = [r for r in results if "prompt_type" in r]
    push = [r for r in results if "pushed_margin" in r]
    assert len(clean) == 4 and len(push) == 2
    for row in clean:
        assert torch.isfinite(torch.tensor(row["dial_entity"]))
        assert torch.isfinite(torch.tensor(row["dial_final"]))
    for row in push:
        assert torch.isfinite(torch.tensor(row["delta_m_dial"]))

    summary = json.loads((run_root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_companies"] == 2
    assert summary["gate_c"]["status"] == "not_evaluated"  # smoke grid
    assert summary["c1_descriptive"]["descriptive_only"] is True
    assert summary["c3_unexplained_gap"]["mean"] is not None
    for ticker in ("NSC", "IT"):
        entry = summary["per_company"][ticker]
        assert torch.isfinite(
            torch.tensor(entry["gap"], dtype=torch.float64)
        )
        assert entry["unexplained_gap"] == pytest.approx(
            entry["gap"] - entry["delta_m_dial"]
        )


# ── no-op enforcement and upstream fail-closed ───────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase_a_noop_enforcement_fails_closed(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    upstream = _fake_upstream(tmp_path, tokenizer)
    _patch_loaders(monkeypatch, _PatchingModel(), tokenizer)
    monkeypatch.setattr(pipeline, "NOOP_TOLERANCE", -1.0)  # force violation

    with pytest.raises(ValueError, match="no-op violated"):
        pipeline.run_phase_a(
            run_id="smoke-fake-a-bad", smoke=True, **_common_kwargs(tmp_path, upstream)
        )
    manifest_path = (
        tmp_path / "artifacts" / model_slug("fake-model") / dataset_slug(pipeline.DATASET)
        / "runs" / "smoke-fake-a-bad" / "manifest.json"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_upstream_verification_fails_closed(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    _patch_loaders(monkeypatch, _PatchingModel(), tokenizer)

    # (a) record count mismatch: one prompt row deleted.
    phase2a, phase2b, rev2 = _fake_upstream(tmp_path / "u1", tokenizer)
    prompts = phase2a / "prepare" / "prompts.jsonl"
    lines = prompts.read_text(encoding="utf-8").splitlines()
    prompts.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record count mismatch"):
        pipeline.run_phase_a(
            run_id="bad-count", smoke=True,
            **_common_kwargs(tmp_path / "a1", (phase2a, phase2b, rev2)),
        )

    # (b) tampered analyze summary: stored margin no longer matches.
    phase2a, phase2b, rev2 = _fake_upstream(tmp_path / "u2", tokenizer)
    summary_path = phase2a / "analyze" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pure_entity_margin_median"]["NSC"] += 0.5
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="pure entity margin mismatch"):
        pipeline.run_phase_a(
            run_id="bad-summary", smoke=True,
            **_common_kwargs(tmp_path / "a2", (phase2a, phase2b, rev2)),
        )

    # (c) group gap below threshold (margins rewritten consistently).
    margins = {
        "NSC": 0.10, "BLK": 0.20, "IT": 0.25, "BDX": 0.30,
        **{t: 0.0 for t in ALL_TICKERS if t not in ("NSC", "BLK", "IT", "BDX")},
    }
    phase2a, phase2b, rev2 = _fake_upstream(
        tmp_path / "u3", tokenizer, margins=margins
    )
    with pytest.raises(ValueError, match="pre-check 1 failed"):
        pipeline.run_phase_a(
            run_id="bad-gap", smoke=True,
            **_common_kwargs(tmp_path / "a3", (phase2a, phase2b, rev2)),
        )

    # (d) upstream manifest not complete.
    phase2a, phase2b, rev2 = _fake_upstream(tmp_path / "u4", tokenizer)
    (phase2a / "manifest.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 'complete'"):
        pipeline.run_phase_a(
            run_id="bad-manifest", smoke=True,
            **_common_kwargs(tmp_path / "a4", (phase2a, phase2b, rev2)),
        )

    # (e) directions do not match the frozen top/bottom pairs.
    phase2a, phase2b, rev2 = _fake_upstream(tmp_path / "u5", tokenizer)
    directions = _frozen_directions()[:-1]
    (phase2b / "pairs" / "directions.json").write_text(
        json.dumps({"directions": directions}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen top/bottom pairs"):
        pipeline.run_phase_a(
            run_id="bad-directions", smoke=True,
            **_common_kwargs(tmp_path / "a5", (phase2a, phase2b, rev2)),
        )


def test_package_does_not_import_other_experiment_packages():
    import ast

    root = Path(__file__).resolve().parents[1] / "llm_bias" / "entity_to_dial"
    forbidden_prefixes = (
        "llm_bias.entity_cell",
        "llm_bias.jspace_intervention",
        "llm_bias.investment_dial",
        "llm_bias.baseline_trial",
        "llm_bias.span_sensitivity",
        "llm_bias.prompt_analysis",
        "llm_bias.financial_soundness",
        "llm_bias.sector_context",
        "llm_bias.balanced_evidence_gap",
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
                assert not any(
                    name == p or name.startswith(p + ".") for p in forbidden_prefixes
                ), f"{path.name} imports {name}"
