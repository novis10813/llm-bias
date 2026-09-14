"""End-to-end fake-model tests for the selective-intervention V1 pipeline.

Monkeypatched load_model/load_tokenizer + deterministic CPU fake model +
fabricated upstream runs (2A with real fake-model margins for the
bit-exact reference; e-01 with a valid fake basis). No checkpoint.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from llm_bias.core.artifact_paths import dataset_slug, model_slug
from llm_bias.core.artifacts.io import _RAW
from llm_bias.selective_intervention import pipeline as sip
from llm_bias.selective_intervention import template as sit
from llm_bias.selective_intervention.scoring import answer_token_ids, margin_forward, margin_from_log_probs, scoring_ids

from test_entity_to_dial_pipeline import (
    _CharTokenizer,
    _fake_upstream,
    _manifest,
    _qwen_fake,
    _read_jsonl,
    _write_jsonl,
)

FAKE_DIAL = 5  # in-range fake dial channel (intermediate 64)


# ── margin semantics regression ───────────────────────────────────────────────


def test_margin_from_log_probs_uses_float64_subtraction():
    """The margin must match the 2A reference (score_single_token_margin_fp32):
    convert-to-float64-then-subtract, not a float32 tensor subtraction.
    """
    log_probs = torch.tensor([[0.3, 0.0, 0.1]], dtype=torch.float32)
    import numpy as np

    a32 = np.float32(0.3)
    b32 = np.float32(0.1)
    fp32_sub = np.float32(a32 - b32)  # the wrong (float32) semantics
    fp64_sub = a32.astype(np.float64) - b32.astype(np.float64)  # the reference
    assert fp32_sub != fp64_sub  # discriminating case
    assert margin_from_log_probs(log_probs, 0, 2) == float(fp64_sub)


# ── fabrication ───────────────────────────────────────────────────────────────


def _fake_e01(tmp_path: Path) -> Path:
    """Fabricate an e-01 run: valid 16 x 32 basis + complete manifest."""
    e01 = tmp_path / "e01"
    (e01 / "analyze").mkdir(parents=True)
    basis = torch.eye(32, dtype=torch.float64)[:16]
    singular = torch.linspace(16.0, 1.0, 16)
    (e01 / "analyze" / "summary.json").write_text(
        json.dumps(
            {
                "pca_basis_vectors": basis.tolist(),
                "pca_singular_values": singular.tolist(),
            }
        ),
        encoding="utf-8",
    )
    (e01 / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return e01


def _real_margin(model, tokenizer, row: dict) -> float:
    """Real fake-model margin for one stored row (bit-exact reference)."""
    tensor = torch.tensor([scoring_ids(tokenizer, row["formatted"])], dtype=torch.long)
    buy_id, sell_id = answer_token_ids(tokenizer, row["formatted"] + sit.DECISION_PREFIX)
    return margin_forward(model, tensor, None, buy_id, sell_id)


def _fake_upstream_with_real_margins(tmp_path: Path, model, tokenizer) -> Path:
    """2A run whose archived margins are real fake-model forwards."""
    phase2a = _fake_upstream(tmp_path, tokenizer)[0]
    rows = _read_jsonl(phase2a / "prepare" / "prompts.jsonl")
    results = _read_jsonl(phase2a / "forward" / "results.jsonl")
    assert len(rows) == len(results) == 64
    for row, result in zip(rows, results):
        result["margin"] = _real_margin(model, tokenizer, row)
    _write_jsonl(phase2a / "forward" / "results.jsonl", results)
    return phase2a


def _patch(monkeypatch, model, tokenizer) -> None:
    monkeypatch.setattr(sip, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(sip, "load_model", lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")))
    monkeypatch.setattr(sit, "DIAL_NEURON", FAKE_DIAL)
    monkeypatch.setattr(sit, "SMOKE_CONTROL_LAYERS", (14,))  # fake model has 16 layers


def _run_smoke(tmp_path: Path, monkeypatch, model, tokenizer, phase2a: Path, e01: Path) -> Path:
    _patch(monkeypatch, model, tokenizer)
    return sip.run_selective_intervention_v1(
        model_path="fake-model",
        run_id="si-v1-fake-smoke",
        phase2a_run=phase2a,
        e01_run=e01,
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )


# ── smoke pipeline ────────────────────────────────────────────────────────────


def _assert_no_reserved_keys(obj) -> None:
    """Core serializer reserved-token guard (recursive; fail-closed)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = str(key).lower()
            assert not (set(normalized.split("_")) & _RAW) and normalized not in _RAW, (
                f"reserved key in serialized output: {key!r}"
            )
            _assert_no_reserved_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            _assert_no_reserved_keys(value)


def test_smoke_pipeline_complete(tmp_path, monkeypatch):
    model = _qwen_fake()
    tokenizer = _CharTokenizer()
    phase2a = _fake_upstream_with_real_margins(tmp_path, model, tokenizer)
    e01 = _fake_e01(tmp_path)
    root = _run_smoke(tmp_path, monkeypatch, model, tokenizer, phase2a, e01)

    manifest = _manifest(root)
    assert manifest["status"] == "complete"
    stages = {name: info["status"] for name, info in manifest["stages"].items()}
    assert stages == {"prepare": "complete", "forward": "complete", "analyze": "complete"}

    records = _read_jsonl(root / "forward" / "records.jsonl")
    arms = Counter(r["arm"] for r in records)
    # 4 group companies x 4 variants = 16 prompts.
    expected = {
        "clean": 16,
        "dose_50": 16,
        "dose_100": 16,
        "center_zero": 16,
        "center_anon": 16,
        "scope_fullseq": 16,
        "ctrl_random": 16,
        "ctrl_layer_14": 4,
        "anon_clean": 1,
        "anon_int": 1,
        "dial_+4_clean": 1,
        "dial_+4_int": 1,
        "dial_-4_clean": 1,
        "dial_-4_int": 1,
    }
    assert dict(arms) == expected

    for r in records:
        assert torch.isfinite(torch.tensor(r["margin"]))
        assert r["decision"] in ("buy", "sell")

    forward_meta = json.loads((root / "forward" / "metadata.json").read_text(encoding="utf-8"))
    # 122 arm forwards + 5 calibration forwards (4 named + 1 anonymous).
    assert forward_meta["n_forwards"] == 122 + 5
    assert forward_meta["n_calib_forwards"] == 5
    assert forward_meta["n_clean_bit_exact"] == 16
    assert forward_meta["clean_max_abs_delta_m"] == 0.0
    for digest in (
        *forward_meta["centers"].values(),
        forward_meta["mu_anon_sha256"],
        forward_meta["mu_full_ref_sha256"],
    ):
        assert len(digest) == 64

    summary = json.loads((root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["smoke"] is True
    assert summary["gate_v1"]["status"] == "not_evaluated"
    assert summary["clean_stats"]["n_prompts"] == 16
    assert len(summary["dose_response"]) == 2
    assert len(summary["per_company"]) == 4
    assert [p["layer"] for p in summary["layer_control"]] == [14, 15]
    assert set(summary["dial_probe"]) == {"+4", "-4"}
    assert set(summary["random_control"]) == {"group_gap", "mean_shift"}
    assert summary["anon_probe"]["clean"] == pytest.approx(forward_meta["m_anon_clean"])

    _assert_no_reserved_keys(summary)
    _assert_no_reserved_keys(records)
    _assert_no_reserved_keys(json.loads((root / "prepare" / "provenance.json").read_text(encoding="utf-8")))


def test_smoke_clean_arm_bit_exact_archive(tmp_path, monkeypatch):
    model = _qwen_fake()
    tokenizer = _CharTokenizer()
    phase2a = _fake_upstream_with_real_margins(tmp_path, model, tokenizer)
    e01 = _fake_e01(tmp_path)
    root = _run_smoke(tmp_path, monkeypatch, model, tokenizer, phase2a, e01)
    archive = {r["id"]: r["margin"] for r in _read_jsonl(phase2a / "forward" / "results.jsonl")}
    records = _read_jsonl(root / "forward" / "records.jsonl")
    clean = [r for r in records if r["arm"] == "clean"]
    assert len(clean) == 16
    for r in clean:
        assert r["margin"] == archive[r["prompt_id"]]  # bit-exact


# ── fail-closed paths ─────────────────────────────────────────────────────────


def test_corrupted_archive_margin_fails_closed(tmp_path, monkeypatch):
    model = _qwen_fake()
    tokenizer = _CharTokenizer()
    phase2a = _fake_upstream_with_real_margins(tmp_path, model, tokenizer)
    results = _read_jsonl(phase2a / "forward" / "results.jsonl")
    target = next(
        r for r in results if r["ticker"] == "NSC" and r["reverse"] is False and r["order"] == 0
    )
    target["margin"] += 0.1
    _write_jsonl(phase2a / "forward" / "results.jsonl", results)
    e01 = _fake_e01(tmp_path)
    _patch(monkeypatch, model, tokenizer)
    with pytest.raises(ValueError, match="bit-exact"):
        sip.run_selective_intervention_v1(
            model_path="fake-model",
            run_id="si-v1-fake-corrupt-archive",
            phase2a_run=phase2a,
            e01_run=e01,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )
    manifest = _manifest(
        tmp_path / "artifacts" / model_slug("fake-model") / dataset_slug(sit.DATASET)
        / "runs" / "si-v1-fake-corrupt-archive"
    )
    assert manifest["status"] == "failed"


def test_non_orthonormal_basis_fails_closed(tmp_path, monkeypatch):
    model = _qwen_fake()
    tokenizer = _CharTokenizer()
    phase2a = _fake_upstream_with_real_margins(tmp_path, model, tokenizer)
    e01 = _fake_e01(tmp_path)
    summary = json.loads((e01 / "analyze" / "summary.json").read_text(encoding="utf-8"))
    summary["pca_basis_vectors"][0] = [x + 0.25 for x in summary["pca_basis_vectors"][0]]
    (e01 / "analyze" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _patch(monkeypatch, model, tokenizer)
    with pytest.raises(ValueError, match="orthonormal"):
        sip.run_selective_intervention_v1(
            model_path="fake-model",
            run_id="si-v1-fake-bad-basis",
            phase2a_run=phase2a,
            e01_run=e01,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )


def test_missing_e01_manifest_fails_closed(tmp_path, monkeypatch):
    model = _qwen_fake()
    tokenizer = _CharTokenizer()
    phase2a = _fake_upstream_with_real_margins(tmp_path, model, tokenizer)
    e01 = _fake_e01(tmp_path)
    (e01 / "manifest.json").unlink()
    _patch(monkeypatch, model, tokenizer)
    with pytest.raises(FileNotFoundError):
        sip.run_selective_intervention_v1(
            model_path="fake-model",
            run_id="si-v1-fake-no-manifest",
            phase2a_run=phase2a,
            e01_run=e01,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )


def test_short_2a_population_fails_closed(tmp_path, monkeypatch):
    model = _qwen_fake()
    tokenizer = _CharTokenizer()
    phase2a = _fake_upstream_with_real_margins(tmp_path, model, tokenizer)
    rows = _read_jsonl(phase2a / "prepare" / "prompts.jsonl")
    _write_jsonl(phase2a / "prepare" / "prompts.jsonl", rows[:-1])
    results = _read_jsonl(phase2a / "forward" / "results.jsonl")
    _write_jsonl(phase2a / "forward" / "results.jsonl", results[:-1])
    e01 = _fake_e01(tmp_path)
    _patch(monkeypatch, model, tokenizer)
    with pytest.raises(ValueError, match="64 prompts"):
        sip.run_selective_intervention_v1(
            model_path="fake-model",
            run_id="si-v1-fake-short-pop",
            phase2a_run=phase2a,
            e01_run=e01,
            artifact_root=tmp_path / "artifacts",
            smoke=True,
        )
