"""M6 fake-model pipeline smoke coverage."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_bias.selective_intervention import m6_pipeline as m6
from llm_bias.selective_intervention.m6_analysis import M6_SECTORS
from llm_bias.selective_intervention.spans import _anonymous_row
from llm_bias.selective_intervention.subspace import calibration_centers, tensor_sha256

from test_entity_to_dial_pipeline import (
    _CharTokenizer,
    _fake_upstream,
    _qwen_fake,
    _read_jsonl,
)


def _write_manifest(path: Path) -> None:
    companies = []
    for sector_index, sector in enumerate(M6_SECTORS):
        for company_index in range(3):
            ticker = f"M{sector_index}{company_index}"
            companies.append({"ticker": ticker, "name": f"M6 {ticker}", "sector": sector})
    path.write_text(
        json.dumps(
            {
                "schema_version": "selective-intervention-m6-external-v1",
                "selection_seed": 42,
                "eligible_pool": companies,
                "exclusion_list": ["NSC", "BLK", "IT", "BDX"],
                "companies": companies,
            }
        ),
        encoding="utf-8",
    )


def _fake_e01(tmp_path: Path) -> Path:
    e01 = tmp_path / "e01"
    (e01 / "analyze").mkdir(parents=True)
    basis = torch.eye(32, dtype=torch.float64)[:16]
    singular = torch.linspace(16.0, 1.0, 16)
    (e01 / "analyze" / "summary.json").write_text(
        json.dumps({"pca_basis_vectors": basis.tolist(), "pca_singular_values": singular.tolist()}),
        encoding="utf-8",
    )
    (e01 / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return e01


def _fake_v1(tmp_path: Path, phase2a: Path, model, tokenizer) -> Path:
    rows = _read_jsonl(phase2a / "prepare" / "prompts.jsonl")
    source_rows = [row for row in rows if not row["reverse"] and row["order"] == 0]
    ref = source_rows[0]
    anon = _anonymous_row(tokenizer, ref)
    ref_len = len(m6.scoring_ids(tokenizer, ref["formatted"]))
    calibration = calibration_centers(
        model,
        tokenizer,
        source_rows,
        ref,
        anon,
        layers=[15],
        anon_layer=15,
        ref_seq_len=ref_len,
        device="cpu",
    )
    v1 = tmp_path / "v1"
    (v1 / "forward").mkdir(parents=True)
    (v1 / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (v1 / "forward" / "metadata.json").write_text(
        json.dumps({"centers": {"15": tensor_sha256(calibration["mu_bar"][15])}}),
        encoding="utf-8",
    )
    return v1


def test_m6_fake_pipeline_reconstructs_frozen_center_and_writes_summary(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    model = _qwen_fake()
    phase2a, _, _ = _fake_upstream(tmp_path, tokenizer)
    e01 = _fake_e01(tmp_path)
    v1 = _fake_v1(tmp_path, phase2a, model, tokenizer)
    external = tmp_path / "external.json"
    _write_manifest(external)

    monkeypatch.setattr(m6, "load_tokenizer_for_inference", lambda _path: tokenizer)
    monkeypatch.setattr(m6, "load_model", lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")))
    monkeypatch.setattr(m6, "_generate_decision", lambda *args, **kwargs: ("sell", True, "parsed"))

    root = m6.run_selective_intervention_m6(
        model_path="fake-model",
        run_id="m6-fake-smoke",
        external_manifest=external,
        phase2a_run=phase2a,
        e01_run=e01,
        v1_run=v1,
        artifact_root=tmp_path / "artifacts",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    records = _read_jsonl(root / "forward" / "records.jsonl")
    assert len([row for row in records if row["arm"] == "clean"]) == 48
    assert len([row for row in records if row["arm"] == "dose_100"]) == 48
    assert len([row for row in records if row["arm"] == "ctrl_random"]) == 48
    summary = json.loads((root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary"]["spread"]["bootstrap"]["samples"] == 10_000
    assert summary["generation"]["main"]["n_prompts"] == 48
    assert summary["metadata"]["center_digest_l15"]


def test_m6_real_model_smoke_path_is_not_formal_analysis(tmp_path, monkeypatch):
    tokenizer = _CharTokenizer()
    model = _qwen_fake()
    phase2a, _, _ = _fake_upstream(tmp_path, tokenizer)
    e01 = _fake_e01(tmp_path)
    v1 = _fake_v1(tmp_path, phase2a, model, tokenizer)
    external = tmp_path / "external.json"
    _write_manifest(external)

    monkeypatch.setattr(m6, "load_tokenizer_for_inference", lambda _path: tokenizer)
    monkeypatch.setattr(m6, "load_model", lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")))
    monkeypatch.setattr(m6, "_generate_decision", lambda *args, **kwargs: ("sell", True, "parsed"))

    root = m6.run_selective_intervention_m6_smoke(
        model_path="fake-model",
        run_id="m6-real-smoke-fake",
        external_manifest=external,
        phase2a_run=phase2a,
        e01_run=e01,
        v1_run=v1,
        artifact_root=tmp_path / "artifacts",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert (root / "forward" / "smoke.json").is_file()
    assert not (root / "analyze" / "summary.json").exists()
    smoke = json.loads((root / "forward" / "smoke.json").read_text(encoding="utf-8"))
    assert smoke["center_digest_l15"] == smoke["expected_center_digest_l15"]
    assert set(smoke["margins"]) == {"clean", "main", "random"}
