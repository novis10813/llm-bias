from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_bias.entity_concept_decision.development import (
    _analyze,
    _bundle_for_rows,
    _fit_all,
    run_development,
)
from llm_bias.entity_concept_decision.development_materials import build_development_materials

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "docs/entity-concept-decision/details/materials-phase1-v2-draft.md"
REVIEW = ROOT / "docs/entity-concept-decision/details/review-materials-phase1-v2.md"


class CharacterTokenizer:
    chat_template = "test"

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, return_special_tokens_mask=False):
        ids = []
        offsets = []
        index = 0
        while index < len(text):
            candidate = next((word for word in ("sell", "buy") if text.startswith(word, index)), None)
            if candidate is not None:
                ids.append(249 if candidate == "buy" else 250)
                offsets.append((index, index + len(candidate)))
                index += len(candidate)
            else:
                ids.append(ord(text[index]) % 240)
                offsets.append((index, index + 1))
                index += 1
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        if return_special_tokens_mask:
            result["special_tokens_mask"] = [False] * len(ids)
        return SimpleNamespace(**result)

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class Block(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, hidden):
        hidden = hidden + (self.index + 1) * 0.01
        if self.index >= 16:
            hidden = hidden + 0.0001 * hidden.square()
        return hidden


class FakeModel(nn.Module):
    def __init__(self, width=4, vocab=251):
        super().__init__()
        self.n_layers = 17
        self.layers = nn.ModuleList(Block(index) for index in range(self.n_layers))
        self.embedding = nn.Embedding(vocab, width)
        self._final_norm = nn.Identity()
        self._lm_head = nn.Linear(width, vocab, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(vocab * width, dtype=torch.float32).reshape(vocab, width) / 100.0)
            self._lm_head.weight.copy_(torch.arange(vocab * width, dtype=torch.float32).reshape(vocab, width) / 100.0)

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self._lm_head(hidden))


def _materials():
    return build_development_materials(SOURCE, REVIEW)


def _provenance():
    return {
        "mode": "fake_smoke",
        "upstream": {"path": str(SOURCE)},
        "lens": {"path": str(REVIEW)},
        "review_status": "ai_reviewed_development",
    }


def test_fake_model_runs_real_recording_and_writes_compact_three_stage_artifacts(tmp_path):
    output = run_development(
        model=FakeModel(), tokenizer=CharacterTokenizer(), device="cpu", basis=torch.eye(4),
        materials=_materials(), instruction_suffix="DECIDE", answer_prefix="Answer: ",
        company_prompts=[], run_id="fake-run", model_name="fake-model", artifact_root=tmp_path,
        provenance=_provenance(), random_count=4,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert set(manifest["stages"]) == {"prepare", "forward", "analyze"}
    assert all(value["status"] == "complete" for value in manifest["stages"].values())
    records = (output / "forward/records.jsonl").read_text()
    assert "tensor" not in records and "residual" not in records and "activation" not in records
    summary = json.loads((output / "analyze/summary.json").read_text())
    assert summary["scientific_status"] == "not_evaluated"
    assert summary["purpose"] == "development"
    assert summary["n_independent_families"] is None
    assert summary["random_controls"]["C"]["random_count"] == 4
    assert summary["random_controls"]["comparison_count"] == 38
    assert len(summary["random_controls"]["comparisons"]) == 38
    assert sum(item["comparison_count"] for item in summary["random_controls"]["by_kind"].values()) == 38
    assert all(item["n_deltas"] == 4 for item in summary["random_controls"]["comparisons"])


def _synthetic_vectors(materials, *, primary_deltas=None, control_value=0.0):
    primary_deltas = primary_deltas or {}
    vectors = {row["id"]: torch.full((4,), control_value) for row in materials["rows"]}
    for row in materials["rows"]:
        if row["role"] != "primary" or row["pole"] != "P":
            continue
        delta = primary_deltas.get(row["family_id"], torch.zeros(4))
        vectors[row["id"]] = torch.as_tensor(delta, dtype=torch.float32)
        negative_id = row["id"].rsplit("-", 1)[0] + "-N"
        vectors[negative_id] = torch.zeros(4)
    return vectors


def test_development_fit_weights_groups_equally_but_keeps_original_family_pair_ids():
    materials = _materials()
    deltas = {
        family: torch.tensor([10.0, 0.0, 0.0, 0.0])
        for family in ("C-F01", "C-F04", "C-F05", "C-V01", "C-V03")
    }
    deltas.update({family: torch.tensor([0.0, 2.0, 0.0, 0.0]) for family in ("C-F02", "C-F03", "C-F06")})
    vectors = _synthetic_vectors(materials, primary_deltas=deltas)
    bundle = _bundle_for_rows(materials["rows"], source_sha256="test")
    fits, _ = _fit_all(bundle, materials["rows"], vectors, torch.eye(4)[:, :2], 1e-6)

    fit = fits["C"]
    expected = torch.tensor([5.0, 1.0, 0.0, 0.0])
    assert torch.allclose(fit.direction, expected / torch.linalg.vector_norm(expected), atol=1e-6, rtol=0.0)
    assert fit.n_families == 2
    assert fit.n_pairs == 8
    assert {pair.id for pair in bundle.pairs if pair.concept_id == "C"} == {
        "C-F01", "C-F02", "C-F03", "C-F04", "C-F05", "C-F06", "C-V01", "C-V03"
    }
    assert {pair.family_id for pair in bundle.pairs if pair.concept_id == "C"} == {"C1", "C2"}


def test_leave_one_group_out_excludes_group_and_deduplicates_pair_deltas():
    materials = _materials()
    vectors = _synthetic_vectors(materials, primary_deltas={family: torch.tensor([1.0, 0.0, 0.0, 0.0]) for family in ("C-F01", "C-F02", "C-F03", "C-F04", "C-F05", "C-F06", "C-V01", "C-V03")})
    bundle = _bundle_for_rows(materials["rows"], source_sha256="test")
    _, loo = _fit_all(bundle, materials["rows"], vectors, torch.eye(4)[:, :2], 1e-6)
    assert loo["C"]["C1"].n_families == 1
    assert loo["C"]["C1"].n_pairs == 3
    assert loo["C"]["C2"].n_families == 1
    assert loo["C"]["C2"].n_pairs == 5

    margins = {row["id"]: 0.0 for row in materials["rows"]}
    output, summary = _analyze(materials["rows"], materials["comparisons"], [], vectors, margins, torch.eye(4)[:, :2], bundle, 1e-6, 1729, 0)
    assert len(output) == 38
    for item in summary["leave_one_group_out"]:
        families = [entry["family_id"] for entry in item["pair_deltas"]]
        assert len(families) == len(set(families))
        assert all(next(row for row in materials["rows"] if row["family_id"] == family and row["role"] == "primary")["group_id"] == item["excluded_group"] for family in families)


def test_controls_do_not_participate_in_direction_fit():
    materials = _materials()
    vectors = _synthetic_vectors(materials, primary_deltas={family: torch.tensor([1.0, 0.0, 0.0, 0.0]) for family in ("C-F01", "C-F02", "C-F03", "C-F04", "C-F05", "C-F06", "C-V01", "C-V03")})
    altered = dict(vectors)
    for row in materials["rows"]:
        if row["role"] != "primary":
            altered[row["id"]] = torch.tensor([0.0, 1000.0, 0.0, 0.0])
    bundle = _bundle_for_rows(materials["rows"], source_sha256="test")
    fit, _ = _fit_all(bundle, materials["rows"], vectors, torch.eye(4)[:, :2], 1e-6)
    altered_fit, _ = _fit_all(bundle, materials["rows"], altered, torch.eye(4)[:, :2], 1e-6)
    assert torch.equal(fit["C"].direction, altered_fit["C"].direction)


def test_all_degenerate_development_still_finalizes(tmp_path):
    model = FakeModel()
    with torch.no_grad():
        model.embedding.weight.zero_()
    output = run_development(
        model=model, tokenizer=CharacterTokenizer(), device="cpu", basis=torch.eye(4),
        materials=_materials(), instruction_suffix="DECIDE", answer_prefix="Answer: ",
        company_prompts=[], run_id="degenerate", model_name="fake-model", artifact_root=tmp_path,
        provenance=_provenance(), random_count=0,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "analyze/summary.json").read_text())
    assert manifest["status"] == "complete"
    assert all(fit["status"] == "degenerate" for fit in summary["fits"].values())
    assert all(fit["reason"] == "source_norm_below_minimum" for fit in summary["fits"].values())


def test_random_controls_are_reproducible_and_run_id_cannot_overwrite(tmp_path):
    kwargs = dict(
        model=FakeModel(), tokenizer=CharacterTokenizer(), device="cpu", basis=torch.eye(4),
        materials=_materials(), instruction_suffix="DECIDE", answer_prefix="Answer: ",
        company_prompts=[], model_name="fake-model", artifact_root=tmp_path,
        provenance=_provenance(), random_count=3, random_seed=123,
    )
    first = run_development(run_id="same", **kwargs)
    first_summary = (first / "analyze/summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_development(run_id="same", **kwargs)
    other = run_development(run_id="other", **kwargs)
    assert (other / "analyze/summary.json").read_bytes() == first_summary


def test_sentiment_residualization_removes_evaluation_keeps_concept():
    materials = _materials()
    c = torch.tensor([1.0, 0.0, 0.0, 0.0])   # pure concept axis
    s = torch.tensor([0.0, 1.0, 0.0, 0.0])   # pure stance/evaluation axis
    basis = torch.eye(4)[:, :2]
    vectors: dict[str, torch.Tensor] = {row["id"]: torch.zeros(4) for row in materials["rows"]}
    # C primary pairs carry a concept+stance mixture so the fitted C direction is mixed.
    for row in materials["rows"]:
        if row["concept_id"] == "C" and row["role"] == "primary" and row["pole"] == "P":
            vectors[row["id"]] = c + s
    # C evaluation families encode a clean 2x2: concept along e1, evaluation along e2.
    cells = {"PH": c + s, "PL": c, "NH": s, "NL": torch.zeros(4)}
    for row in materials["rows"]:
        if row["concept_id"] == "C" and row["role"] == "evaluation":
            vectors[row["id"]] = cells[row["pole"]].clone()
    bundle = _bundle_for_rows(materials["rows"], source_sha256="test")
    margins = {row["id"]: 0.0 for row in materials["rows"]}
    _, summary = _analyze(materials["rows"], materials["comparisons"], [], vectors, margins, basis, bundle, 1e-6, 1729, 0)

    sr = summary["sentiment_residualization"]
    assert sr["status"] == "ok"
    assert sr["stance_n_pairs"] == 8  # 4 C evaluation pairs + 4 S (zero) pairs
    # S primary is all-zero, so S fit is degenerate and reported as such.
    assert sr["concepts"]["S"]["status"] == "degenerate"

    c_entry = sr["concepts"]["C"]
    assert c_entry["status"] == "ok" and c_entry["orthogonalized_status"] == "ok"
    # Fitted C direction is normalize([1,1,0,0]); its cosine with stance [0,1,0,0] is 1/sqrt(2).
    assert c_entry["cosine_with_stance"] == pytest.approx(1.0 / 2.0 ** 0.5, abs=1e-6)
    resp = c_entry["response"]
    # Before: concept and evaluation responses are equal (1/sqrt(2)). After: evaluation removed (0), concept preserved (1).
    assert resp["concept_mean_abs_before"] == pytest.approx(1.0 / 2.0 ** 0.5, abs=1e-6)
    assert resp["evaluation_mean_abs_before"] == pytest.approx(1.0 / 2.0 ** 0.5, abs=1e-6)
    assert resp["concept_mean_abs_after"] == pytest.approx(1.0, abs=1e-6)
    assert resp["evaluation_mean_abs_after"] == pytest.approx(0.0, abs=1e-6)


def test_sentiment_residualization_degenerate_when_no_evaluation_rows():
    materials = _materials()
    basis = torch.eye(4)[:, :2]
    rows = [dict(row, role=("primary" if row["role"] == "primary" else "lexical")) for row in materials["rows"]]
    # Force every evaluation row out so no PH/PL/NH/NL pairs remain.
    from llm_bias.entity_concept_decision import development as dev
    result = dev._sentiment_residualization(rows, {row["id"]: torch.zeros(4) for row in rows}, basis, {}, [], [], 1e-6)
    assert result["status"] == "no_evaluation_rows"


def test_invalid_input_is_rejected_before_run_creation(tmp_path):
    with pytest.raises(ValueError, match="orthonormal"):
        run_development(
            model=FakeModel(), tokenizer=CharacterTokenizer(), device="cpu", basis=torch.ones(4, 1),
            materials=_materials(), instruction_suffix="DECIDE", answer_prefix="Answer: ",
            company_prompts=[], run_id="bad", model_name="fake-model", artifact_root=tmp_path,
            provenance=_provenance(),
        )
    assert not list(tmp_path.rglob("manifest.json"))


def test_failed_forward_leaves_failed_manifest_and_hooks_are_cleaned(tmp_path):
    model = FakeModel()
    original = model.layers[15].forward

    def fail(hidden):
        raise RuntimeError("synthetic forward failure")

    model.layers[15].forward = fail
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        run_development(
            model=model, tokenizer=CharacterTokenizer(), device="cpu", basis=torch.eye(4),
            materials=_materials(), instruction_suffix="DECIDE", answer_prefix="Answer: ",
            company_prompts=[], run_id="failed", model_name="fake-model", artifact_root=tmp_path,
            provenance=_provenance(), random_count=0,
        )
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    assert manifest["status"] == "failed"
    assert model.layers[15]._forward_hooks == {}
    model.layers[15].forward = original
