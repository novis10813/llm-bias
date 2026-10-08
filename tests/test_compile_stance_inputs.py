"""Tiny synthetic compilation fixtures are never approved research inputs."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.population import PopulationMember, PopulationSnapshot
from llm_bias.core.stance_evidence import validate_evidence_pairs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compile_stance_inputs", ROOT / "scripts/compile_stance_inputs.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
CONFIG = ROOT / "configs/concept-cone-steering/rebuild-v1/inputs.json"
CSV = ROOT / "data/sp500_constituents_2020_2025.csv"
PDF = ROOT / "data/concept-cone-steering/rebuild-v1/sources/factset_20241115.pdf"


@pytest.fixture
def fixture_inputs():
    source = b"SYNTHETIC external-review fixture, not research evidence."
    members = tuple(PopulationMember(t, "Fixture Alpha" if t in ("A", "AA") else t,
                                     "Unspecified" if t == "A" else "Fixture sector",
                                     "issuer:alpha" if t in ("A", "AA") else "issuer:" + t)
                    for t in ("A", "AA", "B", "C", "D", "E"))
    snapshot = PopulationSnapshot(members, sha256_bytes(b"fixture CSV"),
                                  sha256_json([asdict(m) for m in members]))
    items = []
    for item_id, polarity in (("p1", "+"), ("p2", "+"), ("n1", "-"), ("n2", "-")):
        text = "Synthetic reviewed observation " + item_id
        items.append(dict(item_id=item_id, text=text, polarity=polarity,
                          source="https://fixture.invalid/source", source_sha256=sha256_bytes(source),
                          content_sha256=sha256_bytes(text.encode()), review_id="synthetic-review",
                          strength="synthetic-factor", numerical_review="Synthetic no-numbers review.",
                          source_quality_review="Synthetic test review only; not research approval."))
    config = dict(schema_version=1,
                  population=dict(source_sha256=snapshot.source_sha256, expected_tickers=6,
                                  expected_issuers=5),
                  issuer_policy=dict(prefix="issuer:", groups=[dict(
                      group_id="issuer:alpha", tickers=["A", "AA"],
                      expected_company_prefix="Fixture Alpha")]),
                  roles=dict(seed=20260930, issuer_counts=dict(fit=1, validation=1,
                                                             calibration=1, evaluation=2)),
                  trials=["fixture-trial"], items=items,
                  pairing={"++": ["p1", "p2"], "+-": ["p1", "n1"],
                           "-+": ["n1", "p1"], "--": ["n1", "n2"]},
                  order_contrasts=[dict(trial_id="fixture-trial", condition1="+-", condition2="-+")],
                  provenance=dict(source_url="https://fixture.invalid/source",
                                  source_pdf_sha256=sha256_bytes(source), source_byte_size=len(source),
                                  acquisition_time_utc="2026-09-30T15:05:00Z",
                                  acquisition_time_is_approximate=True, report_date="2024-11-15",
                                  review_id="synthetic-review"))
    return snapshot, source, config


def _compile(inputs):
    return C.compile_input_records(*inputs)


def _payloads(compiled, config):
    return C.serialize_inputs(compiled, config_bytes=canonical_json_bytes(config) + b"\n")


def test_tiny_complete_isolated_deterministic_compilation(fixture_inputs):
    snapshot, source, config = fixture_inputs
    compiled = _compile(fixture_inputs)
    assert len(compiled.pairs) == 24
    assert compiled.roles.issuer_counts == config["roles"]["issuer_counts"]
    assert compiled.roles.assignments["A"] == compiled.roles.assignments["AA"]
    assert sum(compiled.roles.ticker_counts.values()) == 6
    rows = {(p.ticker, p.condition): p for p in compiled.pairs}
    for member in snapshot.members:
        a, b = rows[member.ticker, "+-"], rows[member.ticker, "-+"]
        assert (a.evidence1, a.evidence2) == (b.evidence2, b.evidence1)
    reversed_snapshot = replace(snapshot, members=tuple(reversed(snapshot.members)))
    assert _payloads(compiled, config) == _payloads(
        C.compile_input_records(reversed_snapshot, source, config), config)
    payloads = _payloads(compiled, config)
    assert set(payloads) == {"population.json", "roles.json", "evidence_pairs.jsonl", "inputs_manifest.json"}
    assert all(p.endswith(b"\n") for p in payloads.values())
    manifest = json.loads(payloads["inputs_manifest.json"])
    assert manifest["counts"] == dict(population_count=6, issuer_count=5, unspecified_sector_count=1,
                                     evidence_pair_count=24, ticker_counts=compiled.roles.ticker_counts,
                                     issuer_counts=compiled.roles.issuer_counts)
    assert manifest["output_hashes"] == {name: sha256_bytes(payload)
                                        for name, payload in payloads.items()
                                        if name != "inputs_manifest.json"}
    assert manifest["config_sha256"] == sha256_bytes(canonical_json_bytes(config) + b"\n")
    assert manifest["semantic_hashes"]["membership_hash"] == snapshot.membership_sha256
    assert manifest["semantic_hashes"]["roles_hash"] == compiled.roles.assignment_hash
    assert manifest["evidence_source_sha256"] == sha256_bytes(source)
    assert not manifest["eligibility"]["research_eligible"]
    assert set(manifest["code_sha256"]) == {"scripts/compile_stance_inputs.py",
                                           "llm_bias/core/population.py", "llm_bias/core/stance_evidence.py",
                                           "llm_bias/core/artifact_paths.py"}


@pytest.mark.parametrize("kind", ["missing", "foreign", "duplicate", "order", "source_hash",
                                  "source_size", "item_source", "item_hash", "duplicate_item",
                                  "population_count", "issuer_count", "role_count", "seed",
                                  "group_missing", "group_name", "group_overlap", "unused_item"])
def test_invalid_config_fails_without_intersection(fixture_inputs, kind):
    snapshot, source, original = fixture_inputs
    config = copy.deepcopy(original)
    if kind == "missing":
        del config["pairing"]["--"]
    elif kind == "foreign":
        config["pairing"]["--"][1] = "foreign"
    elif kind == "duplicate":
        config["pairing"]["++"] = ["p1", "p1"]
    elif kind == "order":
        config["pairing"]["-+"] = ["n1", "p2"]
    elif kind == "source_hash":
        source += b" altered"
    elif kind == "source_size":
        config["provenance"]["source_byte_size"] += 1
    elif kind == "item_source":
        config["items"][0]["source_sha256"] = "a" * 64
    elif kind == "item_hash":
        config["items"][0]["text"] += " altered"
    elif kind == "duplicate_item":
        config["items"].append(config["items"][0])
    elif kind in ("population_count", "issuer_count"):
        config["population"]["expected_tickers" if kind == "population_count" else "expected_issuers"] -= 1
    elif kind == "role_count":
        config["roles"]["issuer_counts"]["fit"] = 0
    elif kind == "seed":
        config["roles"]["seed"] = True
    elif kind == "group_missing":
        config["issuer_policy"]["groups"][0]["tickers"][1] = "FOREIGN"
    elif kind == "group_name":
        config["issuer_policy"]["groups"][0]["expected_company_prefix"] = "Wrong name"
    elif kind == "group_overlap":
        config["issuer_policy"]["groups"].append(config["issuer_policy"]["groups"][0])
    else:
        config["items"].append(config["items"][0] | {"item_id": "unused"})
    with pytest.raises(ValueError):
        C.compile_input_records(snapshot, source, config)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "foreign"])
def test_complete_pair_validator_never_accepts_partial_or_foreign_rows(fixture_inputs, kind):
    compiled = _compile(fixture_inputs)
    pairs = list(compiled.pairs)
    if kind == "duplicate":
        pairs.append(pairs[0])
    elif kind == "missing":
        pairs.pop()
    else:
        pairs[0] = replace(pairs[0], ticker="FOREIGN")
    with pytest.raises(ValueError):
        validate_evidence_pairs(compiled.snapshot.members, ["fixture-trial"], pairs,
                                order_contrasts=compiled.order_contrasts)


def test_snapshot_membership_and_issuer_mapping_revalidated(fixture_inputs):
    snapshot, source, config = fixture_inputs
    for changed in (replace(snapshot, membership_sha256="a" * 64),
                    replace(snapshot, source_sha256="a" * 64),
                    replace(snapshot, members=(replace(snapshot.members[0], issuer_id="foreign"),
                                               *snapshot.members[1:]))):
        with pytest.raises(ValueError):
            C.compile_input_records(changed, source, config)


def test_atomic_identical_resume_and_no_overwrite(fixture_inputs, tmp_path):
    compiled = _compile(fixture_inputs)
    config = fixture_inputs[2]
    payloads = _payloads(compiled, config)
    out = tmp_path / "compiled"
    C.write_outputs(out, payloads)
    C.write_outputs(out, payloads)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == payloads
    changed = _payloads(compiled, config | {"provenance": config["provenance"] | {"note": "changed"}})
    with pytest.raises(ValueError, match="overwrite|identical"):
        C.write_outputs(out, changed)
    (out / "roles.json").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        C.write_outputs(out, payloads)
    assert (out / "roles.json").read_bytes() == b"tampered"


@pytest.mark.parametrize("kind", ["nonempty", "unexpected", "missing", "symlink"])
def test_existing_directory_requires_exact_four_regular_files(fixture_inputs, tmp_path, kind):
    payloads = _payloads(_compile(fixture_inputs), fixture_inputs[2])
    out = tmp_path / "compiled"
    if kind == "nonempty":
        out.mkdir()
        (out / "unrelated").write_text("leave me")
    else:
        C.write_outputs(out, payloads)
        if kind == "unexpected":
            (out / "unexpected").write_text("leave me")
        elif kind == "missing":
            (out / "roles.json").unlink()
        else:
            (out / "roles.json").unlink()
            target = tmp_path / "target"
            target.write_bytes(payloads["roles.json"])
            (out / "roles.json").symlink_to(target)
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(ValueError):
        C.write_outputs(out, payloads)
    assert sorted(p.name for p in out.iterdir()) == before


def test_failed_staging_does_not_publish_partial_output(fixture_inputs, tmp_path, monkeypatch):
    payloads = _payloads(_compile(fixture_inputs), fixture_inputs[2])
    out = tmp_path / "compiled"
    def fail_fsync(fd):
        raise OSError("synthetic durability failure")
    monkeypatch.setattr(C.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="durability"):
        C.write_outputs(out, payloads)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/compile_stance_inputs.py"),
                           *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_cli_wrong_population_digest_and_unavailable_source_fail_before_output(tmp_path):
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("not the frozen population")
    out = tmp_path / "compiled"
    result = _cli("--population-csv", wrong, "--out-dir", out)
    assert result.returncode != 0 and "population" in result.stderr.lower()
    assert not out.exists()
    if not CSV.exists():
        pytest.skip("Frozen population CSV absent; no download permitted.")
    result = _cli("--population-csv", CSV, "--source-pdf", tmp_path / "absent.pdf", "--out-dir", out)
    assert result.returncode != 0 and "source" in result.stderr.lower()
    assert not out.exists()


@pytest.mark.parametrize("kind", ["fixture", "counts", "groups", "roles", "pairing", "source", "text"])
def test_cli_cannot_relax_frozen_policy(fixture_inputs, tmp_path, kind):
    config = json.loads(CONFIG.read_bytes())
    if kind == "fixture":
        config = fixture_inputs[2]
    elif kind == "counts":
        config["population"]["expected_tickers"] = 6
    elif kind == "groups":
        config["issuer_policy"]["groups"] = []
    elif kind == "roles":
        config["roles"]["seed"] += 1
    elif kind == "pairing":
        config["pairing"]["++"] = ["F24-P2", "F24-P1"]
    elif kind == "source":
        config["provenance"]["source_pdf_sha256"] = "a" * 64
    else:
        config["items"][0]["text"] += " Unauthorized conclusion."
        config["items"][0]["content_sha256"] = sha256_bytes(config["items"][0]["text"].encode())
    path = tmp_path / "config.json"
    path.write_bytes(canonical_json_bytes(config))
    out = tmp_path / "compiled"
    result = _cli("--config", path, "--out-dir", out)
    assert result.returncode != 0 and "policy" in result.stderr.lower()
    assert not out.exists()


def test_compiler_import_never_loads_model_frameworks():
    script = (
        "import runpy,sys; "
        f"runpy.run_path({str(ROOT / 'scripts/compile_stance_inputs.py')!r}); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_approved_config_contains_exact_reviewed_text_and_explicit_limitations():
    config = json.loads(CONFIG.read_bytes())
    review = (ROOT / "docs/concept-cone-steering/rebuild-v1/evidence-review.md").read_text()
    reviewed = {line.split("|")[1].strip().split(" / ")[0]: line.split("|")[2].strip()
                for line in review.splitlines() if line.startswith("| F24-")}
    assert {item["item_id"]: item["text"] for item in config["items"]} == reviewed
    for item in config["items"]:
        assert item["content_sha256"] == sha256_bytes(item["text"].encode("utf-8"))
        assert item["review_id"] == "factset-20241115-machine-factual-v1"
        assert item["strength"] == "single_aggregate_factor"
        assert item["numerical_review"].startswith("Machine factual review:")
        assert "no matched persuasive strength" in item["source_quality_review"]
        assert item["source"] == config["provenance"]["source_url"]
        assert item["source_sha256"] == config["provenance"]["source_pdf_sha256"]


def test_cli_has_no_membership_or_digest_override(tmp_path):
    for flag in ("--target-tickers", "--smoke-list", "--expected-count", "--source-sha256"):
        result = _cli("--out-dir", tmp_path / "compiled", flag, "test")
        assert result.returncode != 0 and "unrecognized arguments" in result.stderr


def test_research_assets_exact_503_by_four_and_portable_outputs(tmp_path):
    if not CSV.exists() or not PDF.exists():
        pytest.skip("Acquired frozen CSV/PDF absent; real-source check never downloads assets.")
    out = tmp_path / "compiled"
    result = _cli("--config", CONFIG, "--population-csv", CSV, "--source-pdf", PDF, "--out-dir", out)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "inputs_manifest.json").read_bytes())
    assert manifest["counts"] == dict(population_count=503, issuer_count=500,
                                     unspecified_sector_count=29, evidence_pair_count=2012,
                                     ticker_counts=dict(fit=302, validation=75, calibration=26, evaluation=100),
                                     issuer_counts=dict(fit=300, validation=75, calibration=25, evaluation=100))
    rows = [json.loads(line) for line in (out / "evidence_pairs.jsonl").read_bytes().splitlines()]
    keys = [(r["ticker"], r["trial_id"], r["condition"]) for r in rows]
    assert len(set(keys)) == 2012 and keys == sorted(keys)
    by_key = {(r["ticker"], r["condition"]): r for r in rows}
    for ticker in {r["ticker"] for r in rows}:
        assert by_key[ticker, "+-"]["evidence1"] == by_key[ticker, "-+"]["evidence2"]
        assert by_key[ticker, "+-"]["evidence2"] == by_key[ticker, "-+"]["evidence1"]
    assert manifest["population_source_sha256"] == "27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38"
    assert manifest["evidence_source_sha256"] == "677776cd01ddf02efcfb5fb4b437c336baa825f0dd41f3cdf9f3b1b829078486"
    assert manifest["provenance"]["source_byte_size"] == 1080178
    assert manifest["provenance"]["acquisition_time_is_approximate"] is True
    assert not manifest["eligibility"]["research_eligible"]
    for name, digest in manifest["output_hashes"].items():
        assert sha256_bytes((out / name).read_bytes()) == digest
    # Paths are acquisition provenance, not part of portable output identity.
    copy_csv, copy_pdf = tmp_path / "copy.csv", tmp_path / "copy.pdf"
    copy_csv.write_bytes(CSV.read_bytes())
    copy_pdf.write_bytes(PDF.read_bytes())
    result = _cli("--population-csv", copy_csv, "--source-pdf", copy_pdf, "--out-dir", out)
    assert result.returncode == 0, result.stderr
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    altered = tmp_path / "altered.pdf"
    altered.write_bytes(PDF.read_bytes() + b"altered")
    result = _cli("--source-pdf", altered, "--out-dir", out)
    assert result.returncode != 0 and "source" in result.stderr.lower()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
