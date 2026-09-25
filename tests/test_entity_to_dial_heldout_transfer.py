"""Deterministic Slice 1/2 tests for held-out Entity-to-Dial transfer."""
from __future__ import annotations

import csv
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_bias.entity_to_dial import heldout_transfer
from llm_bias.entity_to_dial.heldout_transfer import (
    CONSTRUCTION_TICKERS,
    HELDOUT_BASIS_K,
    HELDOUT_COHORT_SIZE,
    HELDOUT_SELECTION_VARIANT,
    analyze_heldout_transfer,
    build_cross_sector_graph,
    build_sector_within_graph,
    prepare_heldout_cohort,
    prepare_heldout_prompt_records,
    _load_population_contract,
    _run_heldout_transfer,
    random_orthonormal_bases,
    run_heldout_transfer,
    validate_compact_analysis,
    validate_m6_exclusion_manifest,
    validate_prompt_roles,
    verify_heldout_inputs,
    run_heldout_transfer,
)


def _qwen_fake(num_layers: int = 16, seed: int = 0) -> object:
    """Small real Qwen3.5 fake model."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=9000, hidden_size=32, intermediate_size=64,
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


class _CharTokenizer:
    chat_template = "fake-template-v1"
    name_or_path = "fake-tokenizer"

    def __call__(self, text, *, add_special_tokens=True,
                 return_offsets_mapping=False, return_special_tokens_mask=False, **_kwargs):
        del add_special_tokens
        text = str(text)
        suffix = None
        for candidate, token_id in (("buy", 240), ("sell", 241)):
            if text.endswith(candidate):
                suffix = (candidate, token_id)
                break
        if suffix is None:
            values = [ord(char) for char in text]
        else:
            prefix = text[:-len(suffix[0])]
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


def _write_population(path, n=503):
    sectors = ("Energy", "Financials", "Health Care", "Industrials", "Technology")
    rows = []
    used = set()
    for index, ticker in enumerate(CONSTRUCTION_TICKERS):
        sector = sectors[index % len(sectors)]
        rows.append((ticker, f"Construction {ticker}", sector))
        used.add(ticker)
    index = 0
    while len(rows) < n:
        ticker = f"T{index:04d}"
        index += 1
        if ticker in used:
            continue
        sector = sectors[index % len(sectors)]
        rows.append((ticker, f"Company {ticker}", sector))
        used.add(ticker)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index_name", "year", "ticker", "company_name", "gics_sector"])
        writer.writeheader()
        for ticker, name, sector in rows:
            writer.writerow({"index_name": "S&P 500", "year": "2024", "ticker": ticker,
                             "company_name": name, "gics_sector": sector})


def _m6(*tickers):
    values = list(tickers) or [f"M6{i:02d}" for i in range(12)]
    return {"schema_version": "selective-intervention-m6-external-v1",
            "companies": [{"ticker": ticker, "name": f"M6 {ticker}", "sector": "Other"}
                          for ticker in values]}


def test_cohort_is_deterministic_proportional_and_excludes_construction_and_m6(tmp_path):
    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    rows = _load_population_contract(population_path)
    m6 = _m6("M6-ABSENT", *[f"M6{i:02d}" for i in range(1, 12)])
    first = prepare_heldout_cohort(rows, m6_manifest=m6, population_path=population_path,
                                   selection_seed=123)
    second = prepare_heldout_cohort(rows, m6_manifest=m6, population_path=population_path,
                                    selection_seed=123)
    assert first == second
    assert len(first["companies"]) == HELDOUT_COHORT_SIZE
    tickers = [row["ticker"] for row in first["companies"]]
    assert tickers == sorted(tickers)
    assert len(tickers) == len(set(tickers))
    assert not set(tickers) & set(CONSTRUCTION_TICKERS)
    assert "M6-ABSENT" not in set(tickers)
    assert first["exclusions"]["eligible_ticker_count"] == 503 - 16
    assert sum(item["quota"] for item in first["sector_counts"].values()) == 200
    assert first["raw_runtime_payloads"] is False


def test_m6_validation_is_local_and_fail_closed():
    valid = _m6()
    checked = validate_m6_exclusion_manifest(valid)
    assert len(checked["m6_tickers"]) == 12
    with pytest.raises(ValueError, match="exactly 12"):
        validate_m6_exclusion_manifest({
            "schema_version": valid["schema_version"],
            "companies": valid["companies"][:11],
        })
    duplicate = dict(valid, companies=valid["companies"][:-1] + [valid["companies"][0]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_m6_exclusion_manifest(duplicate)
    malformed = dict(valid, companies=valid["companies"][:-1] + [{"ticker": "X", "name": "", "sector": "S"}])
    with pytest.raises(ValueError, match="non-empty"):
        validate_m6_exclusion_manifest(malformed)


def test_verify_inputs_returns_cpu_v8_and_rejects_overlap(tmp_path):
    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    vectors = torch.eye(16).tolist()
    e01 = {"manifest": {"status": "complete"},
           "summary": {"pca_basis_vectors": vectors,
                       "pca_singular_values": [float(20 - i) for i in range(16)]}}
    result = verify_heldout_inputs(e01, _m6(), population_path)
    assert result["v8"].shape == (16, HELDOUT_BASIS_K)
    assert result["v8"].device.type == "cpu"
    with pytest.raises(ValueError, match="overlap"):
        verify_heldout_inputs(e01, _m6("AMAT", *[f"M{i:02d}" for i in range(11)]), population_path)


def test_m6_validation_rejects_wrong_schema():
    invalid = _m6()
    invalid["schema_version"] = "selective-intervention-m6-external-v2"
    with pytest.raises(ValueError, match="schema_version"):
        validate_m6_exclusion_manifest(invalid)


def test_prompt_records_have_exact_roles_and_no_input_ids():
    cohort = {"companies": [{"ticker": "AAA", "name": "Alpha", "sector": "Technology"}]}
    records = prepare_heldout_prompt_records(cohort, _CharTokenizer())
    assert len(records) == 4
    assert {(r["reverse"], r["order"], r["role"]) for r in records} == {
        (False, 0, "selection"), (True, 0, "evaluation"),
        (False, 1, "evaluation"), (True, 1, "evaluation"),
    }
    assert all("input_ids" not in record for record in records)
    validate_prompt_roles(records, cohort)
    bad = [dict(record) for record in records]
    bad[0]["role"] = "evaluation"
    with pytest.raises(ValueError, match="role"):
        validate_prompt_roles(bad, cohort)
    assert HELDOUT_SELECTION_VARIANT == (False, 0)


def _selection_records(companies):
    return [{"ticker": ticker, "sector": sector, "margin": margin,
             "role": "selection", "reverse": False, "order": 0}
            for ticker, sector, margin in companies]


def test_sector_within_graph_tie_break_and_odd_leftover():
    companies = [
        ("A1", "Alpha", 1.0), ("A2", "Alpha", 1.0), ("A3", "Alpha", 3.0),
        ("B1", "Beta", 0.0), ("B2", "Beta", 2.0), ("B3", "Beta", 4.0), ("B4", "Beta", 5.0),
    ]
    manifest = {"companies": [{"ticker": t, "sector": s} for t, s, _ in companies]}
    graph = build_sector_within_graph(_selection_records(companies), manifest)
    assert [(p["high_ticker"], p["low_ticker"]) for p in graph["pairs"]] == [
        ("A3", "A1"), ("B3", "B2"), ("B4", "B1")
    ]
    assert graph["unmatched_tickers"] == ["A2"]
    assert all("evaluation" not in pair for pair in graph["pairs"])
    bad = _selection_records(companies)
    bad[0]["role"] = "evaluation"
    with pytest.raises(ValueError, match="selection"):
        build_sector_within_graph(bad, manifest)


def test_cross_sector_graph_is_deterministic_and_has_no_same_sector_edges():
    companies = [(f"L{i:03d}", "Target", float(i)) for i in range(100)]
    companies += [(f"S{i:03d}", "Source", float(100 + i)) for i in range(100)]
    manifest = {"companies": [{"ticker": t, "sector": s} for t, s, _ in companies]}
    records = _selection_records(companies)
    first = build_cross_sector_graph(records, manifest)
    second = build_cross_sector_graph(list(reversed(records)), manifest)
    assert first == second
    assert len(first["pairs"]) == 100
    assert all(pair["high_sector"] != pair["low_sector"] for pair in first["pairs"])
    assert len({t for p in first["pairs"] for t in (p["high_ticker"], p["low_ticker"])}) == 200
    assert all("evaluation" not in pair for pair in first["pairs"])


def test_cross_sector_graph_fails_without_perfect_matching():
    companies = [(f"C{i:03d}", "Only", float(i)) for i in range(200)]
    manifest = {"companies": [{"ticker": t, "sector": s} for t, s, _ in companies]}
    with pytest.raises(ValueError, match="perfect"):
        build_cross_sector_graph(_selection_records(companies), manifest)


def _analysis_graph(stratum):
    pairs = [
        {"pair_id": f"{stratum}:p1", "high_ticker": f"{stratum[:2]}H1", "low_ticker": f"{stratum[:2]}L1",
         "high_sector": "High", "low_sector": "Low", "high_margin": 1.0, "low_margin": -1.0, "stratum": stratum},
        {"pair_id": f"{stratum}:p2", "high_ticker": f"{stratum[:2]}H2", "low_ticker": f"{stratum[:2]}L2",
         "high_sector": "High2", "low_sector": "Low2", "high_margin": 1.0, "low_margin": -1.0, "stratum": stratum},
    ]
    provenance = []
    for pair in pairs:
        provenance.extend([
            {"ticker": pair["high_ticker"], "sector": pair["high_sector"], "margin": pair["high_margin"]},
            {"ticker": pair["low_ticker"], "sector": pair["low_sector"], "margin": pair["low_margin"]},
        ])
    return {
        "schema_version": "entity-to-dial-heldout-transfer-v1",
        "stratum": stratum,
        "selection_margin_provenance": provenance,
        "pairs": pairs,
        "unmatched_tickers": [],
        "raw_runtime_payloads": False,
    }


def _analysis_records(graphs, ratios_by_pair):
    records = []
    for graph in graphs:
        stratum = graph["stratum"]
        for pair in graph["pairs"]:
            ratios = ratios_by_pair.get((stratum, pair["pair_id"]), (0.9, 0.2))
            for direction in ("high_to_low", "low_to_high"):
                source = pair["high_ticker"] if direction == "high_to_low" else pair["low_ticker"]
                target = pair["low_ticker"] if direction == "high_to_low" else pair["high_ticker"]
                source_sector = pair["high_sector"] if direction == "high_to_low" else pair["low_sector"]
                target_sector = pair["low_sector"] if direction == "high_to_low" else pair["high_sector"]
                sign = 1.0 if direction == "high_to_low" else -1.0
                for reverse, order in ((True, 0), (False, 1), (True, 1)):
                    for arm in ("clean", "full", "v8", "random_8d_0", "random_8d_1", "random_8d_2", "random_8d_3"):
                        shift = {"clean": 0.0, "full": 1.0, "v8": ratios[0], "random_8d_0": ratios[1],
                                 "random_8d_1": ratios[1], "random_8d_2": ratios[1], "random_8d_3": ratios[1]}[arm]
                        records.append({"stratum": stratum, "pair_id": pair["pair_id"], "direction": direction,
                                        "source_ticker": source, "target_ticker": target,
                                        "source_sector": source_sector, "target_sector": target_sector,
                                        "reverse": reverse, "order": order, "arm": arm,
                                        "source_margin": sign, "target_margin": 0.0,
                                        "patched_margin": sign * shift, "raw_runtime_payloads": False})
    return records


def test_compact_analysis_requires_complete_cells_and_uses_full_only_eligibility():
    graphs = [_analysis_graph("sector_within"), _analysis_graph("cross_sector")]
    records = _analysis_records(graphs, {})
    summary = analyze_heldout_transfer(records, graphs, bootstrap_samples=100, bootstrap_seed=9)
    assert summary["strata"]["sector_within"]["all_directions"]["v8_median_ratio"] == pytest.approx(0.9)
    assert summary["strata"]["sector_within"]["all_directions"]["random_median_ratios"] == pytest.approx([0.2] * 4)
    assert summary["strata"]["sector_within"]["all_directions"]["paired_advantage_median"] == pytest.approx(0.7)
    assert summary["strata"]["sector_within"]["status"] == "supported"
    assert summary["strata"]["cross_sector"]["eligible_bundle_count"] == 2
    with pytest.raises(ValueError, match="seven arms"):
        analyze_heldout_transfer(records[:-1], graphs, bootstrap_samples=10)
    changed = []
    for record in records:
        item = dict(record)
        if (item["stratum"], item["pair_id"]) == ("sector_within", "sector_within:p2") and item["arm"] == "full":
            sign = 1.0 if item["direction"] == "high_to_low" else -1.0
            item["patched_margin"] = sign * 0.1
        changed.append(item)
    excluded_summary = analyze_heldout_transfer(changed, graphs, bootstrap_samples=100)
    excluded = [edge for edge in excluded_summary["edges"]
                if edge["stratum"] == "sector_within" and edge["pair_id"] == "sector_within:p2"]
    assert len(excluded) == 2
    assert all(not edge["eligible"] and edge["exclusion_reason"] == "full_reference_below_threshold" for edge in excluded)
    assert all(edge["ratio_v8"] is None for edge in excluded)
    assert excluded_summary["strata"]["sector_within"]["eligible_bundle_count"] == 1


def test_compact_analysis_keeps_eligible_direction_from_mixed_bundle():
    graphs = [_analysis_graph("sector_within"), _analysis_graph("cross_sector")]
    records = _analysis_records(graphs, {})
    changed = []
    for record in records:
        item = dict(record)
        if (
            item["stratum"], item["pair_id"], item["direction"], item["arm"]
        ) == ("sector_within", "sector_within:p2", "high_to_low", "full"):
            item["patched_margin"] = 0.1
        changed.append(item)
    first = analyze_heldout_transfer(changed, graphs, bootstrap_samples=100, bootstrap_seed=23)
    second = analyze_heldout_transfer(changed, graphs, bootstrap_samples=100, bootstrap_seed=23)
    assert first == second
    pair_edges = [
        edge for edge in first["edges"]
        if edge["stratum"] == "sector_within" and edge["pair_id"] == "sector_within:p2"
    ]
    assert {edge["direction"]: edge["eligible"] for edge in pair_edges} == {
        "high_to_low": False,
        "low_to_high": True,
    }
    metrics = first["strata"]["sector_within"]
    assert metrics["eligible_edge_count"] == 3
    assert metrics["excluded_edge_count"] == 1
    assert metrics["eligible_bundle_count"] == 2
    assert metrics["high_to_low"]["eligible_edge_count"] == 1
    assert metrics["low_to_high"]["eligible_edge_count"] == 2
    assert metrics["high_to_low"]["eligible_bundle_count"] == 1
    assert metrics["low_to_high"]["eligible_bundle_count"] == 2


def test_compact_analysis_separates_directions_strata_and_bootstrap_bundles():
    graphs = [_analysis_graph("sector_within"), _analysis_graph("cross_sector")]
    ratios = {("sector_within", "sector_within:p1"): (0.8, 0.1),
              ("sector_within", "sector_within:p2"): (0.8, 0.1),
              ("cross_sector", "cross_sector:p1"): (0.7, 0.1),
              ("cross_sector", "cross_sector:p2"): (0.7, 0.1)}
    first = analyze_heldout_transfer(_analysis_records(graphs, ratios), graphs, bootstrap_samples=200, bootstrap_seed=31)
    second = analyze_heldout_transfer(_analysis_records(graphs, ratios), graphs, bootstrap_samples=200, bootstrap_seed=31)
    assert first == second
    assert first["strata"]["sector_within"]["status"] == "supported"
    assert first["strata"]["cross_sector"]["status"] == "not_supported"
    assert first["strata"]["sector_within"]["high_to_low"]["eligible_bundle_count"] == 2
    assert first["strata"]["sector_within"]["low_to_high"]["eligible_bundle_count"] == 2
    assert set(first["strata"]["cross_sector"]["sector_breakdown"]) == {"High->Low", "High2->Low2", "Low->High", "Low2->High2"}


def test_compact_analysis_status_boundaries_and_not_evaluable():
    graphs = [_analysis_graph("sector_within"), _analysis_graph("cross_sector")]
    records = _analysis_records(graphs, {("sector_within", "sector_within:p1"): (0.8, 0.0),
                                         ("sector_within", "sector_within:p2"): (0.8, 0.0),
                                         ("cross_sector", "cross_sector:p1"): (0.7, 0.0),
                                         ("cross_sector", "cross_sector:p2"): (0.7, 0.0)})
    summary = analyze_heldout_transfer(records, graphs, bootstrap_samples=100)
    assert summary["strata"]["sector_within"]["status"] == "supported"
    for record in records:
        if record["stratum"] == "sector_within" and record["arm"] == "full":
            sign = 1.0 if record["direction"] == "high_to_low" else -1.0
            record["patched_margin"] = sign * 0.1
    result = analyze_heldout_transfer(records, graphs, bootstrap_samples=100)
    assert result["strata"]["sector_within"]["status"] == "not_evaluable"
    assert result["strata"]["sector_within"]["all_directions"]["v8_median_ratio"] is None


def test_compact_summary_forbids_raw_payload_keys_recursively():
    graphs = [_analysis_graph("sector_within"), _analysis_graph("cross_sector")]
    summary = analyze_heldout_transfer(_analysis_records(graphs, {}), graphs, bootstrap_samples=10)
    validate_compact_analysis(summary)
    bad = dict(summary, nested={"payload": [{"hidden": [1, 2, 3]}]})
    with pytest.raises(ValueError, match="forbidden"):
        validate_compact_analysis(bad)


def _fake_e01_run(tmp_path):
    root = tmp_path / "e01"
    (root / "analyze").mkdir(parents=True)
    basis = torch.eye(32, dtype=torch.float64)[:16]
    (root / "manifest.json").write_text('{"status": "complete"}', encoding="utf-8")
    (root / "analyze" / "summary.json").write_text(
        __import__("json").dumps({
            "pca_basis_vectors": basis.tolist(),
            "pca_singular_values": [float(16 - index) for index in range(16)],
        }),
        encoding="utf-8",
    )
    return root


def test_heldout_fake_workflow_writes_compact_lifecycle_artifacts(tmp_path, monkeypatch):

    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    e01 = _fake_e01_run(tmp_path)
    tokenizer = _CharTokenizer()
    model = _qwen_fake()
    monkeypatch.setattr(heldout_transfer, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(
        heldout_transfer,
        "load_model",
        lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")),
    )

    root = _run_heldout_transfer(
        model_path="fake-model",
        run_id="heldout-fake",
        phase_e_run=e01,
        m6_manifest=_m6(),
        population_csv=population_path,
        artifact_root=tmp_path / "artifacts",
        cohort_size=8,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )
    manifest = __import__("json").loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for relative in (
        "prepare/cohort_manifest.json", "prepare/prompts.jsonl", "prepare/provenance.json",
        "forward/selection.jsonl", "forward/pair_graphs.json", "forward/records.jsonl",
        "forward/noop_records.jsonl", "analyze/summary.json",
    ):
        assert (root / relative).is_file()
    selection = [__import__("json").loads(line) for line in (root / "forward" / "selection.jsonl").read_text(encoding="utf-8").splitlines()]
    graphs = __import__("json").loads((root / "forward" / "pair_graphs.json").read_text(encoding="utf-8"))
    records = [__import__("json").loads(line) for line in (root / "forward" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    noops = [__import__("json").loads(line) for line in (root / "forward" / "noop_records.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = __import__("json").loads((root / "analyze" / "summary.json").read_text(encoding="utf-8"))
    assert len(selection) == 8
    assert set(graphs) == {"sector_within", "cross_sector"}
    assert all("evaluation" not in pair for graph in graphs.values() for pair in graph["pairs"])
    assert len(records) == sum(len(graph["pairs"]) for graph in graphs.values()) * 2 * 3 * 7
    assert {record["arm"] for record in records} == {
        "clean", "full", "v8", "random_8d_0", "random_8d_1", "random_8d_2", "random_8d_3",
    }
    assert all(record["hook_fires"] == 1 for record in records if record["arm"] != "clean")
    assert len(noops) == 8 * 3 * 2
    assert all(record["pass"] and abs(record["delta_m"]) <= 1e-12 for record in noops)
    validate_compact_analysis(summary)
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.json*"))
    for raw_key in ('"activation"', '"residual"', '"hidden"', '"basis"', '"input_ids"', '"logits"'):
        assert raw_key not in artifact_text


def test_public_runner_requires_model_and_population_arguments():
    with pytest.raises(TypeError, match="model_path/model"):
        run_heldout_transfer(
            run_id="missing-model",
            phase_e_run="unused",
            m6_manifest=_m6(),
            population_csv="unused.csv",
        )
    with pytest.raises(TypeError, match="population_csv/population_path"):
        run_heldout_transfer(
            model_path="unused-model",
            run_id="missing-population",
            phase_e_run="unused",
            m6_manifest=_m6(),
        )


def test_heldout_smoke_runs_one_pair_without_graphs_or_analysis(tmp_path, monkeypatch):

    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    e01 = _fake_e01_run(tmp_path)
    tokenizer = _CharTokenizer()
    model = _qwen_fake()
    monkeypatch.setattr(heldout_transfer, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(
        heldout_transfer,
        "load_model",
        lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")),
    )
    monkeypatch.setattr(
        heldout_transfer,
        "build_cross_sector_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("smoke must not build graphs")),
    )
    root = run_heldout_transfer(
        model_path="fake-model",
        run_id="heldout-smoke-fake",
        phase_e_run=e01,
        m6_manifest=_m6(),
        population_csv=population_path,
        artifact_root=tmp_path / "artifacts",
        smoke=True,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    smoke = json.loads((root / "forward" / "smoke.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert smoke["smoke"] is True
    assert set(smoke["margins"]) == {"clean", "full", "v8", "random_8d_0"}
    assert all(value == 1 for value in smoke["hook_fires"].values())
    assert not (root / "forward" / "selection.jsonl").exists()
    assert not (root / "forward" / "pair_graphs.json").exists()
    assert not (root / "analyze" / "summary.json").exists()


def test_heldout_script_requires_run_id_and_forwards_smoke(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    phase_e_run = tmp_path / "e01"
    phase_e_run.mkdir()
    (phase_e_run / "manifest.json").write_text('{"status": "complete"}', encoding="utf-8")
    m6_path = tmp_path / "m6.json"
    m6_path.write_text(json.dumps(_m6()), encoding="utf-8")
    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    script = runpy.run_path("scripts/entity_to_dial_heldout_transfer.py")
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", ["entity_to_dial_heldout_transfer.py"])
        with pytest.raises(SystemExit):
            script["main"]()
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", [
            "entity_to_dial_heldout_transfer.py", "--model", str(model_path),
            "--phase-e-run", str(phase_e_run), "--m6-manifest", str(m6_path),
            "--population-csv", str(population_path),
        ])
        with pytest.raises(SystemExit, match="--run-id"):
            script["main"]()
    calls = []
    smoke_root = tmp_path / "script-smoke"
    (smoke_root / "forward").mkdir(parents=True)
    (smoke_root / "forward" / "smoke.json").write_text(
        json.dumps({"source_ticker": "A", "target_ticker": "B", "margins": {}}),
        encoding="utf-8",
    )
    with monkeypatch.context() as context:
        context.setitem(
            script["main"].__globals__,
            "run_heldout_transfer",
            lambda **kwargs: calls.append(kwargs) or smoke_root,
        )
        context.setattr(sys, "argv", [
            "entity_to_dial_heldout_transfer.py", "--model", str(model_path),
            "--phase-e-run", str(phase_e_run), "--m6-manifest", str(m6_path),
            "--population-csv", str(population_path), "--smoke",
        ])
        script["main"]()
    assert len(calls) == 1
    assert calls[0]["model_path"] == str(model_path)
    assert calls[0]["phase_e_run"] == str(phase_e_run)
    assert calls[0]["smoke"] is True
    assert calls[0]["run_id"].startswith("entity-to-dial-heldout-transfer-smoke-")


def test_heldout_workflow_fails_closed_on_invalid_e01_and_malformed_graph(tmp_path, monkeypatch):

    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    tokenizer = _CharTokenizer()
    model = _qwen_fake()
    monkeypatch.setattr(heldout_transfer, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(
        heldout_transfer,
        "load_model",
        lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")),
    )
    broken = _fake_e01_run(tmp_path)
    (broken / "manifest.json").write_text('{"status": "failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        _run_heldout_transfer(
            model_path="fake-model", run_id="heldout-e01-fail", phase_e_run=broken,
            m6_manifest=_m6(), population_csv=population_path,
            artifact_root=tmp_path / "artifacts", cohort_size=8, bootstrap_samples=10,
        )

    e01 = _fake_e01_run(tmp_path / "valid")
    monkeypatch.setattr(
        heldout_transfer,
        "build_cross_sector_graph",
        lambda *_args, **_kwargs: {"stratum": "cross_sector", "pairs": "not-a-list"},
    )
    with pytest.raises(ValueError, match="pairs must be a list"):
        _run_heldout_transfer(
            model_path="fake-model", run_id="heldout-graph-fail", phase_e_run=e01,
            m6_manifest=_m6(), population_csv=population_path,
            artifact_root=tmp_path / "artifacts", cohort_size=8, bootstrap_samples=10,
        )
    manifest = __import__("json").loads(
        (tmp_path / "artifacts" / "fake-model" / "entity-to-dial-heldout-transfer" / "runs" / "heldout-graph-fail" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"


def test_heldout_workflow_fails_closed_on_noop_violation(tmp_path, monkeypatch):

    population_path = tmp_path / "population.csv"
    _write_population(population_path)
    e01 = _fake_e01_run(tmp_path)
    tokenizer = _CharTokenizer()
    model = _qwen_fake()
    monkeypatch.setattr(heldout_transfer, "load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(
        heldout_transfer,
        "load_model",
        lambda _path, *, dtype=None: (model, tokenizer, torch.device("cpu")),
    )
    original = heldout_transfer._patched_margin

    def corrupt_noop(*args, **kwargs):
        margin, fires = original(*args, **kwargs)
        transform = args[3]
        if args[2]["ticker"].endswith("0001"):
            return margin + 1.0, fires
        return margin, fires

    monkeypatch.setattr(heldout_transfer, "_patched_margin", corrupt_noop)
    with pytest.raises(ValueError, match="self-source no-op violated"):
        _run_heldout_transfer(
            model_path="fake-model",
            run_id="heldout-noop-fail",
            phase_e_run=e01,
            m6_manifest=_m6(),
            population_csv=population_path,
            artifact_root=tmp_path / "artifacts",
            cohort_size=8,
            bootstrap_samples=10,
        )
    manifest = __import__("json").loads(
        (tmp_path / "artifacts" / "fake-model" / "entity-to-dial-heldout-transfer" / "runs" / "heldout-noop-fail" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"


def test_random_bases_are_reproducible_canonical_and_orthonormal():
    first, metadata = random_orthonormal_bases(16, count=4, k=8, seed=77)
    second, metadata_again = random_orthonormal_bases(16, count=4, k=8, seed=77)
    assert metadata == metadata_again
    for left, right, item in zip(first, second, metadata, strict=True):
        assert left.dtype == torch.float32
        assert torch.equal(left, right)
        assert left.shape == (16, 8)
        assert torch.allclose(left.T.double() @ left.double(), torch.eye(8, dtype=torch.float64), atol=1e-6)
        assert item["sha256"] == hashlib.sha256(left.numpy().tobytes()).hexdigest()
        for column in range(left.shape[1]):
            maximum = left[:, column].abs().max()
            first_index = int(torch.nonzero(left[:, column].abs() == maximum, as_tuple=False)[0])
            assert left[first_index, column] > 0
    assert [item["seed"] for item in metadata] == [77, 78, 79, 80]
    for args in ((0, 4, 8), (16, 0, 8), (16, 4, 0), (4, 4, 8)):
        with pytest.raises(ValueError):
            random_orthonormal_bases(args[0], count=args[1], k=args[2])
