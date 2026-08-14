from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_bias.core.artifact_manifest import RunManifest
from llm_bias.synthetic_entity_bias.spec import LABEL_HASH, TEMPLATE_HASH, TEMPLATES
from llm_bias.synthetic_entity_bias.visualization import (
    ArtifactContractError,
    validate_run,
    visualize_run,
)
from llm_bias.synthetic_entity_bias.visualization.chart_specs import build_chart_specs
from llm_bias.synthetic_entity_bias.visualization.contract import (
    BASELINE_FIELDS,
    ENTITY_POOL_FIELDS,
    LOCALIZATION_FIELDS,
    RESULT_FIELDS,
)
from llm_bias.synthetic_entity_bias.visualization.dashboard import _safe_json, render_dashboard
from llm_bias.synthetic_entity_bias.visualization.theme import (
    DARK_COLORS,
    LIGHT_COLORS,
    template_style,
)


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _probability(label: int) -> list[float]:
    values = [0.0] * 9
    values[label] = 1.0
    return values


def _fixture(tmp_path: Path) -> Path:
    manifest = RunManifest.new("org/model", "dataset", "run", artifact_root=tmp_path)
    root = manifest.run_directory
    root.mkdir(parents=True)
    manifest.start()
    entities = []
    for ticker, split, tier in (("AAA", "train", "S&P 500"), ("BBB", "eval", "Russell 2000")):
        entities.append({
            "ticker": ticker, "company_name": f"{ticker} Corp", "latest_year": "2025",
            "years": "2024|2025", "memberships": tier, "membership_years": f"{tier}:2025",
            "sectors": "Technology", "familiarity_tier": tier, "source_row_count": "2",
            "anomalies": "", "split": split,
        })
    config = {
        "model_path": "org/model", "model_config_sha256": "a" * 64,
        "tokenizer_class": "FakeTokenizer", "tokenizer_name_or_path": "org/model",
        "transformers_version": "test", "chat_template_sha256": "b" * 64,
        "lens_binary_sha256": "c" * 64, "lens_metadata_sha256": "d" * 64,
        "input_hashes": {"input.csv": "e" * 64},
        "label_token_ids": {str(i): i for i in range(9)},
        "label_decoded": {str(i): str(i) for i in range(9)},
        "score_mapping": {str(i): i - 4 for i in range(9)}, "templates": TEMPLATES,
        "scoring_instruction": "fixed", "pool_count": 2,
        "tier_counts": {"S&P 500": 1, "Russell 2000": 1},
        "split_counts": {"train": 1, "eval": 1}, "anomaly_count": 0, "seed": 0,
        "split": "stable", "template_hash": TEMPLATE_HASH, "label_hash": LABEL_HASH,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenization = {"label_token_ids": config["label_token_ids"], "decoded": config["label_decoded"], "n_prompts": 9, "anomalies": []}
    (root / "tokenization_validation.json").write_text(json.dumps(tokenization), encoding="utf-8")
    _write_csv(root / "entity_pool.csv", ENTITY_POOL_FIELDS, entities)
    baselines, results = [], []
    template_labels = {"negative": 3, "positive": 5, "neutral": 4}
    for template, label in template_labels.items():
        probabilities = _probability(label)
        expected = label - 4
        baselines.append({"template": template, "entity": "The company", "probabilities": json.dumps(probabilities), "expected_score": expected, "entropy_nats": 0.0, "effective_temperature": 1.0})
        for index, entity in enumerate(entities):
            entity_label = min(8, label + index)
            entity_probabilities = _probability(entity_label)
            results.append({
                "ticker": entity["ticker"], "company_name": entity["company_name"],
                "template": template, "split": entity["split"], "familiarity_tier": entity["familiarity_tier"],
                "entity_probabilities": json.dumps(entity_probabilities), "baseline_probabilities": json.dumps(probabilities),
                "entity_expected_score": entity_label - 4, "baseline_expected_score": expected,
                "entity_entropy_nats": 0.0, "baseline_entropy_nats": 0.0,
                "entity_effective_temperature": 1.0, "baseline_effective_temperature": 1.0,
                "delta_expected_score": entity_label - label, "entity_span_start": 1,
                "entity_span_end": 2, "answer_position": 3,
            })
    _write_csv(root / "no_entity_baselines.csv", BASELINE_FIELDS, baselines)
    _write_csv(root / "raw_entity_template_results.csv", RESULT_FIELDS, results)
    localization = []
    for layer in range(2):
        for template in template_labels:
            localization.append({
                "layer": layer, "template": template, "mean_cosine": 0.1 + layer,
                "pearson_r": 0.2, "spearman_r": 0.3, "linear_r2": 0.4,
                "n_train": 1, "n_eval": 1, "q25": -0.1, "q75": 0.1,
                "n_high": 1, "n_low": 1, "high_ids_sha256": "f" * 64,
                "low_ids_sha256": "1" * 64, "fit_split": "train",
                "direction_sha256": "2" * 64, "statistic_flag": "ok",
            })
    _write_csv(root / "layer_template_localization.csv", LOCALIZATION_FIELDS, localization)
    for stage, count in (("preflight", 9), ("baseline", 3), ("metric", 6), ("localization", 6)):
        manifest.start_stage(stage).finish_stage(stage, record_count=count)
    for artifact_type, filename, stage, count in (
        ("config", "config.json", "preflight", None),
        ("tokenization_validation", "tokenization_validation.json", "preflight", None),
        ("entity_pool", "entity_pool.csv", "preflight", 2),
        ("no_entity_baselines", "no_entity_baselines.csv", "baseline", 3),
        ("raw_entity_template_results", "raw_entity_template_results.csv", "metric", 6),
        ("layer_template_localization", "layer_template_localization.csv", "localization", 6),
    ):
        manifest.register_artifact(root / filename, artifact_type=artifact_type, stage=stage, record_count=count)
    manifest.complete().save()
    return root


def test_visualize_run_writes_auditable_bundle(tmp_path):
    root = _fixture(tmp_path)
    run = validate_run(root)
    assert len(run.results) == 6
    output = visualize_run(root)
    assert not (output / "dashboard.html").exists()
    assert len(list((output / "figures").glob("*.png"))) == 8
    assert len(list((output / "figures").glob("*.svg"))) == 8
    assert len(list((output / "figures").glob("*.pdf"))) == 8
    assert len(list((output / "captions").glob("*.md"))) == 8
    assert len(list((output / "tables").glob("*.csv"))) == 9
    for path in (output / "figures").glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF")
    metadata = json.loads((output / "visualization_metadata.json").read_text())
    assert metadata["validation"]["model_loading_performed"] is False
    assert metadata["records"]["entity_template_results"] == 6
    assert metadata["schema_version"] == 4
    assert metadata["render_profile"] == "paper_first"
    assert metadata["paper_exports"]["formats"] == ["png", "svg", "pdf"]
    assert metadata["dashboard"]["requested"] is False
    assert metadata["accessibility"]["self_explained_static_figures"] is True
    assert metadata["chart_specs"]["count"] == 8
    assert {row["path"] for row in metadata["outputs"]} >= {
        "tables/template_summary.csv",
        "figures/entity_effect_distribution.pdf",
        "captions/entity_effect_distribution.md",
    }
    caption = (output / "captions" / "entity_effect_distribution.md").read_text()
    assert "Figure caption" in caption
    assert "ΔE" in caption
    assert "standalone causal effect" in caption


def test_visualize_optionally_writes_interactive_dashboard(tmp_path):
    output = visualize_run(_fixture(tmp_path), with_dashboard=True)
    dashboard = (output / "dashboard.html").read_text()
    assert "https://" not in dashboard
    assert dashboard.count('class="chart"') == 8
    assert 'href="figures/entity_effect_distribution.pdf"' in dashboard
    assert "forced-colors:active" in dashboard
    assert "theme-toggle" in dashboard
    assert "target.innerHTML" not in dashboard
    assert "textContent" in dashboard


def test_visualize_refuses_existing_bundle_without_explicit_replacement(tmp_path):
    root = _fixture(tmp_path)
    visualize_run(root)
    with pytest.raises(FileExistsError):
        visualize_run(root)
    assert visualize_run(root, replace_existing=True).is_dir()


def test_validate_run_rejects_tampered_artifact(tmp_path):
    root = _fixture(tmp_path)
    with (root / "entity_pool.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ArtifactContractError, match="SHA-256"):
        validate_run(root)


def test_validate_run_rejects_non_complete_manifest(tmp_path):
    root = _fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="complete"):
        validate_run(root)


def test_theme_identity_is_fixed_and_unknown_templates_fail():
    assert LIGHT_COLORS == {
        "negative": "#2a78d6",
        "positive": "#eb6834",
        "neutral": "#1baf7a",
    }
    assert DARK_COLORS == {
        "negative": "#3987e5",
        "positive": "#d95926",
        "neutral": "#199e70",
    }
    assert template_style("positive")["marker"] == "s"
    with pytest.raises(ValueError, match="unknown template"):
        template_style("other")


def test_chart_specs_have_tooltips_tables_and_secondary_encodings(tmp_path):
    run = validate_run(_fixture(tmp_path))
    specs = build_chart_specs(run)
    assert len(specs) == 8
    for spec in specs:
        assert spec["description"]
        assert spec["title"]
        assert spec["subtitle"]
        assert spec["figure_note"]
        assert spec["panels"]
        assert spec["supporting_table"]
        assert spec["tooltip_fields"]
        assert spec["table"]
        assert len({panel["label"] for panel in spec["panels"]}) == len(spec["panels"])
        for series in spec["series"]:
            assert series["style"]["marker"]
            assert series["style"]["dash"]
            assert series["style"]["hatch"]


def test_dashboard_serializes_untrusted_text_without_executable_markup():
    payload = _safe_json({"company_name": '</script><script id="injected">alert(1)</script>'})
    assert '<script id="injected">' not in payload
    assert "\\u003c/script\\u003e" in payload
