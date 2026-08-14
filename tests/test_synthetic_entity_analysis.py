from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from llm_bias.synthetic_entity_bias.analysis import analyze_run
from llm_bias.synthetic_entity_bias.analysis.statistics import (
    descriptive_statistics,
    holm_adjust,
    one_sample_tests,
)
from test_synthetic_entity_visualization import _fixture


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyze_run_writes_statistical_and_diagnostic_tables(tmp_path):
    root = _fixture(tmp_path)
    manifest_hash = _sha256(root / "manifest.json")
    output = analyze_run(root)

    assert {path.name for path in output.glob("*.csv")} == {
        "template_statistics.csv",
        "template_pairwise_tests.csv",
        "familiarity_tier_statistics.csv",
        "familiarity_tier_pairwise_tests.csv",
        "sector_statistics.csv",
        "localization_statistics.csv",
        "baseline_statistics.csv",
        "entity_distribution_diagnostics.csv",
        "temperature_null_diagnostics.csv",
        "localization_transition_diagnostics.csv",
    }
    template = _rows(output / "template_statistics.csv")
    from llm_bias.synthetic_entity_bias.visualization.contract import TEMPLATE_ORDER
    assert [row["template"] for row in template] == list(TEMPLATE_ORDER)
    assert all(row["n"] == "2" for row in template)
    assert all(float(row["mean"]) == pytest.approx(0.5) for row in template)
    assert all(float(row["median"]) == pytest.approx(0.5) for row in template)

    paired = _rows(output / "template_pairwise_tests.csv")
    assert len(paired) == len(TEMPLATE_ORDER) * (len(TEMPLATE_ORDER) - 1) // 2
    assert all(row["n_paired"] == "2" for row in paired)
    assert all(float(row["mean_paired_difference"]) == pytest.approx(0.0) for row in paired)

    sectors = _rows(output / "sector_statistics.csv")
    assert all(row["included_in_primary_analysis"] == "False" for row in sectors)
    assert all(row["t_status"] == "excluded_small_group" for row in sectors)
    assert all(row["t_p_value"] == "" for row in sectors)

    localization = _rows(output / "localization_statistics.csv")
    assert len(localization) == 4 * len(TEMPLATE_ORDER)
    assert {row["metric"] for row in localization} == {
        "mean_cosine", "pearson_r", "spearman_r", "linear_r2"
    }

    baselines = _rows(output / "baseline_statistics.csv")
    assert len(baselines) == len(TEMPLATE_ORDER)
    assert all(float(row["mean_movement_from_baseline"]) == pytest.approx(0.5) for row in baselines)

    diagnostics = _rows(output / "entity_distribution_diagnostics.csv")
    assert all(float(row["mean"]) == pytest.approx(0.5) for row in diagnostics)
    assert all(row["positive_tail_count_ge_0_5"] == "1" for row in diagnostics)

    null = _rows(output / "temperature_null_diagnostics.csv")
    assert len(null) == len(TEMPLATE_ORDER)
    assert all(row["fit_status"] == "ok" for row in null)
    assert all(float(row["temperature_bounds_lower"]) == 0.25 for row in null)

    transitions = _rows(output / "localization_transition_diagnostics.csv")
    assert len(transitions) == 4 * len(TEMPLATE_ORDER)
    cosine = next(row for row in transitions if row["template"] == "negative" and row["metric"] == "mean_cosine")
    assert cosine["sign_change_count"] == "0"
    assert float(cosine["maximum_jump_delta"]) == pytest.approx(1.0)
    assert _sha256(root / "manifest.json") == manifest_hash


def test_analyze_refuses_existing_output_without_replacement(tmp_path):
    root = _fixture(tmp_path)
    analyze_run(root)
    with pytest.raises(FileExistsError):
        analyze_run(root)
    assert analyze_run(root, replace_existing=True).is_dir()


def test_descriptive_statistics_and_bootstrap_inputs_are_not_silently_zeroed():
    summary = descriptive_statistics([0.0])
    assert summary["sample_std"] is None
    assert summary["sem"] is None
    tests = one_sample_tests([0.0])
    assert tests["t_p_value"] is None
    assert tests["t_status"] == "insufficient_sample"
    assert tests["wilcoxon_p_value"] is None
    assert tests["wilcoxon_status"] == "degenerate"


def test_holm_adjustment_is_step_down_and_preserves_missing_values():
    adjusted = holm_adjust([0.01, 0.04, 0.03, None])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, None])
