import json
from pathlib import Path

import pytest

from llm_bias.entity_cell.confirmation import (
    PRIMARY_FAMILY,
    default_confirmation_config,
    evaluate_confirmation,
    evaluate_confirmation_artifact,
    validate_confirmation_config,
)
from llm_bias.core.artifacts.io import write_jsonl


def _config(minimum=2):
    return default_confirmation_config(selected_heads=[(11, 2)], minimum_eligible_tickers=minimum)


def _records(count=3, *, split="calibration", value=1.0):
    rows = []
    for index in range(count):
        row = {
            "artifact_type": "entity_cell_confirmation_record",
            "ticker": f"T{index}", "split": split, "eligible": True,
            "e1": {"eligible": True}, "e2": {"selected_heads": [[11, 2]]},
            "e3": {"alpha": -3.0, "beta": 0.0},
            "direction_consistency": True,
            "metrics": {name: (value + (index % 3) * 0.01) for name in PRIMARY_FAMILY} | {"evidence_gap": value},
        }
        rows.append(row)
    return rows


def test_confirmation_schema_freezes_layers_doses_controls_and_holm_family():
    config = _config()
    assert validate_confirmation_config(config)["statistics"]["holm_family"] == list(PRIMARY_FAMILY)
    args = __import__("llm_bias.entity_cell.cli", fromlist=["build_parser"]).build_parser().parse_args(["analyze", "--experiment", "discovery", "--e1-run-root", "e1", "--e2-run-root", "e2", "--e3-run-root", "e3", "--output", "out"])
    assert args.experiment == "discovery" and args.output == Path("out")
    broken = json.loads(json.dumps(config))
    broken["e3_primary"]["upstream_alpha"] = -2.0
    with pytest.raises(ValueError, match="dose"):
        validate_confirmation_config(broken)


def test_confirmation_statistics_and_holm_are_deterministic():
    result = evaluate_confirmation(_records(8), _config(), split="calibration")
    again = evaluate_confirmation(_records(8), _config(), split="calibration")
    assert result == again
    assert result["success"]
    assert all("bootstrap_95_ci" in result["statistics"][name] for name in PRIMARY_FAMILY)
    assert result["test_authorized"] is True


def test_failed_calibration_blocks_test_and_test_never_authorizes():
    failed = evaluate_confirmation(_records(2, value=0.0), _config(), split="calibration")
    assert failed["success"] is False and failed["test_authorized"] is False
    blocked = evaluate_confirmation(_records(3, split="test"), _config(), split="test", calibration_result=failed)
    assert blocked["blocked"] is True and blocked["success"] is False and blocked["test_authorized"] is False
    with pytest.raises(ValueError):
        evaluate_confirmation(_records(3), _config(), split="discovery")
    successful = evaluate_confirmation(_records(8), _config(), split="calibration")
    authorized = evaluate_confirmation(_records(8, split="test"), _config(), split="test", calibration_result=successful)
    assert authorized["success"] is True and authorized["test_authorized"] is False


def test_confirmation_artifact_is_compact_atomic_and_immutable(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    config = tmp_path / "config.json"
    output = tmp_path / "confirmation.json"
    write_jsonl(records, _records(), overwrite=False)
    config.write_text(json.dumps(_config()), encoding="utf-8")
    result = evaluate_confirmation_artifact(records, config, output, split="calibration")
    assert json.loads(output.read_text())["parent_sha256"]
    assert result["raw_runtime_payloads"] is False
    with pytest.raises(FileExistsError):
        evaluate_confirmation_artifact(records, config, output, split="calibration")
