import csv
import json

import pytest

from llm_bias.prompt_analysis.return_evaluation import LABELS, evaluate_return_predictions


def _row(pair, condition, label="bullish", confidence=80, status="valid"):
    return {
        "pair_id": pair, "filing_date": "2026-01-01", "ticker": "AAA",
        "peer_ticker": "BBB", "condition": condition, "target_label": "bullish",
        "fwd_return_1d": 0.01, "generated_text": "{}", "predicted_label": label,
        "predicted_confidence": confidence, "parse_status": status,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluator_writes_metrics_and_retains_invalid_predictions(tmp_path):
    source = tmp_path / "attribution.jsonl"
    _write(source, [_row("p1", "original", "bullish", 80), _row("p1", "counterfactual", "bearish", 70), _row("p2", "original", "neutral", 101), _row("p2", "counterfactual", "neutral", 60, "invalid")])
    output = evaluate_return_predictions(source, tmp_path / "out", settings={"seed": 1})
    assert output.joinpath("prediction_samples.csv").is_file()
    with output.joinpath("prediction_samples.csv").open() as handle:
        samples = list(csv.DictReader(handle))
    assert len(samples) == 4
    assert sum(row["prediction_valid"] == "False" for row in samples) == 2
    summary = list(csv.DictReader(output.joinpath("prediction_summary.csv").open()))[0]
    assert summary["accuracy"] == "0.5"
    assert summary["invalid_predictions"] == "2"
    pairs = list(csv.DictReader(output.joinpath("pair_flip_summary.csv").open()))
    assert pairs[0]["flip"] == "True" and pairs[0]["valid_pair"] == "True"
    assert pairs[1]["valid_pair"] == "False"
    metadata = json.loads(output.joinpath("metadata.json").read_text())
    assert metadata["input_sha256"]
    assert metadata["settings"] == {"seed": 1}


@pytest.mark.parametrize("field,value", [("predicted_confidence", 1.5), ("predicted_confidence", -1), ("predicted_confidence", 101), ("predicted_label", "unknown")])
def test_invalid_prediction_is_not_batch_failure(tmp_path, field, value):
    source = tmp_path / "input.jsonl"
    row = _row("p", "original"); row[field] = value
    other = _row("p", "counterfactual")
    _write(source, [row, other])
    output = evaluate_return_predictions(source, tmp_path / "out")
    assert json.loads(output.joinpath("metadata.json").read_text())["invalid_predictions"] == 1


def test_duplicate_and_missing_condition_fail_early(tmp_path):
    source = tmp_path / "input.jsonl"
    _write(source, [_row("p", "original"), _row("p", "original")])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_return_predictions(source, tmp_path / "out")
    _write(source, [_row("p", "original")])
    with pytest.raises(ValueError, match="original and counterfactual"):
        evaluate_return_predictions(source, tmp_path / "out2")


def test_all_five_labels_are_declared():
    assert len(LABELS) == 5
