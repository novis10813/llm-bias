import json

import pytest

from llm_bias.prompt_analysis.return_visualization import (
    build_paired_uncertainty_delta_rows,
    build_prediction_flip_rows,
    visualize_return_predictions,
)


LABELS = ["very bearish", "bearish", "neutral", "bullish", "very bullish"]


def _attribution_rows():
    rows = []
    for pair_id, original, counterfactual in (
        ("p1", "very bearish", "bearish"),
        ("p2", "neutral", "neutral"),
        ("p3", "bearish", "bearish"),
        ("p4", "bullish", "bullish"),
        ("p5", "very bullish", "very bullish"),
    ):
        for condition, predicted in (("original", original), ("counterfactual", counterfactual)):
            rows.append(
                {
                    "artifact_type": "generated_outputs",
                    "pair_id": pair_id,
                    "condition": condition,
                    "target_label": original if pair_id == "p1" else "neutral",
                    "generated_text": json.dumps({"label": predicted, "confidence": 80 if condition == "original" else 60}),
                    "predicted_label": predicted,
                    "predicted_confidence": 80 if condition == "original" else 60,
                    "parse_status": "valid",
                    "ticker": "AAA",
                    "peer_ticker": "BBB",
                    "filing_date": "2024-01-02",
                }
            )
    rows[-1]["parse_status"] = "invalid"
    rows[-1]["generated_text"] = "not valid JSON"
    return rows


def _uncertainty_rows():
    rows = []
    for pair_id, entropy in (("p1", 1.0), ("p2", 1.5)):
        for condition, offset in (("original", 0.0), ("counterfactual", 0.25)):
            rows.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "layers": [
                        {"layer": 0, "is_output": False, "entropy_nats": 9},
                        {
                            "layer": 31,
                            "is_output": True,
                            "entropy_nats": entropy + offset,
                            "effective_temperature": 0.5 + offset,
                        },
                    ],
                }
            )
    return rows


def test_prediction_flips_pair_only_by_pair_id_and_retains_invalid_pairs():
    rows = build_prediction_flip_rows(_attribution_rows())
    assert len(rows) == 5
    assert rows[0]["pair_id"] == "p1"
    assert rows[0]["prediction_flip"] is True
    assert sum(row["prediction_flip"] for row in rows) == 1
    assert rows[-1]["valid_counterfactual"] is False
    assert rows[-1]["prediction_flip"] is False


def test_uncertainty_uses_final_output_layer_and_pairs_by_pair_id():
    rows = build_paired_uncertainty_delta_rows(_uncertainty_rows())
    assert len(rows) == 2
    assert rows[0]["entropy_delta_nats"] == pytest.approx(0.25)
    assert rows[1]["effective_temperature_delta"] == pytest.approx(0.25)


def test_visualization_writes_csv_metadata_and_all_pngs(tmp_path):
    attribution = tmp_path / "attribution.jsonl"
    uncertainty = tmp_path / "uncertainty.jsonl"
    attribution.write_text("\n".join(json.dumps(row) for row in _attribution_rows()) + "\n", encoding="utf-8")
    uncertainty.write_text("\n".join(json.dumps(row) for row in _uncertainty_rows()) + "\n", encoding="utf-8")
    output = visualize_return_predictions(
        forward_path=attribution,
        uncertainty_path=uncertainty,
        output_dir=tmp_path / "out",
    )
    assert len(list(output.glob("*.png"))) == 5
    assert len((output / "return_prediction_records.csv").read_text().splitlines()) == 11
    assert len((output / "return_prediction_flips.csv").read_text().splitlines()) == 6
    assert len((output / "return_paired_uncertainty_delta.csv").read_text().splitlines()) == 3
    metadata = json.loads((output / "return_visualization_metadata.json").read_text())
    assert metadata["records"]["invalid_predictions"] == 1
    assert metadata["records"]["uncertainty_pairs"] == 2
    assert metadata["artifact_contract"]["pairing_key"] == "pair_id"


def test_incomplete_pairs_are_reported_not_silently_collapsed(tmp_path):
    rows = _attribution_rows()[:-1]
    attribution = tmp_path / "attribution.jsonl"
    uncertainty = tmp_path / "uncertainty.jsonl"
    attribution.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    uncertainty.write_text("\n".join(json.dumps(row) for row in _uncertainty_rows()) + "\n", encoding="utf-8")
    output = visualize_return_predictions(forward_path=attribution, uncertainty_path=uncertainty, output_dir=tmp_path / "out")
    metadata = json.loads((output / "return_visualization_metadata.json").read_text())
    assert metadata["missing_or_incomplete"]["prediction"]["missing_condition"] == 1
