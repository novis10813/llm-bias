import csv
import json
from pathlib import Path

from llm_bias.prompt_analysis.input_inspection import inspect_input


def _return_row(**changes):
    row = {
        "cik": "1", "filename": "f.txt", "item": "item_5", "filing_date": "2024-01-02",
        "ticker": "AAA", "peer_ticker": "BBB", "system_prompt": "system",
        "prompt": "prompt", "counterfactual_prompt": "counter", "return_label": "bullish",
        "fwd_return_1d": "0.01", "extra": "keep",
    }
    row.update(changes)
    return row


def write_csv(path: Path, rows, *, bom=False, lineterminator="\n"):
    fields = list(rows[0])
    text_path = path
    with text_path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(rows)


def test_return_pairs_report_and_extra_columns_are_allowed(tmp_path):
    path = tmp_path / "returns.csv"
    write_csv(path, [_return_row(), _return_row(cik="2", filename="g.txt", fwd_return_1d="-0.03", return_label="very bearish")], bom=True, lineterminator="\r\n")
    report = inspect_input(path)
    assert report["schema"] == "return-pairs"
    assert report["rows"] == 2
    assert report["counts"]["pair_ids_unique"] is True
    assert report["counts"]["labels"] == {"bullish": 1, "very bearish": 1}
    assert not report["errors"]


def test_legacy_bom_and_quoted_multiline(tmp_path):
    path = tmp_path / "legacy.csv"
    write_csv(path, [{"Date": "2024-01-02", "prompt_with_context_sp500": "hello\nworld", "prompt_without_context_sp500": "bare"}], bom=True, lineterminator="\r\n")
    report = inspect_input(path)
    assert report["schema"] == "legacy-wide"
    assert report["counts"]["nonempty_prompt_cells"] == 2
    assert not report["errors"]


def test_bad_return_quality_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    write_csv(path, [_return_row(fwd_return_1d="nan"), _return_row()])
    report = inspect_input(path)
    assert report["errors"]
    assert any("non-finite" in error for error in report["errors"])
    assert any("duplicate pair IDs" in error for error in report["errors"])


def test_incomplete_or_ambiguous_schema_fails(tmp_path):
    path = tmp_path / "unknown.csv"
    write_csv(path, [{"Date": "2024-01-01", "prompt_with_context_bad": "x", "prompt_without_context_bad": "y", "cik": "1"}])
    report = inspect_input(path)
    assert report["schema"] == "legacy-wide"
    assert not report["errors"]

    ambiguous = tmp_path / "ambiguous.csv"
    row = _return_row(Date="2024-01-01", prompt_with_context_x="x")
    write_csv(ambiguous, [row])
    report = inspect_input(ambiguous)
    assert report["schema"] is None
    assert any("both" in error for error in report["errors"])
