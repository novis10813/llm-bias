import csv
import json
from pathlib import Path

import pytest

from llm_bias.ten_k_change_data.cli import build_parser
from llm_bias.ten_k_change_data.pipeline import TenKChangeDataError, build_change_dataset, validate_change_dataset


def _filing(**overrides):
    row = {"cik": "1", "company": "ACME INC", "filing_type": "10-K", "filing_date": "2021-03-01", "period_of_report": "2020-12-31", "sic": "3571", "state_location": "CA", "state_of_inc": "DE"}
    row.update(overrides)
    return row


def _write(root: Path, name: str, row: dict) -> None:
    (root / name).write_text(json.dumps(row), encoding="utf-8")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _years(source: Path, **changes: object) -> None:
    for year in range(2018, 2023):
        _write(
            source,
            f"{year}.json",
            _filing(
                period_of_report=f"{year}-12-31",
                filing_date=f"{year + 1}-03-01",
                sic="3571",
                **(changes if year >= 2020 else {}),
            ),
        )


def test_build_writes_only_changed_items_for_event_window(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _years(source, company="NEW ACME INC", state_location="NY")
    manifest = build_change_dataset(source, tmp_path / "output")
    rows = _csv(tmp_path / "output" / "change_window_items.csv")
    assert list(rows[0]) == ["year", "cik", "item"]
    assert manifest["counts"]["change_events"] == 1
    assert manifest["counts"]["change_window_item_rows"] == 10
    expected = []
    for year in range(2018, 2023):
        expected.extend(
            [
                {"year": str(year), "cik": "1", "item": f"company={'ACME INC' if year < 2020 else 'NEW ACME INC'}"},
                {"year": str(year), "cik": "1", "item": f"state_location={'CA' if year < 2020 else 'NY'}"},
            ]
        )
    assert rows == expected
    assert validate_change_dataset(tmp_path / "output")["status"] == "passed"


def test_window_excludes_missing_years_and_empty_metadata_is_not_change(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _write(source, "2018.json", _filing(period_of_report="2018-12-31", filing_date="2019-03-01", company="OLD"))
    _write(source, "2020.json", _filing(period_of_report="2020-12-31", company="NEW"))
    _write(source, "2022.json", _filing(period_of_report="2022-12-31", filing_date="2023-03-01", company="NEW", state_location=""))
    manifest = build_change_dataset(source, tmp_path / "output")
    rows = _csv(tmp_path / "output" / "change_window_items.csv")
    assert manifest["counts"]["change_events"] == 1
    assert rows == [
        {"year": "2018", "sic": "3571", "item": "company=OLD"},
        {"year": "2020", "sic": "3571", "item": "company=NEW"},
        {"year": "2022", "sic": "3571", "item": "company=NEW"},
    ]
    event = json.loads((tmp_path / "output" / "change_events.jsonl").read_text())
    assert event["missing_window_years"] == [2019, 2021]


def test_latest_duplicate_is_canonical_and_no_changes_writes_header_only(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _write(source, "old.json", _filing(company="OLD", filing_date="2021-02-01"))
    _write(source, "new.json", _filing(company="NEW", filing_date="2021-03-01"))
    manifest = build_change_dataset(source, tmp_path / "output")
    assert manifest["counts"]["change_events"] == 0
    assert _csv(tmp_path / "output" / "change_window_items.csv") == []
    assert json.loads((tmp_path / "output" / "manifest.json").read_text())["counts"]["canonical_exclusions"] == 1
    assert validate_change_dataset(tmp_path / "output")["status"] == "passed"


def test_input_issues_are_audited_all_invalid_fails_and_strict_publishes(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _write(source, "good.json", _filing())
    (source / "empty.json").write_bytes(b"")
    build_change_dataset(source, tmp_path / "output")
    issue = json.loads((tmp_path / "output" / "input_issues.jsonl").read_text())
    assert issue["reason"] == "zero_byte_json"
    invalid = tmp_path / "invalid"; invalid.mkdir(); (invalid / "bad.json").write_bytes(b"")
    with pytest.raises(TenKChangeDataError, match="no valid"):
        build_change_dataset(invalid, tmp_path / "none")
    with pytest.raises(TenKChangeDataError, match="published dataset"):
        build_change_dataset(source, tmp_path / "strict", fail_on_input_issues=True)
    assert (tmp_path / "strict" / "change_window_items.csv").is_file()


def test_validator_detects_tampered_csv_and_cli_defaults(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _years(source, company="NEW")
    output = tmp_path / "output"; build_change_dataset(source, output)
    with (output / "change_window_items.csv").open("a", encoding="utf-8") as handle:
        handle.write("2020,9999,company=bad\n")
    with pytest.raises(TenKChangeDataError, match="hash mismatch"):
        validate_change_dataset(output)
    args = build_parser().parse_args(["build"])
    assert args.max_files is None and args.fail_on_input_issues is False
    assert args.output == "artifacts/ten_k_change_windows/v1"
