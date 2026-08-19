import json
from pathlib import Path

import pytest

from llm_bias.edgar_preparation.pipeline import (
    DATASET_SCHEMA_VERSION,
    PreparationError,
    analysis_text,
    clean_filings,
    find_entity_mentions,
    find_removed_blocks,
    normalize_text,
    validate_dataset,
)


def _filing(
    *,
    cik="123456",
    company="ACME TECHNOLOGIES INC /DE/",
    filing_date="2012-03-14",
    period_of_report="2012-03-13",
    **items,
):
    return {
        "cik": cik,
        "company": company,
        "filing_type": "8-K",
        "filing_date": filing_date,
        "period_of_report": period_of_report,
        "sic": "3571",
        "state_of_inc": "DE",
        "state_location": "CA",
        "fiscal_year_end": "1231",
        "filing_html_index": "https://example.test/index",
        "htm_filing_link": "https://example.test/filing",
        "complete_text_filing_link": "https://example.test/full",
        **items,
    }


def _write_filing(root: Path, name: str, record: dict):
    (root / name).write_text(json.dumps(record), encoding="utf-8")


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_normalization_and_boilerplate_removal_preserve_auditable_spans():
    raw = (
        "Item 2.02 Results of Operations and Financial Condition\r\n"
        "Acme\u00a0Technologies reported revenue of $12 million, up 20%.\n"
        "A copy of the release is furnished as Exhibit 99.1.\n"
        'The information contained herein shall not be deemed "filed" for '
        "purposes of the Securities Exchange Act."
    )
    normalized = normalize_text(raw)
    removed = find_removed_blocks(normalized, "2.02")
    analyzed = analysis_text(normalized, removed)

    assert "\r" not in normalized
    assert "Acme Technologies" in normalized
    assert {block.kind for block in removed} == {
        "item_heading",
        "exhibit_reference",
        "securities_act_boilerplate",
    }
    assert "reported revenue of $12 million, up 20%." in analyzed
    assert "Exhibit 99.1" not in analyzed
    for block in removed:
        assert normalized[block.start : block.end]


def test_entity_mentions_use_longest_non_overlapping_registrant_aliases():
    text = (
        "Acme Technologies, Inc. announced results. "
        "The Company expects Acme Technologies to grow."
    )
    mentions = find_entity_mentions(text, "ACME TECHNOLOGIES INC /DE/")

    assert [row["text"] for row in mentions] == [
        "Acme Technologies, Inc",
        "The Company",
        "Acme Technologies",
    ]
    assert all(
        text[row["start"] : row["end"]] == row["text"] for row in mentions
    )


def test_boilerplate_patterns_never_cross_a_long_substantive_section():
    text = (
        "Item 8.01 Other Events\n"
        "The information in this operational report describes "
        + ("substantive business facts and customer developments. " * 200)
        + "It is not incorporated by reference into a future registration statement."
    )
    normalized = normalize_text(text)
    removed = find_removed_blocks(normalized, "8.01")
    analyzed = analysis_text(normalized, removed)

    assert [block.kind for block in removed] == [
        "item_heading",
        "incorporation_by_reference",
    ]
    assert max(block.end - block.start for block in removed) < 2_500
    assert analyzed.count("substantive business facts") == 200


def test_clean_filings_writes_all_sections_and_candidate_statuses(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_filing(
        source,
        "123456_8K_2012_0000123456-12-000001.json",
        _filing(
            **{
                "item_2.02": (
                    "Item 2.02 Results of Operations and Financial Condition\n"
                    "Acme Technologies, Inc. reported revenue of $12 million, "
                    "up 20 percent from the prior-year period."
                ),
                "item_8.01": "Item 8.01 Other Events\nNot applicable.",
                "item_9.01": (
                    "Item 9.01 Financial Statements and Exhibits\n"
                    "Exhibit 99.1 Press release dated March 14, 2012."
                ),
            }
        ),
    )
    _write_filing(
        source,
        "654321_8K_2003_0000654321-03-000002.json",
        _filing(
            cik="654321",
            company="LEGACY FOODS CORP",
            filing_date="2003-07-01",
            period_of_report="2003-06-30",
            **{
                "item_5": (
                    "Item 5 Other Events and Regulation FD Disclosure\n"
                    "Legacy Foods Corporation announced a material change to "
                    "its distribution strategy across several regional markets."
                ),
                "item_7": "Item 7 Financial Statements and Exhibits\nExhibit 99.",
            },
        ),
    )

    manifest = clean_filings(source, output)
    validation = validate_dataset(output)
    filings = _jsonl(output / "filings.jsonl")
    sections = _jsonl(output / "sections.jsonl")

    assert manifest["counts"] == {"filings": 2, "sections": 5, "candidates": 2}
    assert validation["valid"] is True
    assert [row["source_file"] for row in filings] == sorted(
        row["source_file"] for row in filings
    )
    by_code = {(row["filing_id"], row["item_code"]): row for row in sections}
    modern = by_code[("8k:0000123456-12-000001", "2.02")]
    assert modern["schema_version"] == DATASET_SCHEMA_VERSION
    assert modern["event_family"] == "financial_results"
    assert modern["candidate_status"] == "candidate"
    assert modern["has_numeric_fact"] is True
    assert modern["entity_mentions"][0]["text"] == "Acme Technologies, Inc"
    assert by_code[("8k:0000123456-12-000001", "8.01")][
        "candidate_status"
    ] == "not_applicable"
    assert by_code[("8k:0000123456-12-000001", "9.01")][
        "candidate_status"
    ] == "supporting_only"
    legacy = by_code[("8k:0000654321-03-000002", "5")]
    assert legacy["item_schema"] == "legacy"
    assert legacy["candidate_status"] == "candidate"
    assert by_code[("8k:0000654321-03-000002", "7")][
        "candidate_status"
    ] == "supporting_only"


def test_clean_filings_refuses_existing_output(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with pytest.raises(PreparationError, match="already exists"):
        clean_filings(source, output)


def test_invalid_source_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_filing(
        source,
        "123456_8K_2012_0000123456-12-000001.json",
        _filing(item_2_02="bad item key"),
    )
    (source / "broken_8K_2012_0000123456-12-000002.json").write_text(
        "{broken", encoding="utf-8"
    )

    with pytest.raises(PreparationError, match="invalid JSON"):
        clean_filings(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_validator_detects_modified_artifact(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_filing(
        source,
        "123456_8K_2012_0000123456-12-000001.json",
        _filing(
            **{
                "item_8.01": (
                    "Item 8.01 Other Events\n"
                    "Acme Technologies announced a material operational event "
                    "affecting several business units and customer contracts."
                )
            }
        ),
    )
    clean_filings(source, output)
    with (output / "sections.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(PreparationError, match="hash mismatch"):
        validate_dataset(output)
