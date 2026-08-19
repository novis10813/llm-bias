import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bias.counterfactual_data.annotation import (
    SEMANTIC_ENTITY_CLASSES,
    _serialize_extraction,
    annotate_events,
)
from llm_bias.counterfactual_data.pipeline import (
    ANNOTATOR_VERSION,
    CounterfactualDataError,
    build_company_history,
    build_context_template,
    build_pairs,
    deterministic_identifier_spans,
    event_excerpt,
    exposure_distance,
    promote_reviewed_annotations,
    resolve_semantic_outcome,
    short_company_name,
)
from llm_bias.counterfactual_patching.data import Pair, load_saved_pairs, save_pairs
from llm_bias.counterfactual_patching.experiment import expected_outcome_margin


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_short_company_name_is_conservative() -> None:
    assert short_company_name("ACME HOLDINGS INC.") == "ACME HOLDINGS"
    assert short_company_name("BANK INC") == "BANK INC"


def test_template_has_one_entity_and_separate_specificity_version() -> None:
    text = "Acme Inc. revenue increased 12% to $4 million on March 3, 2024."
    entities = [
        {
            "extraction_class": "registrant_name",
            "extraction_text": "Acme Inc.",
            "char_interval": {"start_pos": 0, "end_pos": 9},
        }
    ]
    template, _ = build_context_template(text, entities, "Acme Inc.")
    specificity, _ = build_context_template(
        text, entities, "Acme Inc.", redact_quasi_identifiers=False
    )
    assert template.count("{ENTITY}") == 1
    assert "Acme" not in template
    assert "[VALUE]" in template
    assert "12%" in specificity
    assert "$4 million" in specificity


def test_deterministic_identifiers_cover_structured_quasi_entities() -> None:
    text = (
        "Acme Inc. (NASDAQ: ACME) reported $4 million, 12.5%, and "
        "2 million shares on March 3, 2024 under CUSIP 123456789."
    )
    rows = deterministic_identifier_spans(text, "Acme Inc.")
    classes = {row["extraction_class"] for row in rows}
    assert {
        "registrant_name",
        "ticker",
        "exact_amount",
        "exact_percentage",
        "exact_share_count",
        "exact_calendar_date",
        "security_identifier",
    } <= classes
    for row in rows:
        interval = row["char_interval"]
        assert text[interval["start_pos"] : interval["end_pos"]] == row["extraction_text"]


def test_langextract_serialization_rejects_non_exact_grounding() -> None:
    @dataclass
    class Interval:
        start_pos: int
        end_pos: int

    extraction = SimpleNamespace(
        extraction_class="product",
        extraction_text="Widget",
        char_interval=Interval(start_pos=0, end_pos=6),
        alignment_status=None,
        attributes={},
    )
    with pytest.raises(CounterfactualDataError, match="non-exact"):
        _serialize_extraction(
            extraction, "langextract", "Gadget", SEMANTIC_ENTITY_CLASSES
        )


def test_langextract_serialization_canonicalizes_case_to_source() -> None:
    @dataclass
    class Interval:
        start_pos: int
        end_pos: int

    extraction = SimpleNamespace(
        extraction_class="product",
        extraction_text="widget",
        char_interval=Interval(start_pos=0, end_pos=6),
        alignment_status=None,
        attributes={},
    )
    row = _serialize_extraction(
        extraction, "langextract", "Widget", SEMANTIC_ENTITY_CLASSES
    )
    assert row["extraction_text"] == "Widget"
    assert row["model_extraction_text"] == "widget"
    assert row["grounding_canonicalized"] == "case_only"


def test_langextract_serialization_canonicalizes_high_threshold_fuzzy_span() -> None:
    @dataclass
    class Interval:
        start_pos: int
        end_pos: int

    extraction = SimpleNamespace(
        extraction_class="event_fact",
        extraction_text="revenue was $4 million",
        char_interval=Interval(start_pos=0, end_pos=21),
        alignment_status=SimpleNamespace(value="match_fuzzy"),
        attributes={"metric": "revenue", "direction": "increase"},
    )
    row = _serialize_extraction(
        extraction,
        "langextract",
        "revenue of $4 million",
        frozenset({"event_fact"}),
    )
    assert row["extraction_text"] == "revenue of $4 million"
    assert row["model_extraction_text"] == "revenue was $4 million"
    assert row["grounding_canonicalized"] == "high_threshold_fuzzy_alignment"


def test_annotation_resume_rejects_old_annotator_version(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "sampled_events.jsonl", [_draft()])
    _write_jsonl(
        tmp_path / "draft_annotations.jsonl",
        [{**_draft(), "annotator_version": "old", "annotation_status": "complete"}],
    )
    with pytest.raises(CounterfactualDataError, match="older annotator version"):
        annotate_events(tmp_path)


@pytest.mark.parametrize(
    ("metric", "direction", "expected"),
    [
        ("revenue", "increase", "positive"),
        ("revenue", "decrease", "negative"),
        ("net loss", "increase", "negative"),
        ("credit loss", "decrease", "positive"),
    ],
)
def test_semantic_outcome_ontology(metric: str, direction: str, expected: str) -> None:
    facts = [{"attributes": {"metric": metric, "direction": direction}}]
    assert resolve_semantic_outcome(facts) == expected


def test_mixed_semantic_outcome_is_rejected() -> None:
    facts = [
        {"attributes": {"metric": "revenue", "direction": "increase"}},
        {"attributes": {"metric": "loss", "direction": "increase"}},
    ]
    assert resolve_semantic_outcome(facts) is None


def test_event_excerpt_keeps_only_grounded_fact_sentences() -> None:
    text = "Boilerplate sentence. Revenue increased from last year. Legal ending."
    facts = [
        {
            "char_interval": {"start_pos": 22, "end_pos": 54},
            "attributes": {"metric": "revenue", "direction": "increase"},
        }
    ]
    assert event_excerpt(text, facts) == "Revenue increased from last year."


def test_company_history_keeps_filing_exposure_and_checks_indices(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "CIK,Company,Type,Date,SIC,filename\n"
        "1,ACME INC,10-K,2020-01-01,3571,a.txt\n"
        "1,ACME INC,8-K,2020-01-01,3571,b.txt\n",
        encoding="utf-8",
    )
    indices = tmp_path / "indices"
    indices.mkdir()
    (indices / "2020_QTR1.tsv").write_text(
        "1|ACME INC|8-K|2020-01-01|edgar/a.txt|edgar/a-index.html\n",
        encoding="utf-8",
    )
    manifest = build_company_history(metadata, tmp_path / "out", indices)
    row = json.loads((tmp_path / "out/company_history.jsonl").read_text().splitlines()[0])
    assert len(row["observations"]) == 2
    assert manifest["index_8k_metadata_signature_match_rate"] == 1.0


def test_exposure_distance_is_zero_for_equal_profiles() -> None:
    profile = {
        "prior_filing_count": 10,
        "prior_8k_count": 3,
        "trailing_3y_8k_count": 2,
        "years_since_first": 5,
    }
    assert exposure_distance(profile, profile) == 0


def _draft(content_id: str = "content:1") -> dict:
    text = "The Company reported revenue increased 12% from last year."
    return {
        "content_id": content_id,
        "annotator_version": ANNOTATOR_VERSION,
        "annotation_status": "complete",
        "dataset_status": "draft",
        "analysis_text": text,
        "company": "ACME INC",
        "cik": "1",
        "sic": "3571",
        "filing_date": "2020-06-01",
        "expected_outcome": "positive",
        "entities": [
            {
                "extraction_class": "registrant_alias",
                "extraction_text": "The Company",
                "char_interval": {"start_pos": 0, "end_pos": 11},
            }
        ],
        "event_facts": [
            {
                "extraction_class": "event_fact",
                "extraction_text": "revenue increased 12%",
                "char_interval": {"start_pos": 21, "end_pos": 42},
                "attributes": {"metric": "revenue", "direction": "increase"},
            }
        ],
    }


def _review(content_id: str = "content:1") -> dict:
    return {
        "content_id": content_id,
        "registrant_ticker_recall_ok": True,
        "other_entity_true_positive_count": 2,
        "other_entity_false_positive_count": 0,
        "other_entity_false_negative_count": 0,
        "grounded_spans_ok": True,
        "semantic_outcome_ok": True,
        "identity_leakage_found": False,
        "reviewer": "tester",
    }


def test_promotion_refuses_incomplete_review(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "draft_annotations.jsonl", [_draft()])
    review = _review()
    review["semantic_outcome_ok"] = None
    _write_jsonl(tmp_path / "review.jsonl", [review])
    with pytest.raises(CounterfactualDataError, match="incomplete"):
        promote_reviewed_annotations(
            tmp_path / "review.jsonl", tmp_path, minimum_reviews=1
        )


def test_promote_then_build_five_bidirectional_contrasts(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "draft_annotations.jsonl", [_draft()])
    _write_jsonl(tmp_path / "review.jsonl", [_review()])
    promoted = promote_reviewed_annotations(
        tmp_path / "review.jsonl", tmp_path, minimum_reviews=1
    )
    assert promoted["promoted_count"] == 1
    histories = [
        {
            "cik": "1",
            "observations": [
                {
                    "date": "2020-01-01",
                    "legal_name": "ACME INC",
                    "short_name": "ACME",
                    "form": "10-K",
                    "sic": "3571",
                    "source_id": "a",
                }
            ],
        },
        {
            "cik": "2",
            "observations": [
                {
                    "date": "2020-01-01",
                    "legal_name": "BETA CORP",
                    "short_name": "BETA",
                    "form": "10-K",
                    "sic": "3571",
                    "source_id": "b",
                }
            ],
        },
        {
            "cik": "3",
            "observations": [
                {
                    "date": "2020-01-01",
                    "legal_name": "GAMMA BANK",
                    "short_name": "GAMMA BANK",
                    "form": "10-K",
                    "sic": "6021",
                    "source_id": "c",
                }
            ],
        },
    ]
    _write_jsonl(tmp_path / "company_history.jsonl", histories)
    manifest = build_pairs(tmp_path)
    pairs = [
        json.loads(line)
        for line in (tmp_path / "pairs_unrendered.jsonl").read_text().splitlines()
    ]
    assert manifest["contrast_count"] == 5
    assert len(pairs) == 10
    assert {row["direction"] for row in pairs} == {"forward", "reverse"}
    for row in pairs:
        source = row["source_prompt"].replace(row["source_entity"], "{ENTITY}", 1)
        target = row["target_prompt"].replace(row["target_entity"], "{ENTITY}", 1)
        assert source == target
        assert row["expected_outcome"] == "positive"


def test_bias_pair_serialization_and_fixed_margin(tmp_path: Path) -> None:
    pair = Pair(
        pair_id="bias-1",
        category="real_vs_real",
        function="same_industry_matched_exposure",
        source_entity="Acme",
        target_entity="Beta",
        source_prompt="Acme improved. Answer:",
        target_prompt="Beta improved. Answer:",
        source_answer="negative",
        target_answer="positive",
        source_entity_start=1,
        source_entity_end=2,
        target_entity_start=1,
        target_entity_end=2,
        source_entity_token=10,
        target_entity_token=11,
        answer_source_token=12,
        answer_target_token=13,
        task_type="entity_bias",
        expected_outcome="negative",
        margin_definition="logit(positive)-logit(negative)",
    )
    path = tmp_path / "pairs.jsonl"
    save_pairs([pair], path)
    loaded = load_saved_pairs(path)[0]
    assert loaded.task_type == "entity_bias"
    assert loaded.expected_outcome == "negative"
    assert expected_outcome_margin(2.5, "negative") == -2.5
