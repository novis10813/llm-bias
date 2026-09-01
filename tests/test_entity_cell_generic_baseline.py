import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_entity_cell_generic_baseline.py"
_spec = importlib.util.spec_from_file_location("generic_baseline", SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_parse_response_accepts_only_candidate_objects():
    parsed = _mod.parse_response('{"candidates":[{"category":"purpose","prompt":"The purpose is to"},"noise"]}')
    assert parsed == [{"category": "purpose", "prompt": "The purpose is to"}]


def test_validate_candidate_rejects_required_bad_forms():
    base = {"category": "purpose", "prompt": "The purpose of this activity is to"}
    cases = {
        "The purpose of this activity is complete.": "question_or_completed_punctuation",
        "What is the purpose of this activity?": "question_or_completed_punctuation",
        "The purpose of <thing> is to": "placeholder_or_brackets",
        "The purpose of this activity is to\nhelp": "multi_paragraph_or_line",
        "assistant: provide the purpose of this": "dialogue_role",
        "Visit https://example.com to learn": "url_or_email",
        "The company purpose is to": "forbidden_domain_or_instruction_term",
        "The purpose of Alice is to": "named_entity_screen",
        "The purpose is": "length_bounds",
    }
    for prompt, reason in cases.items():
        candidate = {**base, "prompt": prompt}
        accepted, observed, _ = _mod.validate_candidate(candidate)
        assert not accepted
        assert observed == reason


def test_validate_candidate_matches_full_identities_without_bare_ticker_false_positives():
    accepted, reason, _ = _mod.validate_candidate(
        {"category": "purpose", "prompt": "The main goal of this activity is to"},
        forbidden_identities=["A", "The A Company"],
    )
    assert accepted
    assert reason is None
    rejected, reason, _ = _mod.validate_candidate(
        {"category": "purpose", "prompt": "The main goal of A is to"},
        forbidden_identities=["A"],
    )
    assert not rejected
    assert reason == "baseline_identity"


def test_normalization_and_response_shape_fail_closed():
    assert _mod.normalize_prompt("  The\u00a0goal  is to ") == "the goal is to"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _mod.parse_response("not json")
    with pytest.raises(ValueError, match="candidates"):
        _mod.parse_response("{}")


def test_committed_dataset_has_expected_shape_and_hashes():
    path = Path("data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl")
    assert path.is_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 399
    assert len({row["prompt_id"] for row in rows}) == 399
    for row in rows:
        assert row["baseline_identity"] == _mod.ADAPTED_IDENTITY
        assert row["prompt_sha256"] == _mod.sha256_bytes(row["prompt"].encode())
