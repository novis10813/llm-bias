"""Synthetic-only review metadata; these fixtures are not research evidence."""

from dataclasses import fields, replace
from hashlib import sha256

import pytest

from llm_bias.core.population import PopulationMember
from llm_bias.core.stance_evidence import (
    EvidenceItem,
    EvidenceOrderContrast,
    EvidencePair,
    validate_evidence_item,
    validate_evidence_pair,
    validate_evidence_pairs,
)


# Invented bytes ONLY for testing provenance and the numeric-review contract.
_SOURCE_BYTES = (
    b"SYNTHETIC TEST TABLE, NOT AN EXTERNAL RESEARCH SOURCE\n"
    b"cash: 10 -> 12; debt: 8 -> 6; cash: 10 -> 8; debt: 8 -> 10\n"
)


@pytest.fixture
def members():
    return (
        PopulationMember("AAA", "Fixture Alpha", "Technology", "fixture-a"),
        PopulationMember("BBB", "Fixture Beta", "Unspecified", "fixture-b"),
    )


@pytest.fixture
def items():
    descriptions = (
        ("p1", "+", "Fixture cash rose from 10 to 12 units (+20%).",
         "Synthetic audit: (12 - 10) / 10 = 0.20; units and base agree."),
        ("p2", "+", "Fixture debt fell from 8 to 6 units (-25%).",
         "Synthetic audit: (6 - 8) / 8 = -0.25; units and base agree."),
        ("n1", "-", "Fixture cash fell from 10 to 8 units (-20%).",
         "Synthetic audit: (8 - 10) / 10 = -0.20; units and base agree."),
        ("n2", "-", "Fixture debt rose from 8 to 10 units (+25%).",
         "Synthetic audit: (10 - 8) / 8 = 0.25; units and base agree."),
    )
    return {
        item_id: EvidenceItem(
            item_id=item_id, text=text, polarity=polarity,
            source="synthetic test table; not acquired external evidence",
            source_sha256=sha256(_SOURCE_BYTES).hexdigest(),
            content_sha256=sha256(text.encode("utf-8")).hexdigest(),
            review_id=f"synthetic-review-{item_id}",
            strength="fixture-moderate: synthetic test label, not a research rating",
            numerical_review=audit,
            source_quality_review="Synthetic fixture only; no research-quality approval.",
        )
        for item_id, polarity, text, audit in descriptions
    }


@pytest.fixture
def pairs(members, items):
    slots = {"++": ("p1", "p2"), "+-": ("p1", "n1"),
             "-+": ("n1", "p1"), "--": ("n1", "n2")}
    return tuple(
        EvidencePair(member.ticker, "trial-1", condition, items[a], items[b])
        for member in members for condition, (a, b) in slots.items()
    )


def _validate(members, pairs, **kwargs):
    return validate_evidence_pairs(members, ["trial-1"], pairs, **kwargs)


def test_complete_shared_pairs_are_sorted_and_order_independent(members, pairs):
    contrasts = [EvidenceOrderContrast("trial-1", "+-", "-+")]
    result = _validate(members, reversed(pairs), order_contrasts=contrasts)
    assert result == tuple(sorted(pairs, key=lambda p: (p.ticker, p.trial_id, p.condition)))
    assert len(result) == 8
    assert _validate(reversed(members), result, order_contrasts=contrasts) == result


def test_source_hash_is_supplied_byte_provenance_not_a_quality_assertion(items):
    item = items["p1"]
    assert item.source_sha256 == sha256(_SOURCE_BYTES).hexdigest()
    assert item.content_sha256 == sha256(item.text.encode("utf-8")).hexdigest()
    assert item.source_sha256 != item.content_sha256
    assert "0.20" in item.numerical_review
    # The validator cannot approve quality or verify bytes it has not received.
    unverified = replace(item, source_sha256="a" * 64)
    assert validate_evidence_item(unverified) == unverified


def test_exact_utf8_content_hash_preserves_whitespace_and_unicode(items):
    text = "  Fixture café: 10 → 12 units.\n"
    item = replace(items["p1"], text=text, content_sha256=sha256(text.encode()).hexdigest())
    assert validate_evidence_item(item).text == text
    with pytest.raises(ValueError, match="content_sha256"):
        validate_evidence_item(replace(item, content_sha256=sha256(text.strip().encode()).hexdigest()))


@pytest.mark.parametrize("field", [f.name for f in fields(EvidenceItem)])
@pytest.mark.parametrize("value", ["", " \n\t", None, 7])
def test_every_item_field_must_be_a_nonblank_string(items, field, value):
    with pytest.raises(ValueError, match=field):
        validate_evidence_item(replace(items["p1"], **{field: value}))


@pytest.mark.parametrize("field", ["source_sha256", "content_sha256"])
@pytest.mark.parametrize("value", ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n"])
def test_hashes_are_exactly_lowercase_hex(items, field, value):
    with pytest.raises(ValueError, match=field):
        validate_evidence_item(replace(items["p1"], **{field: value}))


def test_recomputed_content_digest_and_polarity_are_required(items):
    with pytest.raises(ValueError, match="content_sha256"):
        validate_evidence_item(replace(items["p1"], text="different exact bytes"))
    with pytest.raises(ValueError, match="polarity"):
        validate_evidence_item(replace(items["p1"], polarity="positive"))


def test_slots_have_independent_item_ids_and_matching_polarities(pairs, items):
    assert validate_evidence_pair(pairs[0]) == pairs[0]
    with pytest.raises(ValueError, match="independent"):
        validate_evidence_pair(replace(pairs[0], evidence2=items["p1"]))
    with pytest.raises(ValueError, match="polarity"):
        validate_evidence_pair(replace(pairs[0], evidence2=items["n1"]))
    with pytest.raises(ValueError, match="EvidenceItem"):
        validate_evidence_pair(replace(pairs[0], evidence1={"text": "legacy pool"}))


@pytest.mark.parametrize("field", ["ticker", "trial_id", "condition"])
@pytest.mark.parametrize("value", ["", " ", None])
def test_pair_identity_fields_are_nonblank(pairs, field, value):
    with pytest.raises(ValueError, match=field):
        validate_evidence_pair(replace(pairs[0], **{field: value}))


def test_invalid_condition_and_legacy_pool_are_not_converted(pairs):
    with pytest.raises(ValueError, match="condition"):
        validate_evidence_pair(replace(pairs[0], condition="positive-positive"))
    with pytest.raises(ValueError, match="EvidencePair"):
        validate_evidence_pair({"positive": ["a", "b"], "negative": ["c", "d"]})


@pytest.mark.parametrize("kind", ["missing", "empty", "duplicate", "foreign_ticker", "foreign_trial"])
def test_exact_coverage_is_required(members, pairs, kind):
    changed = list(pairs)
    if kind == "missing":
        changed.pop()
    elif kind == "empty":
        changed = []
    elif kind == "duplicate":
        changed.append(pairs[0])
    elif kind == "foreign_ticker":
        changed[0] = replace(pairs[0], ticker="OTHER")
    else:
        changed[0] = replace(pairs[0], trial_id="other-trial")
    with pytest.raises(ValueError, match="coverage|duplicate|unknown"):
        _validate(members, changed)


@pytest.mark.parametrize("trial_ids", [[], ["trial-1", "trial-1"], [""], [None], "trial-1"])
def test_trials_are_explicit_nonempty_unique_strings(members, pairs, trial_ids):
    with pytest.raises(ValueError, match="trial"):
        validate_evidence_pairs(members, trial_ids, pairs)


def test_all_declared_trials_and_members_are_validated(members, pairs):
    trial2 = [replace(pair, trial_id="trial-2") for pair in pairs]
    assert len(validate_evidence_pairs(members, ["trial-2", "trial-1"], [*pairs, *trial2])) == 16
    with pytest.raises(ValueError, match="coverage"):
        validate_evidence_pairs(members, ["trial-1", "trial-2"], pairs)
    with pytest.raises(ValueError, match="members"):
        _validate([], pairs)


def test_an_item_id_cannot_change_content_or_review_metadata(members, pairs, items):
    for field, value in [("text", "Fixture changed content."),
                         ("numerical_review", "Different audit"),
                         ("source_sha256", "b" * 64)]:
        changed_item = replace(items["p1"], **{field: value})
        if field == "text":
            changed_item = replace(changed_item, content_sha256=sha256(changed_item.text.encode()).hexdigest())
        changed = [replace(pairs[0], evidence1=changed_item), *pairs[1:]]
        with pytest.raises(ValueError, match="item_id.*inconsistent"):
            _validate(members, changed)


def test_shared_slots_include_ids_bytes_and_all_metadata(members, pairs, items):
    # Even a valid, independently identified replacement is not shared evidence.
    alternative = replace(items["p1"], item_id="p3", review_id="synthetic-review-p3")
    changed = [replace(pair, evidence1=alternative)
               if pair.ticker == "BBB" and pair.condition == "+-" else pair for pair in pairs]
    with pytest.raises(ValueError, match="shared"):
        _validate(members, changed)


def test_explicit_order_contrast_requires_the_same_items_swapped(members, pairs, items):
    # Opposite polarities alone are sufficient without an order-contrast claim.
    changed = [replace(pair, evidence2=items["p2"])
               if pair.condition == "-+" else pair for pair in pairs]
    assert len(_validate(members, changed)) == 8
    with pytest.raises(ValueError, match="swapped"):
        _validate(members, changed, order_contrasts=[EvidenceOrderContrast("trial-1", "+-", "-+")])


def test_identical_content_under_new_item_id_is_not_an_order_contrast(members, pairs, items):
    alias = replace(items["p1"], item_id="p-alias")
    changed = [replace(pair, evidence2=alias) if pair.condition == "-+" else pair for pair in pairs]
    with pytest.raises(ValueError, match="swapped"):
        _validate(members, changed, order_contrasts=[EvidenceOrderContrast("trial-1", "+-", "-+")])


@pytest.mark.parametrize("contrast", [
    EvidenceOrderContrast("unknown", "+-", "-+"),
    EvidenceOrderContrast("trial-1", "++", "--"),
    EvidenceOrderContrast("trial-1", "+-", "+-"),
    EvidenceOrderContrast("trial-1", "invalid", "-+"),
    EvidenceOrderContrast("", "+-", "-+"),
    ("trial-1", "+-", "-+"),
])
def test_order_contrast_declarations_are_explicit_valid_row_keys(members, pairs, contrast):
    with pytest.raises(ValueError, match="contrast"):
        _validate(members, pairs, order_contrasts=[contrast])


def test_duplicate_or_reversed_order_contrasts_are_rejected(members, pairs):
    contrast = EvidenceOrderContrast("trial-1", "+-", "-+")
    for second in [contrast, EvidenceOrderContrast("trial-1", "-+", "+-")]:
        with pytest.raises(ValueError, match="duplicate.*contrast"):
            _validate(members, pairs, order_contrasts=[contrast, second])
