"""Focused regression tests for the round-2 table-format development materials builder."""
from __future__ import annotations

import pytest

from llm_bias.entity_concept_decision.development_materials import build_round2_materials

_G = """
# G
| ID | concept | group | role | pole | 英文描述 |
|---|---|---|---|---|---|
| G-F01-P | G | G1 | primary | P | The company earns most of its revenue in many countries. |
| G-F01-N | G | G1 | primary | N | The company earns most of its revenue in a small number of countries. |
| G-F02-P | G | G2 | primary | P | The company operates in many markets across different countries. |
| G-F02-N | G | G2 | primary | N | The company operates mainly in one home market. |
| G-E01-PH | G | G-E01 | evaluation | PH | The company operates in many countries. An observer describes the outlook as favorable. |
| G-E01-PL | G | G-E01 | evaluation | PL | The company operates in many countries. An observer describes the outlook as unfavorable. |
| G-E01-NH | G | G-E01 | evaluation | NH | The company operates in one country. An observer describes the outlook as favorable. |
| G-E01-NL | G | G-E01 | evaluation | NL | The company operates in one country. An observer describes the outlook as unfavorable. |
| G-L01-A | G | G-L01 | lexical | A | The company lists many office cities worldwide. |
| G-L01-B | G | G-L01 | lexical | B | The company lists one office city. |
| G-X01-A | G | G-X01 | competitor | A | The company employs 20,000 people. |
| G-X01-B | G | G-X01 | competitor | B | The company employs 200 people. |
"""

_C = """
# C
| ID | concept | group | role | pole | 英文描述 |
|---|---|---|---|---|---|
| C-F01-P | C | C1 | primary | P | The company's three largest customers account for 75% of annual revenue. Analysts describe the company as financially healthy. |
| C-F01-N | C | C1 | primary | N | The company's three largest customers account for 15% of annual revenue. Analysts describe the company as financially healthy. |
| C-F02-P | C | C2 | primary | P | A small group of customers provides most of the company's revenue. Analysts describe the company as financially healthy. |
| C-F02-N | C | C2 | primary | N | A broad group of customers provides similar portions of the company's revenue. Analysts describe the company as financially healthy. |
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_round2_builder_parses_g_and_c(tmp_path):
    g = _write(tmp_path, "g.md", _G)
    c = _write(tmp_path, "c.md", _C)
    materials = build_round2_materials([g, c], expected_concepts=["C", "G"])
    assert materials["protocol"] == "phase1-v2-round2"
    assert materials["review_status"] == "ai_reviewed_development"
    rows = {row["id"]: row for row in materials["rows"]}
    assert len(rows) == 16
    # family_id equals group_id in round 2
    assert rows["G-F01-P"]["family_id"] == "G1" and rows["G-F01-P"]["group_id"] == "G1"
    # comparisons: 3 G primary + 4 G eval + 1 G lexical + 1 G competitor + 2 C primary
    kinds = {}
    for comparison in materials["comparisons"]:
        kinds[comparison["id"]] = comparison["kind"]
    assert kinds["G1-primary"] == "primary"
    assert kinds["G-E01-evaluation_at_positive_concept"] == "evaluation_at_positive_concept"
    assert kinds["G-L01-lexical"] == "lexical"
    assert kinds["G-X01-competitor"] == "competitor"
    assert kinds["C1-primary"] == "primary"
    # every row is covered exactly once
    covered = set()
    for comparison in materials["comparisons"]:
        covered.update((comparison["positive_id"], comparison["negative_id"]))
    assert covered == set(rows)


def test_round2_builder_requires_two_primary_groups(tmp_path):
    one_group = """
| ID | concept | group | role | pole | 英文描述 |
|---|---|---|---|---|---|
| C-F01-P | C | C1 | primary | P | The company's three largest customers account for 75% of annual revenue. |
| C-F01-N | C | C1 | primary | N | The company's three largest customers account for 15% of annual revenue. |
| C-E01-PH | C | C-E01 | evaluation | PH | The company's three largest customers account for 75% of annual revenue. An observer describes the outlook as favorable. |
| C-E01-PL | C | C-E01 | evaluation | PL | The company's three largest customers account for 75% of annual revenue. An observer describes the outlook as unfavorable. |
| C-E01-NH | C | C-E01 | evaluation | NH | The company's three largest customers account for 15% of annual revenue. An observer describes the outlook as favorable. |
| C-E01-NL | C | C-E01 | evaluation | NL | The company's three largest customers account for 15% of annual revenue. An observer describes the outlook as unfavorable. |
"""
    c = _write(tmp_path, "c.md", one_group)
    with pytest.raises(ValueError, match="at least two primary groups"):
        build_round2_materials([c], expected_concepts=["C"])


def test_round2_builder_rejects_advice_and_duplicates(tmp_path):
    bad = _G.replace("many office cities worldwide", "a company to buy")
    g = _write(tmp_path, "g.md", bad)
    with pytest.raises(ValueError, match="investment advice"):
        build_round2_materials([g], expected_concepts=["G"])
    dup = _G + "| G-F09-P | G | G1 | primary | P | The company earns most of its revenue in many countries. |\n"
    g2 = _write(tmp_path, "g2.md", dup)
    with pytest.raises(ValueError, match="normalized duplicate"):
        build_round2_materials([g2], expected_concepts=["G"])
