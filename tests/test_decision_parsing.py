"""Complete-object and strict buy/sell decision parsing."""
from __future__ import annotations

import pytest

from llm_bias.core.decision_parsing import parse_complete_decision, parse_strict_decision


@pytest.mark.parametrize("wrapper,kind", [
    ('{\n"decision":"buy", "reason":"evidence"\n}', "bare_json"),
    ('```json\n{"decision":"buy", "reason":"evidence"}\n```', "fenced_json"),
    ('thought\n{"decision":"buy", "reason":"evidence"}', "thought_bare_json"),
    ('thought\n```json\n{"decision":"buy", "reason":"evidence"}\n```', "thought_fenced_json"),
    ('analysisWeigh it. Buy.assistantfinal{"decision":"buy","reason":"evidence"}', "harmony_final_bare_json"),
    ('analysisWeigh.assistantfinal```json\n{"decision":"buy","reason":"evidence"}\n```', "harmony_final_fenced_json"),
])
def test_accepts_only_whole_valid_objects(wrapper, kind):
    assert parse_complete_decision(wrapper) == ("buy", kind)


@pytest.mark.parametrize("text", [
    'prefix {"decision":"buy","reason":"evidence"}',
    '{"decision":"buy","reason":"evidence"} trailing text',
    '```json\n{"decision":"buy","reason":"evidence"}',
    '```json\n{"decision":"buy","reason":"evidence"}\n``` trailing',
    'thought extra\n{"decision":"buy","reason":"evidence"}',
    '{"decision":"buy","reason":"evidence","extra":1}',
    '{"decision":"BUY","reason":"evidence"}',
    '{"decision":"buy","reason":"  "}',
    '{"decision":"buy"}',
    '{"decision":"buy","reason":"incomplete',
    'analysisStill weighing the evidence without reaching a final channel',
    'analysisWeigh.assistantfinal{"decision":"buy","reason":"evidence"} trailing',
    'analysisWeigh.assistantfinalthought\n{"decision":"buy","reason":"evidence"}',
])
def test_rejects_partial_and_invalid(text):
    assert parse_complete_decision(text) == (None, "invalid")


@pytest.mark.parametrize("text,decision", [
    ('{"decision":"sell","reason":"x"}', "sell"),
    ('{"decision":"sell"}', "sell"),
    ('thought\n{"decision":"buy","reason":"x"}', None),
    ('{"decision":"BUY","reason":"x"}', None),
    ('[1]', None),
])
def test_strict_parse_requires_whole_json_object(text, decision):
    assert parse_strict_decision(text) == decision
