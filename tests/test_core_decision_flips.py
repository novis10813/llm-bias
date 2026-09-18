import pytest

from llm_bias.core.analysis import (
    decision_flip_summary,
    generated_decision_flip_summary,
    parse_buy_sell_decision,
)


def test_parse_buy_sell_decision_accepts_json_with_prefix_and_suffix():
    assert parse_buy_sell_decision('answer: {"decision": "buy"} <stop>') == "buy"
    assert parse_buy_sell_decision('{"decision": "sell", "reason": "risk"}') == "sell"


@pytest.mark.parametrize(
    "text",
    [None, "not json", '{"decision": "hold"}', "[]", '{"decision": 1}'],
)
def test_parse_buy_sell_decision_rejects_invalid_outputs(text):
    assert parse_buy_sell_decision(text) is None


def test_decision_flip_summary_reports_total_and_directional_rates():
    summary = decision_flip_summary(
        {"a": "sell", "b": "buy", "c": None, "d": "sell"},
        {"a": "buy", "b": "sell", "c": "buy", "d": "sell"},
    )
    assert summary == {
        "pair_count": 4,
        "valid_pair_count": 3,
        "invalid_pair_count": 1,
        "flip_count": 2,
        "flip_rate": pytest.approx(2 / 3),
        "sell_to_buy_count": 1,
        "sell_to_buy_rate": pytest.approx(1 / 3),
        "buy_to_sell_count": 1,
        "buy_to_sell_rate": pytest.approx(1 / 3),
    }


def test_generated_decision_flip_summary_parses_and_fails_closed_on_duplicates():
    summary = generated_decision_flip_summary(
        [
            {
                "pair_id": "p1",
                "clean_generated_text": '{"decision": "sell"}',
                "intervened_generated_text": '{"decision": "buy"}',
            },
            {
                "pair_id": "p2",
                "clean_generated_text": '{"decision": "buy"}',
                "intervened_generated_text": "malformed",
            },
        ]
    )
    assert summary["flip_count"] == 1
    assert summary["valid_pair_count"] == 1
    assert summary["invalid_pair_count"] == 1

    with pytest.raises(ValueError, match="duplicate pair identifier"):
        generated_decision_flip_summary(
            [
                {
                    "pair_id": "p1",
                    "clean_generated_text": '{"decision": "sell"}',
                    "intervened_generated_text": '{"decision": "buy"}',
                },
                {
                    "pair_id": "p1",
                    "clean_generated_text": '{"decision": "sell"}',
                    "intervened_generated_text": '{"decision": "sell"}',
                },
            ]
        )


def test_decision_flip_summary_rejects_mismatched_pair_keys():
    with pytest.raises(ValueError, match="keys mismatch"):
        decision_flip_summary({"a": "sell"}, {"b": "buy"})
