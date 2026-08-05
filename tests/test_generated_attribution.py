from llm_bias.prompt_analysis.attribution import parse_generated_return_answer


def test_parse_generated_return_answer_is_strict_and_nonfatal():
    assert parse_generated_return_answer('{"label":"neutral","confidence":80}') == {
        "predicted_label": "neutral",
        "predicted_confidence": 80,
        "parse_status": "valid",
        "parse_reason": None,
    }
    invalid = parse_generated_return_answer(
        '{"label":"neutral","confidence":80.0}'
    )
    assert invalid["parse_status"] == "invalid"
    assert invalid["parse_reason"] == "invalid_confidence"
