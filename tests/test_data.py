from llm_bias.analysis import normalized_transfer
from llm_bias.data import calibration_prompts


def test_calibration_prompts_are_deterministic_and_long_enough():
    prompts = calibration_prompts(8)
    assert prompts == calibration_prompts(8)
    assert len(prompts) == 8
    assert all(len(prompt.split()) > 10 for prompt in prompts)


def test_normalized_transfer_uses_source_as_zero_and_target_as_one():
    assert normalized_transfer(10.0, 2.0, 10.0) == 1.0
    assert normalized_transfer(2.0, 2.0, 10.0) == 0.0
    assert normalized_transfer(6.0, 2.0, 10.0) == 0.5
    assert normalized_transfer(1.0, 1.0, 1.0) is None


def test_pair_dataclass_serialization():
    from llm_bias.data import Pair

    pair = Pair(
        pair_id="x",
        category="c",
        function="f",
        source_entity="A",
        target_entity="B",
        source_prompt="A is",
        target_prompt="B is",
        source_answer="one",
        target_answer="two",
        source_entity_start=0,
        source_entity_end=1,
        target_entity_start=0,
        target_entity_end=1,
        source_entity_token=3,
        target_entity_token=4,
        answer_source_token=1,
        answer_target_token=2,
    )
    assert pair.to_dict()["target_entity"] == "B"
