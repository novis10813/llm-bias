import json
import re

from llm_bias.counterfactual_patching.data import load_pairs
from llm_bias.counterfactual_patching.experiment import normalized_transfer
from llm_bias.lens_fitting.calibration import builtin_calibration_prompts


class _Encoded(dict):
    def __init__(self, input_ids, *, offsets=None, specials=None):
        super().__init__()
        self.input_ids = input_ids
        if offsets is not None:
            self["offset_mapping"] = offsets
            self["special_tokens_mask"] = specials


class _WhitespaceTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
    ):
        matches = list(re.finditer(r"\S+", text))
        ids = [index + 10 for index, _match in enumerate(matches)]
        if add_special_tokens:
            ids = [1, *ids]
        if not return_offsets_mapping:
            return _Encoded(ids)
        offsets = [(0, 0)] * int(add_special_tokens) + [
            (match.start(), match.end()) for match in matches
        ]
        specials = [True] * int(add_special_tokens) + [False] * len(matches)
        return _Encoded(ids, offsets=offsets, specials=specials)


def test_calibration_prompts_are_deterministic_and_long_enough():
    prompts = builtin_calibration_prompts(8)
    assert prompts == builtin_calibration_prompts(8)
    assert len(prompts) == 8
    assert all(len(prompt.split()) > 10 for prompt in prompts)


def test_normalized_transfer_uses_source_as_zero_and_target_as_one():
    assert normalized_transfer(10.0, 2.0, 10.0) == 1.0
    assert normalized_transfer(2.0, 2.0, 10.0) == 0.0
    assert normalized_transfer(6.0, 2.0, 10.0) == 0.5
    assert normalized_transfer(1.0, 1.0, 1.0) is None


def test_pair_dataclass_serialization():
    from llm_bias.counterfactual_patching.data import Pair

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


def test_load_pairs_keeps_variable_length_entity_spans(tmp_path):
    spec = {
        "categories": [
            {
                "name": "companies",
                "args": ["Acme Corporation", "Company"],
                "funcs": [
                    {
                        "name": "risk",
                        "template": "Headline about {arg} says",
                        "answers": {"Acme Corporation": "safe", "Company": "risky"},
                    }
                ],
            }
        ]
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    pairs = load_pairs(_WhitespaceTokenizer(), spec_path=spec_path)

    pair = pairs[0]
    assert pair.source_entity_end - pair.source_entity_start == 2
    assert pair.target_entity_end - pair.target_entity_start == 1
    assert len(pair.source_prompt.split()) != len(pair.target_prompt.split())
    assert len(pair.source_entity_token_ids) == 2
    assert len(pair.target_entity_token_ids) == 1
