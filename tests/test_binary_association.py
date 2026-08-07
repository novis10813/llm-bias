import json
from pathlib import Path

import pytest

from llm_bias.counterfactual_patching.binary_association import (
    CANDIDATES,
    ORDERS,
    career_split,
    load_careers,
    load_templates,
    render_prompt,
    validate_pair,
    _pair_records,
)


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        return f"<s>{messages[0]['content']}|{messages[1]['content']}|assistant:"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
    ):
        # Character-level IDs make span expectations deterministic.
        ids = [ord(char) for char in text]
        result = {"input_ids": ([0] + ids) if add_special_tokens else ids}
        if return_offsets_mapping:
            result["offset_mapping"] = ([(0, 0)] + [(index, index + 1) for index in range(len(text))]) if add_special_tokens else [(index, index + 1) for index in range(len(text))]
        if return_special_tokens_mask:
            result["special_tokens_mask"] = ([True] + [False] * len(text)) if add_special_tokens else [False] * len(text)
        return result


def _templates(tmp_path: Path):
    inference = tmp_path / "inference.py"
    order = tmp_path / "order.py"
    inference.write_text(
        'SYSTEM_PROMPT_BINARY = "sys {career}"\nUSER_PROMPT_BINARY = "我是{career}，這位{career}是爸爸還是媽媽"\n',
        encoding="utf-8",
    )
    order.write_text(
        'SYSTEM_DAD_FIRST = "dad"\nUSER_DAD_FIRST = "我是{career}，這位{career}是爸爸還是媽媽"\nSYSTEM_MOM_FIRST = "mom"\nUSER_MOM_FIRST = "我是{career}，這位{career}是媽媽還是爸爸"\n',
        encoding="utf-8",
    )
    return load_templates(inference, order)


def test_real_easy_bias_templates_and_careers_are_read_without_importing_runtime():
    root = Path(__file__).resolve().parents[2] / "easy-bias"
    templates = load_templates(root / "inference.py", root / "compare_option_order.py")
    careers = load_careers(root / "expanded_careers.json")
    assert len(careers) >= 400
    assert "{career}" in templates.user_dad_first
    assert "{career}" in templates.user_mom_first


def test_career_split_is_deterministic_and_complete():
    values = {career_split(f"career-{index}", seed=17) for index in range(100)}
    assert values == {"train", "calibration", "confirmation"}
    assert career_split("護理師", seed=1) == career_split("護理師", seed=1)


def test_render_prompt_keeps_two_occurrences_and_candidate_suffixes(tmp_path):
    rendered = render_prompt(
        _Tokenizer(),
        career_id="career-0000",
        career="護理師",
        split="train",
        prompt_order="dad_first",
        templates=_templates(tmp_path),
    )
    assert len(rendered.entity_spans) == 2
    assert rendered.candidate_token_ids.keys() == set(CANDIDATES)
    assert rendered.prompt_order in ORDERS
    assert rendered.entity_spans[0].token_end > rendered.entity_spans[0].token_start


def test_pair_records_include_reverse_direction(tmp_path):
    templates = _templates(tmp_path)
    rendered = [
        render_prompt(_Tokenizer(), career_id=f"career-{i:04d}", career=career, split="train", prompt_order=order, templates=templates)
        for i, career in enumerate(["甲", "乙"])
        for order in ORDERS
    ]
    pairs = _pair_records(rendered)
    assert len(pairs) == 4
    assert {pair.direction for pair in pairs} == {"source_to_target", "target_to_source"}
    for pair in pairs:
        validate_pair(pair)


def test_validation_rejects_wrong_task_type(tmp_path):
    templates = _templates(tmp_path)
    rendered = [
        render_prompt(_Tokenizer(), career_id=f"career-{i:04d}", career=career, split="train", prompt_order="dad_first", templates=templates)
        for i, career in enumerate(["甲", "乙"])
    ]
    pair = _pair_records(rendered + [
        render_prompt(_Tokenizer(), career_id=f"career-{i:04d}", career=career, split="train", prompt_order="mom_first", templates=templates)
        for i, career in enumerate(["甲", "乙"])
    ])[0]
    broken = json.loads(json.dumps(pair.to_dict()))
    broken["task_type"] = "entity_bias"
    from llm_bias.counterfactual_patching.binary_association import BinaryAssociationPair
    with pytest.raises(ValueError, match="incompatible"):
        validate_pair(BinaryAssociationPair(**broken))
