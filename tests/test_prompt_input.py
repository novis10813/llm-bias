from types import SimpleNamespace

import pytest

from llm_bias.core.prompt_input import (
    all_occurrences,
    continuation_token_ids,
    find_token_subsequence,
    format_messages,
    input_ids,
    span_record,
    token_span,
)


class _Tokenizer:
    chat_template = "available"

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False,
                 return_special_tokens_mask=False):
        ids = list(range(len(text)))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        if return_special_tokens_mask:
            result["special_tokens_mask"] = [False] * len(text)
        return SimpleNamespace(**result)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"


def test_input_ids_normalizes_mapping_and_object_outputs():
    tokenizer = _Tokenizer()
    assert input_ids(tokenizer, "abc", add_special_tokens=False) == [0, 1, 2]


def test_format_messages_preserves_system_user_boundaries():
    tokenizer = _Tokenizer()
    assert format_messages(
        tokenizer,
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}],
        use_chat_template=True,
    ) == "system:rules|user:question|assistant:"
    assert format_messages(
        tokenizer,
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}],
        use_chat_template=False,
    ) == "rules\n\nquestion"


def test_token_span_and_record_cover_overlapping_tokens():
    tokenizer = _Tokenizer()
    assert token_span(tokenizer, "abcdef", 2, 5) == (2, 5)
    record = span_record(tokenizer, "abcdef", (2, 5))
    assert record.to_dict() == {
        "char_start": 2, "char_end": 5, "token_start": 2, "token_end": 5,
        "token_ids": [2, 3, 4],
    }


def test_token_span_rejects_invalid_character_ranges():
    with pytest.raises(ValueError, match="invalid character span"):
        token_span(_Tokenizer(), "abc", 0, 4)


def test_shared_continuation_contract_rejects_non_prefix_tokenization():
    tokenizer = _Tokenizer()
    assert continuation_token_ids(tokenizer, "ab", "c") == [2]
    assert all_occurrences("aba", "a") == [(0, 1), (2, 3)]
    assert find_token_subsequence([1, 2, 3], [2, 3]) == (1, 3)
