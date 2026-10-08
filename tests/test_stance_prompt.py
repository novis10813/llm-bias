"""Generation-only prompts with synthetic source and review metadata."""

import json
from collections import Counter
from dataclasses import asdict, fields, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.population import PopulationMember
from llm_bias.core.prompt_input.decision_prompt import (
    DECISION_INSTRUCTION,
    DECISION_SCHEMA,
    DECISION_TEMPLATE,
    DecisionPrompt,
    UnsupportedTokenizerError,
    WrapperPolicy,
    render_decision_prompt,
)
from llm_bias.core.stance_evidence import EvidenceItem, EvidencePair


class _Tokenizer:
    """Character tokenizer with real wrapper characters and BOS/EOS offsets."""

    chat_template = "fixture-chat-v1"

    def __init__(self):
        self.calls = []
        self.chat_calls = []

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False,
                 return_special_tokens_mask=False):
        self.calls.append((text, add_special_tokens, return_offsets_mapping))
        ids = [ord(c) + 3 for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        mask = [0] * len(text)
        if add_special_tokens:
            ids, offsets, mask = [1, *ids, 2], [(0, 0), *offsets, (0, 0)], [1, *mask, 1]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        if return_special_tokens_mask:
            result["special_tokens_mask"] = mask
        return result

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages) + "<assistant>\n"


def _bytelevel_tokenizer(*, boundary_merges=False):
    """Actual fast BPE tokenizer built locally, with no checkpoint or download."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    merges = [("Ċ", "C"), ("Ċ", "F")] if boundary_merges else []
    vocab = {token: i for i, token in enumerate([*alphabet, *(a + b for a, b in merges),
                                                "[BOS]", "[EOS]"])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=not boundary_merges,
    )
    backend.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=True),
        processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
        ),
    ])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="[BOS]", eos_token="[EOS]",
        chat_template=("{% for message in messages %}<{{ message['role'] }}>\n"
                       "{{ message['content'] }}\n</{{ message['role'] }}>\n{% endfor %}"
                       "{% if add_generation_prompt %}<assistant>\n{% endif %}"),
    )


@pytest.fixture
def member():
    return PopulationMember("AAA", "Fixture Café", "Unspecified", "fixture-a")


def _item(item_id, polarity, text):
    return EvidenceItem(
        item_id=item_id, text=text, polarity=polarity,
        source="synthetic unit-test source only",
        source_sha256=sha256(b"SYNTHETIC SOURCE BYTES: not research evidence").hexdigest(),
        content_sha256=sha256(text.encode()).hexdigest(),
        review_id=f"synthetic-{item_id}",
        strength="fixture-only: unranked synthetic label",
        numerical_review="Synthetic review: no numerical claims in this fixture.",
        source_quality_review="Synthetic review: not approved research evidence.",
    )


@pytest.fixture
def pair():
    return EvidencePair("AAA", "trial-1", "+-",
                        _item("fixture-p", "+", "Fixture favorable observation."),
                        _item("fixture-n", "-", "Fixture unfavorable observation."))


def _render(member, pair, tokenizer=None, **policy):
    return render_decision_prompt(
        tokenizer or _Tokenizer(), member, pair,
        wrapper_policy=WrapperPolicy(use_chat_template=False, add_special_tokens=False, **policy),
    )


def _assert_span(prompt, role, text, shift=0):
    span = getattr(prompt, f"{role}_span")
    assert span.role == role
    assert prompt.rendered_text[span.char_start:span.char_end] == text
    assert span.token_start == span.char_start + shift
    assert span.token_end == span.char_end + shift
    assert span.token_ids == prompt.inference_token_ids[span.token_start:span.token_end]
    assert span.token_ids == tuple(ord(c) + 3 for c in text)
    assert span.token_sha256 == sha256_json(span.token_ids)
    assert span.text_sha256 == sha256(text.encode()).hexdigest()


def test_fixed_order_body_and_generation_only_contract(member, pair):
    prompt = _render(member, pair)
    entity = "Company ticker: AAA\nCompany name: Fixture Café\nCompany sector: Unspecified"
    expected = (f"{entity}\n\nEvidence slot 1:\n{pair.evidence1.text}"
                f"\n\nEvidence slot 2:\n{pair.evidence2.text}\n\n{DECISION_INSTRUCTION}")
    assert prompt.raw_text == prompt.rendered_text == expected
    assert prompt.rendered_text.count(DECISION_INSTRUCTION) == 1
    assert '"buy"' in DECISION_INSTRUCTION and '"sell"' in DECISION_INSTRUCTION
    assert DECISION_INSTRUCTION.index('"decision"') < DECISION_INSTRUCTION.index('"reason"')
    assert prompt.raw_text.endswith(DECISION_INSTRUCTION)
    assert not {"score_ids", "buy_id", "sell_id", "margin", "answer_prefix"} & {f.name for f in fields(DecisionPrompt)}
    for role, text in [("entity", entity), ("evidence1", pair.evidence1.text),
                       ("evidence2", pair.evidence2.text), ("instruction", DECISION_INSTRUCTION)]:
        _assert_span(prompt, role, text)


@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("system_message", [None, "You are a careful analyst."])
def test_wrapped_spans_use_actual_inference_offsets(member, pair, add_special_tokens, system_message):
    tokenizer = _Tokenizer()
    policy = WrapperPolicy(use_chat_template=True, add_special_tokens=add_special_tokens,
                           system_message=system_message, enable_thinking=False,
                           chat_template_kwargs={"fixture_mode": "safe"})
    prompt = render_decision_prompt(tokenizer, member, pair, wrapper_policy=policy)
    body_offset = prompt.rendered_text.index(prompt.raw_text)
    assert body_offset > 0
    assert prompt.rendered_text.count(prompt.raw_text) == 1
    raw = _render(member, pair)
    for role in ["entity", "evidence1", "evidence2", "instruction"]:
        raw_span = getattr(raw, f"{role}_span")
        span = getattr(prompt, f"{role}_span")
        assert span.char_start == raw_span.char_start + body_offset
        assert span.char_end == raw_span.char_end + body_offset
        _assert_span(prompt, role, raw.raw_text[raw_span.char_start:raw_span.char_end], int(add_special_tokens))
    assert prompt.inference_token_ids == tuple(tokenizer(prompt.rendered_text, add_special_tokens=add_special_tokens)["input_ids"])
    assert tokenizer.chat_calls[0][1] == {
        "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        "fixture_mode": "safe",
    }
    assert tokenizer.chat_calls[0][0] == ([{"role": "system", "content": system_message}] if system_message else []) + [
        {"role": "user", "content": prompt.raw_text},
    ]


def test_raw_explicit_system_message_offsets(member, pair):
    prompt = _render(member, pair, system_message="You are a careful analyst.")
    assert prompt.rendered_text == "You are a careful analyst.\n\n" + prompt.raw_text
    assert prompt.entity_span.char_start == len("You are a careful analyst.\n\n")


def test_repeated_text_and_delimiters_do_not_drive_span_search(member, pair):
    text = "Company ticker: AAA\n\nEvidence slot 1:\nEvidence slot 2:\nFixture Café"
    repeated = replace(pair, evidence1=_item("fixture-p", "+", text),
                       evidence2=_item("fixture-n", "-", text))
    prompt = _render(member, repeated)
    _assert_span(prompt, "evidence1", text)
    _assert_span(prompt, "evidence2", text)
    assert prompt.evidence1_span.char_start < prompt.evidence2_span.char_start
    assert prompt.raw_text.count("Evidence slot 1:") == 3
    assert prompt.entity_span.char_start == 0


def test_evidence_order_contrast_changes_only_slot_order(member, pair):
    forward = _render(member, pair)
    reverse = _render(member, replace(pair, condition="-+", evidence1=pair.evidence2, evidence2=pair.evidence1))
    assert forward.raw_text == DECISION_TEMPLATE.format(
        entity=forward.raw_text[:forward.entity_span.char_end],
        evidence1=pair.evidence1.text, evidence2=pair.evidence2.text,
        instruction=DECISION_INSTRUCTION,
    )
    assert reverse.raw_text == DECISION_TEMPLATE.format(
        entity=forward.raw_text[:forward.entity_span.char_end],
        evidence1=pair.evidence2.text, evidence2=pair.evidence1.text,
        instruction=DECISION_INSTRUCTION,
    )
    assert forward.template_sha256 == reverse.template_sha256
    assert forward.schema_sha256 == reverse.schema_sha256
    assert forward.prompt_sha256 != reverse.prompt_sha256


def test_prompt_template_schema_and_wrapper_hashes_are_recomputable(member, pair):
    prompt = _render(member, pair)
    assert prompt.template_sha256 == sha256_json({
        "template": DECISION_TEMPLATE, "instruction": DECISION_INSTRUCTION,
        "entity_template": "Company ticker: {ticker}\nCompany name: {name}\nCompany sector: {sector}",
    })
    assert prompt.schema_sha256 == sha256_json({"schema": DECISION_SCHEMA, "key_order": ["decision", "reason"]})
    assert DECISION_SCHEMA["required"] == ["decision", "reason"]
    assert DECISION_SCHEMA["properties"]["decision"]["enum"] == ["buy", "sell"]
    assert DECISION_SCHEMA["additionalProperties"] is False
    assert DECISION_SCHEMA["properties"]["reason"] == {"type": "string", "minLength": 1}
    assert "shared market context" in DECISION_INSTRUCTION
    assert "not company-specific financial results" in DECISION_INSTRUCTION
    assert prompt.prompt_sha256 == sha256_json({
        "raw_text": prompt.raw_text, "rendered_text": prompt.rendered_text,
        "inference_token_ids": prompt.inference_token_ids,
        "template_sha256": prompt.template_sha256, "schema_sha256": prompt.schema_sha256,
        "wrapper_policy": prompt.wrapper_policy,
    })
    assert prompt.wrapper_policy == {
        "use_chat_template": False, "add_special_tokens": False,
        "system_message": None, "enable_thinking": False,
        "chat_template_kwargs": {}, "tokenizer_chat_template": None,
    }


def test_wrapper_settings_and_template_enter_provenance_even_if_output_is_equal(member, pair):
    def render(**kwargs):
        return render_decision_prompt(_Tokenizer(), member, pair, wrapper_policy=WrapperPolicy(
            use_chat_template=True, add_special_tokens=False, **kwargs))
    plain = render()
    for other in [render(enable_thinking=True), render(chat_template_kwargs={"fixture_mode": "safe"})]:
        assert other.rendered_text == plain.rendered_text
        assert other.prompt_sha256 != plain.prompt_sha256
    other_tokenizer = _Tokenizer()
    other_tokenizer.chat_template = "fixture-chat-v2"
    other = render_decision_prompt(other_tokenizer, member, pair, wrapper_policy=WrapperPolicy(True, False))
    assert other.rendered_text == plain.rendered_text
    assert other.prompt_sha256 != plain.prompt_sha256
    special = render_decision_prompt(_Tokenizer(), member, pair, wrapper_policy=WrapperPolicy(True, True))
    assert special.prompt_sha256 != plain.prompt_sha256


def test_wrapper_policy_records_a_copy_of_nested_kwargs(member, pair):
    kwargs = {"fixture": {"label": "before"}}
    prompt = _render(member, pair, chat_template_kwargs=kwargs)
    kwargs["fixture"]["label"] = "after"
    assert prompt.wrapper_policy["chat_template_kwargs"] == {"fixture": {"label": "before"}}


@pytest.mark.parametrize("mutation", ["top_level", "nested_dict", "nested_list", "template"])
def test_returned_wrapper_policy_cannot_mutate_prompt_identity_or_export(member, pair, mutation):
    tokenizer = _Tokenizer()
    tokenizer.chat_template = {"default": "fixture-chat-v1"}
    kwargs = {"fixture": {"label": "before", "labels": ["original"]}}
    prompt = render_decision_prompt(tokenizer, member, pair,
                                    wrapper_policy=WrapperPolicy(True, False, chat_template_kwargs=kwargs))
    original = asdict(prompt)
    returned = prompt.wrapper_policy
    if mutation == "top_level":
        returned["add_special_tokens"] = True
    elif mutation == "nested_dict":
        returned["chat_template_kwargs"]["fixture"]["label"] = "changed"
    elif mutation == "nested_list":
        returned["chat_template_kwargs"]["fixture"]["labels"].append("changed")
    else:
        returned["tokenizer_chat_template"]["default"] = "changed"
    assert prompt.wrapper_policy == original["wrapper_policy"]
    assert prompt.wrapper_policy is not returned
    assert asdict(prompt) == original
    assert not any(key.startswith("_wrapper") for key in original)
    exported = json.loads(json.dumps(asdict(prompt)))
    assert exported["wrapper_policy"] == original["wrapper_policy"]
    assert prompt.prompt_sha256 == sha256_json({
        key: original[key] for key in ["raw_text", "rendered_text", "inference_token_ids",
                                       "template_sha256", "schema_sha256", "wrapper_policy"]
    })
    assert replace(prompt).wrapper_policy == original["wrapper_policy"]
    constructed = DecisionPrompt(**{field.name: getattr(prompt, field.name) for field in fields(prompt)})
    assert constructed == prompt
    kwargs["fixture"]["labels"].append("input changed")
    tokenizer.chat_template["default"] = "input changed"
    exported["wrapper_policy"]["chat_template_kwargs"]["fixture"]["label"] = "export changed"
    assert asdict(prompt) == original


@pytest.mark.parametrize("behavior", ["rewrite", "duplicate", "duplicate_instruction", "missing"])
def test_chat_wrappers_cannot_rewrite_or_duplicate_body_or_instruction(member, pair, behavior):
    class BadWrapper(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            body = messages[-1]["content"]
            return {"rewrite": body.replace("Company ticker", "Ticker"),
                    "duplicate": body + body,
                    "duplicate_instruction": body + DECISION_INSTRUCTION,
                    "missing": "discarded"}[behavior]
    with pytest.raises(ValueError, match="body|instruction"):
        render_decision_prompt(BadWrapper(), member, pair, wrapper_policy=WrapperPolicy(True, False))


@pytest.mark.parametrize("system_message", ["", " \n", 1, DECISION_INSTRUCTION,
                                          "Return JSON with decision and reason.",
                                          "Return only a JSON object.",
                                          "Choose BUY or SELL."])
def test_system_message_is_explicit_nonblank_and_not_a_second_task(member, pair, system_message):
    with pytest.raises(ValueError, match="system_message"):
        _render(member, pair, system_message=system_message)


def test_evidence_cannot_duplicate_fixed_instruction(member, pair):
    changed = replace(pair, evidence1=_item("fixture-p", "+", DECISION_INSTRUCTION))
    with pytest.raises(ValueError, match="instruction"):
        _render(member, changed)


def test_member_pair_and_policy_are_validated_before_rendering(member, pair):
    with pytest.raises(ValueError, match="ticker"):
        _render(member, replace(pair, ticker="OTHER"))
    with pytest.raises(ValueError, match="text"):
        _render(member, replace(pair, evidence1=replace(pair.evidence1, text=" ")))
    with pytest.raises(ValueError, match="sector"):
        _render(replace(member, sector=""), pair)
    with pytest.raises(ValueError, match="WrapperPolicy"):
        render_decision_prompt(_Tokenizer(), member, pair, wrapper_policy={})


@pytest.mark.parametrize("kwargs", [{"tokenize": True}, {"add_generation_prompt": False},
                                    {"enable_thinking": True}, {"messages": []}])
def test_reserved_chat_kwargs_are_not_accepted(member, pair, kwargs):
    with pytest.raises(ValueError, match="reserved"):
        _render(member, pair, chat_template_kwargs=kwargs)


@pytest.mark.parametrize("field", ["use_chat_template", "add_special_tokens", "enable_thinking"])
def test_policy_booleans_are_explicit(member, pair, field):
    policy = replace(WrapperPolicy(False, False), **{field: "false"})
    with pytest.raises(ValueError, match=field):
        render_decision_prompt(_Tokenizer(), member, pair, wrapper_policy=policy)


def test_non_json_chat_kwargs_and_missing_chat_template_are_rejected(member, pair):
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        _render(member, pair, chat_template_kwargs={"unsupported": object()})
    tokenizer = _Tokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        render_decision_prompt(tokenizer, member, pair, wrapper_policy=WrapperPolicy(True, False))


@pytest.mark.parametrize("problem", ["no_offsets", "no_mask", "wrong_length", "out_of_range",
                                    "backwards", "zero_width", "gap", "batch",
                                    "noninteger", "different_ids", "interleaved_special", "throws",
                                    "repeated_full_prompt", "zero_width_after_whitespace"])
def test_unreliable_offsets_are_explicitly_unsupported(member, pair, problem):
    class BadOffsets(_Tokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if not kwargs.get("return_offsets_mapping"):
                return result
            if problem == "no_offsets":
                result.pop("offset_mapping")
            elif problem == "no_mask":
                result.pop("special_tokens_mask")
            elif problem == "wrong_length":
                result["offset_mapping"].pop()
            elif problem == "out_of_range":
                result["offset_mapping"][0] = (0, len(text) + 1)
            elif problem == "backwards":
                result["offset_mapping"][0] = (2, 1)
            elif problem == "zero_width":
                result["offset_mapping"][0] = (0, 0)
            elif problem == "gap":
                result["offset_mapping"][1] = (2, 3)
            elif problem == "batch":
                result["offset_mapping"] = [result["offset_mapping"]]
            elif problem == "noninteger":
                result["offset_mapping"][0] = (0, 1.5)
            elif problem == "different_ids":
                result["input_ids"][0] += 1
            elif problem == "interleaved_special":
                result["special_tokens_mask"][1] = 1
                result["offset_mapping"][1] = (0, 0)
            elif problem == "repeated_full_prompt":
                result["offset_mapping"] = [(0, len(text))] * len(result["input_ids"])
            elif problem == "zero_width_after_whitespace":
                position = text.index("ticker")
                result["offset_mapping"][position] = (position, position)
            else:
                raise NotImplementedError("offsets unavailable")
            return result
    with pytest.raises(UnsupportedTokenizerError, match="offset|token"):
        _render(member, pair, tokenizer=BadOffsets())


def test_object_tokenizer_outputs_and_overlapping_token_boundaries(member, pair):
    class ChunkTokenizer(_Tokenizer):
        def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False,
                     return_special_tokens_mask=False):
            offsets = [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]
            ids = list(range(10, 10 + len(offsets)))
            mask = [0] * len(ids)
            if add_special_tokens:
                ids, offsets, mask = [1, *ids, 2], [(0, 0), *offsets, (0, 0)], [1, *mask, 1]
            result = {"input_ids": ids}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets
            if return_special_tokens_mask:
                result["special_tokens_mask"] = mask
            return SimpleNamespace(**result)
    tokenizer = ChunkTokenizer()
    prompt = render_decision_prompt(tokenizer, member, pair, wrapper_policy=WrapperPolicy(True, True))
    encoded = tokenizer(prompt.rendered_text, return_offsets_mapping=True, return_special_tokens_mask=True)
    for role in ["entity", "evidence1", "evidence2", "instruction"]:
        span = getattr(prompt, f"{role}_span")
        selected = [i for i, (a, b) in enumerate(encoded.offset_mapping)
                    if not encoded.special_tokens_mask[i] and a < span.char_end and b > span.char_start]
        assert span.token_start == min(selected)
        assert span.token_end == max(selected) + 1
        assert span.token_ids == prompt.inference_token_ids[min(selected):max(selected) + 1]


@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("use_chat_template", [False, True])
@pytest.mark.parametrize("boundary_merges", [False, True])
def test_real_trimmed_bytelevel_unicode_offsets_and_wrappers(
    member, pair, add_special_tokens, use_chat_template, boundary_merges,
):
    tokenizer = _bytelevel_tokenizer(boundary_merges=boundary_merges)
    unicode_member = replace(member, name="Fixture Café 中 😀")
    unicode_pair = replace(
        pair,
        evidence1=_item("fixture-p", "+", "Fixture favorable Café 中 😀  observation. "),
        evidence2=_item("fixture-n", "-", " Fixture unfavorable e\u0301 中 😀  observation."),
    )
    policy = WrapperPolicy(use_chat_template, add_special_tokens,
                           system_message="You are a careful analyst.")
    # Assert that the fixture really exercises trimmed non-special whitespace and
    # repeated UTF-8 byte offsets, independently of the renderer under test.
    raw = _render(unicode_member, unicode_pair).raw_text
    if use_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": policy.system_message}, {"role": "user", "content": raw}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    else:
        text = policy.system_message + "\n\n" + raw
    encoded = tokenizer(text, add_special_tokens=add_special_tokens,
                        return_offsets_mapping=True, return_special_tokens_mask=True)
    offsets, mask = encoded["offset_mapping"], encoded["special_tokens_mask"]
    assert any(a == b and not special for (a, b), special in zip(offsets, mask, strict=True))
    counts = Counter((a, b) for (a, b), special in zip(offsets, mask, strict=True) if not special and a < b)
    assert any(count > 1 and text[a:b] == "中" for (a, b), count in counts.items())
    assert any(count == 4 and text[a:b] == "😀" for (a, b), count in counts.items())
    if boundary_merges:
        entity_start = text.index(raw)
        assert any(a < entity_start < b for a, b in offsets)
    prompt = render_decision_prompt(tokenizer, unicode_member, unicode_pair, wrapper_policy=policy)
    assert prompt.rendered_text == text
    assert prompt.inference_token_ids == tuple(encoded["input_ids"])
    expected_texts = {
        "entity": "Company ticker: AAA\nCompany name: Fixture Café 中 😀\nCompany sector: Unspecified",
        "evidence1": unicode_pair.evidence1.text, "evidence2": unicode_pair.evidence2.text,
        "instruction": DECISION_INSTRUCTION,
    }
    ranges = []
    for role, expected in expected_texts.items():
        span = getattr(prompt, f"{role}_span")
        assert text[span.char_start:span.char_end] == expected
        # Zero-width whitespace between substantive tokens belongs in the
        # contiguous inference slice; it must not introduce missing token IDs.
        substantive = [i for i, ((a, b), special) in enumerate(zip(offsets, mask, strict=True))
                       if not special and a < b and a < span.char_end and b > span.char_start]
        assert (span.token_start, span.token_end) == (substantive[0], substantive[-1] + 1)
        assert span.token_ids == tuple(encoded["input_ids"][span.token_start:span.token_end])
        assert span.token_sha256 == sha256_json(span.token_ids)
        assert span.text_sha256 == sha256(expected.encode()).hexdigest()
        assert any(offsets[i][0] == offsets[i][1] for i in range(span.token_start, span.token_end))
        assert not any(mask[span.token_start:span.token_end])
        ranges.append((span.token_start, span.token_end))
    assert all(left_end <= right_start for (_, left_end), (right_start, _) in zip(ranges, ranges[1:]))
