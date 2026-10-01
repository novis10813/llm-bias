"""Generation-only, uniquely rendered stance prompts with audited token spans.

This renderer consumes reviewed EvidencePair records, not legacy scoring pools.
It does not constrain or parse generation: the schema and fixed instruction are
input provenance only. Ordered output/schema enforcement belongs to decoding.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.population import PopulationMember, validate_members
from llm_bias.core.stance_evidence import EvidencePair, validate_evidence_pair

DECISION_INSTRUCTION = (
    'Choose an investment stance for the named stock using the two evidence slots '
    'as shared market context, not company-specific financial results. '
    'Return only a JSON object with exactly these keys in this order: "decision", '
    'then "reason". The "decision" must be either "buy" or "sell". The "reason" '
    'must be a nonblank string explaining the decision using the company and '
    'the two evidence slots. Do not include other keys or text outside the JSON object.'
)
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["buy", "sell"]},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}
DECISION_TEMPLATE = (
    "{entity}\n\nEvidence slot 1:\n{evidence1}"
    "\n\nEvidence slot 2:\n{evidence2}\n\n{instruction}"
)
_ENTITY_TEMPLATE = "Company ticker: {ticker}\nCompany name: {name}\nCompany sector: {sector}"
_RESERVED_CHAT_KWARGS = {"messages", "tokenize", "add_generation_prompt", "enable_thinking"}


class UnsupportedTokenizerError(ValueError):
    """The tokenizer cannot reliably map the actual inference input to offsets."""


@dataclass(frozen=True)
class WrapperPolicy:
    """Explicit inference wrapper and special-token policy, recorded in provenance.

    ``use_chat_template`` and ``add_special_tokens`` have no inferred defaults.
    A supplied ``system_message`` is context only: task vocabulary JSON, decision,
    reason, buy and sell is reserved to the one fixed user instruction. This is
    a conservative lexical guard, not semantic approval of arbitrary prose.
    Chat kwargs must be JSON-compatible and cannot override messages, tokenize,
    add_generation_prompt, or enable_thinking. Their values and the tokenizer's
    actual chat template are copied into DecisionPrompt provenance.
    """

    use_chat_template: bool
    add_special_tokens: bool
    system_message: str | None = None
    enable_thinking: bool = False
    chat_template_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class DecisionSpan:
    """Named rendered-text range and overlapping inference-token range.

    Character offsets refer to ``DecisionPrompt.rendered_text`` (wrapper included)
    and token offsets index ``inference_token_ids`` (special tokens included per
    policy). ``token_sha256`` hashes the canonical JSON list of these IDs;
    ``text_sha256`` hashes exact UTF-8 span text. Boundary-crossing tokens may
    include adjacent delimiters; the mapping never retokenizes span text alone.
    """

    role: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_ids: tuple[int, ...]
    token_sha256: str
    text_sha256: str


class _CanonicalWrapperPolicy:
    """Defensive JSON copies, including dataclass construction/read/export.

    A descriptor keeps ``wrapper_policy`` as a public dataclass field (and the
    constructor/asdict format unchanged), while only immutable canonical bytes
    are stored on the frozen instance.
    """

    def __get__(self, instance: Any, owner: Any = None) -> dict[str, Any]:
        if instance is None:
            raise AttributeError("wrapper_policy requires an instance")
        return json.loads(instance._wrapper_policy_json)

    def __set__(self, instance: Any, value: dict[str, Any]) -> None:
        object.__setattr__(instance, "_wrapper_policy_json", canonical_json_bytes(value))


@dataclass(frozen=True)
class DecisionPrompt:
    """Generation input and four separately named, tokenizer-aligned spans.

    ``raw_text`` is the fixed body before wrappers; ``rendered_text`` is the exact
    inference string. ``template_sha256`` covers body/entity templates and the
    instruction. ``schema_sha256`` covers DECISION_SCHEMA and explicit key order.
    ``prompt_sha256`` covers both strings, inference IDs, those hashes and the
    immutable canonical wrapper policy. ``wrapper_policy`` returns a defensive
    JSON copy, including nested kwargs/templates; dataclass exports retain the
    same public dict field. No answer prefix, answer IDs or scoring fields exist.
    """

    raw_text: str
    rendered_text: str
    inference_token_ids: tuple[int, ...]
    entity_span: DecisionSpan
    evidence1_span: DecisionSpan
    evidence2_span: DecisionSpan
    instruction_span: DecisionSpan
    prompt_sha256: str
    template_sha256: str
    schema_sha256: str
    wrapper_policy: dict[str, Any] = _CanonicalWrapperPolicy()


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-compatible") from exc


def _wrapper_record(tokenizer: Any, policy: WrapperPolicy) -> dict[str, Any]:
    if not isinstance(policy, WrapperPolicy):
        raise ValueError("wrapper_policy must be a WrapperPolicy record")
    for field in ("use_chat_template", "add_special_tokens", "enable_thinking"):
        if not isinstance(getattr(policy, field), bool):
            raise ValueError(f"{field} must be a boolean")
    system = policy.system_message
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("system_message must be an explicit nonblank string or None")
        if re.search(r"\b(json|decision|reason|buy|sell)\b", system, re.IGNORECASE):
            raise ValueError("system_message must not duplicate task instructions or task vocabulary")
    kwargs = policy.chat_template_kwargs
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping) or any(not isinstance(key, str) for key in kwargs):
        raise ValueError("chat_template_kwargs must be a string-keyed JSON mapping")
    if _RESERVED_CHAT_KWARGS & set(kwargs):
        raise ValueError("chat_template_kwargs cannot override reserved wrapper arguments")
    template = None
    if policy.use_chat_template:
        template = getattr(tokenizer, "chat_template", None)
        if not template:
            raise ValueError("use_chat_template=True but tokenizer has no chat template")
    return {
        "use_chat_template": policy.use_chat_template,
        "add_special_tokens": policy.add_special_tokens,
        "system_message": system,
        "enable_thinking": policy.enable_thinking,
        "chat_template_kwargs": _json_copy(dict(kwargs), "chat_template_kwargs"),
        "tokenizer_chat_template": _json_copy(template, "tokenizer chat template"),
    }


def _field(encoded: Any, name: str) -> Any:
    return encoded.get(name) if isinstance(encoded, Mapping) else getattr(encoded, name, None)


def _sequence(value: Any, label: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise UnsupportedTokenizerError(f"tokenizer {label} must be a one-dimensional sequence")
    return list(value)


def _ids(encoded: Any) -> tuple[int, ...]:
    values = _sequence(_field(encoded, "input_ids"), "input_ids")
    if not values or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in values):
        raise UnsupportedTokenizerError("tokenizer input_ids must be nonempty integer tokens")
    return tuple(values)


def _inference_encoding(
    tokenizer: Any, text: str, add_special_tokens: bool,
) -> tuple[tuple[int, ...], list[tuple[int, int]], list[bool]]:
    try:
        ids = _ids(tokenizer(text, add_special_tokens=add_special_tokens))
        encoded = tokenizer(text, add_special_tokens=add_special_tokens,
                            return_offsets_mapping=True, return_special_tokens_mask=True)
    except (TypeError, ValueError, NotImplementedError, AttributeError) as exc:
        raise UnsupportedTokenizerError(f"tokenizer inference offsets are unsupported: {exc}") from exc
    if _ids(encoded) != ids:
        raise UnsupportedTokenizerError("offset-mapped token IDs differ from actual inference tokens")
    raw_offsets = _sequence(_field(encoded, "offset_mapping"), "offset_mapping")
    raw_mask = _sequence(_field(encoded, "special_tokens_mask"), "special_tokens_mask")
    if len(raw_offsets) != len(ids) or len(raw_mask) != len(ids):
        raise UnsupportedTokenizerError("tokenizer offsets/mask length differs from inference tokens")
    if any(not isinstance(flag, (int, bool)) or flag not in (0, 1) for flag in raw_mask):
        raise UnsupportedTokenizerError("tokenizer special-token mask must contain only 0 or 1")
    mask = [bool(flag) for flag in raw_mask]
    offsets: list[tuple[int, int]] = []
    previous_start = previous_end = 0
    previous_content: tuple[int, int] | None = None
    repeated = 0
    for raw, special in zip(raw_offsets, mask, strict=True):
        values = _sequence(raw, "offset pair")
        if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise UnsupportedTokenizerError("tokenizer offset pairs must contain two integer character offsets")
        start, end = values
        if not 0 <= start <= end <= len(text):
            raise UnsupportedTokenizerError("tokenizer offsets are outside rendered text or backwards")
        if not special:
            if start < previous_start or end < previous_end:
                raise UnsupportedTokenizerError("non-special token offsets must be monotonic")
            if start == end:
                # ByteLevel trimming can place a whitespace token at the right
                # edge of its source whitespace. An empty offset is not itself
                # coverage: _span still requires all substantive characters.
                if not ((start > 0 and text[start - 1].isspace())
                        or (start < len(text) and text[start].isspace())):
                    raise UnsupportedTokenizerError("zero-width token offsets must adjoin whitespace")
            else:
                repeated = repeated + 1 if previous_content == (start, end) else 1
                if repeated > 1:
                    # UTF-8 byte splits may share one character's offsets, but
                    # many tokens sharing a broad range provide no locations.
                    if end - start != 1 or repeated > len(text[start:end].encode("utf-8")):
                        raise UnsupportedTokenizerError("repeated token offsets are not reliable Unicode byte splits")
                previous_content = (start, end)
            previous_start, previous_end = start, end
        offsets.append((start, end))
    return ids, offsets, mask


def _span(
    role: str, text: str, start: int, end: int,
    ids: tuple[int, ...], offsets: list[tuple[int, int]], specials: list[bool],
) -> DecisionSpan:
    selected = [i for i, ((a, b), special) in enumerate(zip(offsets, specials, strict=True))
                if not special and a < b and a < end and b > start]
    # Retain trimmed whitespace IDs between substantive overlapping tokens, but
    # never bridge an interleaved special token or count empty offsets as cover.
    if not selected or any(specials[selected[0]:selected[-1] + 1]):
        raise UnsupportedTokenizerError(f"tokenizer offsets cannot map contiguous {role} tokens")
    covered = start
    for i in selected:
        a, b = offsets[i]
        # Some tokenizers omit separator whitespace from their offsets. That is
        # harmless; missing any substantive span character is not reliable.
        if a > covered and text[covered:min(a, end)].strip():
            raise UnsupportedTokenizerError(f"tokenizer offsets leave a gap in {role}")
        covered = max(covered, b)
    if covered < end and text[covered:end].strip():
        raise UnsupportedTokenizerError(f"tokenizer offsets do not cover {role}")
    token_start, token_end = selected[0], selected[-1] + 1
    span_ids = ids[token_start:token_end]
    return DecisionSpan(role, start, end, token_start, token_end, span_ids,
                        sha256_json(span_ids), sha256_bytes(text[start:end].encode("utf-8")))


def render_decision_prompt(
    tokenizer: Any,
    member: PopulationMember,
    pair: EvidencePair,
    *,
    wrapper_policy: WrapperPolicy,
) -> DecisionPrompt:
    """Render one validated row and audit offsets against actual inference IDs.

    Population coverage is enforced separately by validate_evidence_pairs. Body
    offsets are accumulated while rendering, never found by searching evidence
    or delimiters. Only the whole exact body is located in the final wrapper;
    missing/rewritten/duplicate bodies and duplicate fixed instructions fail.
    Unsupported/missing/unreliable offsets raise UnsupportedTokenizerError;
    there is no token subsequence, full-range, or span-retokenization fallback.
    """
    member = validate_members([member])[0]
    pair = validate_evidence_pair(pair)
    if member.ticker != pair.ticker:
        raise ValueError("evidence pair ticker differs from population member ticker")
    provenance = _wrapper_record(tokenizer, wrapper_policy)
    entity = _ENTITY_TEMPLATE.format(ticker=member.ticker, name=member.name, sector=member.sector)
    pieces = [("entity", entity), (None, "\n\nEvidence slot 1:\n"),
              ("evidence1", pair.evidence1.text), (None, "\n\nEvidence slot 2:\n"),
              ("evidence2", pair.evidence2.text), (None, "\n\n"),
              ("instruction", DECISION_INSTRUCTION)]
    ranges: dict[str, tuple[int, int]] = {}
    parts: list[str] = []
    cursor = 0
    for role, value in pieces:
        if role is not None:
            ranges[role] = (cursor, cursor + len(value))
        parts.append(value)
        cursor += len(value)
    raw = "".join(parts)
    messages = []
    if wrapper_policy.system_message is not None:
        messages.append({"role": "system", "content": wrapper_policy.system_message})
    messages.append({"role": "user", "content": raw})
    if wrapper_policy.use_chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=wrapper_policy.enable_thinking,
            **_json_copy(provenance["chat_template_kwargs"], "chat_template_kwargs"),
        )
    else:
        rendered = "\n\n".join(message["content"] for message in messages)
    if not isinstance(rendered, str):
        raise ValueError("tokenizer chat template must render a text body")
    body_offset = rendered.find(raw)
    # Check overlapping duplicates too, rather than relying on str.count alone.
    if body_offset < 0 or rendered.find(raw, body_offset + 1) >= 0:
        raise ValueError("rendered text must contain the exact unchanged body once")
    if rendered.count(DECISION_INSTRUCTION) != 1:
        raise ValueError("rendered text must contain the fixed instruction exactly once")
    ids, offsets, specials = _inference_encoding(tokenizer, rendered, wrapper_policy.add_special_tokens)
    spans = {role: _span(role, rendered, start + body_offset, end + body_offset, ids, offsets, specials)
             for role, (start, end) in ranges.items()}
    template_hash = sha256_json({"template": DECISION_TEMPLATE, "entity_template": _ENTITY_TEMPLATE,
                                "instruction": DECISION_INSTRUCTION})
    schema_hash = sha256_json({"schema": DECISION_SCHEMA, "key_order": ["decision", "reason"]})
    prompt_hash = sha256_json({"raw_text": raw, "rendered_text": rendered, "inference_token_ids": ids,
                              "template_sha256": template_hash, "schema_sha256": schema_hash,
                              "wrapper_policy": provenance})
    return DecisionPrompt(
        raw_text=raw, rendered_text=rendered, inference_token_ids=ids,
        entity_span=spans["entity"], evidence1_span=spans["evidence1"],
        evidence2_span=spans["evidence2"], instruction_span=spans["instruction"],
        prompt_sha256=prompt_hash, template_sha256=template_hash, schema_sha256=schema_hash,
        wrapper_policy=provenance,
    )
