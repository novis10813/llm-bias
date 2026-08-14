"""Shared prompt rendering and tokenizer contract.

This module contains only model/tokenizer-neutral mechanics. Experiment packages
own the meaning of prompts, entities, answers, and spans; they consume these
helpers and retain compatibility facades where their public APIs predate this
contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TokenSpan:
    """A character occurrence and its contiguous token representation."""

    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(encoded: Any, name: str, default: Any = None) -> Any:
    if hasattr(encoded, name):
        return getattr(encoded, name)
    if isinstance(encoded, dict):
        return encoded.get(name, default)
    return default


def input_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> list[int]:
    """Return one-dimensional integer IDs for one input string."""
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    values = _field(encoded, "input_ids")
    if values is None:
        raise TypeError("tokenizer output does not contain input_ids")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]


def format_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    use_chat_template: bool,
    enable_thinking: bool = False,
) -> str:
    """Render messages while preserving the raw-prompt compatibility mode."""
    if not use_chat_template:
        return "\n\n".join(message["content"] for message in messages)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "use_chat_template=True but the tokenizer has no chat template; "
            "pass --raw-prompt for a base model"
        )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer chat template must return text when tokenize=False")
    return rendered


def format_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
    enable_thinking: bool = False,
) -> str:
    """Render one user prompt; retained as the canonical legacy facade."""
    return format_messages(
        tokenizer,
        [{"role": "user", "content": prompt}],
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )


def decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token without special-token or whitespace normalization."""
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def token_span(
    tokenizer: Any, text: str, start: int, end: int, *, add_special_tokens: bool = True
) -> tuple[int, int] | None:
    """Map a character range to the overlapping contiguous token range."""
    if start < 0 or end <= start or end > len(text):
        raise ValueError(f"invalid character span {(start, end)} for text of length {len(text)}")
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = _field(encoded, "offset_mapping")
    specials = _field(encoded, "special_tokens_mask", [False] * len(offsets))
    if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets, specials = offsets[0], specials[0]
    selected = [
        index
        for index, ((token_start, token_end), special) in enumerate(zip(offsets, specials, strict=True))
        if not special and token_end > token_start and token_start < end and token_end > start
    ]
    return (min(selected), max(selected) + 1) if selected else None


def span_record(tokenizer: Any, text: str, occurrence: tuple[int, int]) -> TokenSpan:
    """Create a serializable token span record for one character occurrence."""
    char_start, char_end = occurrence
    span = token_span(tokenizer, text, char_start, char_end)
    if span is None:
        raise ValueError(f"could not map character span {occurrence}")
    ids = input_ids(tokenizer, text)
    return TokenSpan(char_start, char_end, span[0], span[1], ids[span[0] : span[1]])


def all_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Return non-overlapping occurrences of value in text."""
    if not value:
        raise ValueError("value must not be empty")
    occurrences: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = text.find(value, offset)
        if start < 0:
            return occurrences
        occurrences.append((start, start + len(value)))
        offset = start + len(value)


def continuation_token_ids(tokenizer: Any, prompt: str, candidate: str) -> list[int]:
    """Return IDs added by appending candidate to an already-tokenized prompt."""
    prefix = input_ids(tokenizer, prompt, add_special_tokens=False)
    combined = input_ids(tokenizer, prompt + candidate, add_special_tokens=False)
    if combined[: len(prefix)] != prefix:
        raise ValueError("prompt is not an exact token prefix of prompt+candidate")
    suffix = combined[len(prefix) :]
    if not suffix:
        raise ValueError("candidate produced no continuation token")
    return suffix


def find_token_subsequence(sequence: list[int], target: list[int]) -> tuple[int, int]:
    """Locate target IDs inside sequence, or return the conservative full range."""
    if not target:
        return 0, len(sequence)
    width = len(target)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == target:
            return start, start + width
    return 0, len(sequence)
