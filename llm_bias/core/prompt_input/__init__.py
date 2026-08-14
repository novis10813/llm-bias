"""Shared prompt rendering and tokenizer-alignment contract.

The package exposes stable imports while keeping data contracts separate from
encoding mechanics. Experiment packages own research semantics and consume only
these model/tokenizer-neutral primitives.
"""

from .contracts import TokenSpan
from .encoding import (
    all_occurrences,
    continuation_token_ids,
    decode_token,
    find_token_subsequence,
    format_messages,
    format_prompt,
    input_ids,
    span_record,
    token_span,
)

__all__ = [
    "TokenSpan",
    "all_occurrences",
    "continuation_token_ids",
    "decode_token",
    "find_token_subsequence",
    "format_messages",
    "format_prompt",
    "input_ids",
    "span_record",
    "token_span",
]
