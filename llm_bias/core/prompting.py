"""Backward-compatible facade for shared prompt/tokenization helpers."""

from .prompt_input import (
    decode_token,
    find_token_subsequence,
    format_messages,
    format_prompt,
)

__all__ = [
    "decode_token",
    "find_token_subsequence",
    "format_messages",
    "format_prompt",
]
