"""Shared prompt formatting and token-alignment helpers."""

from __future__ import annotations

from typing import Any


def decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token without hiding special tokens or normalizing spaces."""
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def format_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
    enable_thinking: bool = False,
) -> str:
    """Optionally wrap a raw prompt as one user turn for a chat model."""
    if not use_chat_template:
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "use_chat_template=True but the tokenizer has no chat template; "
            "pass --raw-prompt for a base model"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def find_token_subsequence(
    sequence: list[int], target: list[int]
) -> tuple[int, int]:
    """Locate raw-message token IDs inside a formatted prompt."""
    if not target:
        return 0, len(sequence)
    width = len(target)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == target:
            return start, start + width
    return 0, len(sequence)
