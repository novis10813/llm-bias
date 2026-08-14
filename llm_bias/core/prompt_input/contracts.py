"""Data contracts for tokenizer-aligned prompt input."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
