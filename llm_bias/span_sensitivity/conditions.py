"""Deterministic header-span conditions for the sensitivity pilot."""
from __future__ import annotations

import codecs
import hashlib
import re
from dataclasses import dataclass

CONDITIONS = (
    "original",
    "anonymous_ticker",
    "anonymous_name",
    "anonymous_identity",
    "same_sector_swap",
    "constructed_identity",
    "name_form_control",
)

_CONSTRUCTED_IDENTITIES = (
    ("AVSY", "Avenor Systems, Inc."),
    ("BLDG", "Bluegate Digital, Inc."),
    ("CRVX", "Corvexia Technologies, Inc."),
    ("DLPH", "Delphi Ridge Software, Inc."),
    ("ELMR", "Elmriver Networks, Inc."),
    ("FSTN", "Foxton Data Systems, Inc."),
    ("GRDN", "Greystone Logic, Inc."),
    ("HLIX", "Helix North Computing, Inc."),
)
_TICKER_PATTERN = re.compile(r"^Stock Ticker: \[(?P<value>[^\]\r\n]+)\]$", re.MULTILINE)
_NAME_PATTERN = re.compile(r"^Stock Name: \[(?P<value>[^\]\r\n]+)\]$", re.MULTILINE)


@dataclass(frozen=True)
class Identity:
    ticker: str
    name: str


@dataclass(frozen=True)
class PromptCondition:
    condition: str
    prompt: str
    replacement: Identity
    mutation_fields: tuple[str, ...]
    ticker_mentions_outside_header: int
    name_mentions_outside_header: int


def _stable_index(value: str, size: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) % size


def _match_header(prompt: str, pattern: re.Pattern[str], label: str) -> re.Match[str]:
    matches = list(pattern.finditer(prompt))
    if len(matches) != 1:
        raise ValueError(f"prompt must contain exactly one bracketed {label} header")
    return matches[0]


def parse_header_identity(prompt: str) -> Identity:
    """Return the identity from the two canonical bracketed header lines."""
    ticker = _match_header(prompt, _TICKER_PATTERN, "ticker")
    name = _match_header(prompt, _NAME_PATTERN, "name")
    return Identity(ticker.group("value"), name.group("value"))


def _replace_header(prompt: str, *, ticker: str | None = None, name: str | None = None) -> str:
    if ticker is not None:
        match = _match_header(prompt, _TICKER_PATTERN, "ticker")
        prompt = prompt[: match.start("value")] + ticker + prompt[match.end("value") :]
    if name is not None:
        match = _match_header(prompt, _NAME_PATTERN, "name")
        prompt = prompt[: match.start("value")] + name + prompt[match.end("value") :]
    return prompt


def _outside_header(prompt: str) -> str:
    return _TICKER_PATTERN.sub("", _NAME_PATTERN.sub("", prompt))


def _literal_count(text: str, value: str) -> int:
    return len(re.findall(re.escape(value), text, flags=re.IGNORECASE))


def name_form_identity(source: Identity) -> Identity:
    """Apply ROT13 while preserving character class, case, punctuation, and length."""
    return Identity(codecs.decode(source.ticker, "rot_13"), codecs.decode(source.name, "rot_13"))


def constructed_identity(key: str, *, seed: int = 0) -> Identity:
    ticker, name = _CONSTRUCTED_IDENTITIES[_stable_index(key, len(_CONSTRUCTED_IDENTITIES), seed)]
    return Identity(ticker, name)


def build_conditions(
    prompt: str,
    *,
    source: Identity,
    peer: Identity,
    key: str,
    seed: int = 0,
) -> list[PromptCondition]:
    """Build header-only identity variants; evidence text is never modified."""
    parsed = parse_header_identity(prompt)
    if parsed != source:
        raise ValueError(
            "prompt header identity does not match CSV metadata: "
            f"header={parsed!r}, row={source!r}"
        )
    if peer == source:
        raise ValueError("same-sector peer must differ from the source identity")
    outside = _outside_header(prompt)
    ticker_mentions = _literal_count(outside, source.ticker)
    name_mentions = _literal_count(outside, source.name)
    anonymous = Identity("ANON", "Anonymous Company")
    fabricated = constructed_identity(key, seed=seed)
    form = name_form_identity(source)
    definitions = (
        ("original", source, ()),
        ("anonymous_ticker", Identity(anonymous.ticker, source.name), ("ticker",)),
        ("anonymous_name", Identity(source.ticker, anonymous.name), ("name",)),
        ("anonymous_identity", anonymous, ("ticker", "name")),
        ("same_sector_swap", peer, ("ticker", "name")),
        ("constructed_identity", fabricated, ("ticker", "name")),
        ("name_form_control", form, ("ticker", "name")),
    )
    rows = []
    for condition, replacement, fields in definitions:
        rendered = _replace_header(
            prompt,
            ticker=replacement.ticker if "ticker" in fields else None,
            name=replacement.name if "name" in fields else None,
        )
        rows.append(
            PromptCondition(
                condition,
                rendered,
                replacement,
                fields,
                ticker_mentions,
                name_mentions,
            )
        )
    return rows


__all__ = [
    "CONDITIONS",
    "Identity",
    "PromptCondition",
    "build_conditions",
    "constructed_identity",
    "name_form_identity",
    "parse_header_identity",
]
