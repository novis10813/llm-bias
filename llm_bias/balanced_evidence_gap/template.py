"""Frozen Phase 2 constants and prompt-family construction.

All frozen values (evidence sentences, companies, variant grid) are defined
in docs/balanced-evidence-gap/details/proposal-phase2.md; changes require a new
protocol version, not an edit here.
"""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "balanced-evidence-gap-phase2-v1"
DATASET = "balanced-evidence-gap-phase2"

DECISION_PREFIX = '{"decision": "'
POSITIVE_CANDIDATE = "buy"
NEGATIVE_CANDIDATE = "sell"

# proposal-phase2.md §3.1 — 16 companies, investment-dial test split.
TARGET_TICKERS: dict[str, list[str]] = {
    "Information Technology": ["AMAT", "GLW", "HPE", "IT"],
    "Financials": ["AXP", "BLK", "C", "GS"],
    "Health Care": ["ABT", "BDX", "DHR", "SYK"],
    "Industrials": ["CSX", "DE", "HON", "NSC"],
}
ALL_TICKERS: list[str] = [t for tickers in TARGET_TICKERS.values() for t in tickers]
SECTOR_OF: dict[str, str] = {t: s for s, ts in TARGET_TICKERS.items() for t in ts}

# proposal-phase2.md §4.2 — frozen shared-evidence template (verbatim).
EVIDENCE_P1 = "Q3 revenue grew 14% year over year, exceeding analyst consensus by 3%."
EVIDENCE_P2 = "Free cash flow reached a record quarterly high, up 22% year over year."
EVIDENCE_N1 = "Gross margin contracted 300 basis points due to rising input costs."
EVIDENCE_N2 = "Full-year revenue guidance was revised downward by 6%."

# order 0 = listed order (1-4); order 1 = exact reverse (4-1).
EVIDENCE_ORDERS: tuple[tuple[str, str, str, str], ...] = (
    (EVIDENCE_P1, EVIDENCE_P2, EVIDENCE_N1, EVIDENCE_N2),
    (EVIDENCE_N2, EVIDENCE_N1, EVIDENCE_P2, EVIDENCE_P1),
)

REVERSE_OPTIONS: tuple[bool, ...] = (False, True)

# L15/n8490 investment-dial coordinate (H4 descriptive readout).
DIAL_LAYER = 15
DIAL_NEURON = 8490

EVIDENCE_MARKER = "— Evidence —"
EVIDENCE_CLOSE = "\n\n—\n\n"
TICKER_LINE_PREFIX = "Stock Ticker: ["
NAME_LINE_PREFIX = "Stock Name: ["


def build_prompt(ticker: str, name: str, order: int, reverse: bool) -> str:
    """One cross-entity probe prompt (shared evidence, frozen template)."""
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}")
    options = '"sell" or "buy"' if reverse else '"buy" or "sell"'
    items = EVIDENCE_ORDERS[order]
    evidence = "\n".join(f"- {item}" for item in items)
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"Stock Ticker: [{ticker}]\n\nStock Name: [{name}]\n\n"
        f"{EVIDENCE_MARKER}\n\n"
        f"{evidence}\n\n"
        "—\n\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        f'"decision": {options}\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


@dataclass(frozen=True)
class ProbeRow:
    """One prepared prompt of the cross-entity probe family."""

    id: str
    ticker: str
    name: str
    sector: str
    reverse: bool
    order: int
    prompt: str
    formatted: str
    prompt_ids: list[int]
    entity_span: tuple[int, int]
    evidence_span: tuple[int, int]
    instruction_span: tuple[int, int]
    final_position: int
    entity_position: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "reverse": self.reverse,
            "order": self.order,
            "prompt": self.prompt,
            "formatted": self.formatted,
            "prompt_ids": self.prompt_ids,
            "entity_span": list(self.entity_span),
            "evidence_span": list(self.evidence_span),
            "instruction_span": list(self.instruction_span),
            "final_position": self.final_position,
            "entity_position": self.entity_position,
        }


def variant_id(ticker: str, reverse: bool, order: int) -> str:
    return f"{ticker}:rev{int(reverse)}:ord{order}"
