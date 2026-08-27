"""Deterministic positive/negative valence pairing for vocabulary readout.

Raw baseline trial rows (``condition == "attribute"``) carry exactly four
evidence items: buy/sell x qual/quant.  This module validates those rows,
deterministically selects source trials per ticker under a stable hash seed,
and renders one canonical two-item user prompt per valence condition:
``positive`` uses buy qual + buy quant, ``negative`` uses sell qual + sell
quant.  The original four-item prompt body is never parsed or reused; only the
structured evidence fields are consumed.  Source trial identity (trial
key/index/set index) and evidence item hashes are preserved for provenance;
unneeded raw metadata (marketcap, original prompt text, per-row seed) is not.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from llm_bias.core.artifact_paths import sha256_json, stable_record_id

SIDES = ("buy", "sell")
KINDS = ("qual", "quant")
CONDITIONS = ("positive", "negative")
CONDITION_SIDES = {"positive": "buy", "negative": "sell"}

PROMPT_TEMPLATE_VERSION = "canonical-valence-v1"
PROMPT_HEADER = (
    "Refer to the evidence below to make a final investment decision.\n"
    "Stock Ticker: [{ticker}]\n"
    "Stock Name: [{name}]\n"
    "--- Evidence ---\n"
)
PROMPT_INSTRUCTION = (
    '---\nRespond with one valid JSON object containing only the keys '
    '"decision" (buy | sell) and "reason" (brief justification). Do not choose hold.'
)


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"raw trial row is missing field {field!r}")
    return value


def _evidence_items(evidence: Any) -> list[tuple[str, str, str]]:
    if not isinstance(evidence, list):
        raise ValueError("raw trial row evidence must be a list")
    items = []
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("evidence items must be objects")
        side = item.get("side")
        kind = item.get("kind")
        if side not in SIDES or kind not in KINDS:
            raise ValueError(f"evidence item has invalid side/kind: {side!r}/{kind!r}")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"evidence item {side}/{kind} has empty text")
        items.append((side, kind, text))
    return items


def has_balanced_valence_evidence(evidence: Any) -> bool:
    """Return whether evidence has exactly one item for every side/kind cell."""
    items = _evidence_items(evidence)
    keys = [(side, kind) for side, kind, _text in items]
    required = [(side, kind) for side in SIDES for kind in KINDS]
    return len(keys) == len(required) and sorted(keys) == sorted(required)


def _validate_evidence(evidence: Any) -> dict[tuple[str, str], str]:
    items = _evidence_items(evidence)
    result: dict[tuple[str, str], str] = {}
    for side, kind, text in items:
        if (side, kind) in result:
            raise ValueError(f"evidence has a duplicate item for {side}/{kind}")
        result[(side, kind)] = text
    missing = [(side, kind) for side in SIDES for kind in KINDS if (side, kind) not in result]
    if missing:
        raise ValueError(f"evidence is missing items: {missing}")
    return result


def validate_raw_trial_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one raw trial row for valence pairing and return its identity."""
    condition = row.get("condition")
    if condition != "attribute":
        raise ValueError(
            f"raw trial row has condition {condition!r}; valence readout requires 'attribute'"
        )
    ticker = _require_text(row, "ticker")
    name = _require_text(row, "name")
    sector = _require_text(row, "sector")
    _require_text(row, "prompt")
    trial_key = _require_text(row, "trial_key")
    trial_index = row.get("trial_index")
    set_index = row.get("set_index")
    if not isinstance(trial_index, int) or isinstance(trial_index, bool):
        raise ValueError("raw trial row trial identity requires integer trial_index")
    if not isinstance(set_index, int) or isinstance(set_index, bool):
        raise ValueError("raw trial row trial identity requires integer set_index")
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "trial_key": trial_key,
        "trial_index": trial_index,
        "set_index": set_index,
        "evidence": _validate_evidence(row.get("evidence")),
    }


def evidence_item_hash(side: str, kind: str, text: str) -> str:
    """Stable SHA-256 of one structured evidence item."""
    return sha256_json({"side": side, "kind": kind, "text": text})


def _selection_key(ticker: str, trial_key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{ticker}:{trial_key}".encode()).hexdigest()


def select_valence_trials(
    rows: Sequence[Mapping[str, Any]],
    *,
    sector: str,
    split_assignments: Mapping[str, str],
    split_name: str,
    trials_per_ticker: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Deterministically select up to ``trials_per_ticker`` attribute rows per ticker.

    Selection is stable-hash ordered (``sha256(seed:ticker:trial_key)``), so the
    same inputs and seed always yield the same trials in the same order.
    """
    if trials_per_ticker < 1:
        raise ValueError("trials_per_ticker must be positive")
    if not sector.strip():
        raise ValueError("sector must be a non-empty string")
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        # The raw trial plan mixes conditions, sectors, and valid attribute
        # compositions. Rows outside the requested scope are skipped. Once a
        # row has the requested balanced composition, all fields fail closed.
        if row.get("condition") != "attribute":
            continue
        ticker = _require_text(row, "ticker")
        row_sector = _require_text(row, "sector")
        if row_sector != sector or split_assignments.get(ticker) != split_name:
            continue
        if not has_balanced_valence_evidence(row.get("evidence")):
            continue
        identity = validate_raw_trial_row(row)
        by_ticker[identity["ticker"]].append(dict(row))
    selected: list[dict] = []
    for ticker in sorted(by_ticker):
        ordered = sorted(
            by_ticker[ticker],
            key=lambda row: _selection_key(ticker, str(row["trial_key"]), seed),
        )
        selected.extend(ordered[:trials_per_ticker])
    if not selected:
        raise ValueError(f"no {sector!r} {split_name!r} attribute rows match the raw trial input")
    return selected


def render_valence_prompt_with_spans(
    *, ticker: str, name: str, qual_text: str, quant_text: str
) -> tuple[str, dict[str, list[int]]]:
    """Render one prompt and return exact character spans for both evidence items."""
    header = PROMPT_HEADER.format(ticker=ticker, name=name)
    prefix = header + "1. "
    qual_start = len(prefix)
    middle = "\n2. "
    quant_start = qual_start + len(qual_text) + len(middle)
    prompt = prefix + qual_text + middle + quant_text + "\n" + PROMPT_INSTRUCTION
    return prompt, {
        "qual": [qual_start, qual_start + len(qual_text)],
        "quant": [quant_start, quant_start + len(quant_text)],
    }


def render_valence_prompt(*, ticker: str, name: str, qual_text: str, quant_text: str) -> str:
    """Render the canonical two-item user prompt for one valence condition."""
    prompt, _spans = render_valence_prompt_with_spans(
        ticker=ticker, name=name, qual_text=qual_text, quant_text=quant_text
    )
    return prompt


def build_valence_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build one positive/negative raw-prompt pair from a validated source row."""
    identity = validate_raw_trial_row(row)
    evidence = identity["evidence"]
    prompts = {}
    evidence_char_spans = {}
    for condition, side in sorted(CONDITION_SIDES.items()):
        prompt, spans = render_valence_prompt_with_spans(
            ticker=identity["ticker"],
            name=identity["name"],
            qual_text=evidence[(side, "qual")],
            quant_text=evidence[(side, "quant")],
        )
        prompts[condition] = prompt
        evidence_char_spans[condition] = spans
    return {
        "record_id": stable_record_id("valence-pair", identity["trial_key"]),
        "source_trial_key": identity["trial_key"],
        "source_trial_index": identity["trial_index"],
        "set_index": identity["set_index"],
        "ticker": identity["ticker"],
        "name": identity["name"],
        "sector": identity["sector"],
        "evidence_char_spans": evidence_char_spans,
        "evidence_item_hashes": {
            f"{side}_{kind}": evidence_item_hash(side, kind, text)
            for (side, kind), text in sorted(evidence.items())
        },
        "prompts": prompts,
    }


def resolve_valence_layers(layers: Sequence[int], final_layer: int) -> list[int]:
    """Return the requested layers plus the final model layer, sorted and unique."""
    resolved = sorted({int(layer) for layer in layers})
    for layer in resolved:
        if layer < 0 or layer > final_layer:
            raise ValueError(f"valence layer {layer} out of range 0..{final_layer}")
    if final_layer not in resolved:
        resolved.append(final_layer)
    return sorted(resolved)


__all__ = [
    "CONDITIONS",
    "CONDITION_SIDES",
    "KINDS",
    "PROMPT_HEADER",
    "PROMPT_INSTRUCTION",
    "PROMPT_TEMPLATE_VERSION",
    "SIDES",
    "build_valence_pair",
    "evidence_item_hash",
    "has_balanced_valence_evidence",
    "render_valence_prompt",
    "render_valence_prompt_with_spans",
    "resolve_valence_layers",
    "select_valence_trials",
    "validate_raw_trial_row",
]
