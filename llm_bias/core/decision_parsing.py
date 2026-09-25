"""Buy/sell decision parsing for generated investment-decision JSON."""

from __future__ import annotations

import json
from typing import Any

FORMATS = (
    "bare_json", "fenced_json", "thought_bare_json", "thought_fenced_json",
    "harmony_final_bare_json", "harmony_final_fenced_json",
)

# skip_special_tokens=True removes Harmony's <|end|><|start|>...<|channel|>...<|message|> control
# tokens but keeps the plain role/channel words, so GPT-OSS's switch to the final channel always
# decodes as this literal, just as Gemma's thinking block decodes as a "thought\n" prefix.
HARMONY_FINAL_MARKER = "assistantfinal"


def parse_strict_decision(text: str) -> str | None:
    """Decision only when the whole generated text is one JSON object."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    decision = obj.get("decision") if isinstance(obj, dict) else None
    return decision if decision in ("buy", "sell") else None


def _unwrap(body: str, *, allow_thought: bool, kind_prefix: str) -> tuple[dict[str, Any] | None, str]:
    thought = allow_thought and body.startswith("thought\n")
    if thought:
        body = body[len("thought\n"):]
    fenced = body.startswith("```json\n")
    if fenced:
        if not body.endswith("\n```"):
            return None, "invalid"
        body = body[len("```json\n"):-len("\n```")]
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None, "invalid"
    if (not isinstance(obj, dict) or set(obj) != {"decision", "reason"}
            or obj["decision"] not in ("buy", "sell")
            or not isinstance(obj["reason"], str) or not obj["reason"].strip()):
        return None, "invalid"
    kind = kind_prefix + ("thought_" if thought else "") + ("fenced_json" if fenced else "bare_json")
    return obj, kind


def parse_complete_decision(text: str) -> tuple[str | None, str]:
    """Accept one whole two-key JSON object behind at most one exact chat-format wrapper."""
    if not isinstance(text, str):
        return None, "invalid"
    body = text.strip()
    obj, kind = _unwrap(body, allow_thought=True, kind_prefix="")
    if obj is None and (marker_at := body.rfind(HARMONY_FINAL_MARKER)) >= 0:
        remainder = body[marker_at + len(HARMONY_FINAL_MARKER):].strip()
        obj, kind = _unwrap(remainder, allow_thought=False, kind_prefix="harmony_final_")
    return (obj["decision"], kind) if obj is not None else (None, "invalid")
