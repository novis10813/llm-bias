"""Immutable synthetic pilot constants."""
from __future__ import annotations
import hashlib
LABELS = tuple(str(i) for i in range(9))
SCORES = tuple(range(-4,5))
TEMPLATES = {
 "negative": "[ENTITY] announced that next quarter's revenue will fall short of market expectations.",
 "positive": "[ENTITY] announced that next quarter's revenue will exceed market expectations.",
 "neutral": "[ENTITY] released its revenue forecast for next quarter.",
}
SCORING_INSTRUCTION = "Rate the expected next-quarter revenue outcome from -4 (most negative) to +4 (most positive), with 0 meaning neutral. Respond with exactly one score label according to this mapping: 0=-4, 1=-3, 2=-2, 3=-1, 4=0, 5=+1, 6=+2, 7=+3, 8=+4.\nScore label:"
BASELINE_ENTITY = "The company"
TEMPLATE_HASH = hashlib.sha256("\n".join(f"{k}:{v}" for k,v in TEMPLATES.items()).encode()).hexdigest()
LABEL_HASH = hashlib.sha256(("|".join(LABELS)+"|"+SCORING_INSTRUCTION).encode()).hexdigest()
