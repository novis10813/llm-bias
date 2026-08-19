"""Immutable synthetic pilot constants."""
from __future__ import annotations
import hashlib
LABELS = tuple(str(i) for i in range(9))
SCORES = tuple(range(-4,5))
TEMPLATES = {
    "negative": "[ENTITY] announced that next quarter's revenue will fall short of market expectations.",
    "positive": "[ENTITY] announced that next quarter's revenue will exceed market expectations.",
    "neutral": "[ENTITY] released its revenue forecast for next quarter.",
    "positive_guidance_raise": "[ENTITY] raised its next-quarter financial guidance well above consensus estimates due to surging customer demand.",
    "positive_record_bookings": "[ENTITY] reported record quarterly order bookings and projected substantial revenue expansion for next quarter.",
    "positive_momentum": "[ENTITY] announced robust sales momentum across all core business lines, forecasting double-digit revenue growth next quarter.",
    "neutral_reaffirm": "[ENTITY] reaffirmed its previous financial guidance and expects next-quarter revenue to remain in line with current forecasts.",
    "neutral_scheduled": "[ENTITY] scheduled its upcoming quarterly earnings release and provided preliminary baseline financial metrics.",
    "neutral_seasonal": "[ENTITY] reported steady operational performance and projected next-quarter revenue consistent with historical seasonal trends.",
    "negative_guidance_cut": "[ENTITY] lowered its next-quarter revenue guidance, citing widespread customer spending cutbacks and macroeconomic headwinds.",
    "negative_order_cancellations": "[ENTITY] reported significant cancellations of major customer orders and forecasted a steep decline in next-quarter revenue.",
    "negative_competitive_drop": "[ENTITY] announced that next-quarter sales and revenue are expected to drop sharply due to intensifying competitive pressure.",
}
TEMPLATE_SENTIMENTS = {
    "negative": "negative",
    "positive": "positive",
    "neutral": "neutral",
    "positive_guidance_raise": "positive",
    "positive_record_bookings": "positive",
    "positive_momentum": "positive",
    "neutral_reaffirm": "neutral",
    "neutral_scheduled": "neutral",
    "neutral_seasonal": "neutral",
    "negative_guidance_cut": "negative",
    "negative_order_cancellations": "negative",
    "negative_competitive_drop": "negative",
}
SCORING_INSTRUCTION = "Rate the expected next-quarter revenue outcome from -4 (most negative) to +4 (most positive), with 0 meaning neutral. Respond with exactly one score label according to this mapping: 0=-4, 1=-3, 2=-2, 3=-1, 4=0, 5=+1, 6=+2, 7=+3, 8=+4.\nScore label:"
BASELINE_ENTITY = "The company"
TEMPLATE_HASH = hashlib.sha256("\n".join(f"{k}:{v}" for k,v in TEMPLATES.items()).encode()).hexdigest()
LABEL_HASH = hashlib.sha256(("|".join(LABELS)+"|"+SCORING_INSTRUCTION).encode()).hexdigest()

