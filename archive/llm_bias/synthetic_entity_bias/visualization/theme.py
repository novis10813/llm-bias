"""Shared, validated visual tokens for synthetic entity-bias charts."""

from __future__ import annotations

from typing import Any

from .contract import TEMPLATE_ORDER

LIGHT_COLORS = {
    "negative": "#2a78d6",
    "positive": "#eb6834",
    "neutral": "#1baf7a",
    "positive_guidance_raise": "#d95926",
    "positive_record_bookings": "#f18b62",
    "positive_momentum": "#b84818",
    "neutral_reaffirm": "#127854",
    "neutral_scheduled": "#26d698",
    "neutral_seasonal": "#42b883",
    "negative_guidance_cut": "#1e5699",
    "negative_order_cancellations": "#4c93eb",
    "negative_competitive_drop": "#143d73",
}
DARK_COLORS = {
    "negative": "#3987e5",
    "positive": "#d95926",
    "neutral": "#199e70",
    "positive_guidance_raise": "#f18b62",
    "positive_record_bookings": "#fa9d78",
    "positive_momentum": "#c95a2b",
    "neutral_reaffirm": "#26d698",
    "neutral_scheduled": "#42b883",
    "neutral_seasonal": "#5cdbb5",
    "negative_guidance_cut": "#4c93eb",
    "negative_order_cancellations": "#6ba5f2",
    "negative_competitive_drop": "#2a59a8",
}
MARKERS = {
    "negative": "o",
    "positive": "s",
    "neutral": "^",
    "positive_guidance_raise": "v",
    "positive_record_bookings": "<",
    "positive_momentum": ">",
    "neutral_reaffirm": "D",
    "neutral_scheduled": "d",
    "neutral_seasonal": "p",
    "negative_guidance_cut": "P",
    "negative_order_cancellations": "X",
    "negative_competitive_drop": "*",
}
SVG_MARKERS = {
    "negative": "circle",
    "positive": "square",
    "neutral": "triangle",
    "positive_guidance_raise": "triangle-down",
    "positive_record_bookings": "triangle-left",
    "positive_momentum": "triangle-right",
    "neutral_reaffirm": "diamond",
    "neutral_scheduled": "diamond-thin",
    "neutral_seasonal": "pentagon",
    "negative_guidance_cut": "cross",
    "negative_order_cancellations": "x",
    "negative_competitive_drop": "star",
}
DASHES = {
    "negative": "solid",
    "positive": "dashed",
    "neutral": "dashdot",
    "positive_guidance_raise": "solid",
    "positive_record_bookings": "dashed",
    "positive_momentum": "dotted",
    "neutral_reaffirm": "solid",
    "neutral_scheduled": "dashed",
    "neutral_seasonal": "dotted",
    "negative_guidance_cut": "solid",
    "negative_order_cancellations": "dashed",
    "negative_competitive_drop": "dotted",
}
HATCHES = {
    "negative": "///",
    "positive": "\\\\\\",
    "neutral": "xxx",
    "positive_guidance_raise": "///",
    "positive_record_bookings": "\\\\\\",
    "positive_momentum": "++",
    "neutral_reaffirm": "///",
    "neutral_scheduled": "\\\\\\",
    "neutral_seasonal": "++",
    "negative_guidance_cut": "///",
    "negative_order_cancellations": "\\\\\\",
    "negative_competitive_drop": "++",
}
LINE_WIDTH = 2.0
MARKER_SIZE = 8.0
PAPER_DPI = 300
PAPER_WIDTH_IN = 7.2
PAPER_FONT_SIZE = 9.0
PALETTE_VALIDATION = {
    "source": "dataviz reference palette",
    "pair_scope": "all three categorical slots",
    "light": "passed",
    "dark": "passed",
}


def template_style(template: str) -> dict[str, Any]:
    if template not in TEMPLATE_ORDER:
        raise ValueError(f"unknown template visual identity: {template}")
    return {
        "light": LIGHT_COLORS[template],
        "dark": DARK_COLORS[template],
        "marker": MARKERS[template],
        "svg_marker": SVG_MARKERS[template],
        "dash": DASHES[template],
        "hatch": HATCHES[template],
    }


def palette_metadata() -> dict[str, Any]:
    return {
        "mapping": {template: template_style(template) for template in TEMPLATE_ORDER},
        "validation": PALETTE_VALIDATION,
        "marks": {"line_width_px": LINE_WIDTH, "minimum_marker_size_px": MARKER_SIZE},
    }
