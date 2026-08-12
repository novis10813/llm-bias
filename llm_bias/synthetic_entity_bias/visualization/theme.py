"""Shared, validated visual tokens for synthetic entity-bias charts."""

from __future__ import annotations

from typing import Any

from .contract import TEMPLATE_ORDER

LIGHT_COLORS = dict(zip(TEMPLATE_ORDER, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True))
DARK_COLORS = dict(zip(TEMPLATE_ORDER, ("#3987e5", "#d95926", "#199e70"), strict=True))
MARKERS = dict(zip(TEMPLATE_ORDER, ("o", "s", "^"), strict=True))
DASHES = dict(zip(TEMPLATE_ORDER, ("solid", "dashed", "dashdot"), strict=True))
HATCHES = dict(zip(TEMPLATE_ORDER, ("///", "\\\\\\", "xxx"), strict=True))
SVG_MARKERS = dict(zip(TEMPLATE_ORDER, ("circle", "square", "triangle"), strict=True))
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
