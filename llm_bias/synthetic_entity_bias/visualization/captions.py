"""Deterministic publication captions for synthetic entity-bias figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_caption(spec: dict[str, Any], run: Any) -> str:
    panels = " ".join(
        f"**{panel['label']} {panel['title']}.** "
        + (f"{panel['statistic']}. " if panel.get("statistic") else "")
        + f"Sample: {panel['sample_size']}. {panel['reference']}."
        for panel in spec["panels"]
    )
    return (
        f"# {spec['title']}\n\n"
        f"**Figure caption.** {spec['subtitle']} {panels}\n\n"
        f"**Figure note.** {spec['figure_note']}\n\n"
        f"**Source.** Model `{run.manifest['model']}`; dataset `{run.manifest['dataset']}`; "
        f"run `{run.manifest['run_id']}`. Figure stem: `{spec['id']}`. "
        f"Supporting table: `tables/{spec['supporting_table']}`.\n"
    )


def write_captions(specs: list[dict[str, Any]], run: Any, output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for spec in specs:
        path = directory / f"{spec['id']}.md"
        path.write_text(render_caption(spec, run), encoding="utf-8")
        paths[spec["id"]] = path
    return paths
