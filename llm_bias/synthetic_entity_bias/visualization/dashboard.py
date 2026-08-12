"""Self-contained static dashboard for compact synthetic summaries."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .summaries import summarize_all

_CSS = """
:root{color-scheme:light;--ink:#17212b;--muted:#64717d;--surface:#f5f7f9;--line:#d9e0e6;--blue:#35608f}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:14px system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:28px}h1{margin:0 0 6px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;margin:18px 0}
.card{background:white;border:1px solid var(--line);border-radius:10px;padding:16px;overflow:auto}.card h2{margin-top:0;font-size:17px}
img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--line)}th{color:var(--muted)}
input{width:100%;padding:9px;border:1px solid var(--line);border-radius:6px;margin-bottom:8px}@media print{body{background:white}.card{break-inside:avoid}}
"""


def _table(rows: list[dict[str, Any]], *, table_id: str | None = None) -> str:
    if not rows:
        return "<p>No rows.</p>"
    fields = list(rows[0])
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>" for row in rows)
    identity = f' id="{table_id}"' if table_id else ""
    return f"<table{identity}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard(run: Any, *, plot_names: list[str] | None = None) -> str:
    summaries = summarize_all(run)
    plot_names = plot_names or []
    overview = {
        "model": run.manifest["model"], "dataset": run.manifest["dataset"],
        "run_id": run.manifest["run_id"], "pool_count": len(run.entity_pool),
        "metric_rows": len(run.results), "localization_rows": len(run.localization),
        "lens_sha256": run.config["lens_binary_sha256"],
    }
    figures = "".join(f'<section class="card"><h2>{html.escape(name.replace("_", " ").title())}</h2><img src="{html.escape(name)}.png" alt="{html.escape(name)}"></section>' for name in plot_names)
    payload = json.dumps(summaries["ticker"], ensure_ascii=True, separators=(",", ":"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic entity-bias visualization</title><style>{_CSS}</style></head><body><main>
<h1>Synthetic entity-bias visualization</h1><p class="muted">Artifact-only descriptive report. Delta-E is entity minus baseline expected score; it is not a standalone causal effect. Input membership years preserve source provenance and are not independently verified historical membership evidence.</p>
<section class="card"><h2>Run provenance</h2>{_table([overview])}</section><div class="grid">{figures}</div>
<div class="grid"><section class="card"><h2>Template summary</h2>{_table(summaries["template"])}</section><section class="card"><h2>Familiarity tier summary</h2>{_table(summaries["tier"])}</section></div>
<section class="card"><h2>Ticker search</h2><input id="query" placeholder="Search ticker or company"><div id="ticker-table"></div></section>
<section class="card"><h2>Audit</h2><p>All required source artifacts passed manifest status, stage, path, SHA-256, schema, count, and numeric-domain validation. No model or lens was loaded.</p></section>
<script>const rows={payload};const q=document.getElementById('query'),target=document.getElementById('ticker-table');function draw(){{const s=q.value.toLowerCase();const selected=rows.filter(r=>r.ticker.toLowerCase().includes(s)||r.company_name.toLowerCase().includes(s)).slice(0,100);if(!selected.length){{target.innerHTML='<p>No rows.</p>';return}}const keys=Object.keys(selected[0]);target.innerHTML='<table><thead><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr></thead><tbody>'+selected.map(r=>'<tr>'+keys.map(k=>'<td>'+String(r[k])+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}}q.addEventListener('input',draw);draw();</script>
</main></body></html>'''


def write_dashboard(run: Any, output_path: str | Path, *, plot_names: list[str] | None = None) -> Path:
    path = Path(output_path)
    path.write_text(render_dashboard(run, plot_names=plot_names), encoding="utf-8")
    return path
