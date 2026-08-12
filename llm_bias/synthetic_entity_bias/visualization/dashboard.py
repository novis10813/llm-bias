"""Self-contained, accessible dashboard for compact synthetic summaries."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .chart_specs import build_chart_specs
from .summaries import summarize_all
from .theme import DARK_COLORS, LIGHT_COLORS

_CSS = f"""
:root{{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--negative:{LIGHT_COLORS['negative']};--positive:{LIGHT_COLORS['positive']};--neutral:{LIGHT_COLORS['neutral']};color-scheme:light}}
:root[data-theme=dark]{{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--negative:{DARK_COLORS['negative']};--positive:{DARK_COLORS['positive']};--neutral:{DARK_COLORS['neutral']};color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:14px system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1440px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}.muted{{color:var(--secondary)}}
.controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}}button,input{{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:9px}}button:focus-visible,input:focus-visible,summary:focus-visible,.hit:focus{{outline:3px solid var(--negative);outline-offset:2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px;margin:18px 0}}.card{{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;overflow:auto}}.card h2{{margin-top:0;font-size:17px}}.chart{{width:100%;min-width:320px;height:auto;display:block}}.gridline{{stroke:var(--grid);stroke-width:1}}.axis{{stroke:var(--axis);stroke-width:1}}.zero{{stroke:var(--muted);stroke-width:1}}.series{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}.mark{{stroke:var(--surface);stroke-width:2}}.hit{{fill:transparent;stroke:transparent;pointer-events:all}}.crosshair{{stroke:var(--muted);stroke-width:1;pointer-events:none}}.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--secondary);font-size:12px}}.key{{display:inline-flex;align-items:center;gap:5px}}.key svg{{width:24px;height:12px}}.downloads{{display:flex;gap:10px;font-size:12px}}a{{color:var(--negative)}}
.tooltip{{position:absolute;z-index:5;max-width:260px;padding:9px;background:var(--surface);border:1px solid var(--axis);border-radius:6px;box-shadow:0 4px 18px var(--border);pointer-events:none}}.tooltip strong{{display:block}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:7px;border-bottom:1px solid var(--grid)}}th{{color:var(--secondary)}}details{{margin-top:12px}}summary{{cursor:pointer;color:var(--secondary)}}
.dash-negative{{stroke-dasharray:none}}.dash-positive{{stroke-dasharray:8 5}}.dash-neutral{{stroke-dasharray:10 4 2 4}}.fill-negative{{fill:var(--negative);stroke:var(--negative)}}.fill-positive{{fill:var(--positive);stroke:var(--positive)}}.fill-neutral{{fill:var(--neutral);stroke:var(--neutral)}}.series.fill-negative{{fill:none;stroke:var(--negative)}}.series.fill-positive{{fill:none;stroke:var(--positive)}}.series.fill-neutral{{fill:none;stroke:var(--neutral)}}
@media(max-width:640px){{main{{padding:14px}}.grid{{display:block}}.card{{margin-bottom:14px}}}}
@media print{{:root,:root[data-theme=dark]{{--page:#fff;--surface:#fff;--ink:#000;--secondary:#222;--grid:#ddd;--axis:#777}}.controls,.tooltip,.downloads{{display:none!important}}details{{display:block}}details>summary{{display:none}}details>div{{display:block!important}}.card{{break-inside:avoid;border-color:#aaa}}}}
@media(forced-colors:active){{.series,.mark{{forced-color-adjust:auto;stroke:CanvasText}}.mark{{fill:Canvas}}.gridline{{display:none}}details{{display:block}}}}
"""

_JS = r"""
const NS='http://www.w3.org/2000/svg';
const data=JSON.parse(document.getElementById('dashboard-data').textContent);
const el=(name,attrs={})=>{const node=document.createElementNS(NS,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));return node};
const htmlEl=(name,text)=>{const node=document.createElement(name);if(text!==undefined)node.textContent=String(text);return node};
const fmt=value=>typeof value==='number'?Number(value).toLocaleString(undefined,{maximumFractionDigits:4}):String(value);
function bounds(spec){const points=spec.series.flatMap(s=>s.points);const xs=points.map(p=>Number(p.x)).filter(Number.isFinite),ys=points.map(p=>Number(p.y)).filter(Number.isFinite);let x0=Math.min(...xs,0),x1=Math.max(...xs,0),y0=Math.min(...ys,0),y1=Math.max(...ys,0);if(x0===x1){x0-=1;x1+=1}if(y0===y1){y0-=1;y1+=1}const xp=(x1-x0)*.08,yp=(y1-y0)*.12;return [x0-xp,x1+xp,y0-yp,y1+yp]}
function tooltip(card,row,event){const tip=card.querySelector('.tooltip');tip.replaceChildren();for(const [key,value] of Object.entries(row)){const line=htmlEl('div');const strong=htmlEl('strong',fmt(value));line.append(strong,document.createTextNode(' '+key.replaceAll('_',' ')));tip.append(line)}tip.hidden=false;if(event){const box=card.getBoundingClientRect();tip.style.left=`${Math.min(event.clientX-box.left+12,box.width-270)}px`;tip.style.top=`${event.clientY-box.top+12}px`}}
function hide(card){card.querySelector('.tooltip').hidden=true;const cross=card.querySelector('.crosshair');if(cross)cross.hidden=true}
function symbol(template,x,y){if(template==='positive')return el('rect',{x:x-5,y:y-5,width:10,height:10,rx:1,class:`mark fill-${template}`});if(template==='neutral')return el('polygon',{points:`${x},${y-6} ${x+6},${y+5} ${x-6},${y+5}`,class:`mark fill-${template}`});return el('circle',{cx:x,cy:y,r:5,class:`mark fill-${template}`})}
function legend(card,spec){const root=card.querySelector('.legend');for(const series of spec.series){const template=series.key.split('_')[0],item=htmlEl('span');item.className='key';const swatch=el('svg',{viewBox:'0 0 24 12','aria-hidden':'true'});swatch.append(el('line',{x1:1,y1:6,x2:23,y2:6,class:`series fill-${template} dash-${template}`}));swatch.append(symbol(template,12,6));item.append(swatch,document.createTextNode(series.label));root.append(item)}}
function table(card,spec){const root=card.querySelector('.table-root'),table=htmlEl('table'),head=htmlEl('thead'),body=htmlEl('tbody'),tr=htmlEl('tr');for(const key of spec.tooltip_fields)tr.append(htmlEl('th',key.replaceAll('_',' ')));head.append(tr);for(const row of spec.table){const line=htmlEl('tr');for(const key of spec.tooltip_fields)line.append(htmlEl('td',fmt(row[key])));body.append(line)}table.append(head,body);root.append(table)}
function render(card,spec){const svg=card.querySelector('svg'),W=720,H=360,L=65,R=24,T=24,B=54,[x0,x1,y0,y1]=bounds(spec),sx=x=>L+(Number(x)-x0)/(x1-x0)*(W-L-R),sy=y=>H-B-(Number(y)-y0)/(y1-y0)*(H-T-B);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.append(el('title',{id:`${spec.id}-title`})).textContent=spec.title;svg.append(el('desc',{id:`${spec.id}-desc`})).textContent=spec.description;
for(let i=0;i<=4;i++){const y=T+i*(H-T-B)/4;svg.append(el('line',{x1:L,y1:y,x2:W-R,y2:y,class:'gridline'}))}svg.append(el('line',{x1:L,y1:H-B,x2:W-R,y2:H-B,class:'axis'}));svg.append(el('line',{x1:L,y1:T,x2:L,y2:H-B,class:'axis'}));if(x0<=0&&x1>=0)svg.append(el('line',{x1:sx(0),y1:T,x2:sx(0),y2:H-B,class:'zero'}));if(y0<=0&&y1>=0)svg.append(el('line',{x1:L,y1:sy(0),x2:W-R,y2:sy(0),class:'zero'}));
for(const series of spec.series){const template=series.key.split('_')[0],points=series.points;if(spec.mark==='bar'){for(const point of points){const y=sy(point.y),zero=sx(0),x=sx(point.x),rect=el('rect',{x:Math.min(zero,x),y:y-8,width:Math.max(2,Math.abs(x-zero)),height:16,rx:4,class:`mark fill-${template}`,tabindex:0});rect.addEventListener('pointermove',e=>tooltip(card,{series:series.label,...point},e));rect.addEventListener('focus',()=>tooltip(card,{series:series.label,...point}));rect.addEventListener('blur',()=>hide(card));svg.append(rect)}}else if(spec.mark==='scatter'){for(const point of points){const dot=symbol(template,sx(point.x),sy(point.y)),hit=el('circle',{cx:sx(point.x),cy:sy(point.y),r:12,class:'hit',tabindex:0});for(const target of [dot,hit])target.addEventListener('pointermove',e=>tooltip(card,{series:series.label,...point},e));hit.addEventListener('focus',()=>tooltip(card,{series:series.label,...point}));hit.addEventListener('blur',()=>hide(card));svg.append(dot,hit)}}else{const ordered=[...points].sort((a,b)=>Number(a.x)-Number(b.x));if(ordered.length){const path=el('path',{d:ordered.map((p,i)=>`${i?'L':'M'}${sx(p.x)},${sy(p.y)}`).join(' '),class:`series fill-${template} dash-${template}`});svg.append(path)}for(const [index,point] of ordered.entries()){if(index%Math.max(1,Math.ceil(ordered.length/8))===0)svg.append(symbol(template,sx(point.x),sy(point.y)));const hit=el('circle',{cx:sx(point.x),cy:sy(point.y),r:12,class:'hit',tabindex:0});hit.addEventListener('pointermove',e=>{let row={x:point.x};for(const candidate of spec.series){const nearest=candidate.points.reduce((best,p)=>Math.abs(Number(p.x)-Number(point.x))<Math.abs(Number(best.x)-Number(point.x))?p:best,candidate.points[0]);row[candidate.label]=nearest.y}tooltip(card,row,e)});hit.addEventListener('focus',()=>tooltip(card,{series:series.label,...point}));hit.addEventListener('blur',()=>hide(card));svg.append(hit)}}}
legend(card,spec);table(card,spec)}
for(const spec of data.charts)render(document.querySelector(`[data-chart="${spec.id}"]`),spec);
const q=document.getElementById('query'),target=document.getElementById('ticker-table');function draw(){const search=q.value.toLowerCase(),selected=data.tickers.filter(r=>r.ticker.toLowerCase().includes(search)||r.company_name.toLowerCase().includes(search)).slice(0,100);target.replaceChildren();if(!selected.length){target.append(htmlEl('p','No rows.'));return}const keys=Object.keys(selected[0]),table=htmlEl('table'),head=htmlEl('thead'),body=htmlEl('tbody'),tr=htmlEl('tr');for(const key of keys)tr.append(htmlEl('th',key.replaceAll('_',' ')));head.append(tr);for(const row of selected){const line=htmlEl('tr');for(const key of keys)line.append(htmlEl('td',fmt(row[key])));body.append(line)}table.append(head,body);target.append(table)}q.addEventListener('input',draw);draw();
const toggle=document.getElementById('theme-toggle');function setTheme(theme){document.documentElement.dataset.theme=theme;toggle.setAttribute('aria-pressed',String(theme==='dark'));toggle.textContent=theme==='dark'?'Use light theme':'Use dark theme';localStorage.setItem('synthetic-viz-theme',theme)}const saved=localStorage.getItem('synthetic-viz-theme'),initial=saved||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');setTheme(initial);toggle.addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
"""


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    fields = list(rows[0])
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def render_dashboard(run: Any, *, plot_names: list[str] | None = None) -> str:
    summaries = summarize_all(run)
    specs = build_chart_specs(run)
    available = set(plot_names or [spec["id"] for spec in specs])
    overview = {"model": run.manifest["model"], "dataset": run.manifest["dataset"], "run_id": run.manifest["run_id"], "pool_count": len(run.entity_pool), "metric_rows": len(run.results), "localization_rows": len(run.localization), "lens_sha256": run.config["lens_binary_sha256"]}
    cards = "".join(f'''<section class="card" data-chart="{html.escape(spec['id'])}"><h2>{html.escape(spec['title'])}</h2><p class="muted">{html.escape(spec['description'])}</p><div class="legend" aria-label="Series legend"></div><svg class="chart" role="img" aria-labelledby="{spec['id']}-title {spec['id']}-desc"></svg><div class="tooltip" role="status" aria-live="polite" hidden></div><p class="downloads"><a href="figures/{spec['id']}.png">PNG</a><a href="figures/{spec['id']}.svg">SVG</a><a href="figures/{spec['id']}.pdf">PDF</a></p><details><summary>Show accessible data table</summary><div class="table-root"></div></details><noscript><img src="figures/{spec['id']}.png" alt="{html.escape(spec['description'])}"></noscript></section>''' for spec in specs if spec["id"] in available)
    payload = _safe_json({"charts": specs, "tickers": summaries["ticker"]})
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic entity-bias visualization</title><style>{_CSS}</style></head><body><main><h1>Synthetic entity-bias visualization</h1><p class="muted">Artifact-only descriptive report. Delta-E is entity minus baseline expected score; it is not a standalone causal effect. Localization is transported representation evidence, not chain-of-thought.</p><div class="controls" aria-label="Visualization controls"><button id="theme-toggle" type="button" aria-pressed="false">Use dark theme</button><label for="query">Ticker or company</label><input id="query" placeholder="Search ticker or company"></div><section class="card"><h2>Run provenance</h2>{_table([overview])}</section><div class="grid">{cards}</div><div class="grid"><section class="card"><h2>Template summary</h2>{_table(summaries['template'])}</section><section class="card"><h2>Familiarity tier summary</h2>{_table(summaries['tier'])}</section></div><section class="card"><h2>Ticker results</h2><div id="ticker-table"></div></section><section class="card"><h2>Audit</h2><p>All required source artifacts passed manifest status, stage, path, SHA-256, schema, count, and numeric-domain validation. No model or lens was loaded.</p></section><script type="application/json" id="dashboard-data">{payload}</script><script>{_JS}</script></main></body></html>'''


def write_dashboard(run: Any, output_path: str | Path, *, plot_names: list[str] | None = None) -> Path:
    path = Path(output_path)
    path.write_text(render_dashboard(run, plot_names=plot_names), encoding="utf-8")
    return path
