"""Compact, schema-checked synthetic pilot artifacts."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any, Iterable
from llm_bias.core.artifact_manifest import RunManifest

FORBIDDEN={"activation","activations","residual","residuals","gradient","gradients","hidden_state","raw"}
def _check(value: Any, key: str="") -> None:
 if any(x in key.lower() for x in FORBIDDEN): raise ValueError(f"forbidden raw payload field: {key}")
 if hasattr(value,"detach") or value.__class__.__module__.startswith("numpy"): raise ValueError("tensor/ndarray payloads are forbidden")
 if isinstance(value,dict):
  for k,v in value.items(): _check(v,str(k))
 elif isinstance(value,(list,tuple)):
  for v in value: _check(v,key)

def write_json(path: str|Path, value: Any) -> Path:
 _check(value); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8"); return p

def write_csv(path: str|Path, rows: Iterable[dict[str,Any]], fields: list[str]) -> Path:
 rows=list(rows)
 for row in rows:
  _check(row)
  extra=set(row)-set(fields)
  if extra: raise ValueError(f"unexpected output fields: {sorted(extra)}")
  for field in fields:
   value=row.get(field)
   if isinstance(value,float) and not math.isfinite(value): raise ValueError(f"non-finite output value in {field}")
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="raise"); w.writeheader(); w.writerows(rows)
 return p

def start_manifest(model,dataset,run_id,artifact_root):
 m=RunManifest.new(model,dataset,run_id,artifact_root=artifact_root)
 if m.run_directory.exists() or m.manifest_path.exists(): raise FileExistsError(f"refusing to overwrite existing run: {m.run_directory}")
 m.start(); m.save(); return m

def fail_manifest(m,error):
 if m.status not in {"complete","failed"}: m.fail(str(error)); m.save()

def complete_manifest(m, *, required_stages: set[str], postcheck: bool):
 if m.status!="running": raise ValueError("manifest must be running")
 if not postcheck or any(m.stages.get(s,{}).get("status")!="complete" for s in required_stages): raise ValueError("cannot complete manifest before successful postchecks")
 m.complete(); m.save()
