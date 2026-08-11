"""Explicit constituent CSV loading and ticker-level aggregation."""
from __future__ import annotations
import csv, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

TIERS = {"sp500":3, "s&p500":3, "s&p 500":3, "russell1000":2, "russell 1000":2, "russell2000":1, "russell 2000":1}
NAMES = {3:"S&P 500",2:"Russell 1000",1:"Russell 2000"}
@dataclass(frozen=True)
class EntityRecord:
 ticker: str; company_name: str; latest_year: int; years: tuple[int,...]; memberships: tuple[str,...]; membership_years: tuple[str,...]; sectors: tuple[str,...]; familiarity_tier: str; source_row_count: int; anomalies: tuple[str,...]; split: str = ""
 def to_dict(self): return asdict(self) | {"years":"|".join(map(str,self.years)),"memberships":"|".join(self.memberships),"membership_years":"|".join(self.membership_years),"sectors":"|".join(self.sectors),"anomalies":"|".join(self.anomalies)}

def normalize_ticker(value: str) -> str:
 value = value.strip().upper().replace(".", "-")
 if not value or any(ord(c)<32 for c in value): raise ValueError(f"invalid ticker {value!r}")
 return value

def _index_name(path: Path, row: dict[str,str]) -> str:
 raw = (row.get("index_name") or "").strip().lower()
 if raw in TIERS: return NAMES[TIERS[raw]]
 stem = path.stem.lower().replace("_constituents_2020_2025","").replace("-constituents-2020-2025","")
 for key,tier in TIERS.items():
  if key.replace(" ","") in stem.replace("_","").replace("-",""): return NAMES[tier]
 raise ValueError(f"cannot determine constituent index for {path}")

def load_entity_pool(paths: Iterable[str|Path], *, start_year=2020, end_year=2025, seed=0) -> list[EntityRecord]:
 paths = [Path(p) for p in paths]
 if not paths: raise ValueError("at least one explicit constituent CSV is required")
 rows=[]; identities=set()
 for path in paths:
  if not path.is_file(): raise FileNotFoundError(path)
  with path.open(newline="",encoding="utf-8-sig") as f:
   reader=csv.DictReader(f)
   required={"year","ticker","company_name"}
   if not reader.fieldnames or not required <= set(reader.fieldnames): raise ValueError(f"{path} missing required columns {sorted(required-set(reader.fieldnames or []))}")
   for n,row in enumerate(reader):
    year=int(row["year"]);
    if not start_year <= year <= end_year: continue
    ticker=normalize_ticker(row["ticker"]); name=row["company_name"].strip()
    if not name: raise ValueError(f"empty company_name in {path}:{n+2}")
    index=_index_name(path,row); identity=(index,year,ticker,name,row.get("gics_sector","").strip())
    if identity in identities: continue
    identities.add(identity); rows.append((index,year,ticker,name,row.get("gics_sector","").strip()))
 grouped={}
 for index,year,ticker,name,sector in rows: grouped.setdefault(ticker,[]).append((index,year,name,sector))
 out=[]
 for ticker, values in grouped.items():
  years=sorted({v[1] for v in values}); memberships=sorted({v[0] for v in values}); max_tier=max(TIERS[m.lower()] for m in memberships)
  latest=max(years); candidates=[v for v in values if v[1]==latest]; names=sorted({v[2] for v in candidates}); chosen=sorted(names, key=lambda x:x)[0]
  # prefer highest tier at latest year, then lexical
  chosen=sorted(names,key=lambda x:(-max(TIERS[v[0].lower()] for v in candidates if v[2]==x),x))[0]
  anomalies=[]
  if len(names)>1: anomalies.append("conflicting_company_name")
  sectors=sorted({v[3] for v in values if v[3]});
  if len(sectors)>1: anomalies.append("conflicting_sector")
  membership_years=sorted({f"{v[0]}:{v[1]}" for v in values})
  out.append(EntityRecord(ticker,chosen,latest,tuple(years),tuple(memberships),tuple(membership_years),tuple(sectors),NAMES[max_tier],len(values),tuple(anomalies),""))
 by_tier={tier:sorted([e for e in out if e.familiarity_tier==NAMES[tier]],key=lambda e:hashlib.sha256(f"{seed}:{e.ticker}".encode()).hexdigest()) for tier in NAMES}
 assigned={e.ticker:("train" if i < max(1,int(len(group)*.8)) else "eval") for group in by_tier.values() for i,e in enumerate(group)}
 return sorted([EntityRecord(**(e.to_dict() | {"years":e.years,"memberships":e.memberships,"membership_years":e.membership_years,"sectors":e.sectors,"anomalies":e.anomalies,"split":assigned[e.ticker]})) for e in out],key=lambda e:e.ticker)
