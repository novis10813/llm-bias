"""Layer/template localization with bounded-memory statistics."""
from __future__ import annotations
from typing import Any, Iterable
import torch
from llm_bias.core.directions import OnlineDirection, quantile_bounds, cosine_and_statistics, direction_hash
from llm_bias.core.analysis.transport import transport_residual_delta

def answer_residual(activation: torch.Tensor, position: int) -> torch.Tensor:
 if activation.ndim==3: return activation[:,position,:].float()
 if activation.ndim==2: return activation[position,:].float()
 raise ValueError("activation must be [batch, sequence, d_model] or [sequence, d_model]")

def transported_delta(entity: torch.Tensor, baseline: torch.Tensor, *, layer: int, final_layer: int, lens: Any|None=None, jacobian_cache: dict[int,torch.Tensor]|None=None) -> torch.Tensor:
 """Compatibility wrapper; method identity is ``jacobian_transport``."""
 return transport_residual_delta(entity, baseline, layer=layer, final_layer=final_layer, lens=lens, jacobian_cache=jacobian_cache)

def fit_layer_direction(vectors: Iterable[torch.Tensor], targets: Iterable[float], *, ids: Iterable[str]|None=None, splits: Iterable[str]|None=None, seed: int|None=None) -> tuple[torch.Tensor,dict[str,Any]]:
 vectors=list(vectors); targets=list(targets)
 if len(vectors)!=len(targets) or len(vectors)<2: raise ValueError("direction fitting requires matching vectors and targets")
 if not all(torch.isfinite(v).all() for v in vectors): raise ValueError("direction vectors must be finite")
 if ids is not None:
  ids=list(ids)
  if len(ids)!=len(vectors) or len(set(ids))!=len(ids): raise ValueError("direction IDs must be unique")
 if splits is not None:
  splits=list(splits)
  if len(splits)!=len(vectors) or any(s!="train" for s in splits): raise ValueError("direction fitting accepts train split only")
 q25,q75=quantile_bounds(targets)
 direction=OnlineDirection(torch.zeros_like(vectors[0].flatten().cpu()),torch.zeros_like(vectors[0].flatten().cpu()))
 for vector,target in zip(vectors,targets,strict=True):
  if target>=q75: direction.add(vector,high=True)
  elif target<=q25: direction.add(vector,high=False)
 unit,norm=direction.normalized()
 high_ids=[str(i) for i,y in zip(ids or range(len(vectors)),targets,strict=True) if y>=q75]
 low_ids=[str(i) for i,y in zip(ids or range(len(vectors)),targets,strict=True) if y<=q25]
 if len(high_ids)<2 or len(low_ids)<2 or set(high_ids)&set(low_ids): raise ValueError("high/low groups require at least two disjoint IDs each")
 import hashlib
 hash_ids=lambda values: hashlib.sha256("\\n".join(sorted(values)).encode()).hexdigest()
 return unit,{"q25":q25,"q75":q75,"n_train":len(vectors),"n_high":len(high_ids),"n_low":len(low_ids),"high_ids_sha256":hash_ids(high_ids),"low_ids_sha256":hash_ids(low_ids),"fit_split":"train","seed":seed,"direction_norm":norm,"direction_sha256":direction_hash(unit)}

def evaluate_layer_direction(vectors: Iterable[torch.Tensor], targets: Iterable[float], direction: torch.Tensor, metadata: dict[str,Any], *, ids: Iterable[str]|None=None, splits: Iterable[str]|None=None) -> dict[str,Any]:
 vals=list(vectors); ys=list(targets)
 if len(vals)!=len(ys) or not vals: raise ValueError("evaluation requires matching nonempty values")
 if ids is not None:
  ids=list(ids)
  if len(ids)!=len(vals) or len(set(ids))!=len(ids): raise ValueError("evaluation IDs must be unique")
 if splits is not None:
  splits=list(splits)
  if len(splits)!=len(vals) or any(s!="eval" for s in splits): raise ValueError("evaluation accepts eval split only")
 if not all(torch.isfinite(v).all() for v in vals) or not all(float(y)==float(y) and abs(float(y))<float("inf") for y in ys): raise ValueError("evaluation vectors and targets must be finite")
 d=direction.flatten().float(); norm=d.norm()
 if not torch.isfinite(norm) or norm<=1e-8: raise ValueError("direction has near-zero norm")
 cos=[float(torch.dot(v.flatten().float(),d)/(v.flatten().float().norm().clamp_min(1e-12)*norm)) for v in vals]
 result=cosine_and_statistics(cos,ys); result.update(metadata); result["n_eval"]=len(vals); return result
