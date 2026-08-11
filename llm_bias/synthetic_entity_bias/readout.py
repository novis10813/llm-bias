"""Nine-label score readout."""
from __future__ import annotations
from typing import Any
import torch
from llm_bias.core.readout import distribution_stats, restricted_softmax
from .spec import SCORES

def score_distribution(logits: torch.Tensor, label_token_ids: list[int]|dict[str,int], *, effective_temperature_value: float|None=None, residual: torch.Tensor|None=None, final_norm: Any|None=None) -> dict[str,Any]:
 ids=list(label_token_ids.values()) if isinstance(label_token_ids,dict) else list(label_token_ids)
 if len(ids)!=9 or len(set(ids))!=9: raise ValueError("exactly nine distinct label token IDs are required")
 probs=restricted_softmax(logits,ids)
 stats=distribution_stats(probs,SCORES)
 if effective_temperature_value is not None:
  stats["effective_temperature"]=float(effective_temperature_value)
 else:
  if residual is None: raise ValueError("effective temperature requires explicit value or transported answer residual")
  from llm_bias.core.readout import effective_temperature
  temp=effective_temperature(residual.reshape(1,-1),final_norm=final_norm)
  stats["effective_temperature"]=float(temp.reshape(-1)[0])
 if not torch.isfinite(torch.tensor(list(stats.values()),dtype=torch.float32)).all(): raise ValueError("non-finite restricted readout statistic")
 return {"probabilities":[float(x) for x in probs.detach().cpu()],**stats,"label_token_ids":ids}

def delta_expected_score(entity: dict[str,Any], baseline: dict[str,Any]) -> float:
 return float(entity["expected_score"])-float(baseline["expected_score"])
