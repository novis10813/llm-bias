"""Bounded-memory synthetic entity-bias pipeline."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from typing import Any
import torch
from .spec import *
from .entities import load_entity_pool, EntityRecord
from .prompts import render_prompt, validate_token_contract
from .readout import score_distribution
from .localization import transported_delta, fit_layer_direction, evaluate_layer_direction
from llm_bias.core.directions import OnlineDirection, quantile_bounds, direction_hash, cosine_and_statistics
from .artifacts import write_csv, write_json, start_manifest, fail_manifest, complete_manifest

RAW_FIELDS=["ticker","company_name","template","split","familiarity_tier","entity_probabilities","baseline_probabilities","entity_expected_score","baseline_expected_score","entity_entropy_nats","baseline_entropy_nats","entity_effective_temperature","baseline_effective_temperature","delta_expected_score","entity_span_start","entity_span_end","answer_position"]
BASE_FIELDS=["template","entity","probabilities","expected_score","entropy_nats","effective_temperature"]
LOC_FIELDS=["layer","template","mean_cosine","pearson_r","spearman_r","linear_r2","n_train","n_eval","q25","q75","n_high","n_low","high_ids_sha256","low_ids_sha256","fit_split","direction_sha256","statistic_flag"]

def _forward_batch(model, ids: list[list[int]], layers: list[int], device: Any, *, pad_token_id: int = 0, final_norm: Any|None = None, keep_activations_device: bool = False):
 if not ids: raise ValueError("empty batch")
 width=max(map(len,ids)); x=torch.full((len(ids),width),pad_token_id,dtype=torch.long,device=device); mask=torch.zeros_like(x)
 for i,row in enumerate(ids):
  if not row: raise ValueError("all-padding row")
  x[i,:len(row)]=torch.as_tensor(row,dtype=torch.long,device=device); mask[i,:len(row)]=1
 positions=mask.sum(-1)-1
 if (positions<0).any(): raise ValueError("all-padding batch row")
 if hasattr(model,"layers"):
  from jlens.hooks import ActivationRecorder
  with torch.no_grad(), ActivationRecorder(model.layers,at=layers) as rec:
   try: out=model.forward(x,attention_mask=mask)
   except TypeError: out=model.forward(x)
   answer={int(k):v[torch.arange(len(ids),device=v.device),positions.to(v.device)].float() for k,v in rec.activations.items()}
  if hasattr(out,"logits"): logits=out.logits[torch.arange(len(ids),device=device),positions].float()
  elif isinstance(out,dict) and "logits" in out: logits=out["logits"][torch.arange(len(ids),device=device),positions].float()
  elif hasattr(model,"unembed"): logits=model.unembed(answer[max(layers)]).float()
  else: raise ValueError("model output has no logits")
  final_residual=answer[max(layers)]
  if final_norm is not None:
   params=list(final_norm.parameters())
   norm_device=params[0].device if params else final_residual.device
   normalized=final_norm(final_residual.to(norm_device))
  else: normalized=final_residual
  temperatures=normalized.float().norm(dim=-1).reciprocal()
  if not torch.isfinite(temperatures).all() or (temperatures<=0).any(): raise ValueError("non-finite effective temperature")
  stored={k:v.detach() if keep_activations_device else v.detach().cpu() for k,v in answer.items()}
  return logits.detach().cpu(),stored,temperatures.detach().cpu()
 with torch.no_grad():
  out=model(x,attention_mask=mask)
 logits=out.logits if hasattr(out,"logits") else out["logits"]
 selected=logits[torch.arange(len(ids),device=device),positions].float().detach().cpu()
 temperatures=torch.ones(len(ids))
 return selected,{},temperatures

def _forward(model, ids: list[int], layers: list[int], device: Any):
 logits,acts,temps=_forward_batch(model,[ids],layers,device,final_norm=getattr(model,"_final_norm",None))
 return logits[0],{k:v.unsqueeze(1) for k,v in acts.items()},float(temps[0])

def _resolve_device(model, explicit=None):
 if explicit is not None: return torch.device(explicit)
 value=getattr(model,"device",None)
 if value is None and hasattr(model,"_hf_model"):
  value=next(model._hf_model.parameters()).device
 if value is None: raise ValueError("device must be explicit when model exposes no reliable device")
 return torch.device(value)

def _flat(value): return json.dumps(value,separators=(",",":"))

def run_pipeline(*, constituents, model_path, lens_path, artifact_root="artifacts", dataset="synthetic-entity-bias-2020-2025", run_id="run", model=None, tokenizer=None, lens=None, device=None, seed=0, max_seq_len=2048, batch_size=16, use_chat_template=True) -> Path:
 pool=load_entity_pool(constituents,seed=seed)
 if not pool: raise ValueError("entity pool is empty")
 if model is None:
  from llm_bias.core.model import load_model
  model,tokenizer,device=load_model(model_path)
 else:
  device=_resolve_device(model,device)
 if tokenizer is None: raise ValueError("tokenizer is required")
 if not hasattr(model,"n_layers") or not hasattr(model,"d_model"): raise ValueError("loaded model lacks jlens layer metadata")
 if lens is None:
  from llm_bias.core.lens_loader import load_validated_lens
  lens=load_validated_lens(model=model,model_name=model_path,lens_path=lens_path,require_complete=True).lens
 else:
  from llm_bias.core.lens_artifacts import validate_lens_for_model
  validate_lens_for_model(model=model,lens=lens,model_name=model_path,lens_path=lens_path,require_complete=True)
 layers=list(range(int(getattr(model,"n_layers",1)))); final=layers[-1]
 rendered=[]
 for e in pool:
  for t in TEMPLATES: rendered.append(render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len))
 for t in TEMPLATES: rendered.append(render_prompt(tokenizer,t,entity=BASELINE_ENTITY,use_chat_template=use_chat_template,max_seq_len=max_seq_len))
 token_report=validate_token_contract(tokenizer,rendered)
 label_ids=list(token_report["label_token_ids"].values())
 m=start_manifest(model_path,dataset,run_id,artifact_root)
 root=m.run_directory; root.mkdir(parents=True,exist_ok=True)
 try:
  model_config=Path(model_path)/"config.json"; lens_meta=Path(str(lens_path)+".metadata.json")
  hash_file=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
  chat_hash=hashlib.sha256(str(getattr(tokenizer,"chat_template","")).encode()).hexdigest()
  provenance={"model_path":model_path,"model_config_sha256":hash_file(model_config),"tokenizer_class":type(tokenizer).__name__,"tokenizer_name_or_path":getattr(tokenizer,"name_or_path",None),"transformers_version":__import__("transformers").__version__,"chat_template_sha256":chat_hash,"lens_binary_sha256":hash_file(lens_path),"lens_metadata_sha256":hash_file(lens_meta),"input_hashes":{str(p):hash_file(p) for p in constituents},"label_token_ids":token_report["label_token_ids"],"label_decoded":token_report["decoded"],"score_mapping":dict(zip(LABELS,SCORES)),"templates":TEMPLATES,"scoring_instruction":SCORING_INSTRUCTION,"pool_count":len(pool),"tier_counts":{tier:sum(e.familiarity_tier==tier for e in pool) for tier in set(e.familiarity_tier for e in pool)},"split_counts":{s:sum(e.split==s for e in pool) for s in ("train","eval")},"anomaly_count":sum(bool(e.anomalies) for e in pool)}
  write_json(root/"config.json",provenance | {"seed":seed,"split":"stable sha256(seed:ticker), 80/20 within tier","template_hash":TEMPLATE_HASH,"label_hash":LABEL_HASH})
  write_csv(root/"entity_pool.csv",[e.to_dict() for e in pool],list(pool[0].to_dict()))
  write_json(root/"tokenization_validation.json",token_report)
  m.start_stage("preflight"); m.finish_stage("preflight",record_count=len(rendered)); m.save()
  localization_jacobians={layer: lens.jacobians[layer].detach().float().cpu() for layer in getattr(lens,"source_layers",[]) } if lens is not None else {}
  baselines={}; baseline_vectors={}
  for t in TEMPLATES:
   p=render_prompt(tokenizer,t,entity=BASELINE_ENTITY,use_chat_template=use_chat_template,max_seq_len=max_seq_len)
   logits,acts,temp=_forward_batch(model,[list(p.input_ids)],layers,device,final_norm=getattr(model,"_final_norm",None),keep_activations_device=True)
   acts={k:v for k,v in acts.items()}; temp=float(temp[0])
   baselines[t]=score_distribution(logits,label_ids,effective_temperature_value=temp); baseline_vectors[t]=acts
  base_rows=[{"template":t,"entity":BASELINE_ENTITY,"probabilities":_flat(baselines[t]["probabilities"]),"expected_score":baselines[t]["expected_score"],"entropy_nats":baselines[t]["entropy_nats"],"effective_temperature":baselines[t]["effective_temperature"]} for t in TEMPLATES]
  write_csv(root/"no_entity_baselines.csv",base_rows,BASE_FIELDS); m.start_stage("baseline"); m.finish_stage("baseline",record_count=3); m.save()
  rows=[]
  metric_items=[(e,t,render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len)) for e in pool for t in TEMPLATES]
  pad=getattr(tokenizer,"pad_token_id",None) or getattr(tokenizer,"eos_token_id",0)
  for start in range(0,len(metric_items),batch_size):
   batch=metric_items[start:start+batch_size]; logits_batch,acts_batch,temp_batch=_forward_batch(model,[list(p.input_ids) for _,_,p in batch],[final],device,pad_token_id=pad,final_norm=getattr(model,"_final_norm",None))
   for i,(e,t,p) in enumerate(batch):
    score=score_distribution(logits_batch[i],label_ids,effective_temperature_value=float(temp_batch[i])); base=baselines[t]
    rows.append({"ticker":e.ticker,"company_name":e.company_name,"template":t,"split":e.split,"familiarity_tier":e.familiarity_tier,"entity_probabilities":_flat(score["probabilities"]),"baseline_probabilities":_flat(base["probabilities"]),"entity_expected_score":score["expected_score"],"baseline_expected_score":base["expected_score"],"entity_entropy_nats":score["entropy_nats"],"baseline_entropy_nats":base["entropy_nats"],"entity_effective_temperature":score["effective_temperature"],"baseline_effective_temperature":base["effective_temperature"],"delta_expected_score":score["expected_score"]-base["expected_score"],"entity_span_start":p.entity_span[0],"entity_span_end":p.entity_span[1],"answer_position":p.answer_position})
  write_csv(root/"raw_entity_template_results.csv",rows,RAW_FIELDS); m.start_stage("metric"); m.finish_stage("metric",record_count=len(rows)); m.save()
  loc=[]
  for t in TEMPLATES:
   targets={e.ticker:next(r["delta_expected_score"] for r in rows if r["ticker"]==e.ticker and r["template"]==t) for e in pool}; train=[e for e in pool if e.split=="train"]; ev=[e for e in pool if e.split=="eval"]
   if set(e.ticker for e in train) & set(e.ticker for e in ev): raise ValueError("train/eval ticker leakage")
   q25,q75=quantile_bounds([targets[e.ticker] for e in train]); high_ids={e.ticker for e in train if targets[e.ticker]>=q75}; low_ids={e.ticker for e in train if targets[e.ticker]<=q25}
   if q25>=q75 or len(high_ids)<2 or len(low_ids)<2 or high_ids & low_ids: raise ValueError("degenerate train high/low groups")
   id_hash=lambda values: hashlib.sha256("\\n".join(sorted(values)).encode()).hexdigest()
   d_model=int(baseline_vectors[t][final].shape[-1]); directions={layer:OnlineDirection(torch.zeros(d_model,dtype=torch.float32),torch.zeros(d_model,dtype=torch.float32)) for layer in layers}; pad=getattr(tokenizer,"pad_token_id",None) or getattr(tokenizer,"eos_token_id",0)
   for start in range(0,len(train),batch_size):
    batch=train[start:start+batch_size]; prompts=[render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len) for e in batch]; _,acts_batch,_temps=_forward_batch(model,[list(p.input_ids) for p in prompts],layers,device,pad_token_id=pad,keep_activations_device=True)
    for layer in layers:
     delta=transported_delta(acts_batch[layer],baseline_vectors[t][layer][0].expand(len(batch),-1),layer=layer,final_layer=final,lens=lens,jacobian_cache=localization_jacobians)
     for i,e in enumerate(batch):
      if targets[e.ticker]>=q75: directions[layer].add(delta[i],high=True)
      elif targets[e.ticker]<=q25: directions[layer].add(delta[i],high=False)
   fitted={}
   for layer in layers:
    direction,norm=directions[layer].normalized(); fitted[layer]=(direction,{"q25":q25,"q75":q75,"n_train":len(train),"n_high":directions[layer].n_high,"n_low":directions[layer].n_low,"high_ids_sha256":id_hash(high_ids),"low_ids_sha256":id_hash(low_ids),"direction_norm":norm,"direction_sha256":direction_hash(direction),"template":t,"layer":layer,"fit_split":"train","source_train_row_count":len(train)})
   eval_directions={layer:fitted[layer][0].to(device=baseline_vectors[t][layer].device,dtype=torch.float32) for layer in layers}
   eval_cos={layer:[] for layer in layers}; eval_y=[]
   for start in range(0,len(ev),batch_size):
    batch=ev[start:start+batch_size]; prompts=[render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len) for e in batch]; _,acts_batch,_temps=_forward_batch(model,[list(p.input_ids) for p in prompts],layers,device,pad_token_id=pad,keep_activations_device=True); eval_y.extend(targets[e.ticker] for e in batch)
    for layer in layers:
     delta=transported_delta(acts_batch[layer],baseline_vectors[t][layer][0].expand(len(batch),-1),layer=layer,final_layer=final,lens=lens,jacobian_cache=localization_jacobians); direction=eval_directions[layer]; denom=delta.norm(dim=1).clamp_min(1e-12)*direction.norm().clamp_min(1e-12); eval_cos[layer].extend((delta@direction/denom).tolist())
   for layer in layers:
    direction,meta=fitted[layer]
    stats=cosine_and_statistics(eval_cos[layer],eval_y); stats.update(meta); loc.append({"layer":layer,"template":t,**{k:stats.get(k) for k in LOC_FIELDS if k not in {"layer","template"}}})
  write_csv(root/"layer_template_localization.csv",loc,LOC_FIELDS); m.start_stage("localization"); m.finish_stage("localization",record_count=len(loc)); m.save()
  write_json(root/"README.md.json",{"note":"See README.md; constituent source files may be current snapshots copied across years and years are source-row provenance, not independently verified historical membership.","pool_count":len(pool),"anomaly_count":sum(bool(e.anomalies) for e in pool)})
  (root/"README.md").write_text("# Synthetic entity-bias pilot\n\n"+json.dumps(provenance,ensure_ascii=False,indent=2)+"\n\n## Metrics\nRestricted nine-label softmax only; expected score is the probability-weighted -4..+4 score, entropy is categorical nats, effective temperature is reciprocal final-normalized answer residual norm, and signed delta-E is entity minus baseline. Localization uses answer-position entity-minus-baseline residuals, canonical Jacobian transport for non-final layers, train-only q25/q75 high-minus-low directions, and eval-only cosine/Pearson/Spearman/linear-R2.\n\n## Limitations\nInput years and memberships preserve source-file provenance; source CSV history may be snapshot-derived and is not independently verified historical membership evidence. No per-example activations, residuals, hidden states, or gradients are written.\n",encoding="utf-8")
  for source in constituents:
   source_path=Path(source)
   m.register_artifact(source_path,artifact_type="constituent_input",stage="preflight",role="input",metadata={"sha256":hashlib.sha256(source_path.read_bytes()).hexdigest()})
  if lens_path: m.register_artifact(lens_path,artifact_type="jacobian_lens",stage="preflight",role="lens")
  m.register_artifact(root/"config.json",artifact_type="config",stage="preflight")
  m.register_artifact(root/"tokenization_validation.json",artifact_type="tokenization_validation",stage="preflight")
  m.register_artifact(root/"entity_pool.csv",artifact_type="entity_pool",stage="preflight",record_count=len(pool)); m.register_artifact(root/"raw_entity_template_results.csv",artifact_type="raw_entity_template_results",stage="metric",record_count=len(rows)); m.register_artifact(root/"no_entity_baselines.csv",artifact_type="no_entity_baselines",stage="baseline",record_count=3); m.register_artifact(root/"layer_template_localization.csv",artifact_type="layer_template_localization",stage="localization",record_count=len(loc)); m.register_artifact(root/"README.md",artifact_type="run_readme",stage="localization")
  complete_manifest(m,required_stages={"preflight","baseline","metric","localization"},postcheck=len(rows)==len(pool)*3 and len(base_rows)==3 and len(loc)==len(layers)*3); return root
 except Exception as exc:
  fail_manifest(m,exc); raise
