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
from .artifacts import write_csv, write_json, start_manifest, fail_manifest, complete_manifest

RAW_FIELDS=["ticker","company_name","template","split","familiarity_tier","entity_probabilities","baseline_probabilities","entity_expected_score","baseline_expected_score","entity_entropy_nats","baseline_entropy_nats","entity_effective_temperature","baseline_effective_temperature","delta_expected_score","entity_span_start","entity_span_end","answer_position"]
BASE_FIELDS=["template","entity","probabilities","expected_score","entropy_nats","effective_temperature"]
LOC_FIELDS=["layer","template","mean_cosine","pearson_r","spearman_r","linear_r2","n_train","n_eval","q25","q75","direction_sha256","statistic_flag"]

def _forward_batch(model, ids: list[list[int]], layers: list[int], device: Any, *, pad_token_id: int = 0):
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
   answer={int(k):v[torch.arange(len(ids),device=device),positions].float() for k,v in rec.activations.items()}
  if hasattr(out,"logits"): logits=out.logits[torch.arange(len(ids),device=device),positions].float()
  elif isinstance(out,dict) and "logits" in out: logits=out["logits"][torch.arange(len(ids),device=device),positions].float()
  elif hasattr(model,"unembed"): logits=model.unembed(answer[max(layers)]).float()
  else: raise ValueError("model output has no logits")
  return logits.detach().cpu(),{k:v.detach().cpu() for k,v in answer.items()}
 with torch.no_grad():
  out=model(x,attention_mask=mask)
 logits=out.logits if hasattr(out,"logits") else out["logits"]
 return logits[torch.arange(len(ids),device=device),positions].float().detach().cpu(),{}

def _forward(model, ids: list[int], layers: list[int], device: Any):
 logits,acts=_forward_batch(model,[ids],layers,device)
 return logits[0],{k:v.unsqueeze(1) for k,v in acts.items()}

def _flat(value): return json.dumps(value,separators=(",",":"))

def run_pipeline(*, constituents, model_path, lens_path, artifact_root="artifacts", dataset="synthetic-entity-bias-2020-2025", run_id="run", model=None, tokenizer=None, lens=None, seed=0, max_seq_len=2048, batch_size=16, use_chat_template=True) -> Path:
 pool=load_entity_pool(constituents,seed=seed)
 if not pool: raise ValueError("entity pool is empty")
 if model is None:
  from llm_bias.core.model import load_model
  model,tokenizer,device=load_model(model_path)
 else: device=getattr(model,"device","cpu")
 if tokenizer is None: raise ValueError("tokenizer is required")
 if lens is None and lens_path:
  from jspace_viz.lens import JacobianLens
  lens=JacobianLens.load(str(lens_path))
 if not hasattr(model,"n_layers") or not hasattr(model,"d_model"): raise ValueError("loaded model lacks jlens layer metadata")
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
  write_json(root/"config.json",{"model":model_path,"lens":str(lens_path),"tokenizer_class":type(tokenizer).__name__,"tokenizer_name_or_path":getattr(tokenizer,"name_or_path",None),"input_hashes":{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in constituents},"seed":seed,"split":"stable sha256(seed:ticker), 80/20 within tier","templates":TEMPLATES,"template_hash":TEMPLATE_HASH,"label_hash":LABEL_HASH,"score_mapping":dict(zip(LABELS,SCORES))})
  write_csv(root/"entity_pool.csv",[e.to_dict() for e in pool],list(pool[0].to_dict()))
  write_json(root/"tokenization_validation.json",token_report)
  m.start_stage("preflight"); m.finish_stage("preflight",record_count=len(rendered)); m.save()
  baselines={}; baseline_vectors={}
  for t in TEMPLATES:
   p=render_prompt(tokenizer,t,entity=BASELINE_ENTITY,use_chat_template=use_chat_template,max_seq_len=max_seq_len)
   logits,acts=_forward(model,list(p.input_ids),layers,device)
   baselines[t]=score_distribution(logits,label_ids,residual=(acts.get(final)[:,-1,:] if acts else None),final_norm=getattr(model,"_final_norm",None)); baseline_vectors[t]=acts
  base_rows=[{"template":t,"entity":BASELINE_ENTITY,"probabilities":_flat(baselines[t]["probabilities"]),"expected_score":baselines[t]["expected_score"],"entropy_nats":baselines[t]["entropy_nats"],"effective_temperature":baselines[t]["effective_temperature"]} for t in TEMPLATES]
  write_csv(root/"no_entity_baselines.csv",base_rows,BASE_FIELDS); m.start_stage("baseline"); m.finish_stage("baseline",record_count=3); m.save()
  rows=[]
  metric_items=[(e,t,render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len)) for e in pool for t in TEMPLATES]
  pad=getattr(tokenizer,"pad_token_id",None) or getattr(tokenizer,"eos_token_id",0)
  for start in range(0,len(metric_items),batch_size):
   batch=metric_items[start:start+batch_size]; logits_batch,acts_batch=_forward_batch(model,[list(p.input_ids) for _,_,p in batch],layers,device,pad_token_id=pad)
   for i,(e,t,p) in enumerate(batch):
    acts={k:v[i:i+1].unsqueeze(1) for k,v in acts_batch.items()}; score=score_distribution(logits_batch[i],label_ids,residual=(acts.get(final)[:,0,:] if acts else None),final_norm=getattr(model,"_final_norm",None)); base=baselines[t]
    rows.append({"ticker":e.ticker,"company_name":e.company_name,"template":t,"split":e.split,"familiarity_tier":e.familiarity_tier,"entity_probabilities":_flat(score["probabilities"]),"baseline_probabilities":_flat(base["probabilities"]),"entity_expected_score":score["expected_score"],"baseline_expected_score":base["expected_score"],"entity_entropy_nats":score["entropy_nats"],"baseline_entropy_nats":base["entropy_nats"],"entity_effective_temperature":score["effective_temperature"],"baseline_effective_temperature":base["effective_temperature"],"delta_expected_score":score["expected_score"]-base["expected_score"],"entity_span_start":p.entity_span[0],"entity_span_end":p.entity_span[1],"answer_position":p.answer_position})
  write_csv(root/"raw_entity_template_results.csv",rows,RAW_FIELDS); m.start_stage("metric"); m.finish_stage("metric",record_count=len(rows)); m.save()
  loc=[]
  for t in TEMPLATES:
   targets={e.ticker:next(r["delta_expected_score"] for r in rows if r["ticker"]==e.ticker and r["template"]==t) for e in pool}; train=[e for e in pool if e.split=="train"]; ev=[e for e in pool if e.split=="eval"]
   train_prompts=[render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len) for e in train]; eval_prompts=[render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,use_chat_template=use_chat_template,max_seq_len=max_seq_len) for e in ev]; pad=getattr(tokenizer,"pad_token_id",None) or getattr(tokenizer,"eos_token_id",0)
   _,train_acts=_forward_batch(model,[list(p.input_ids) for p in train_prompts],layers,device,pad_token_id=pad); _,eval_acts=_forward_batch(model,[list(p.input_ids) for p in eval_prompts],layers,device,pad_token_id=pad) if ev else (None,{})
   for layer in layers:
    b=baseline_vectors[t][layer][0,0]; vecs=[transported_delta(train_acts[layer][i],b,layer=layer,final_layer=final,lens=lens) for i in range(len(train))]; ys=[targets[e.ticker] for e in train]
    if len(vecs)<2: continue
    direction,meta=fit_layer_direction(vecs,ys,ids=[e.ticker for e in train],splits=[e.split for e in train],seed=seed); meta.update({"template":t,"layer":layer,"source_train_row_count":len(train),"baseline_template":t}); eval_vec=[transported_delta(eval_acts[layer][i],b,layer=layer,final_layer=final,lens=lens) for i in range(len(ev))]
    if not eval_vec: continue
    stats=evaluate_layer_direction(eval_vec,[targets[e.ticker] for e in ev],direction,meta,ids=[e.ticker for e in ev],splits=[e.split for e in ev]); loc.append({"layer":layer,"template":t,**{k:stats.get(k) for k in LOC_FIELDS if k not in {"layer","template"}}})
  write_csv(root/"layer_template_localization.csv",loc,LOC_FIELDS); m.start_stage("localization"); m.finish_stage("localization",record_count=len(loc)); m.save()
  write_json(root/"README.md.json",{"note":"See README.md; constituent source files may be current snapshots copied across years and years are source-row provenance, not independently verified historical membership.","pool_count":len(pool),"anomaly_count":sum(bool(e.anomalies) for e in pool)})
  (root/"README.md").write_text(f"# Synthetic entity-bias pilot\n\nModel: `{model_path}`\n\nBaseline: `{BASELINE_ENTITY}`. Nine labels 0..8 map to scores -4..4. Templates and scoring instruction are immutable. Pool rows preserve source years and memberships; source CSV history may be snapshot-derived and is not independently verified historical membership evidence. No per-example activations, residuals, hidden states, or gradients are written.\n",encoding="utf-8")
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
