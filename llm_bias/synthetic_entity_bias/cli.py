"""CLI for the synthetic entity-bias pilot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .entities import load_entity_pool
from .spec import TEMPLATES, BASELINE_ENTITY
from .prompts import render_prompt, validate_token_contract

def build_parser():
 p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
 for name in ("validate","run"):
  q=sub.add_parser(name); q.add_argument("--constituents",action="append",required=True); q.add_argument("--model",required=True); q.add_argument("--lens",required=True); q.add_argument("--seed",type=int,default=0); q.add_argument("--max-seq-len",type=int,default=2048)
  q.add_argument("--batch-size",type=int,default=16)
  if name=="run": q.add_argument("--artifact-root",default="artifacts"); q.add_argument("--dataset",default="synthetic-entity-bias-2020-2025"); q.add_argument("--run-id",required=True)
 return p

def _load(args):
 from llm_bias.core.model import load_model
 model,tokenizer,device=load_model(args.model); return model,tokenizer,device

def main():
 args=build_parser().parse_args(); pool=load_entity_pool(args.constituents,seed=args.seed)
 model,tokenizer,device=_load(args)
 if args.command=="validate":
  from jspace_viz.lens import JacobianLens
  from llm_bias.core.lens_artifacts import validate_lens_for_model
  lens_path=Path(args.lens)
  if not lens_path.is_file(): raise FileNotFoundError(lens_path)
  lens=JacobianLens.load(str(lens_path)); validate_lens_for_model(model=model,lens=lens,model_name=args.model,lens_path=lens_path,require_complete=True)
  rendered=[render_prompt(tokenizer,t,entity=e.company_name,ticker=e.ticker,max_seq_len=args.max_seq_len) for e in pool for t in TEMPLATES]
  rendered += [render_prompt(tokenizer,t,entity=BASELINE_ENTITY,max_seq_len=args.max_seq_len) for t in TEMPLATES]
  result=validate_token_contract(tokenizer,rendered); result.update(pool_count=len(pool),anomaly_count=sum(bool(e.anomalies) for e in pool),model=args.model,lens=args.lens)
  print(json.dumps(result,ensure_ascii=False,indent=2)); return
 from .pipeline import run_pipeline
 root=run_pipeline(constituents=args.constituents,model_path=args.model,lens_path=args.lens,artifact_root=args.artifact_root,dataset=args.dataset,run_id=args.run_id,model=model,tokenizer=tokenizer,device=device,seed=args.seed,max_seq_len=args.max_seq_len,batch_size=args.batch_size)
 print(root)
