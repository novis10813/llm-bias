#!/usr/bin/env python
"""Evaluate one preselected Qwen27B lens without candidate selection."""
from __future__ import annotations
import argparse, json, os
from llm_bias.lens_fitting.evaluation import evaluate_single_lens
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--lens',required=True); p.add_argument('--holdout',required=True); p.add_argument('--output',required=True); p.add_argument('--max-seq-len',type=int,default=128)
a=p.parse_args(); result=evaluate_single_lens(model_name=a.model,lens_path=a.lens,holdout_path=a.holdout,output_path=a.output,max_seq_len=a.max_seq_len,use_chat_template=True,expected_n_prompts=128,device_map='qwen27b_two_gpu',max_memory={0:os.environ.get('GPU_MEMORY','30GiB'),1:os.environ.get('GPU_MEMORY','30GiB')}); print(json.dumps(result,ensure_ascii=False,indent=2))
