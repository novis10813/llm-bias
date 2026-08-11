#!/usr/bin/env python
"""GPU-only Qwen 27B placement and short autograd probe (never fits a lens)."""
from __future__ import annotations
import argparse, json, time, subprocess
from pathlib import Path
import torch
from llm_bias.core.model import load_model

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--model', default='.cache/models/qwen3.6-27b')
    p.add_argument('--max-seq-len', type=int, default=64)
    p.add_argument('--dim-batch', type=int, default=1)
    p.add_argument('--gpu-memory', default='28GiB')
    p.add_argument('--load-only', action='store_true')
    p.add_argument('--output', type=Path, default=Path('artifacts/qwen3.6-27b/jacobian-lens/candidates/chinese_simplified/benchmarks/probe.json'))
    args=p.parse_args()
    if torch.cuda.device_count() < 2: raise RuntimeError('probe requires two CUDA devices')
    free_before = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip().splitlines()
    model, tokenizer, device = load_model(args.model, device_map='qwen27b_two_gpu', max_memory={0:args.gpu_memory,1:args.gpu_memory})
    text='请简要说明一个城市如何改善公共交通。'
    if args.load_only:
        result={'model':args.model,'input_device':str(device),'load_only':True,'diagnostics':model.model_diagnostics.as_dict(),'free_vram_before_mib':[int(x) for x in free_before],'peak_vram_bytes':{str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}}
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(result,ensure_ascii=False,indent=2)); return
    ids=model.encode(text, max_length=args.max_seq_len)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.repeat(args.dim_batch, 1)
    layers=[0,31,62,63]
    records={}
    started=time.time()
    from jlens.hooks import ActivationRecorder
    from llm_bias.lens_fitting.device_probe import validate_probe_gradients
    with ActivationRecorder(model.layers, at=layers, start_graph_at=0) as recorder:
        output=model.forward(ids)
        for layer in layers:
            value=recorder.activations[layer]
            records[str(layer)]={'device':str(value.device),'shape':list(value.shape),'requires_grad':bool(value.requires_grad)}
        gradient_result=validate_probe_gradients(activations=recorder.activations, source_layers=[0,31,62], target_layer=63, source_devices={layer: records[str(layer)]['device'] for layer in [0,31,62]})
    result={'model':args.model,'input_device':str(device),'dim_batch':args.dim_batch,'input_batch':args.dim_batch,'elapsed_seconds':time.time()-started,'free_vram_before_mib':[int(x) for x in free_before],'peak_vram_bytes':{str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())},'diagnostics':model.model_diagnostics.as_dict(),'activations':records,'gradient_result':gradient_result,'output_shape':list(output.shape) if hasattr(output,'shape') else None,'probe_only':True}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
