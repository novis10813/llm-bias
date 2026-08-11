#!/usr/bin/env python
from __future__ import annotations
import argparse, json, hashlib, os, subprocess, resource
from pathlib import Path
import torch
import jlens
p=argparse.ArgumentParser(); p.add_argument('--lens',required=True); p.add_argument('--prompts',type=int,required=True); p.add_argument('--output',required=True); a=p.parse_args()
path=Path(a.lens); meta=Path(str(path)+'.metadata.json'); finite=False; complete=False; metadata_ok=False; resume_ok=False
try:
 lens=jlens.JacobianLens.load(str(path)); finite=all(torch.isfinite(v).all().item() for v in lens.jacobians.values()); complete=(lens.d_model==5120 and lens.source_layers==list(range(63))); m=json.loads(meta.read_text()); metadata_ok=(m.get('calibration_count')==a.prompts and m.get('d_model')==5120 and m.get('source_layers')==list(range(63))); resume_ok=bool(m.get('checkpoint_path') and Path(m['checkpoint_path']).is_file())
except Exception as exc: m={'error':repr(exc)}
try: vram=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines(); vram=[{'used_mib':int(x.split(',')[0]),'free_mib':int(x.split(',')[1])} for x in vram]
except Exception: vram=[]
result={'prompts':a.prompts,'lens_finite':finite,'complete_63_layers_d5120':complete,'metadata_ok':metadata_ok,'resume_ok':resume_ok,'stable_resources':bool(vram and min(x['free_mib'] for x in vram)>=512),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'vram':vram,'lens_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,'metadata':m}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(0 if all(result[k] for k in ('lens_finite','complete_63_layers_d5120','metadata_ok','resume_ok','stable_resources')) else 2)
