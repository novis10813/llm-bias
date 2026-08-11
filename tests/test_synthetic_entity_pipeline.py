from types import SimpleNamespace
import torch
from llm_bias.synthetic_entity_bias.pipeline import _forward_batch

class DeviceSensitiveFinalNorm:
 def __init__(self): self.seen=[]
 def __call__(self,x):
  self.seen.append(x.device)
  if x.device.type != 'cpu': return x
  return x
class FakeModel:
 def __init__(self): self.calls=[]; self._final_norm=DeviceSensitiveFinalNorm()
 def __call__(self,x,attention_mask=None):
  self.calls.append(int(x.shape[0])); return SimpleNamespace(logits=torch.zeros(x.shape[0],x.shape[1],32,device=x.device))

def test_forward_batch_computes_temperature_before_cpu_transfer():
 model=FakeModel(); logits,acts,temps=_forward_batch(model,[[1,2],[3]], [0], 'cpu', final_norm=model._final_norm)
 assert model.calls == [2] and temps.shape == (2,) and all(t > 0 for t in temps.tolist())

def test_forward_batch_never_exceeds_requested_batch():
 model=FakeModel(); rows=[[i] for i in range(5)]; batch_size=2
 for start in range(0,len(rows),batch_size): _forward_batch(model,rows[start:start+batch_size],[0],'cpu')
 assert model.calls == [2,2,1] and max(model.calls) <= batch_size
