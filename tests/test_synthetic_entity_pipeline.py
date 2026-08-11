from types import SimpleNamespace
import torch
from llm_bias.synthetic_entity_bias.pipeline import _forward_batch, _resolve_device

class DeviceSensitiveFinalNorm:
 def __init__(self): self.seen=[]
 def __call__(self,x):
  self.seen.append(x.device)
  if x.device.type != 'cpu': return x
  return x
class FakeModel:
 def __init__(self): self.calls=[]; self.seen_devices=[]; self._final_norm=DeviceSensitiveFinalNorm()
 def __call__(self,x,attention_mask=None):
  self.calls.append(int(x.shape[0])); self.seen_devices.append(x.device); return SimpleNamespace(logits=torch.zeros(x.shape[0],x.shape[1],32,device=x.device))

def test_wrapper_device_comes_from_underlying_model_parameter():
 class Wrapper:
  _hf_model=torch.nn.Linear(2,2)
 assert _resolve_device(Wrapper()) == next(Wrapper._hf_model.parameters()).device

def test_forward_batch_computes_temperature_before_cpu_transfer():
 model=FakeModel(); logits,acts,temps=_forward_batch(model,[[1,2],[3]], [0], 'cpu', final_norm=model._final_norm)
 assert model.calls == [2] and temps.shape == (2,) and all(t > 0 for t in temps.tolist())

def test_forward_batch_never_exceeds_requested_batch():
 model=FakeModel(); rows=[[i] for i in range(5)]; batch_size=2
 for start in range(0,len(rows),batch_size): _forward_batch(model,rows[start:start+batch_size],[0],'cpu')
 assert model.calls == [2,2,1] and max(model.calls) <= batch_size

def test_eval_localization_uses_cached_jacobian_on_residual_device():
 from llm_bias.synthetic_entity_bias.localization import transported_delta
 residual=torch.tensor([[1.,2.],[3.,4.]])
 result=transported_delta(residual,torch.zeros_like(residual),layer=0,final_layer=1,jacobian_cache={0:torch.eye(2)})
 assert result.device == residual.device and torch.equal(result,residual)
