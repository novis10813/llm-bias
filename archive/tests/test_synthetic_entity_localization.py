import pytest, torch
from llm_bias.synthetic_entity_bias.localization import fit_layer_direction, evaluate_layer_direction

def test_direction_rejects_tied_quantiles():
 with pytest.raises(ValueError): fit_layer_direction([torch.ones(2),torch.ones(2)],[1.,1.])

def test_eval_is_eval_only():
 d,m=fit_layer_direction([torch.tensor([1.,0.]),torch.tensor([0.,1.]),torch.tensor([2.,0.]),torch.tensor([0.,2.])],[1.,1.,4.,4.],ids=['a','b','c','d'],splits=['train']*4)
 with pytest.raises(ValueError): evaluate_layer_direction([torch.ones(2)],[1.],d,m,splits=['train'])
