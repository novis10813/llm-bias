import torch
from llm_bias.synthetic_entity_bias.readout import score_distribution

def test_restricted_readout_uses_only_labels():
 r=score_distribution(torch.tensor([0.,1.,2.,3.,4.,5.,6.,7.,8.,100.]),list(range(9)))
 assert len(r['probabilities'])==9 and abs(sum(r['probabilities'])-1)<1e-6
 assert r['expected_score']>0
