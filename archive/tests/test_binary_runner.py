import torch
from torch import nn

from llm_bias.counterfactual_patching.binary_association import (
    BinaryAssociationPair,
    MARGIN_DEFINITION,
    TASK_TYPE,
)
from llm_bias.counterfactual_patching.binary_runner import (
    patch_pair_single_layer,
    steer_pair,
)


class _Tokenizer:
    def __init__(self):
        self.vocab = {char: index for index, char in enumerate("P甲乙媽媽爸爸", start=1)}

    def __call__(self, text, *, add_special_tokens=True):
        ids = [self.vocab[char] for char in text]
        return {"input_ids": ([99, *ids] if add_special_tokens else ids)}


class _Block(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class _Model:
    n_layers = 2

    def __init__(self):
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    def forward(self, input_ids):
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def unembed(self, residual):
        return residual @ self.weight.T


def _pair() -> BinaryAssociationPair:
    return BinaryAssociationPair(
        pair_id="pair",
        contrast_id="contrast",
        direction="source_to_target",
        prompt_order="dad_first",
        split="confirmation",
        source_career_id="source",
        target_career_id="target",
        source_career="甲",
        target_career="乙",
        source_prompt_id="source-dad_first",
        target_prompt_id="target-dad_first",
        source_prompt="P甲甲",
        target_prompt="P乙乙",
        source_entity_spans=[
            {"token_start": 1, "token_end": 2},
            {"token_start": 2, "token_end": 3},
        ],
        target_entity_spans=[
            {"token_start": 1, "token_end": 2},
            {"token_start": 2, "token_end": 3},
        ],
        source_entity_token_ids=[[1], [1]],
        target_entity_token_ids=[[2], [2]],
        candidate_spec=["媽媽", "爸爸"],
    )


def _record() -> dict:
    return {
        "record_id": "source-dad_first",
        "career_id": "source",
        "split": "confirmation",
        "prompt_order": "dad_first",
        "formatted_prompt": "P甲甲",
        "entity_spans": [
            {"token_start": 1, "token_end": 2},
            {"token_start": 2, "token_end": 3},
        ],
        "mother_score": {"token_ids": [4, 5]},
        "father_score": {"token_ids": [6, 7]},
        "margin": 0.0,
        "margin_definition": MARGIN_DEFINITION,
        "task_type": TASK_TYPE,
    }


def test_patch_and_steering_score_multi_token_candidates():
    model = _Model()
    tokenizer = _Tokenizer()
    patched = patch_pair_single_layer(model, tokenizer, _pair(), layer=0, device="cpu")
    steered = steer_pair(
        model,
        tokenizer,
        _record(),
        layer=0,
        direction=torch.ones(2),
        alpha=1.0,
        device="cpu",
    )
    assert isinstance(patched["patched_margin"], float)
    assert isinstance(steered["mother_minus_father_logprob_margin"], float)
    assert "logit" not in " ".join(steered)
