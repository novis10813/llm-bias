from types import SimpleNamespace

import torch

from llm_bias.core.inference import GenerationConfig, encode_batch, extract_logits, finish_reason


def test_encode_batch_tracks_each_final_non_padding_position():
    encoded = encode_batch([[1, 2, 3], [4]], "cpu", pad_token_id=99)
    assert encoded.input_ids.tolist() == [[1, 2, 3], [4, 99, 99]]
    assert encoded.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert encoded.final_positions.tolist() == [2, 0]


def test_extract_logits_accepts_hf_output_shapes():
    logits = torch.randn(2, 3, 5)
    assert torch.equal(extract_logits(SimpleNamespace(logits=logits)), logits)
    assert torch.equal(extract_logits((logits, "cache")), logits)
    assert torch.equal(extract_logits({"logits": logits}), logits)


def test_generation_config_only_adds_sampling_controls_when_sampling():
    assert GenerationConfig(pad_token_id=0).as_kwargs() == {
        "max_new_tokens": 64,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": 0,
    }
    assert GenerationConfig(temperature=0.7, top_p=0.9, top_k=10).as_kwargs()["top_k"] == 10


def test_finish_reason_distinguishes_eos_and_length():
    assert finish_reason([], eos_token_id=0, max_new_tokens=2) == "empty"
    assert finish_reason([3, 0], eos_token_id=0, max_new_tokens=2) == "eos_token"
    assert finish_reason([3, 4], eos_token_id=0, max_new_tokens=2) == "max_new_tokens"
