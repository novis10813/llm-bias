"""Deterministic placement checks for the named Qwen27B two-GPU device maps."""

from llm_bias.core.model import (
    qwen27b_two_gpu_16_device_map,
    qwen27b_two_gpu_24_device_map,
    qwen27b_two_gpu_device_map,
)


def test_explicit_cpu_dtype_preserves_default(monkeypatch):
    import torch
    from types import SimpleNamespace
    from llm_bias.core import model as module
    seen = []
    def load(name, **kwargs):
        seen.append(kwargs["dtype"])
        return torch.nn.Linear(2, 2, dtype=kwargs["dtype"])
    monkeypatch.setattr(module, "load_tokenizer", lambda _: object())
    monkeypatch.setattr(module, "_is_conditional_generation_checkpoint", lambda _: False)
    monkeypatch.setattr(module.transformers.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(module.jlens, "from_hf", lambda raw, *a, **kw: SimpleNamespace(_hf_model=raw, layers=[]))
    module.load_model("fake")
    module.load_model("fake", dtype=torch.bfloat16)
    assert seen == [torch.float32, torch.bfloat16]


def test_balanced_map_splits_32_32():
    mapping = qwen27b_two_gpu_device_map()
    layers = {
        int(name.split(".")[-1]): device
        for name, device in mapping.items()
        if name.startswith("model.language_model.layers.")
    }
    assert len(layers) == 64
    assert [device for device in layers.values()] == [0] * 32 + [1] * 32
    assert mapping["model.language_model.embed_tokens"] == 0
    assert mapping["model.language_model.norm"] == 1
    assert mapping["lm_head"] == 1
    assert mapping["model.visual"] == 0


def test_asymmetric_map_splits_24_40_and_keeps_head_on_gpu1():
    mapping = qwen27b_two_gpu_24_device_map()
    layers = {
        int(name.split(".")[-1]): device
        for name, device in mapping.items()
        if name.startswith("model.language_model.layers.")
    }
    assert len(layers) == 64
    assert [device for device in layers.values()] == [0] * 24 + [1] * 40
    # The lm_head, final norm and rotary embedding stay on the lens GPU.
    assert mapping["model.language_model.norm"] == 1
    assert mapping["lm_head"] == 1
    assert mapping["model.language_model.rotary_emb"] == 1
    assert mapping["model.language_model.embed_tokens"] == 0
    assert mapping["model.visual"] == 0


def test_aggressive_map_splits_16_48_and_keeps_head_on_gpu1():
    mapping = qwen27b_two_gpu_16_device_map()
    layers = {
        int(name.split(".")[-1]): device
        for name, device in mapping.items()
        if name.startswith("model.language_model.layers.")
    }
    assert len(layers) == 64
    assert [device for device in layers.values()] == [0] * 16 + [1] * 48
    assert mapping["model.language_model.norm"] == 1
    assert mapping["lm_head"] == 1
    assert mapping["model.language_model.rotary_emb"] == 1
    assert mapping["model.language_model.embed_tokens"] == 0
    assert mapping["model.visual"] == 0
