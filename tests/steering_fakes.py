"""Checkpoint-free fakes for confirmation-v1 steering tests: a character tokenizer and a tiny real model."""
from __future__ import annotations

import string
from types import SimpleNamespace

import torch

SPECIALS = ("<pad>", "<eos>", "<bos>", "<unk>")
WORDS = ("buy", "sell")
ALPHABET = sorted(set(string.printable) | set("—–«»’"))
VOCAB = list(SPECIALS) + list(WORDS) + ALPHABET
TOKEN_ID = {token: i for i, token in enumerate(VOCAB)}


class CharTokenizer:
    """Characters are tokens except the whole words ``buy`` and ``sell``; BOS is prepended."""

    pad_token_id, eos_token_id, bos_token_id, unk_token_id = 0, 1, 2, 3
    bos_token = "<bos>"
    name_or_path = "fake-char-tokenizer"

    def __init__(self, chat_template: str = "fake") -> None:
        self.chat_template = chat_template

    def _encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets, i = [], [], 0
        while i < len(text):
            for word in WORDS:
                if text.startswith(word, i):
                    ids.append(TOKEN_ID[word])
                    offsets.append((i, i + len(word)))
                    i += len(word)
                    break
            else:
                if text[i] not in TOKEN_ID:
                    raise ValueError(f"character {text[i]!r} not in fake vocabulary")
                ids.append(TOKEN_ID[text[i]])
                offsets.append((i, i + 1))
                i += 1
        return ids, offsets

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False,
                 return_special_tokens_mask=False, return_tensors=None, **_):
        ids, offsets = self._encode(str(text))
        specials = [False] * len(ids)
        if add_special_tokens:
            ids, offsets, specials = [self.bos_token_id] + ids, [(0, 0)] + offsets, [True] + specials
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        out = SimpleNamespace(input_ids=ids)
        if return_offsets_mapping:
            out.offset_mapping = offsets
        if return_special_tokens_mask:
            out.special_tokens_mask = specials
        return out

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        date = kwargs["strftime_now"]("%Y-%m-%d") if "strftime_now" in kwargs else ""
        effort = kwargs.get("reasoning_effort", "")
        return f"{date}{effort}«" + messages[0]["content"] + "»"

    def decode(self, ids, skip_special_tokens=False, **_):
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and i < len(SPECIALS)))

    def convert_ids_to_tokens(self, index):
        return VOCAB[int(index)]

    def convert_tokens_to_ids(self, token):
        return TOKEN_ID.get(token, self.unk_token_id)


class FakeJlens:
    """jlens-style wrapper over a tiny real Qwen3.5 causal LM (CPU, fp32)."""

    def __init__(self, num_layers: int = 3, seed: int = 0, *, full_attention_only: bool = False) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

        torch.manual_seed(seed)
        if full_attention_only:     # plain decoder: fast, standard KV cache
            config = LlamaConfig(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=num_layers, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=16, pad_token_id=0, eos_token_id=1, bos_token_id=2)
            self._hf_model = LlamaForCausalLM(config).eval()
            self._finish()
            self.n_layers = num_layers
            return
        config = Qwen3_5TextConfig(
            vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=(["linear_attention", "full_attention"] * num_layers)[:num_layers],
            pad_token_id=0, eos_token_id=1, bos_token_id=2,
        )
        self._hf_model = Qwen3_5ForCausalLM(config).eval()
        self._finish()
        self.n_layers = num_layers

    def _finish(self) -> None:
        self.hf_model = self._hf_model
        self.layers = self._hf_model.model.layers
        self._final_norm = self._hf_model.model.norm
        self._lm_head = self._hf_model.lm_head
        self.input_device = torch.device("cpu")

    def forward(self, input_ids, **kwargs):
        return self._hf_model(input_ids, **kwargs)
