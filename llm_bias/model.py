"""HuggingFace model loading and device checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import transformers
import jlens


DEFAULT_MODEL = ".cache/models/llama-3.2-1b-instruct"


def resolve_model_name(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path)
    if model == DEFAULT_MODEL:
        raise FileNotFoundError(
            f"Local model is not ready at {path}. Check artifacts/llama_download.log "
            "or pass --model with a HuggingFace model id."
        )
    return model


def load_model(model: str = DEFAULT_MODEL) -> tuple[Any, Any, torch.device]:
    """Load a decoder and wrap it in jlens' HF adapter."""
    name = resolve_model_name(model)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"Loading {name} on {device} with {dtype} (transformers {transformers.__version__})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, use_fast=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model = jlens.from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    return model, tokenizer, device
