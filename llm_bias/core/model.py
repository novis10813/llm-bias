"""Hugging Face model loading and device checks shared by all workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import transformers
import jlens

DEFAULT_MODEL = ".cache/models/llama-3.2-1b-instruct"


@dataclass(frozen=True)
class ModelDiagnostics:
    """Resolved placement facts persisted by sharded workflows."""

    hf_device_map: dict[str, str]
    layer_devices: dict[str, str]
    embedding_device: str
    final_norm_device: str
    lm_head_device: str
    dtype: str
    parameter_bytes_by_device: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "hf_device_map": self.hf_device_map,
            "layer_devices": self.layer_devices,
            "embedding_device": self.embedding_device,
            "final_norm_device": self.final_norm_device,
            "lm_head_device": self.lm_head_device,
            "dtype": self.dtype,
            "parameter_bytes_by_device": self.parameter_bytes_by_device,
        }


def _is_conditional_generation_checkpoint(name: str, *, trust_remote_code: bool = True) -> bool:
    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
    architectures = getattr(config, "architectures", None) or ()
    return any("ForConditionalGeneration" in architecture for architecture in architectures)


def resolve_model_name(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path)
    if model == DEFAULT_MODEL:
        raise FileNotFoundError(f"Local model is not ready at {path}. Check artifacts/llama_download.log or pass --model with a HuggingFace model id.")
    return model


def load_tokenizer(model: str = DEFAULT_MODEL) -> Any:
    """Load only the tokenizer for data preparation workflows."""
    name = resolve_model_name(model)
    return transformers.AutoTokenizer.from_pretrained(name, use_fast=True)


def load_tokenizer_for_inference(model: str = DEFAULT_MODEL) -> Any:
    """Tokenizer whose encodings match ``load_model``'s inference-time ones.

    ``load_model`` wraps the HF model with ``jlens.from_hf(..., force_bos=True)``,
    which mutates the shared tokenizer in place (``add_bos_token = True`` when
    the tokenizer has a ``bos_token_id``). Any span/position table that must
    index inference-time sequences (capture positions, patch coordinates)
    must be derived with this loader or it will be off by the forced BOS for
    tokenizers whose default is ``add_bos_token=False``.
    """
    tokenizer = load_tokenizer(model)
    if (
        getattr(tokenizer, "bos_token_id", None) is not None
        and hasattr(tokenizer, "add_bos_token")
    ):
        tokenizer.add_bos_token = True
    return tokenizer


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("meta")


def model_diagnostics(model: Any) -> ModelDiagnostics:
    """Inspect the resolved HF map and the wrapped text-module placement."""
    hf_model = getattr(model, "_hf_model", model)
    raw = getattr(hf_model, "hf_device_map", {}) or {}
    device_map = {str(k): str(v) for k, v in raw.items()}
    layers = {
        str(index): str(_module_device(layer))
        for index, layer in enumerate(getattr(model, "layers", ()))
    }
    embedding = getattr(model, "_embed_tokens", None)
    norm = getattr(model, "_final_norm", None)
    head = getattr(model, "_lm_head", None)
    by_device: dict[str, int] = {}
    for parameter in hf_model.parameters():
        key = str(parameter.device)
        by_device[key] = by_device.get(key, 0) + parameter.numel() * parameter.element_size()
    parameters = list(hf_model.parameters())
    dtype = str(parameters[0].dtype) if parameters else "unknown"
    return ModelDiagnostics(
        device_map,
        layers,
        str(_module_device(embedding) if embedding else torch.device("meta")),
        str(_module_device(norm) if norm else torch.device("meta")),
        str(_module_device(head) if head else torch.device("meta")),
        dtype,
        by_device,
    )


def _validate_gpu_only(hf_model: torch.nn.Module, diagnostics: ModelDiagnostics) -> None:
    bad = {device for device in diagnostics.hf_device_map.values() if device in {"cpu", "disk", "meta"} or device.startswith("cpu")}
    bad.update(device for device in diagnostics.parameter_bytes_by_device if device in {"cpu", "meta"})
    if bad:
        raise RuntimeError(f"GPU-only sharded loading forbids CPU/disk/meta placement: {sorted(bad)}")
    if not diagnostics.hf_device_map:
        raise RuntimeError("Sharded loading did not produce a resolved hf_device_map")
    actual_cuda = {device for device in diagnostics.parameter_bytes_by_device if device.startswith("cuda:")}
    if len(actual_cuda) < 2:
        raise RuntimeError(f"Sharded loading did not use two GPUs: {sorted(actual_cuda)}")
    if diagnostics.embedding_device.startswith(("cpu", "meta")) or diagnostics.final_norm_device.startswith(("cpu", "meta")) or diagnostics.lm_head_device.startswith(("cpu", "meta")):
        raise RuntimeError("embedding, final norm, and lm head must remain on GPU")
    if not diagnostics.layer_devices:
        raise RuntimeError("resolved layer map is incomplete")


def qwen27b_two_gpu_device_map() -> dict[str, int]:
    """Return the deterministic GPU-only Qwen27B split used by the workflow."""
    mapping = {"model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.norm": 1, "model.language_model.rotary_emb": 1, "lm_head": 1}
    mapping.update({f"model.language_model.layers.{i}": (0 if i <= 31 else 1) for i in range(64)})
    return mapping


def qwen27b_two_gpu_24_device_map() -> dict[str, int]:
    """Asymmetric GPU-only Qwen27B split for busy machines.

    Keeps only layers 0-23 on GPU 0 (~18 GB with embed + visual tower) so the
    lower shard fits in ~22 GB of free VRAM; GPU 1 hosts layers 24-63, the
    final norm and the lm_head (~31 GB), leaving room for the 6.6 GB Jacobian
    lens alongside the upper shard.
    """
    mapping = {"model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.norm": 1, "model.language_model.rotary_emb": 1, "lm_head": 1}
    mapping.update({f"model.language_model.layers.{i}": (0 if i <= 23 else 1) for i in range(64)})
    return mapping


def qwen27b_two_gpu_16_device_map() -> dict[str, int]:
    """Aggressively asymmetric GPU-only Qwen27B split for very busy machines.

    Keeps only layers 0-15 on GPU 0 (~13 GB with embed + visual tower) so the
    lower shard fits in ~17 GB of free VRAM; GPU 1 hosts layers 16-63, the
    final norm and the lm_head (~35 GB), leaving room for the 6.6 GB Jacobian
    lens alongside the upper shard.
    """
    mapping = {"model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.norm": 1, "model.language_model.rotary_emb": 1, "lm_head": 1}
    mapping.update({f"model.language_model.layers.{i}": (0 if i <= 15 else 1) for i in range(64)})
    return mapping


def gptoss20b_device_map() -> dict[str, int]:
    """Return the deterministic GPU-only GPT-OSS-20B split used by the workflow.

    24 MoE layers split 14/10 across the two GPUs. The GPU 1 shard (upper
    layers + lm_head, ~17.5 GB measured) co-fits with the ~1.9 GB robot
    process only; the GPU 0 shard (~21 GB) co-fits with the ~7 GB
    llama-server. Measured against the 2026-09-23 OOMs: the upper layers are
    the heavier ones (~1.6 GB vs ~0.8 GB per layer), so the heavy tail stays
    on the GPU that hosts no other experiment.
    """
    mapping = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    mapping.update({f"model.layers.{i}": (0 if i <= 13 else 1) for i in range(24)})
    return mapping


def load_model(
    model: str = DEFAULT_MODEL,
    *,
    device_map: str | Mapping[str, int | str] | None = None,
    max_memory: Mapping[int | str, int | str] | None = None,
    dtype: torch.dtype | str | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any, torch.device]:
    """Load a decoder and wrap it in jlens' HF adapter.

    ``device_map=None`` retains the historical single-device behavior. Passing a
    map opts into GPU-only Accelerate dispatch and fails closed on offload.
    An explicit floating dtype supports bounded CPU integration checks; omitting
    it preserves the historical CUDA-bf16 / CPU-fp32 defaults.
    ``dtype="native"`` omits the dtype argument entirely so the checkpoint keeps
    its stored representation (e.g. gpt-oss-20b's packed MXFP4 expert weights,
    which a bf16 cast would dequantize into a much larger live footprint).

    ``trust_remote_code=True`` (default) lets checkpoints that bundle custom
    modeling code (e.g. MiMo) load; it is a no-op for standard architectures.
    """
    name = resolve_model_name(model)
    sharded = device_map is not None
    if sharded and torch.cuda.device_count() < 2:
        raise RuntimeError("Sharded loading requires at least two CUDA GPUs")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    native = dtype == "native"
    if dtype is None:
        dtype = torch.bfloat16 if use_cuda else torch.float32
    if not native and dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise ValueError("model dtype must be floating point")
    dtype_label = "native (checkpoint dtypes)" if native else dtype
    print(f"Loading {name} on {device} with {dtype_label} (transformers {transformers.__version__})")
    tokenizer = load_tokenizer(name)
    auto_model = transformers.AutoModelForCausalLM
    if _is_conditional_generation_checkpoint(name, trust_remote_code=trust_remote_code):
        auto_model = getattr(transformers, "AutoModelForMultimodalLM", None)
        if auto_model is None:
            raise RuntimeError("This checkpoint requires transformers.AutoModelForMultimodalLM")
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
    if not native:
        kwargs["dtype"] = dtype
    requested_map = device_map
    if device_map == "qwen27b_two_gpu":
        requested_map = qwen27b_two_gpu_device_map()
    elif device_map == "qwen27b_two_gpu_24":
        requested_map = qwen27b_two_gpu_24_device_map()
    elif device_map == "qwen27b_two_gpu_16":
        requested_map = qwen27b_two_gpu_16_device_map()
    elif device_map == "gptoss20b_two_gpu":
        requested_map = gptoss20b_device_map()
    if sharded:
        if device_map == "auto":
            raise ValueError("Use an explicit GPU-only map or qwen27b_two_gpu; auto may offload")
        named_budgets = {
            "qwen27b_two_gpu": {0: "28GiB", 1: "28GiB"},
            "qwen27b_two_gpu_24": {0: "24GiB", 1: "36GiB"},
            "qwen27b_two_gpu_16": {0: "16GiB", 1: "40GiB"},
        }
        kwargs.update(
            device_map=requested_map,
            max_memory=max_memory or named_budgets.get(device_map, {0: "28GiB", 1: "28GiB"}),
        )
    hf_model = auto_model.from_pretrained(name, **kwargs)
    if not sharded:
        hf_model = hf_model.to(device)
    wrapped = jlens.from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    diagnostics = model_diagnostics(wrapped)
    if sharded:
        _validate_gpu_only(hf_model, diagnostics)
    setattr(wrapped, "model_diagnostics", diagnostics)
    setattr(wrapped, "requested_device_map", requested_map)
    return wrapped, tokenizer, (wrapped.input_device if sharded else device)
