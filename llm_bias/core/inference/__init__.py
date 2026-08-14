"""Model-independent inference primitives shared by experiment workflows."""

from .adapter import InjectedModelAdapter, ModelAdapter
from .forward import EncodedBatch, encode_batch, forward_batch, capture_final_residuals
from .generation import GenerationConfig, generate_tokens, finish_reason
from .logits import extract_logits, final_position_logits

__all__ = [
    "InjectedModelAdapter",
    "ModelAdapter",
    "EncodedBatch",
    "GenerationConfig",
    "capture_final_residuals",
    "extract_logits",
    "final_position_logits",
    "finish_reason",
    "forward_batch",
    "encode_batch",
    "generate_tokens",
]
