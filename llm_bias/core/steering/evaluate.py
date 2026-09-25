"""Inject, score and generate: one row per (prompt, operator, alpha).

The operator is a precomputed per-token base shift ``B[p]`` (the alpha=1 dose) added after block
``layer`` at the prompt's steer-suffix positions during prefill only (decode tokens are untouched,
exactly as the V2 tokenwise arm). Each row carries the fixed-prefix margin (secondary readout), the
greedy generation with its complete-object decision (primary), and the realized-path margin read
from the same generation's logits.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from llm_bias.core.decision_parsing import parse_complete_decision, parse_strict_decision
from llm_bias.core.decision_readout import generation_finish, path_class, realized_margin, unparsed_kind
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.generation import GenerationConfig, generate_with_logits
from llm_bias.core.inference.interventions import residual_interventions

from .prompts import FormattedPrompt
from .protocol import MAX_NEW_TOKENS

ROW_FIELDS = ("alpha", "margin", "generated_text", "decision", "format", "strict_decision", "finish",
              "n_new_tokens", "path_class", "unparsed_kind", "realized_margin", "realized_status",
              "realized_step", "realized_token_ids")
BASELINE_EXTRA = ("new_ids",)


def parse_fields(text: str, finish: str) -> dict[str, Any]:
    """Everything derivable from the stored text and finish kind (re-derived on resume)."""
    decision, kind = parse_complete_decision(text)
    decision = decision or "unparsed"
    return {"decision": decision, "format": kind, "strict_decision": parse_strict_decision(text) or "unparsed",
            "path_class": path_class(text), "unparsed_kind": unparsed_kind(decision, finish)}


def validate_row(row: Any, alpha: float, where: str, *, baseline: bool = False) -> None:
    fields = set(ROW_FIELDS) | (set(BASELINE_EXTRA) if baseline else set())
    if not isinstance(row, dict) or set(row) != fields or row["alpha"] != alpha:
        raise ValueError(f"row fields or alpha mismatch at {where}")
    if isinstance(row["margin"], bool) or not isinstance(row["margin"], (int, float)) or not math.isfinite(row["margin"]):
        raise ValueError(f"nonfinite fixed-prefix margin at {where}")
    if not isinstance(row["generated_text"], str) or row["finish"] not in ("eos", "max_new_tokens", "model_stop", "empty"):
        raise ValueError(f"invalid generation record at {where}")
    derived = parse_fields(row["generated_text"], row["finish"])
    if any(row[k] != v for k, v in derived.items()):
        raise ValueError(f"decision fields do not re-derive from the generated text at {where}")
    realized = row["realized_margin"]
    if (realized is None) != (row["realized_status"] != "ok") or (
            realized is not None and not math.isfinite(realized)):
        raise ValueError(f"realized-path margin inconsistent at {where}")


def fp32_margin(model: Any, residual_final: torch.Tensor, buy_id: int, sell_id: int, *,
                chunk: int = 32768) -> float:
    """log p(buy) - log p(sell) from a [1, d] final residual with an FP32 norm and FP32 head.

    Numerically the V2 ``fp32_next_token_log_probs`` tail, but the unembedding is cast to FP32 in
    vocabulary chunks so a 262k-vocabulary head never needs a whole FP32 copy.
    """
    with torch.no_grad():
        return _fp32_margin(model, residual_final, buy_id, sell_id, chunk)


def _fp32_margin(model: Any, residual_final: torch.Tensor, buy_id: int, sell_id: int, chunk: int) -> float:
    normalized = model._final_norm(residual_final.float()).float()
    head = model._lm_head
    weight = head.weight
    bias = getattr(head, "bias", None)
    maxima, sums = [], []
    picked: dict[int, torch.Tensor] = {}
    for start in range(0, weight.shape[0], chunk):
        w = weight[start:start + chunk].float().to(normalized.device)
        logits = normalized @ w.T
        if bias is not None:
            logits = logits + bias[start:start + chunk].float().to(normalized.device)
        m = logits.max(dim=-1, keepdim=True).values
        maxima.append(m)
        sums.append((logits - m).exp().sum(dim=-1, keepdim=True))
        for token in (buy_id, sell_id):
            if start <= token < start + logits.shape[-1]:
                picked[token] = logits[:, token - start]
        del w, logits
    top = torch.cat(maxima, dim=-1).max(dim=-1, keepdim=True).values
    total = sum(s * (m - top).exp() for s, m in zip(sums, maxima))
    lse = (top + total.log()).squeeze(-1)
    value = float((picked[buy_id] - lse) - (picked[sell_id] - lse))
    if not math.isfinite(value):
        raise ValueError("nonfinite margin")
    return value


def suffix_shift_transform(span: tuple[int, int], shift: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add ``shift`` ([span_len, d] or [d]) at prompt positions ``span``; decode steps pass through."""
    start, end = span

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor
        out = tensor.clone()
        out[:, start:end, :] = out[:, start:end, :] + shift.to(device=tensor.device, dtype=tensor.dtype)
        return out

    return transform


@dataclass
class Evaluator:
    """Model-bound kernel. ``transforms_for`` lets arms inject arbitrary residual edits (patching)."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = MAX_NEW_TOKENS
    printer: Callable[[str], None] | None = print

    def __post_init__(self) -> None:
        self.adapter = InjectedModelAdapter(self.model, hf_model=getattr(self.model, "_hf_model", self.model))
        self.config = GenerationConfig(max_new_tokens=self.max_new_tokens, temperature=0.0)
        eos: set[int] = set()
        for source in (getattr(getattr(self.adapter.hf_model, "generation_config", None), "eos_token_id", None),
                       getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(source, int):
                eos.add(source)
            elif isinstance(source, (list, tuple)):
                eos.update(int(x) for x in source)
        self.eos_ids = eos
        self.final_layer = int(self.model.n_layers) - 1
        self.device = getattr(self.model, "input_device", None) or next(self.adapter.hf_model.parameters()).device

    def _ids(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(ids)], dtype=torch.long, device=self.device)

    def fixed_prefix_margin(self, fp: FormattedPrompt, transforms: Mapping[int, Any] | None = None) -> float:
        context = residual_interventions(self.model, transforms) if transforms else nullcontext()
        with context:
            res = record_residuals(self.model, self._ids(fp.score_ids), [self.final_layer])[self.final_layer]
        return fp32_margin(self.model, res[:, -1, :], fp.buy_id, fp.sell_id)

    def generate(self, fp: FormattedPrompt, transforms: Mapping[int, Any] | None = None) -> tuple[list[int], tuple]:
        ids = self._ids(fp.ids)
        context = residual_interventions(self.model, transforms) if transforms else nullcontext()
        with context:
            seq, logits = generate_with_logits(self.adapter, ids, self.config)
        return seq[0, ids.shape[1]:].tolist(), logits

    def row(self, fp: FormattedPrompt, alpha: float, transforms: Mapping[int, Any] | None = None, *,
            baseline: bool = False, label: str = "") -> dict[str, Any]:
        """One complete row: fixed-prefix margin, greedy text, decisions, realized-path margin."""
        margin = self.fixed_prefix_margin(fp, transforms)
        new_ids, logits = self.generate(fp, transforms)
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        finish = generation_finish(new_ids, self.eos_ids, self.max_new_tokens)
        fields = parse_fields(text, finish)
        realized = realized_margin(self.tokenizer, new_ids, logits, fields["decision"])
        row = {"alpha": float(alpha), "margin": margin, "generated_text": text, **fields, "finish": finish,
               "n_new_tokens": len(new_ids), **realized}
        if baseline:
            row["new_ids"] = new_ids
        del logits
        if self.printer:
            self.printer(f"  {label} {fp.key} a={alpha:+g} M={margin:+.3f} dec={fields['decision']} "
                         f"path={fields['path_class']} fin={finish} realized={realized['realized_margin']}")
        return {k: row[k] for k in (*ROW_FIELDS, *(BASELINE_EXTRA if baseline else ()))}

    def steered_row(self, fp: FormattedPrompt, layer: int, base_shift: torch.Tensor, alpha: float, *,
                    label: str = "") -> dict[str, Any]:
        """Suffix-steered row with dose alpha * base_shift at ``layer``; alpha 0 means no hook at all."""
        if alpha == 0:
            return self.row(fp, 0.0, None, label=label)
        span = fp.spans["steer_suffix"]
        if base_shift.ndim == 2 and base_shift.shape[0] != span[1] - span[0]:
            raise ValueError("operator token count differs from the steer suffix")
        transform = suffix_shift_transform(span, alpha * base_shift)
        return self.row(fp, alpha, {layer: transform}, label=label)

    def teacher_forced_margin(self, fp: FormattedPrompt, baseline_row: Mapping[str, Any],
                              transforms: Mapping[int, Any] | None = None) -> float | None:
        """Clean-path margin: prompt + the prompt's own alpha-0 tokens up to the decision value."""
        step, pair = baseline_row.get("realized_step"), baseline_row.get("realized_token_ids")
        if baseline_row.get("realized_status") != "ok" or step is None or not pair:
            return None
        ids = list(fp.ids) + list(baseline_row["new_ids"][:step])
        context = residual_interventions(self.model, transforms) if transforms else nullcontext()
        with context:
            res = record_residuals(self.model, self._ids(ids), [self.final_layer])[self.final_layer]
        return fp32_margin(self.model, res[:, -1, :], int(pair[0]), int(pair[1]))


__all__ = ["BASELINE_EXTRA", "Evaluator", "ROW_FIELDS", "fp32_margin", "parse_fields", "suffix_shift_transform",
           "validate_row"]
