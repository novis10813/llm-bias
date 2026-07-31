"""Sampled generated-token Semantic Scope attribution."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.model import WrappedModel

from llm_bias.core.model import DEFAULT_MODEL, load_model as load_lens_model
from llm_bias.core.prompting import decode_token, find_token_subsequence, format_prompt
from llm_bias.prompt_analysis.readout import (
    DEFAULT_INPUT,
    discover_prompt_columns,
)

DEFAULT_OUTPUT_DIR = "artifacts/prompt_analysis/generated_attribution"


def _sample_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("sample count must be positive")
    if len(rows) <= count:
        return rows
    indices = [
        round(index * (len(rows) - 1) / (count - 1))
        for index in range(count)
    ] if count > 1 else [0]
    return [rows[index] for index in indices]


def _find_subsequence(sequence: list[int], target: list[int]) -> tuple[int, int]:
    """Backward-compatible alias for shared token alignment."""
    return find_token_subsequence(sequence, target)


def _semantic_scope_scores(gradient: torch.Tensor) -> torch.Tensor:
    """Return one Semantic Scope influence score per input position."""
    if gradient.ndim != 3:
        raise ValueError(
            f"expected [batch, sequence, hidden] gradient, got {gradient.shape}"
        )
    return torch.linalg.vector_norm(gradient.float(), ord=2, dim=-1)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return reader.fieldnames, list(reader)


@torch.enable_grad()
def _attribute_generated_token(
    *,
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    generated_prefix: torch.Tensor,
    target_id: int,
    input_top_k: int | None,
    input_span: tuple[int, int],
) -> dict[str, Any]:
    full_ids = torch.cat([prompt_ids, generated_prefix], dim=1)
    attention_mask = torch.ones_like(full_ids)
    embedding_box: dict[str, torch.Tensor] = {}

    def root_embedding(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        rooted = output.detach().requires_grad_(True)
        embedding_box["value"] = rooted
        return rooted

    final_layer = model.n_layers - 1
    handle = model._embed.register_forward_hook(root_embedding)
    try:
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model._decoder(
                input_ids=full_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            residual = recorder.activations[final_layer][0, -1].float()
        logits = model.unembed(residual).float()
        target_logit = logits[target_id]
        log_probability = logits.log_softmax(dim=-1)[target_id]
        embedding = embedding_box["value"]
        gradient = torch.autograd.grad(target_logit, embedding)[0]
    finally:
        handle.remove()

    input_start, input_end = input_span
    contribution = _semantic_scope_scores(gradient)[0, input_start:input_end]
    top_k = input_end - input_start if input_top_k is None else min(
        input_top_k, input_end - input_start
    )
    top = contribution.topk(top_k)
    return {
        "token_id": target_id,
        "token": decode_token(model.tokenizer, target_id),
        "logit": float(target_logit.detach()),
        "log_probability": float(log_probability.detach()),
        "top_input_tokens": [
            {
                "rank": rank,
                "position": input_start + int(position),
                "prompt_position": int(position),
                "token_id": int(prompt_ids[0, input_start + position]),
                "token": decode_token(
                    model.tokenizer, int(prompt_ids[0, input_start + position])
                ),
                "attribution": float(value),
            }
            for rank, (position, value) in enumerate(
                zip(top.indices.tolist(), top.values.tolist(), strict=True),
                start=1,
            )
        ],
    }


@torch.no_grad()
def _greedy_generate(
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    return model.hf_model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=model.tokenizer.eos_token_id,
    )


def analyze_generated_attribution(
    *,
    input_path: str = DEFAULT_INPUT,
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_per_condition: int = 32,
    max_new_tokens: int = 64,
    input_top_k: int | None = None,
    max_seq_len: int = 256,
    prompt_columns: Iterable[str] | None = None,
    dates: Iterable[str] | None = None,
) -> Path:
    if sample_per_condition < 1 or max_new_tokens < 1:
        raise ValueError("sample_per_condition and max_new_tokens must be positive")
    if input_top_k is not None and input_top_k < 1:
        raise ValueError("input_top_k must be positive when provided")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    fieldnames, rows = _read_rows(source)
    columns = discover_prompt_columns(fieldnames, prompt_columns)
    selected_dates = set(dates or ())
    lens_model, tokenizer, _device = load_lens_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Sample the same dates for every condition.  This is important for real
    # market tables where a ticker can have missing prompt rows: sampling each
    # column independently would leave the dashboard with too few common dates
    # to select crash/normal examples.
    effective_dates = set(selected_dates)
    if not selected_dates:
        date_sets = [
            {
                row.get("Date", "")
                for row in rows
                if (row.get(column.name) or "").strip() and row.get("Date", "")
            }
            for column in columns
        ]
        common_dates = set.intersection(*date_sets) if date_sets else set()
        if not common_dates:
            raise ValueError("prompt columns have no common non-empty dates to sample")
        shared_rows = [{"Date": date} for date in sorted(common_dates)]
        effective_dates = {
            row["Date"] for row in _sample_rows(shared_rows, sample_per_condition)
        }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "generated_token_attribution.jsonl"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    total = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for column in columns:
            candidates = [
                row
                for row in rows
                if (row.get(column.name) or "").strip()
                and (
                    not effective_dates
                    or row.get("Date", "") in effective_dates
                )
            ]
            sampled = candidates
            for sample_index, row in enumerate(sampled):
                prompt = (row.get(column.name) or "").strip()
                formatted = format_prompt(
                    tokenizer,
                    prompt,
                    use_chat_template=True,
                    enable_thinking=False,
                )
                encoded = tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                prompt_ids = encoded.input_ids.to(model.device)
                raw_encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                input_span = _find_subsequence(
                    prompt_ids[0].tolist(), raw_encoded.input_ids[0].tolist()
                )
                with torch.no_grad():
                    generated = _greedy_generate(model, prompt_ids, max_new_tokens)
                generated_ids = generated[:, prompt_ids.shape[1] :]
                token_records = []
                for position, target_id in enumerate(generated_ids[0].tolist()):
                    prefix = generated_ids[:, :position]
                    token_records.append(
                        {
                            "position": position,
                            **_attribute_generated_token(
                                model=model,
                                prompt_ids=prompt_ids,
                                generated_prefix=prefix,
                                target_id=int(target_id),
                                input_top_k=input_top_k,
                                input_span=input_span,
                            ),
                        }
                    )
                handle.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "date": row.get("Date", ""),
                            "prompt_column": column.name,
                            "index": column.index,
                            "context": column.context,
                            "prompt": prompt,
                            "generated_text": tokenizer.decode(
                                generated_ids[0].tolist(), skip_special_tokens=False
                            ),
                            "generated_tokens": token_records,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                total += 1
                if total % 8 == 0 or total == len(columns) * min(sample_per_condition, len(rows)):
                    print(f"processed sampled prompts: {total}", flush=True)
    temporary.replace(output_path)
    metadata = {
        "input": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": model_name,
        "prompt_columns": [column.name for column in columns],
        "sample_per_condition": sample_per_condition,
        "selected_dates": sorted(effective_dates),
        "max_new_tokens": max_new_tokens,
        "input_top_k": input_top_k,
        "input_attribution_storage": "all_input_tokens" if input_top_k is None else "top_k",
        "max_seq_len": max_seq_len,
        "generation": "greedy",
        "thinking": False,
        "method": "semantic_scope_target_logit_gradient_l2_norm",
        "batch_size": 1,
        "attribution_scope": "raw_user_message_tokens_inside_chat_formatted_prompt",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote generated-token attribution to {destination}", flush=True)
    return destination
