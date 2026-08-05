"""Sampled generated-token Semantic Scope attribution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.model import WrappedModel

from llm_bias.core.model import DEFAULT_MODEL, load_model as load_lens_model
from llm_bias.core.prompting import decode_token, find_token_subsequence, format_messages, format_prompt
from llm_bias.prompt_analysis.readout import DEFAULT_INPUT, load_prompt_table

DEFAULT_OUTPUT_DIR = "artifacts/prompt_analysis/generated_attribution"
RETURN_LABELS = {"very bullish", "bullish", "neutral", "bearish", "very bearish"}


def parse_generated_return_answer(text: Any) -> dict[str, Any]:
    """Parse the strict categorical return answer without aborting generation."""
    result = {
        "predicted_label": None,
        "predicted_confidence": None,
        "parse_status": "invalid",
        "parse_reason": None,
    }
    if not isinstance(text, str):
        result["parse_reason"] = "generated_text_not_string"
        return result
    try:
        start = text.find("{")
        if start < 0:
            raise ValueError("json_object_not_found")
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except (ValueError, json.JSONDecodeError):
        result["parse_reason"] = "invalid_json"
        return result
    if not isinstance(payload, dict):
        result["parse_reason"] = "json_not_object"
        return result
    label = payload.get("label")
    confidence = payload.get("confidence")
    if label not in RETURN_LABELS:
        result["parse_reason"] = "invalid_label"
        return result
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
        result["parse_reason"] = "invalid_confidence"
        return result
    return {
        "predicted_label": label,
        "predicted_confidence": confidence,
        "parse_status": "valid",
        "parse_reason": None,
    }


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
def _generate_tokens(
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
        "pad_token_id": model.tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return model.hf_model.generate(prompt_ids, **generation_kwargs)


def _seed_run(base_seed: int | None, run_index: int) -> int | None:
    if base_seed is None:
        return None
    run_seed = base_seed + run_index
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    return run_seed


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
    runs: int = 1,
    temperature: float = 0.0,
    seed: int | None = None,
    top_p: float = 1.0,
    top_k: int = 0,
    backprop: bool = False,
    dataset_format: str = "auto",
) -> Path:
    if not backprop:
        raise ValueError(
            "generated-token attribution requires backprop=True; pass --backprop from the CLI"
        )
    if sample_per_condition < 1 or max_new_tokens < 1:
        raise ValueError("sample_per_condition and max_new_tokens must be positive")
    if input_top_k is not None and input_top_k < 1:
        raise ValueError("input_top_k must be positive when provided")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if runs < 1:
        raise ValueError("runs must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    if runs > 1 and temperature == 0:
        raise ValueError("runs greater than one require temperature greater than zero")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and between zero and one")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative when provided")

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    columns, rows = load_prompt_table(source, prompt_columns, dataset_format=dataset_format)
    is_return_pairs = bool(rows and rows[0].get("input_schema") == "return-pairs")
    selected_dates = set(dates or ())
    lens_model, tokenizer, _device = load_lens_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Legacy samples shared dates; return-pairs sample pair IDs so duplicate
    # filing dates never collapse distinct filing-item identities.
    effective_dates = set(selected_dates)
    effective_pairs: set[str] = set()
    if is_return_pairs:
        pair_ids = sorted({row["pair_id"] for row in rows})
        effective_pairs = {row["pair_id"] for row in _sample_rows([{"pair_id": p} for p in pair_ids], sample_per_condition)} if not selected_dates else {row["pair_id"] for row in rows if row.get("filing_date") in selected_dates}
    elif not selected_dates:
        date_sets = [{row.get("Date", "") for row in rows if (row.get(column.name) or "").strip() and row.get("Date", "")} for column in columns]
        common_dates = set.intersection(*date_sets) if date_sets else set()
        if not common_dates:
            raise ValueError("prompt columns have no common non-empty dates to sample")
        effective_dates = {row["Date"] for row in _sample_rows([{"Date": date} for date in sorted(common_dates)], sample_per_condition)}

    candidates_by_column = {
        column.name: [
            row for row in rows
            if (row.get(column.name) or row.get("prompt", "")).strip()
            and (not effective_pairs or row.get("pair_id") in effective_pairs)
            and (not effective_dates or row.get("Date", row.get("filing_date", "")) in effective_dates)
            and (row.get("condition") in (None, column.condition or column.context))
        ] for column in columns
    }
    condition_counts = {
        name: len(candidates) for name, candidates in candidates_by_column.items()
    }
    records_per_run = sum(condition_counts.values())
    generation_strategy = "sampling" if temperature > 0 else "greedy"
    generation_settings = {
        "strategy": generation_strategy,
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "base_seed": seed,
        "seed_policy": "base_seed_plus_run_index" if seed is not None else None,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if runs > 1 and any(destination.iterdir()):
        raise FileExistsError(
            f"multi-run output directory must be empty: {destination}"
        )
    run_directories = (
        [destination / f"run_{run_index:03d}" for run_index in range(runs)]
        if runs > 1
        else [destination]
    )
    run_manifest: list[dict[str, Any]] = []
    for run_index, run_destination in enumerate(run_directories):
        run_destination.mkdir(parents=True, exist_ok=True)
        output_path = run_destination / "generated_token_attribution.jsonl"
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        run_seed = _seed_run(seed, run_index)
        run_generation = {
            **generation_settings,
            "run_index": run_index,
            "run_seed": run_seed,
        }
        total = 0
        with temporary.open("w", encoding="utf-8") as handle:
            for column in columns:
                for sample_index, row in enumerate(candidates_by_column[column.name]):
                    prompt = (row.get(column.name) or row.get("prompt", "")).strip()
                    system_prompt = row.get("system_prompt") or None
                    formatted = (format_messages(
                        tokenizer,
                        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                        use_chat_template=True,
                        enable_thinking=False,
                    ) if system_prompt else format_prompt(
                        tokenizer, prompt, use_chat_template=True, enable_thinking=False
                    ))
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
                    generated = _generate_tokens(
                        model,
                        prompt_ids,
                        max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
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
                                "run_index": run_index,
                                "sample_index": sample_index,
                                "date": row.get("Date", ""),
                                "prompt_column": column.name,
                                "index": column.index,
                                "context": column.context,
                                "temperature": temperature,
                                "generation": run_generation,
                                "prompt": prompt,
                                "generated_text": tokenizer.decode(
                                    generated_ids[0].tolist(), skip_special_tokens=False
                                ),
                                "generated_tokens": token_records,
                                **(parse_generated_return_answer(tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)) if is_return_pairs else {}),
                                **({key: row[key] for key in ("input_schema", "pair_id", "filing_date", "ticker", "peer_ticker", "condition", "target_label", "fwd_return_1d", "system_prompt") if key in row}),
                                **({"attribution_scope": "user_message"} if is_return_pairs else {}),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    total += 1
                    if total % 8 == 0 or total == records_per_run:
                        print(
                            f"run {run_index + 1}/{runs}: processed sampled prompts: {total}",
                            flush=True,
                        )
        temporary.replace(output_path)
        run_metadata = {
            "input": str(source),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model": model_name,
            "prompt_columns": [column.name for column in columns],
            "sample_per_condition": sample_per_condition,
            "selected_dates": sorted(effective_dates),
            "condition_counts": condition_counts,
            "records_written": total,
            "max_new_tokens": max_new_tokens,
            "input_top_k": input_top_k,
            "input_attribution_storage": (
                "all_input_tokens" if input_top_k is None else "top_k"
            ),
            "max_seq_len": max_seq_len,
            "generation": generation_strategy,
            "generation_config": run_generation,
            "backpropagation": True,
            "thinking": False,
            "method": "semantic_scope_target_logit_gradient_l2_norm",
            "batch_size": 1,
            "attribution_scope": "raw_user_message_tokens_inside_chat_formatted_prompt",
        }
        (run_destination / "metadata.json").write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run_manifest.append(
            {
                "run_index": run_index,
                "run_seed": run_seed,
                "directory": run_destination.relative_to(destination).as_posix()
                if runs > 1
                else ".",
                "records_written": total,
            }
        )

    if runs > 1:
        manifest = {
            "input": str(source),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model": model_name,
            "runs": runs,
            "run_indices": list(range(runs)),
            "generation": generation_strategy,
            "generation_config": generation_settings,
            "backpropagation": True,
            "sample_per_condition": sample_per_condition,
            "selected_dates": sorted(effective_dates),
            "condition_counts": condition_counts,
            "records_per_run": records_per_run,
            "run_directories": run_manifest,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote generated-token attribution to {destination}", flush=True)
    return destination
