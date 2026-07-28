"""Faithfulness validation for generated-token Semantic Scope scores."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.model import WrappedModel

from llm_bias.core.model import DEFAULT_MODEL, load_model
from llm_bias.core.prompting import find_token_subsequence, format_prompt

DEFAULT_ATTRIBUTION = (
    "artifacts/prompt_analysis/generated_attribution/"
    "generated_token_attribution.jsonl"
)
DEFAULT_OUTPUT_DIR = "artifacts/prompt_analysis/attribution_validation"
ABLATION_RATES = (0.0, 0.05, 0.10, 0.20)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            rows.append(value)
    return rows


def _aopc(rates: Iterable[float], deltas: Iterable[float]) -> float:
    rate_values = list(rates)
    delta_values = list(deltas)
    if len(rate_values) != len(delta_values) or len(rate_values) < 2:
        raise ValueError("AOPC needs at least two rate/delta points")
    return sum(
        0.5
        * (rate_values[index + 1] - rate_values[index])
        * (delta_values[index] + delta_values[index + 1])
        for index in range(len(rate_values) - 1)
    )


@torch.no_grad()
def _score_target_token(
    *,
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    generated_prefix: torch.Tensor,
    target_id: int,
    ablation_positions: Iterable[int] = (),
) -> dict[str, float]:
    """Score one target token after zeroing selected prompt embeddings."""
    full_ids = torch.cat([prompt_ids, generated_prefix], dim=1)
    attention_mask = torch.ones_like(full_ids)
    positions = tuple(sorted(set(int(position) for position in ablation_positions)))

    def zero_prompt_embeddings(
        _module: Any, _inputs: Any, output: torch.Tensor
    ) -> torch.Tensor:
        if not positions:
            return output
        ablated = output.clone()
        ablated[:, list(positions)] = 0
        return ablated

    handle = model._embed.register_forward_hook(zero_prompt_embeddings)
    try:
        final_layer = model.n_layers - 1
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model._decoder(
                input_ids=full_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            residual = recorder.activations[final_layer][0, -1].float()
        logits = model.unembed(residual).float()
        log_probability = logits.log_softmax(dim=-1)[target_id]
        return {
            "target_logit": float(logits[target_id]),
            "log_probability": float(log_probability),
        }
    finally:
        handle.remove()


def _positions_for_rate(
    ranked_positions: list[int], input_length: int, rate: float
) -> list[int]:
    if rate <= 0:
        return []
    count = min(input_length, max(1, math.ceil(input_length * rate)))
    return ranked_positions[:count]


def _evaluate_method(
    *,
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    generated_prefix: torch.Tensor,
    target_id: int,
    baseline: dict[str, float],
    ranked_positions: list[int],
    input_length: int,
) -> dict[str, Any]:
    log_probability_deltas = [0.0]
    target_logit_deltas = [0.0]
    for rate in ABLATION_RATES[1:]:
        score = _score_target_token(
            model=model,
            prompt_ids=prompt_ids,
            generated_prefix=generated_prefix,
            target_id=target_id,
            ablation_positions=_positions_for_rate(ranked_positions, input_length, rate),
        )
        log_probability_deltas.append(
            score["log_probability"] - baseline["log_probability"]
        )
        target_logit_deltas.append(score["target_logit"] - baseline["target_logit"])
    return {
        "rates": list(ABLATION_RATES),
        "log_probability_delta": log_probability_deltas,
        "target_logit_delta": target_logit_deltas,
        "aopc": _aopc(ABLATION_RATES, log_probability_deltas),
    }


def _visible_output_token(token: str) -> bool:
    return not token.startswith("<|")


def evaluate_semantic_scope(
    *,
    attribution_path: str | Path = DEFAULT_ATTRIBUTION,
    model_name: str = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 0,
) -> Path:
    """Evaluate Semantic Scope and a random baseline with prompt ablations."""
    source = Path(attribution_path)
    rows = _read_jsonl(source)
    if not rows:
        raise ValueError(f"no attribution rows found in {source}")

    lens_model, tokenizer, _device = load_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "semantic_scope_aopc.jsonl"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    total_tokens = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"attribution row {row_index} has no prompt")
            formatted = format_prompt(
                tokenizer,
                prompt,
                use_chat_template=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids.to(model.device)
            raw_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids
            input_start, input_end = find_token_subsequence(
                prompt_ids[0].tolist(), raw_ids[0].tolist()
            )
            input_positions = list(range(input_start, input_end))
            if not input_positions:
                raise ValueError(f"attribution row {row_index} has empty input span")

            generated_tokens = row.get("generated_tokens")
            if not isinstance(generated_tokens, list):
                raise ValueError(f"attribution row {row_index} has no generated tokens")
            generated_ids = [int(token["token_id"]) for token in generated_tokens]
            random_order = input_positions[:]
            random.Random(seed + row_index).shuffle(random_order)
            evaluated_tokens: list[dict[str, Any]] = []
            for position, generated_token in enumerate(generated_tokens):
                token = str(generated_token.get("token", ""))
                if not _visible_output_token(token):
                    continue
                target_id = int(generated_token["token_id"])
                prefix = torch.tensor(
                    [generated_ids[:position]], dtype=torch.long, device=model.device
                )
                baseline = _score_target_token(
                    model=model,
                    prompt_ids=prompt_ids,
                    generated_prefix=prefix,
                    target_id=target_id,
                )
                attribution_items = generated_token.get("top_input_tokens")
                if not isinstance(attribution_items, list) or len(attribution_items) < len(
                    input_positions
                ):
                    raise ValueError(
                        "Semantic Scope validation requires complete input-token scores"
                    )
                semantic_order = [
                    int(item["position"])
                    for item in sorted(
                        attribution_items,
                        key=lambda item: float(item["attribution"]),
                        reverse=True,
                    )
                ]
                semantic = _evaluate_method(
                    model=model,
                    prompt_ids=prompt_ids,
                    generated_prefix=prefix,
                    target_id=target_id,
                    baseline=baseline,
                    ranked_positions=semantic_order,
                    input_length=len(input_positions),
                )
                random_baseline = _evaluate_method(
                    model=model,
                    prompt_ids=prompt_ids,
                    generated_prefix=prefix,
                    target_id=target_id,
                    baseline=baseline,
                    ranked_positions=random_order,
                    input_length=len(input_positions),
                )
                evaluated_tokens.append(
                    {
                        "position": position,
                        "token_id": target_id,
                        "token": token,
                        "target_logit": baseline["target_logit"],
                        "log_probability": baseline["log_probability"],
                        "semantic_scope": semantic,
                        "random": random_baseline,
                    }
                )
                total_tokens += 1
            semantic_aopcs = [
                float(token["semantic_scope"]["aopc"]) for token in evaluated_tokens
            ]
            random_aopcs = [float(token["random"]["aopc"]) for token in evaluated_tokens]
            handle.write(
                json.dumps(
                    {
                        "date": row.get("date", ""),
                        "index": row.get("index", ""),
                        "context": row.get("context", ""),
                        "prompt_column": row.get("prompt_column", ""),
                        "input_span_length": len(input_positions),
                        "ablation": "zero_input_embedding",
                        "rates": list(ABLATION_RATES),
                        "generated_tokens": evaluated_tokens,
                        "summary": {
                            "semantic_scope_aopc_mean": sum(semantic_aopcs)
                            / len(semantic_aopcs),
                            "random_aopc_mean": sum(random_aopcs) / len(random_aopcs),
                            "n_output_tokens": len(evaluated_tokens),
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if (row_index + 1) % 2 == 0 or row_index + 1 == len(rows):
                print(
                    f"validated attribution rows: {row_index + 1}/{len(rows)} "
                    f"({total_tokens} output tokens)",
                    flush=True,
                )
    temporary.replace(output_path)
    metadata = {
        "input": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": model_name,
        "method": "semantic_scope_top_input_ablation",
        "baseline": "deterministic_random_input_positions",
        "ablation": "zero_input_embedding",
        "rates": list(ABLATION_RATES),
        "aopc": "trapezoidal_integral_of_log_probability_delta_from_zero_rate",
        "target_scope": "visible_greedy_generated_output_tokens",
        "seed": seed,
        "n_rows": len(rows),
        "n_output_tokens": total_tokens,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote Semantic Scope validation to {destination}", flush=True)
    return destination
