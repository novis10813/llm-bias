#!/usr/bin/env python
"""Per-(layer, position) first-order attribution of Buy/Sell decision tokens.

For every prompt of a frozen split, backpropagates the decision-position
``log P(buy)`` and ``log P(sell)`` (FP32 final norm + unembedding tail, the
V2 scoring target) through the full decoder and collects per-position L2-norm
gradient scores with respect to

  * the embedding output (the input-token view used by the
    ``prompt_analysis`` Semantic Scope attribution), and
  * every decoder layer's residual output (the internal (layer, position)
    view this operator exists to answer).

This is the internal-state generalization of the generated-token
attribution in ``llm_bias/prompt_analysis/attribution.py``: same
single-target log-probability backprop, same per-position norm scoring
convention, gradient collection moved from the embedding root to each layer
residual.  Prompt preparation reuses the V2 outcome-flip pipeline
(scoring prompt, candidate tokens, position rules) so numbers stay
comparable with V2.

Interpretation boundary (repo semantic rules): first-order local
sensitivity at the clean point.  Not an attention map, not a standalone
causal claim.  Raw norms are not calibrated across layers; read the global
top-k together with the layer x position-class aggregate.

Only compact derived outputs are written (top-k lists, per-class means/max,
token text, provenance); no raw activations, residuals, or gradients are
persisted.

Usage (from repository root):

    uv run python scripts/jspace_outcome_token_attribution.py \
        --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
        --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
        --config artifacts/qwen3.5-4b/jspace-outcome-direction-flip/config-technology-draft1.json \
        --model .cache/models/qwen3.5-4b \
        --split discovery \
        --output-dir artifacts/qwen3.5-4b/jspace-outcome-token-attribution \
        --top-k 20 --embedding-top-k 10
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import write_json, write_metadata
from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.continuation_scoring import (
    continuation_token_ids,
    fp32_next_token_log_probs,
)
from llm_bias.core.lens_artifacts import model_slug
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input import decode_token
from llm_bias.core.prompt_input.encoding import token_span
from llm_bias.jspace_intervention.outcome_flip import (
    OUTCOME_FLIP_DEFAULT_PROMPT_COLUMN,
    _enable_deterministic_gpu,
    _materialize_records,
    _prompt_records,
    evidence_item_end_positions,
)
from llm_bias.jspace_intervention.schemas import OutcomeFlipConfig


def _candidate_single_token_id(tokenizer: Any, prompt: str, candidate: str) -> int:
    suffix = continuation_token_ids(tokenizer, prompt, candidate)[1]
    if len(suffix) != 1:
        raise ValueError(f"answer candidate {candidate!r} must be a single token")
    return int(suffix[0])


def _position_classes(
    *,
    tokenizer: Any,
    scoring_prompt: str,
    raw_prompt: str,
    prompt_ids: list[int],
    evidence_span: tuple[int, int],
    decision_prefix: str,
) -> list[str]:
    """Classify every prompt position (item end / span / prefix / other)."""
    sequence_length = len(prompt_ids)
    item_ends = set(
        evidence_item_end_positions(tokenizer, scoring_prompt, raw_prompt)
    )
    span_start, span_end = int(evidence_span[0]), int(evidence_span[1])
    prefix_span = token_span(
        tokenizer,
        scoring_prompt,
        len(scoring_prompt) - len(decision_prefix),
        len(scoring_prompt),
    )
    prefix_positions = (
        set(range(int(prefix_span[0]), int(prefix_span[1]))) if prefix_span else set()
    )
    classes: list[str] = []
    for position in range(sequence_length):
        if position in item_ends:
            classes.append("evidence_item_end")
        elif span_start <= position < span_end:
            classes.append("evidence_span")
        elif position in prefix_positions:
            classes.append("decision_prefix")
        else:
            classes.append("other")
    return classes


@torch.enable_grad()
def attribute_target_logprob(
    model: Any,
    input_tensor: torch.Tensor,
    *,
    target_id: int,
) -> dict[str, Any]:
    """One forward+backward: per-position norms of d log P(target) / d h.

    The embedding output is re-rooted as a leaf (same convention as the
    prompt_analysis attribution), every decoder layer's residual output is
    captured as an intermediate autograd target, and a single backward
    yields the per-position L2-norm score for the embedding and all layers.
    Returns CPU norm vectors only; gradients never leave this function.
    """
    n_layers = int(getattr(model, "n_layers", len(model.layers)))
    if n_layers != len(model.layers):
        raise ValueError("model.n_layers does not match len(model.layers)")
    captured: dict[int, torch.Tensor] = {}
    embed_box: dict[str, torch.Tensor] = {}

    def embed_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        rooted = tensor.detach().requires_grad_(True)
        embed_box["value"] = rooted
        return rooted

    def make_layer_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            captured[layer_id] = tensor
            return output

        return hook

    handles = [model._embed_tokens.register_forward_hook(embed_hook)]
    handles.extend(
        model.layers[layer].register_forward_hook(make_layer_hook(layer))
        for layer in range(n_layers)
    )
    try:
        output = model.forward(input_tensor)
        final_residual = getattr(output, "last_hidden_state", None)
        if not torch.is_tensor(final_residual):
            raise TypeError("model forward must return last_hidden_state")
        log_probs = fp32_next_token_log_probs(model, final_residual[:, -1, :])
        target = log_probs[0, int(target_id)]
        if not torch.isfinite(target):
            raise ValueError(f"target log prob is not finite: {float(target)}")
        roots = [embed_box["value"], *[captured[layer] for layer in range(n_layers)]]
        grads = torch.autograd.grad(target, roots)
    finally:
        for handle in handles:
            handle.remove()
    if "value" not in embed_box:
        raise ValueError("embedding hook did not fire during forward")
    missing = [layer for layer in range(n_layers) if layer not in captured]
    if missing:
        raise ValueError(f"layer hooks did not fire for layers: {missing}")
    if any(grad is None for grad in grads):
        raise ValueError("a root received no gradient; attribution is incomplete")
    return {
        "log_prob": float(target.detach().cpu()),
        "embedding": grads[0].detach().float().cpu().norm(dim=-1)[0],
        "layers": {
            layer: grads[layer + 1].detach().float().cpu().norm(dim=-1)[0]
            for layer in range(n_layers)
        },
    }


def _top_entries(
    *,
    layer_norms: dict[int, torch.Tensor],
    classes: list[str],
    prompt_ids: list[int],
    tokenizer: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    flat = torch.stack([layer_norms[layer] for layer in sorted(layer_norms)], dim=0)
    count = min(top_k, flat.numel())
    values, indices = flat.flatten().topk(count)
    n_positions = flat.shape[1]
    entries: list[dict[str, Any]] = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist(), strict=True), start=1):
        layer, position = divmod(index, n_positions)
        entries.append(
            {
                "rank": rank,
                "layer": layer,
                "position": position,
                "token": decode_token(tokenizer, int(prompt_ids[position])),
                "position_class": classes[position],
                "attribution": value,
            }
        )
    return entries


def _embedding_top_entries(
    *,
    norms: torch.Tensor,
    classes: list[str],
    prompt_ids: list[int],
    tokenizer: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    count = min(top_k, norms.numel())
    values, indices = norms.topk(count)
    return [
        {
            "rank": rank,
            "position": position,
            "token": decode_token(tokenizer, int(prompt_ids[position])),
            "position_class": classes[position],
            "attribution": value,
        }
        for rank, (position, value) in enumerate(
            zip(indices.tolist(), values.tolist(), strict=True), start=1
        )
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("discovery", "calibration", "test"), default="discovery")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--embedding-top-k", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args(argv)

    if args.top_k <= 0 or args.embedding_top_k <= 0:
        raise ValueError("top-k values must be positive")

    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    if config_payload.get("artifact_type") != "outcome_flip_config":
        raise ValueError("config artifact_type must be 'outcome_flip_config'")
    config = OutcomeFlipConfig.from_dict(config_payload)
    if config.model != args.model:
        raise ValueError(
            f"run model {args.model!r} does not match model {config.model!r} frozen in config"
        )
    if config.split_manifest_sha256 != sha256_file(args.split_manifest):
        raise ValueError("split manifest does not match the manifest frozen in config")

    _enable_deterministic_gpu()
    model, tokenizer, fallback_device = load_model(args.model)
    device = getattr(model, "input_device", fallback_device)

    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    assignments = {str(key): str(value) for key, value in split_payload["assignments"].items()}
    records = _prompt_records(
        args.input,
        assignments=assignments,
        split_name=args.split,
        source_sector=config.source_sector,
        prompt_columns={OUTCOME_FLIP_DEFAULT_PROMPT_COLUMN},
        max_records=args.max_records,
    )
    if not records:
        raise ValueError("no prompt records match the requested sector and split")
    materialized = _materialize_records(records, tokenizer, config)
    for record in materialized:
        if len(record["prompt_ids"]) > args.max_seq_len:
            raise ValueError(
                f"record {record['record_id']} has {len(record['prompt_ids'])} tokens "
                f"exceeding --max-seq-len {args.max_seq_len}"
            )

    target_ids = {
        "buy": _candidate_single_token_id(
            tokenizer, materialized[0]["scoring_prompt"], config.positive_candidate
        ),
        "sell": _candidate_single_token_id(
            tokenizer, materialized[0]["scoring_prompt"], config.negative_candidate
        ),
    }

    record_entries: list[dict[str, Any]] = []
    # (target, source, class) -> [sum, max, count] over all records/positions.
    aggregate: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in materialized:
        scoring_prompt = record["scoring_prompt"]
        prompt_ids = record["prompt_ids"]
        classes = _position_classes(
            tokenizer=tokenizer,
            scoring_prompt=scoring_prompt,
            raw_prompt=record["prompt"],
            prompt_ids=prompt_ids,
            evidence_span=record["evidence_span"],
            decision_prefix=config.decision_prefix,
        )
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        per_target: dict[str, Any] = {}
        for name in ("buy", "sell"):
            result = attribute_target_logprob(model, tensor, target_id=target_ids[name])
            per_target[name] = {
                "log_prob": result["log_prob"],
                "top": _top_entries(
                    layer_norms=result["layers"],
                    classes=classes,
                    prompt_ids=prompt_ids,
                    tokenizer=tokenizer,
                    top_k=args.top_k,
                ),
            }
            embed_norms = result["embedding"]
            per_target[name]["embedding_top"] = _embedding_top_entries(
                norms=embed_norms,
                classes=classes,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                top_k=args.embedding_top_k,
            )
            for source_key, norms in (
                ("embedding", embed_norms),
                *[
                    (f"layer_{layer}", value)
                    for layer, value in sorted(result["layers"].items())
                ],
            ):
                bucket = aggregate.setdefault(name, {}).setdefault(source_key, {})
                for position in range(len(prompt_ids)):
                    value = float(norms[position])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"non-finite attribution at {source_key} position {position}"
                        )
                    cls = classes[position]
                    cell = bucket.setdefault(cls, [0.0, 0.0, 0])
                    cell[0] += value
                    cell[1] = max(cell[1], value)
                    cell[2] += 1
        margin = per_target["buy"]["log_prob"] - per_target["sell"]["log_prob"]
        record_entries.append(
            {
                "record_id": record["record_id"],
                "ticker": record["ticker"],
                "sector": record["sector"],
                "prompt_column": record["prompt_column"],
                "sequence_length": len(prompt_ids),
                "clean_margin": margin,
                "clean_decision": "buy" if margin > 0 else ("sell" if margin < 0 else "tie"),
                "targets": per_target,
            }
        )

    def _rank_layers(name: str) -> list[dict[str, Any]]:
        ranking = []
        for source_key in aggregate[name]:
            if not source_key.startswith("layer_"):
                continue
            cell = aggregate[name][source_key]
            total = sum(value[2] for value in cell.values())
            if total == 0:
                continue
            mean = sum(value[0] for value in cell.values()) / total
            maximum = max(value[1] for value in cell.values())
            ranking.append(
                {"layer": int(source_key.split("_")[1]), "mean": mean, "max": maximum}
            )
        ranking.sort(key=lambda item: item["mean"], reverse=True)
        return ranking

    layer_ranking = {
        name: _rank_layers(name) for name in ("buy", "sell")
    }
    aggregate_out: dict[str, Any] = {}
    for name in ("buy", "sell"):
        aggregate_out[name] = {
            source_key: {
                cls: {
                    "mean": cell[0] / cell[2],
                    "max": cell[1],
                    "count": cell[2],
                }
                for cls, cell in sorted(aggregate[name][source_key].items())
            }
            for source_key in sorted(aggregate[name], key=lambda key: key != "embedding")
        }

    result = {
        "artifact_type": "outcome_token_attribution",
        "schema_version": 1,
        "interpretation": (
            "first-order local sensitivity at the clean point; not an attention "
            "map and not a standalone causal claim; raw norms are not calibrated "
            "across layers"
        ),
        "score_definition": (
            "per-position L2 norm of the gradient of the decision-position "
            "log P(target) (Semantic Scope convention; cf. prompt_analysis attribution)"
        ),
        "scoring": "fp32_next_token_logprob_decision_position",
        "model": args.model,
        "model_slug": model_slug(args.model),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "split": args.split,
        "source_sector": config.source_sector,
        "prompt_columns": [OUTCOME_FLIP_DEFAULT_PROMPT_COLUMN],
        "targets": {
            name: {
                "token_id": target_ids[name],
                "token": decode_token(tokenizer, target_ids[name]),
            }
            for name in ("buy", "sell")
        },
        "n_layers": len(model.layers),
        "top_k": args.top_k,
        "embedding_top_k": args.embedding_top_k,
        "record_count": len(record_entries),
        "record_ids": [entry["record_id"] for entry in record_entries],
        "records": record_entries,
        "layer_ranking": layer_ranking,
        "aggregate": aggregate_out,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "outcome_token_attribution.json"
    write_json(output_path, result, overwrite=True)
    write_metadata(
        args.output_dir / "outcome_token_attribution.json.metadata.json",
        {
            "artifact_type": "outcome_token_attribution_metadata",
            "schema_version": 1,
            "created_by": "scripts/jspace_outcome_token_attribution.py",
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "record_count": len(record_entries),
            "top_k": args.top_k,
            "embedding_top_k": args.embedding_top_k,
        },
        overwrite=True,
    )
    print(f"wrote {output_path}")
    for name in ("buy", "sell"):
        top_layers = ", ".join(
            f"L{item['layer']}({item['mean']:.2e})" for item in layer_ranking[name][:5]
        )
        print(f"{name}: top-5 layers by mean norm: {top_layers}")


if __name__ == "__main__":
    main()
