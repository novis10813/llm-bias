"""First-order attribution of the L15 stance readout and the buy/sell margin to
company-description input tokens.

This is a development-only diagnostic for the stance-channel research. For each
company it roots the input embedding with ``requires_grad``, runs one forward
pass that captures the analysis-layer and final residuals (kept in the autograd
graph), forms two differentiable scalars -- the stance readout
(mean analysis-layer instruction-span state dot the eval-stance direction) and
the buy/sell log-prob margin -- and back-propagates each to the rooted
embedding. The per-token attribution is the gradient-by-input saliency (plus the
gradient L2 norm as a secondary). Only compact derived scalars and token text
are persisted; raw embeddings/gradients/activations are never written.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from jspace_viz.hooks import ActivationRecorder

from llm_bias.core.artifacts.io import write_json, write_jsonl
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.forward import record_residuals
from llm_bias.entity_concept_decision.development import (
    _answer_ids,
    _company_records,
    _json_number,
    _material_prompt_records,
    _register_reference,
)
from llm_bias.entity_concept_decision.layer_scan import _fit_stance_direction


def _token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _attribute_company(
    model: Any,
    device: torch.device,
    item: Mapping[str, Any],
    stance_dir: torch.Tensor,
    layer: int,
    last: int,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Root the input embedding and attribute the stance readout and the margin
    to the input tokens (gradient-by-input saliency + gradient L2 norm)."""
    ids, buy_id, sell_id = _answer_ids(model.tokenizer, item["formatted"])
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    input_start, input_end = item["instruction_span"]
    # The company description (the facts to attribute) is everything BEFORE the
    # instruction span; the instruction span itself is the shared "respond buy/sell
    # in JSON" suffix, whose tokens dominate a naive saliency and are not company-specific.
    company_end = int(input_start)
    if company_end <= 0:
        raise ValueError(f"{item['id']}: instruction span starts at {input_start}; no company-description span to attribute")
    tokenizer = model.tokenizer
    stance_gpu = stance_dir.to(device=device, dtype=torch.float32)

    embedding_box: dict[str, torch.Tensor] = {}

    def root_embedding(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        rooted = output.detach().requires_grad_(True)
        embedding_box["value"] = rooted
        return rooted

    handle = model._embed_tokens.register_forward_hook(root_embedding)
    try:
        with torch.enable_grad():
            with ActivationRecorder(model.layers, at=[layer, last]) as recorder:
                model._text_module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                layer_res = recorder.activations[layer]
                last_res = recorder.activations[last]
            block = layer_res[:, input_start:input_end, :]
            state = block.float().mean(dim=1)
            stance_score = (state @ stance_gpu).squeeze()
            log_probs = fp32_next_token_log_probs(model, last_res[:, -1, :])
            margin = log_probs[0, buy_id] - log_probs[0, sell_id]
            embedding = embedding_box["value"]
            grad_stance = torch.autograd.grad(stance_score, embedding, retain_graph=True)[0]
            grad_margin = torch.autograd.grad(margin, embedding)[0]
    finally:
        handle.remove()

    sal_stance = ((grad_stance.float() * embedding.float()).sum(dim=-1))[0, 0:company_end]
    sal_margin = ((grad_margin.float() * embedding.float()).sum(dim=-1))[0, 0:company_end]
    norm_stance = torch.linalg.vector_norm(grad_stance.float(), dim=-1)[0, 0:company_end]
    norm_margin = torch.linalg.vector_norm(grad_margin.float(), dim=-1)[0, 0:company_end]

    span_ids = ids[0:company_end]
    token_texts = [_token_text(tokenizer, int(t)) for t in span_ids]

    def _top(saliency: torch.Tensor) -> list[dict[str, Any]]:
        values = saliency.abs()
        k = min(top_k, values.numel())
        idx = values.topk(k).indices.tolist()
        out = []
        for rank, rel in enumerate(sorted(idx, key=lambda i: -float(values[i]))):
            out.append(
                {
                    "rank": rank,
                    "position": int(rel),
                    "span_offset": int(rel),
                    "token_id": int(span_ids[rel]),
                    "token": token_texts[rel],
                    "saliency": _json_number(float(saliency[int(rel)]), name="saliency"),
                    "grad_norm": _json_number(float((norm_stance if saliency is sal_stance else norm_margin)[int(rel)]), name="grad_norm"),
                }
            )
        return out

    result = {
        "id": item["id"],
        "input_length": len(ids),
        "company_span": [0, int(company_end)],
        "instruction_span": [int(input_start), int(input_end)],
        "stance_score": _json_number(float(stance_score.detach()), name="stance_score"),
        "margin": _json_number(float(margin.detach()), name="margin"),
        "token_texts": token_texts,
        "stance_saliency": [_json_number(float(v), name="s") for v in sal_stance.tolist()],
        "margin_saliency": [_json_number(float(v), name="m") for v in sal_margin.tolist()],
        "stance_grad_norm": [_json_number(float(v), name="n") for v in norm_stance.tolist()],
        "margin_grad_norm": [_json_number(float(v), name="n") for v in norm_margin.tolist()],
        "top_stance_tokens": _top(sal_stance),
        "top_margin_tokens": _top(sal_margin),
    }
    # release references before returning (gradients/graph freed by autograd.grad)
    del grad_stance, grad_margin, embedding, layer_res, last_res, block, state
    return result


def run_attribution(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    materials: Mapping[str, Any],
    instruction_suffix: str,
    answer_prefix: str,
    company_prompts: Any,
    layer: int,
    run_id: str,
    model_name: str,
    artifact_root: str | Path,
    provenance: Mapping[str, Any],
    random_seed: int = 1729,
    min_norm: float = 1e-6,
    top_k: int = 12,
) -> Path:
    """prepare -> forward (material, no-grad, fit stance dir) -> attribute
    (company, grad) -> analyze, then finalize."""
    concept_ids = [str(c) for c in materials.get("concepts", [])]
    if not concept_ids:
        raise ValueError("materials must declare concepts")
    layer = int(layer)
    last = int(model.n_layers) - 1
    if layer < 0 or layer >= int(model.n_layers):
        raise ValueError(f"layer {layer} out of range for {model.n_layers} layers")
    rows = [dict(r) for r in materials["rows"]]
    comparisons = [dict(c) for c in materials["comparisons"]]
    prepared_rows = _material_prompt_records(tokenizer, rows, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix)
    expected_instruction_ids = list(prepared_rows[0]["instruction_ids"])
    prepared_companies = _company_records(tokenizer, company_prompts, instruction_suffix=instruction_suffix, answer_prefix=answer_prefix, expected_instruction_ids=expected_instruction_ids)
    company_ids = [c["id"] for c in prepared_companies]

    run = ArtifactRun.create(model_name, "entity-concept-attribution", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            prepare_dir = run.run_directory / "prepare"
            write_json(prepare_dir / "materials.json", dict(materials), overwrite=False)
            write_jsonl(prepare_dir / "prompts.jsonl", [*prepared_rows, *prepared_companies], overwrite=False)
            metadata = {"purpose": "development", "provenance": dict(provenance), "sources": {k: materials.get(k) for k in ("source_paths", "source_sha256", "concepts") if materials.get(k) is not None}, "seed": random_seed, "layer": layer, "final_layer": last, "min_norm": min_norm, "top_k": top_k}
            write_json(prepare_dir / "metadata.json", metadata, overwrite=False)
            _register_reference(run, provenance["upstream"], role="upstream", stage="prepare")
            _register_reference(run, provenance["lens"], role="lens", stage="prepare")
            stage.count(len(prepared_rows) + len(prepared_companies))

        # Forward over material prompts (no grad) to fit the eval-stance direction.
        material_states: dict[str, torch.Tensor] = {}
        with run.stage("forward") as stage:
            for item in prepared_rows:
                ids, _b, _s = _answer_ids(tokenizer, item["formatted"])
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                captured = record_residuals(model, tensor, [layer, last])
                block = captured[layer][:, item["instruction_span"][0]:item["instruction_span"][1], :]
                if block.ndim != 3 or block.shape[1] <= 0:
                    raise ValueError(f"invalid material capture for {item['id']}")
                material_states[item["id"]] = block.float().mean(dim=1).detach().cpu().reshape(-1)
                print(f"material forward {len(material_states)}/{len(prepared_rows)}: {item['id']}", flush=True)
            stage.count(len(prepared_rows))
        stance_dir, stance_n_pairs = _fit_stance_direction(material_states, comparisons, min_norm)
        if stance_dir is None:
            raise ValueError("could not fit a non-degenerate eval-stance direction from the materials")
        stance_meta = {
            "layer": layer,
            "n_pairs": stance_n_pairs,
            "stance_norm": _json_number(float(stance_dir.norm()), name="stance_norm"),
        }
        write_json(run.run_directory / "forward" / "stance_metadata.json", stance_meta, overwrite=False)
        run.manifest.register_artifact(run.run_directory / "forward" / "stance_metadata.json", artifact_type="attribution_stance_metadata", stage="forward")

        # Attribute each company (grad forward + two backward passes).
        attributions: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []
        with run.stage("attribute") as stage:
            for item in prepared_companies:
                started = time.perf_counter()
                result = _attribute_company(model, torch.device(device), item, stance_dir, layer, last, top_k=top_k)
                result["elapsed_seconds"] = _json_number(time.perf_counter() - started, name="elapsed")
                attributions[item["id"]] = result
                records.append({"id": item["id"], "stance_score": result["stance_score"], "margin": result["margin"], "elapsed_seconds": result["elapsed_seconds"]})
                print(f"attribute {len(records)}/{len(prepared_companies)}: {item['id']} stance={result['stance_score']:.4f} margin={result['margin']:.4f}", flush=True)
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            write_jsonl(run.run_directory / "attribute" / "records.jsonl", records, overwrite=False)
            stage.count(len(records))

        # Per-company attribution details (compact: saliency scalars + top tokens).
        with run.stage("analyze") as stage:
            details_path = run.run_directory / "analyze" / "attribution_details.json"
            write_json(details_path, {cid: attributions[cid] for cid in company_ids}, overwrite=False)
            run.manifest.register_artifact(details_path, artifact_type="attribution_details", stage="analyze", record_count=len(company_ids))

            # Aggregate: stance-vs-margin R^2 (reconfirm the decision proxy), and the
            # per-company correlation between the stance and margin token attributions
            # (do the same tokens drive both?).
            xs = [attributions[c]["stance_score"] for c in company_ids]
            ys = [attributions[c]["margin"] for c in company_ids]
            summary = _summarize(company_ids, attributions, xs, ys, layer=layer, final_layer=last, stance_meta=stance_meta, top_k=top_k)
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary, overwrite=False)
            run.manifest.register_artifact(summary_path, artifact_type="attribution_summary", stage="analyze")
            stage.count(len(company_ids))

        run.finalize(
            required_stages={"prepare", "forward", "attribute", "analyze"},
        )
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _summarize(
    company_ids: list[str],
    attributions: dict[str, dict[str, Any]],
    xs: list[float],
    ys: list[float],
    *,
    layer: int,
    final_layer: int,
    stance_meta: Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    r = _pearson_r(xs, ys)
    per_company: dict[str, Any] = {}
    for cid in company_ids:
        a = attributions[cid]
        sal_s = [float(v) for v in a["stance_saliency"]]
        sal_m = [float(v) for v in a["margin_saliency"]]
        rs = _pearson_r(sal_s, sal_m)
        per_company[cid] = {
            "stance_score": a["stance_score"],
            "margin": a["margin"],
            "stance_vs_margin_saliency_r": _json_number(rs, name="r") if rs is not None else None,
            "top_stance_tokens": a["top_stance_tokens"][:top_k],
            "top_margin_tokens": a["top_margin_tokens"][:top_k],
        }
    return {
        "schema_version": 1,
        "scientific_status": "not_evaluated",
        "purpose": "development",
        "claim_status": "development_only",
        "layer": layer,
        "final_layer": final_layer,
        "stance": dict(stance_meta),
        "n_companies": len(company_ids),
        "company_margins": {cid: attributions[cid]["margin"] for cid in company_ids},
        "stance_score_vs_margin_r2": _json_number(r * r, name="r2") if r is not None else None,
        "stance_score_vs_margin_r": _json_number(r, name="r") if r is not None else None,
        "per_company": per_company,
    }
