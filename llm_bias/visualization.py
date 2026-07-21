"""Interactive jspace-viz adapter for entity counterfactuals."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jspace_viz.lens import JacobianLens
from jspace_viz.model import WrappedModel, load_model as load_jspace_model

from llm_bias.data import Pair, load_saved_pairs
from llm_bias.interventions import patched_residuals, record_residuals

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PAIRS = "artifacts/entity_control/pairs.jsonl"
DEFAULT_LENS = "artifacts/entity_control/jacobian_lens.pt"


@dataclass
class VisualizationState:
    model: WrappedModel
    lens: JacobianLens
    pairs: dict[str, Pair]
    model_id: str
    lens_path: str


class CounterfactualRequest(BaseModel):
    pair_id: str
    patch_layer: int = Field(ge=0)
    patch_position: int | None = Field(default=None, ge=0)
    mode: str = "jlens"
    top_k: int = Field(default=8, ge=1, le=32)


def _excess_kurtosis(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(-1, keepdim=True)
    centered = values - mean
    variance = centered.pow(2).mean(-1)
    return centered.pow(4).mean(-1) / variance.pow(2).clamp_min(1e-12) - 3.0


def _rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def _record_dependency_versions() -> None:
    """Record ignored checkout revisions so local figures remain attributable."""
    root = Path(__file__).resolve().parents[1]
    versions: dict[str, str] = {}
    for name in ("jacobian-lens", "jspace-viz"):
        checkout = root / "third_party" / name
        try:
            versions[name] = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            versions[name] = "unknown"
    destination = root / "artifacts" / "visualization" / "dependencies.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def _grid_from_residuals(
    model: WrappedModel,
    lens: JacobianLens,
    input_ids: torch.Tensor,
    activations: dict[int, torch.Tensor],
    *,
    mode: str,
    top_k: int,
    pinned_ids: list[int],
) -> dict[str, Any]:
    """Convert recorded residuals into jspace-viz's JSON grid schema."""
    final_layer = model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final_layer})
    ids = input_ids[0].tolist()
    seq_len = len(ids)
    grid: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    seen = set(ids) | set(pinned_ids)
    pinned = torch.tensor(pinned_ids, dtype=torch.long, device=model.device) if pinned_ids else None
    next_ids = input_ids[0, 1:]
    valid_start = 0 if seq_len <= 6 else 4

    for layer in layers:
        residual = activations[layer][0].float()
        if mode == "jlens" and layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits = model.unembed(residual).float()
        probabilities = logits.softmax(-1)
        top = probabilities.topk(top_k, dim=-1)
        top_ids = top.indices
        seen.update(top_ids.flatten().tolist())
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
        kurtosis = _excess_kurtosis(logits)
        row: dict[str, Any] = {
            "layer": layer,
            "is_output": layer == final_layer,
            "top_ids": top_ids.cpu().tolist(),
            "top_probs": [[round(value, 5) for value in values] for values in top.values.cpu().tolist()],
            "entropy": [round(value, 3) for value in entropy.cpu().tolist()],
            "kurtosis": [round(value, 2) for value in kurtosis.cpu().tolist()],
        }
        if pinned is not None:
            pinned_logits = logits[:, pinned]
            ranks = (logits.unsqueeze(-1) > pinned_logits.unsqueeze(1)).sum(1)
            row["pinned_ranks"] = ranks.cpu().tolist()
        grid.append(row)
        top1 = top_ids[:, 0]
        valid = min(max(seq_len - 1, 0), len(next_ids))
        metrics.append(
            {
                "layer": layer,
                "next_token_acc": round((top1[:valid] == next_ids[:valid]).float().mean().item(), 4)
                if valid else 0.0,
                "mean_kurtosis": round(kurtosis[valid_start:max(seq_len - 1, valid_start)].mean().item(), 2)
                if seq_len > valid_start + 1 else 0.0,
                "top1_autocorr": round((top1[valid_start + 1 : seq_len] == top1[valid_start : seq_len - 1]).float().mean().item(), 4)
                if seq_len > valid_start + 1 else 0.0,
            }
        )

    decode = model.tokenizer.decode
    vocab = {int(token_id): decode([int(token_id)]) for token_id in seen}
    return {
        "continuation": None,
        "prompt_len": seq_len,
        "mode": mode,
        "seq_len": seq_len,
        "layers": layers,
        "context_ids": ids,
        "vocab": vocab,
        "grid": grid,
        "layer_metrics": metrics,
        "pinned_ids": pinned_ids,
    }


def _top1_comparison(*grids: dict[str, Any]) -> list[dict[str, Any]]:
    source, target, patched = grids
    comparison: list[dict[str, Any]] = []
    for source_row, target_row, patched_row in zip(
        source["grid"], target["grid"], patched["grid"], strict=True
    ):
        for position in range(source["seq_len"]):
            comparison.append(
                {
                    "layer": source_row["layer"],
                    "position": position,
                    "source_top1": source_row["top_ids"][position][0],
                    "target_top1": target_row["top_ids"][position][0],
                    "patched_top1": patched_row["top_ids"][position][0],
                    "source_top1_prob": source_row["top_probs"][position][0],
                    "target_top1_prob": target_row["top_probs"][position][0],
                    "patched_top1_prob": patched_row["top_probs"][position][0],
                }
            )
    return comparison


def read_counterfactual(
    state: VisualizationState,
    pair: Pair,
    *,
    patch_layer: int,
    patch_position: int | None,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    if mode not in {"jlens", "logit"}:
        raise ValueError("mode must be 'jlens' or 'logit'")
    if patch_layer >= state.model.n_layers - 1:
        raise ValueError("patch_layer must leave at least one transformer layer above the patch")

    model, lens = state.model, state.lens
    source_ids = model.encode(pair.source_prompt, max_length=256)
    target_ids = model.encode(pair.target_prompt, max_length=256)
    if source_ids.shape != target_ids.shape:
        raise ValueError("source and target prompts are not token-aligned")
    position = pair.source_entity_start if patch_position is None else patch_position
    if position >= source_ids.shape[1]:
        raise ValueError(f"patch_position {position} is outside the prompt")

    layers = sorted(set(lens.source_layers) | {model.n_layers - 1})
    record_layers = sorted(set(layers) | {patch_layer})
    source_acts = record_residuals(model, source_ids, record_layers)
    target_acts = record_residuals(model, target_ids, record_layers)
    patched_acts = patched_residuals(
        model,
        source_ids,
        layers=record_layers,
        patch_layer=patch_layer,
        position=position,
        replacement=target_acts[patch_layer][0, pair.target_entity_start, :],
    )
    pinned_ids = [
        pair.source_entity_token,
        pair.target_entity_token,
        pair.answer_source_token,
        pair.answer_target_token,
    ]
    source = _grid_from_residuals(model, lens, source_ids, source_acts, mode=mode, top_k=top_k, pinned_ids=pinned_ids)
    target = _grid_from_residuals(model, lens, target_ids, target_acts, mode=mode, top_k=top_k, pinned_ids=pinned_ids)
    patched = _grid_from_residuals(model, lens, source_ids, patched_acts, mode=mode, top_k=top_k, pinned_ids=pinned_ids)

    answer_layer = model.n_layers - 1
    source_logits = model.unembed(source_acts[answer_layer][0, -1].float()).float().cpu()
    target_logits = model.unembed(target_acts[answer_layer][0, -1].float()).float().cpu()
    patched_logits = model.unembed(patched_acts[answer_layer][0, -1].float()).float().cpu()
    source_margin = float(source_logits[pair.answer_target_token] - source_logits[pair.answer_source_token])
    target_margin = float(target_logits[pair.answer_target_token] - target_logits[pair.answer_source_token])
    patched_margin = float(patched_logits[pair.answer_target_token] - patched_logits[pair.answer_source_token])
    denominator = target_margin - source_margin
    transfer = (patched_margin - source_margin) / denominator if abs(denominator) > 1e-6 else None
    return {
        "pair": pair.to_dict(),
        "patch": {
            "layer": patch_layer,
            "position": position,
            "replacement_position": pair.target_entity_start,
            "replacement": "target entity residual",
        },
        "source": source,
        "target": target,
        "patched": patched,
        "comparison": _top1_comparison(source, target, patched),
        "metrics": {
            "source_margin": source_margin,
            "target_margin": target_margin,
            "patched_margin": patched_margin,
            "transfer": transfer,
            "source_answer_rank": _rank(source_logits, pair.answer_source_token),
            "target_answer_rank": _rank(target_logits, pair.answer_target_token),
            "patched_target_answer_rank": _rank(patched_logits, pair.answer_target_token),
        },
    }


def build_app(
    model_name: str,
    lens_path: str,
    pairs_path: str = DEFAULT_PAIRS,
) -> FastAPI:
    """Build a local app while keeping the model resident in memory."""
    _record_dependency_versions()
    model = load_jspace_model(model_name, dtype="auto")
    lens = JacobianLens.load(lens_path)
    if lens.d_model != model.d_model:
        raise ValueError(f"lens d_model={lens.d_model} does not match model d_model={model.d_model}")
    pairs = {pair.pair_id: pair for pair in load_saved_pairs(pairs_path)}
    state = VisualizationState(model, lens, pairs, model_name, lens_path)
    app = FastAPI(title="Entity Bias · J-space")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "model_id": state.model_id,
            "device": str(state.model.device),
            "n_layers": state.model.n_layers,
            "d_model": state.model.d_model,
            "fitted_layers": state.lens.source_layers,
            "lens_n_prompts": state.lens.n_prompts,
            "lens_source": state.lens_path,
            "pairs": len(state.pairs),
        }

    @app.get("/api/pairs")
    def pairs() -> list[dict[str, Any]]:
        return [
            {
                "pair_id": pair.pair_id,
                "category": pair.category,
                "function": pair.function,
                "source_entity": pair.source_entity,
                "target_entity": pair.target_entity,
                "source_prompt": pair.source_prompt,
                "target_prompt": pair.target_prompt,
            }
            for pair in state.pairs.values()
        ]

    @app.post("/api/counterfactual")
    def counterfactual(request: CounterfactualRequest) -> dict[str, Any]:
        pair = state.pairs.get(request.pair_id)
        if pair is None:
            raise HTTPException(404, f"unknown pair_id: {request.pair_id}")
        try:
            return read_counterfactual(
                state,
                pair,
                patch_layer=request.patch_layer,
                patch_position=request.patch_position,
                mode=request.mode,
                top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("counterfactual read failed")
            raise HTTPException(500, str(exc)) from exc

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "counterfactual.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/llama-3.2-1b-instruct")
    parser.add_argument("--lens", default=DEFAULT_LENS)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = build_app(args.model, args.lens, args.pairs)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
