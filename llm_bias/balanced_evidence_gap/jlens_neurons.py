"""J-lens structural readout of the Phase 3 neuron coordinates.

Post-line-closing diagnostic (docs/balanced-evidence-gap/details/diagnostic-jlens-neurons.md,
frozen 2026-09-10). For each (layer, neuron) coordinate, the injected
residual direction (the down-projection row, same channel index as
``mlp_addition``) is transported to the final-layer basis with the
canonical Jacobian lens and decoded with the model's unembed path.

Descriptive, first-order, non-causal: the readout is the isolated
direction's vocabulary signature, not a prompt-conditioned state and not
a causal claim. No forwards, no prompts, no raw vector persistence.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import write_json, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.core.lens_artifacts import canonical_lens_path, load_lens_metadata
from llm_bias.core.lens_loader import load_validated_lens

ENTITY_CANDIDATES = (
    ("L19_n6334", 19, 6334),
    ("L20_n6520", 20, 6520),
    ("L26_n2394", 26, 2394),
)
DIAL = ("L15_n8490", 15, 8490)
CONTROL_LAYERS = (15, 19, 20, 26)
N_CONTROLS = 10
CONTROL_SEED_BASE = 42
MOE_WIDTH = 9216
SCORING_PREFIX = '{"decision": "'
TOP_K = 10


def control_neurons_for_layer(layer: int, n: int = N_CONTROLS) -> list[int]:
    return random.Random(CONTROL_SEED_BASE + layer).sample(range(MOE_WIDTH), n)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _decode(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return f"<id:{int(token_id)}>"


def readout_one(
    model: Any,
    lens: Any,
    tokenizer: Any,
    layer: int,
    neuron: int,
    *,
    buy_id: int,
    sell_id: int,
) -> dict[str, Any]:
    """Transport one neuron's down-projection row and decode it.

    Returns compact scalars plus token ids only (no vectors).
    """
    # down_proj.weight is [d_model, intermediate]: neuron n's per-unit
    # activation direction in residual space is COLUMN n.
    weight = dense_down_projection(model.layers[layer]).weight
    if neuron >= weight.shape[1]:
        raise ValueError(f"neuron {neuron} out of range for layer {layer}")
    w = weight[:, neuron].detach().float().cpu()
    jacobian = lens.lens.jacobians[layer].detach().float().cpu()
    transported = w @ jacobian.T  # [d_model], final-layer basis
    norm = float(transported.norm())
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"degenerate transported direction for L{layer}/n{neuron}")

    device = next(model._lm_head.parameters()).device
    dtype = model._lm_head.weight.dtype
    with torch.no_grad():
        logits = model.unembed(
            transported.reshape(1, -1).to(dtype=dtype, device=device)
        ).float().cpu()[0]
    if not torch.isfinite(logits).all():
        raise ValueError(f"non-finite logits for L{layer}/n{neuron}")

    margin = float(logits[buy_id] - logits[sell_id])
    probs = torch.softmax(logits, dim=0)
    top_probs, top_ids = torch.topk(probs, TOP_K)
    top10 = [
        {"id": int(tid), "text": _decode(tokenizer, int(tid)), "prob": float(p)}
        for tid, p in zip(top_ids.tolist(), top_probs.tolist())
    ]
    return {
        "layer": layer,
        "neuron": neuron,
        "transported_norm": norm,
        "margin_buy_sell": margin,
        "prob_buy": float(probs[buy_id]),
        "prob_sell": float(probs[sell_id]),
        "top10": top10,
    }


def pairwise_cosine(rows: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = list(rows)
    out: dict[str, Any] = {}
    for a in names:
        out[a] = {}
        for b in names:
            va = rows[a] / rows[a].norm()
            vb = rows[b] / rows[b].norm()
            out[a][b] = float(va @ vb)
    return out


def transported_rows(
    model: Any,
    lens: Any,
    coordinates: list[tuple[str, int, int]],
) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    for name, layer, neuron in coordinates:
        weight = dense_down_projection(model.layers[layer]).weight
        w = weight[:, neuron].detach().float().cpu()
        jacobian = lens.lens.jacobians[layer].detach().float().cpu()
        rows[name] = w @ jacobian.T
    return rows


def run_jlens_neurons_diagnostic(
    *,
    model_name: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
) -> Path:
    from llm_bias.core.model import load_model

    model, tokenizer, _device = load_model(model_name)
    lens = load_validated_lens(model=model, model_name=model_name)

    # buy/sell ids: single-token continuations of the 2A scoring prefix
    from llm_bias.core.prompt_input.encoding import continuation_token_ids

    buy = continuation_token_ids(tokenizer, SCORING_PREFIX, "buy")
    sell = continuation_token_ids(tokenizer, SCORING_PREFIX, "sell")
    if len(buy) != 1 or len(sell) != 1 or buy == sell:
        raise ValueError("buy/sell must be distinct single-token continuations")
    buy_id, sell_id = int(buy[0]), int(sell[0])

    primary = [(name, layer, neuron) for name, layer, neuron in [*ENTITY_CANDIDATES, DIAL]]
    controls = [
        (f"L{layer}_ctl{i:02d}", layer, n)
        for layer in CONTROL_LAYERS
        for i, n in enumerate(control_neurons_for_layer(layer))
    ]
    coordinates = [(n, l, m, "entity_2c_candidate" if (l, m) in {(19, 6334), (20, 6520), (26, 2394)} else ("dial_positive_control" if (l, m) == (15, 8490) else "matched_control")) for n, l, m in primary + controls]

    run = ArtifactRun.create(model_name, "balanced-evidence-gap-phase3", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            metadata = load_lens_metadata(canonical_lens_path(model_name, artifact_root=artifact_root))
            provenance = {
                "schema_version": 1,
                "artifact_type": "balanced_evidence_gap_jlens_neurons_provenance",
                "diagnostic_only": True,
                "raw_runtime_payloads": False,
                "no_forwards": True,
                "protocol": "docs/balanced-evidence-gap/details/diagnostic-jlens-neurons.md (frozen 2026-09-10)",
                "lens": {
                    "path": str(lens.path),
                    "binary_sha256": _sha256(lens.path),
                    "binary_sha256_matches_metadata": metadata is not None
                    and metadata.get("binary_sha256") == _sha256(lens.path),
                    "metadata_sha256": metadata.get("metadata_sha256") if metadata else None,
                    "source_layers": [int(x) for x in lens.lens.source_layers],
                    "n_prompts": lens.lens.n_prompts,
                    "d_model": int(lens.lens.d_model),
                    "calibration_source": (metadata or {}).get("calibration_source"),
                },
                "model": {
                    "name": model_name,
                    "diagnostics": model.model_diagnostics.as_dict(),
                },
                "scoring_prefix": SCORING_PREFIX,
                "buy_token": {"id": buy_id, "text": _decode(tokenizer, buy_id)},
                "sell_token": {"id": sell_id, "text": _decode(tokenizer, sell_id)},
                "coordinates": [
                    {"name": n, "layer": l, "neuron": m, "role": role}
                    for n, l, m, role in coordinates
                ],
                "control_rule": "random.Random(42+layer).sample(range(9216), 10)",
            }
            prov_path = run.run_directory / "prepare" / "provenance.json"
            write_json(prov_path, provenance)
            run.manifest.register_artifact(
                prov_path,
                artifact_type="balanced_evidence_gap_jlens_neurons_provenance",
                stage="prepare", role="output",
            )
            write_metadata(
                run.run_directory / "prepare" / "metadata.json",
                {"artifact_type": "balanced_evidence_gap_jlens_neurons_prepare",
                 "n_coordinates": len(coordinates)},
            )
            run.manifest.register_artifact(
                run.run_directory / "prepare" / "metadata.json",
                artifact_type="balanced_evidence_gap_jlens_neurons_prepare_metadata",
                stage="prepare", role="output",
            )
            stage.count(len(coordinates))

        with run.stage("analyze") as stage:
            readouts = {}
            for name, layer, neuron, role in coordinates:
                readouts[name] = readout_one(
                    model, lens, tokenizer, layer, neuron,
                    buy_id=buy_id, sell_id=sell_id,
                )
                readouts[name]["role"] = role

            rows = transported_rows(model, lens, [p for p in primary])
            cos_primary = pairwise_cosine(rows)

            entity_ctl_max: dict[str, float] = {}
            for name, layer, neuron in ENTITY_CANDIDATES:
                ent = rows[name] / rows[name].norm()
                best = -1.0
                for n in control_neurons_for_layer(layer):
                    w = dense_down_projection(model.layers[layer]).weight[:, n].detach().float().cpu()
                    t = w @ lens.lens.jacobians[layer].detach().float().cpu().T
                    best = max(best, float(ent @ (t / t.norm())))
                entity_ctl_max[name] = best

            summary = {
                "artifact_type": "balanced_evidence_gap_jlens_neurons_summary",
                "descriptive_non_causal": True,
                "buy_token": {"id": buy_id, "text": _decode(tokenizer, buy_id)},
                "sell_token": {"id": sell_id, "text": _decode(tokenizer, sell_id)},
                "readouts": readouts,
                "cosine_primary": cos_primary,
                "entity_vs_control_max_cosine": entity_ctl_max,
            }
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary)
            run.manifest.register_artifact(
                summary_path,
                artifact_type="balanced_evidence_gap_jlens_neurons",
                stage="analyze", role="output",
            )
            write_metadata(
                run.run_directory / "analyze" / "metadata.json",
                {"artifact_type": "balanced_evidence_gap_jlens_neurons_analyze",
                 "n_coordinates": len(coordinates)},
            )
            run.manifest.register_artifact(
                run.run_directory / "analyze" / "metadata.json",
                artifact_type="balanced_evidence_gap_jlens_neurons_analyze_metadata",
                stage="analyze", role="output",
            )
            stage.count(1)

        run.finalize(required_stages={"prepare", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory
