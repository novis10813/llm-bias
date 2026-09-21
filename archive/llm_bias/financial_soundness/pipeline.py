"""Independent localization and causal runs, consuming shared workflow mechanics."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.artifacts.io import write_json, write_jsonl
from llm_bias.core.artifacts.lifecycle import run_context
from llm_bias.core.artifacts.provenance import local_model_identity, object_sha256, source_identity
from llm_bias.core.inference.continuations import score_encoded_candidates
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.mlp import mlp_coordinates
from llm_bias.core.model import load_model
from .analysis import group_key, rank_candidates, summarize_behavior, summarize_effects
from .prompts import VERSION, prepare_encoded, validate_prompts


def _write(run, relative: str, value: Any) -> Path:
    path = run.run_directory / relative
    if path.suffix == ".jsonl":
        write_jsonl(path, value)
    else:
        write_json(path, value)
    run.manifest.register_artifact(path=path, artifact_type=path.stem, stage=relative.split("/")[0])
    run.save()
    return path


def _score(model, row, device) -> dict:
    pos, neg = score_encoded_candidates(model, row["prompt_ids"], [row["positive_ids"], row["negative_ids"]], device)
    margin = pos - neg
    return {"positive_log_probability": pos, "negative_log_probability": neg, "margin": margin,
            "correct": None if row["expected"] is None else int(row["expected"] * margin > 0)}


def _capture(model, row, layers, device):
    with mlp_coordinates(model, layers, row["position"]) as captured:
        record_residuals(model, torch.tensor([row["prompt_ids"]], device=device), [len(model.layers) - 1])
    if set(captured) != set(layers):
        raise RuntimeError("MLP hook did not fire on all selected layers")
    return captured


def _protocol(model_path, tokenizer, data, *, layers, top_k):
    return {"schema_version": 1, "protocol_version": VERSION, "status": "exploratory",
            "formal_authorized": False, "data_sha256": object_sha256(data),
            "model_identity": local_model_identity(model_path, tokenizer),
            "source_identity": source_identity("llm_bias/financial_soundness", "llm_bias/core"),
            "layers": layers, "top_k": top_k, "seed": 0, "scope": "prompt_final_position",
            "score": "exact_continuation_sum_fp32", "max_tokens": 512}


def run_localization(prompts: str | Path, model_path: str, run_id: str, *,
                     artifact_root: str | Path = "artifacts", layers: list[int] | None = None,
                     top_k: int = 3, loaded: tuple | None = None) -> Path:
    data = json.loads(Path(prompts).read_text())
    validate_prompts(data)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model, tokenizer, device = loaded if loaded is not None else load_model(model_path)
    getattr(model, "_hf_model", model).eval()
    layers = list(range(len(model.layers))) if layers is None else layers
    if not layers or len(layers) != len(set(layers)) or any(l < 0 or l >= len(model.layers) for l in layers):
        raise ValueError("invalid selected layers")
    encoded = prepare_encoded(data, tokenizer)
    protocol = _protocol(model_path, tokenizer, data, layers=layers, top_k=top_k)
    with run_context(model_path, "financial-soundness-localization", run_id, artifact_root=artifact_root) as run:
        with run.stage("prepare") as stage:
            _write(run, "prepare/prompts.json", encoded)
            _write(run, "prepare/protocol.json", protocol)
            stage.count(len(encoded["pairs"]))
        responses = defaultdict(list)
        behavior = []
        with run.stage("forward") as stage:
            for pair in encoded["pairs"]:
                values = {}
                for name in ("a", "b"):
                    row = pair[name]
                    values[name] = _capture(model, row, layers, device)
                    behavior.append({k: pair[k] for k in ("pair_id", "family", "concept", "answer_mode", "split", "group_id")} |
                                    {"member": name} | _score(model, row, device))
                for layer in layers:
                    responses[(*group_key(pair), pair["split"], layer)].append((values["a"][layer], values["b"][layer]))
            _write(run, "forward/behavior.jsonl", behavior)
            stage.count(len(behavior))
        with run.stage("analyze") as stage:
            candidates = rank_candidates(responses, top_k)
            del responses
            _write(run, "analyze/candidates.json", {"schema_version": 1, "protocol_sha256": object_sha256(protocol), "candidates": candidates})
            _write(run, "analyze/summary.json", {"status": "exploratory", "certified": False,
                                               "candidate_count": len(candidates), "behavior": summarize_behavior(behavior)})
            stage.count(len(candidates))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory


def _load_source(source_run: str | Path) -> tuple[dict, dict, list[dict], str]:
    source = Path(source_run).resolve()
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("dataset") != "financial-soundness-localization":
        raise ValueError("source must be a completed financial localization run")
    refs = {ref["path"]: ref for ref in manifest.get("output_refs", [])}
    required = {"prepare/prompts.json", "prepare/protocol.json", "analyze/candidates.json", "analyze/summary.json", "forward/behavior.jsonl"}
    if not required <= set(refs):
        raise ValueError("missing required source output references")
    for name, ref in refs.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or file_sha256(path) != ref["sha256"]:
            raise ValueError("source artifact hash mismatch")
    data = json.loads((source / "prepare/prompts.json").read_text())
    protocol = json.loads((source / "prepare/protocol.json").read_text())
    candidates = json.loads((source / "analyze/candidates.json").read_text())
    if candidates.get("protocol_sha256") != object_sha256(protocol) or not candidates.get("candidates"):
        raise ValueError("empty candidates or protocol hash mismatch")
    validate_prompts(data)
    return data, protocol, candidates["candidates"], file_sha256(manifest_path)


def run_causal_validation(source_run: str | Path, model_path: str, run_id: str, *,
                          artifact_root: str | Path = "artifacts", loaded: tuple | None = None) -> Path:
    data, source_protocol, candidates, manifest_hash = _load_source(source_run)
    model, tokenizer, device = loaded if loaded is not None else load_model(model_path)
    getattr(model, "_hf_model", model).eval()
    if local_model_identity(model_path, tokenizer) != source_protocol["model_identity"]:
        raise ValueError("source model/tokenizer identity mismatch")
    # Re-tokenize rather than trusting stored positions or continuation suffixes.
    encoded = prepare_encoded(data, tokenizer)
    if encoded != data:
        raise ValueError("source encoding differs from current tokenizer")
    coordinates = sorted({(c["layer"], c["neuron"]) for c in candidates})
    if any(not isinstance(l, int) or not isinstance(n, int) or l < 0 or l >= len(model.layers) or n < 0 for l, n in coordinates):
        raise ValueError("invalid source coordinates")
    layers = sorted({l for l, _ in coordinates})
    pairs = [p for p in encoded["pairs"] if p["split"] != "discovery" and p["family"] in {"evidence", "comparison"}]
    if not pairs:
        raise ValueError("no causal evaluation pairs")
    protocol = {"schema_version": 1, "protocol_version": VERSION, "status": "exploratory", "formal_authorized": False,
                "source_run": str(Path(source_run).resolve()), "source_manifest_sha256": manifest_hash,
                "source_protocol_sha256": object_sha256(source_protocol), "model_identity": source_protocol["model_identity"],
                "source_identity": source_identity("llm_bias/financial_soundness", "llm_bias/core"),
                "candidates": candidates, "scales": [0.5, 0.0], "seed": 0, "scope": "prompt_final_position"}
    with run_context(model_path, "financial-soundness-causal-validation", run_id, artifact_root=artifact_root) as run:
        with run.stage("prepare") as stage:
            _write(run, "prepare/prompts.json", encoded)
            _write(run, "prepare/protocol.json", protocol)
            stage.count(len(pairs))
        effects = []
        controls = {}
        with run.stage("forward") as stage:
            for pair in pairs:
                captured = {name: _capture(model, pair[name], layers, device) for name in ("a", "b")}
                clean = {name: _score(model, pair[name], device)["margin"] for name in ("a", "b")}
                for layer, neuron in coordinates:
                    width = captured["a"][layer].numel()
                    if neuron >= width:
                        raise ValueError("candidate neuron out of range")
                    if layer not in controls:
                        eligible = [n for n in range(width) if (layer, n) not in coordinates]
                        if not eligible:
                            raise ValueError("no noncandidate random control in layer")
                        controls[layer] = random.Random(layer).choice(eligible)
                    control = controls[layer]
                    for name, other in (("a", "b"), ("b", "a")):
                        row = pair[name]
                        target = float(captured[name][layer][neuron])
                        donor = float(captured[other][layer][neuron])
                        arms = [("clean", None, neuron, 1.0, None),
                                ("scale", 0.5, neuron, 0.5, None), ("scale", 0.0, neuron, 0.0, None),
                                ("random", 0.5, control, 0.5, None), ("random", 0.0, control, 0.0, None),
                                ("restore", 0.0, neuron, 0.0, target), ("donor", None, neuron, 1.0, donor)]
                        for condition, scale, channel, factor, replacement in arms:
                            if condition == "clean":
                                margin = clean[name]
                            else:
                                with mlp_coordinates(model, [layer], row["position"], edits={layer: {channel: (factor, replacement)}}):
                                    margin = _score(model, row, device)["margin"]
                            tolerance = max(1e-5, 1e-4 * abs(clean[name]))
                            if condition == "restore" and abs(margin - clean[name]) > tolerance:
                                raise ValueError("restore margin exceeds engineering tolerance")
                            denominator = clean[other] - clean[name]
                            transfer, reason = None, "not_donor"
                            if condition == "donor":
                                if abs(denominator) <= tolerance:
                                    reason = "near_zero_clean_contrast"
                                else:
                                    transfer, reason = (margin - clean[name]) / denominator, None
                            effects.append({k: pair[k] for k in ("pair_id", "group_id", "split", "family", "answer_mode")} |
                                           {"schema_version": 1, "member": name, "layer": layer, "neuron": neuron,
                                            "control_neuron": control, "condition": condition, "scale": scale,
                                            "clean_margin": clean[name], "margin": margin,
                                            "correct_margin_delta": row["expected"] * (margin - clean[name]),
                                            "transfer_fraction": transfer, "transfer_reason": reason})
            _write(run, "forward/effects.jsonl", effects)
            stage.count(len(effects))
        with run.stage("analyze") as stage:
            summary = summarize_effects(effects)
            _write(run, "analyze/summary.json", {"status": "exploratory", "certified": False,
                                               "restore_is_engineering_control": True, "groups": summary})
            stage.count(len(summary))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
