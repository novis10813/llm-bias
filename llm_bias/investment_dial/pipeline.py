"""Local V1: separate screening, A/B calibration, and untouched-company tests."""
import json
import math
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import transformers

from llm_bias.core.artifact_paths import file_sha256
from llm_bias.core.artifacts.lifecycle import run_context
from llm_bias.core.artifacts.provenance import local_model_identity, object_sha256, source_identity
from llm_bias.core.artifacts.registered import write_registered as write, verified_run
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.coordinate_screen import coordinate_derivatives, coordinate_finite_difference, frozen_eval
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens, finish_reason
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.core.inference.mlp_addition import mlp_addition
from llm_bias.core.model import load_model
from .analysis import parse_response, summary, inverse_curve, feasible
from .prompts import VERSION, PREFIX, build_trials, encode_trials


REQUIRED = {"prepare/protocol.json", "prepare/trials.json", "analyze/result.json"}


def _load(model_path, loaded):
    return loaded if loaded is not None else load_model(model_path)


def _identity(model_path, tokenizer):
    return local_model_identity(model_path, tokenizer)


def _runtime(model, tokenizer, device):
    raw = getattr(model, "_hf_model", model)
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "device": str(device), "dtype": str(next(raw.parameters()).dtype),
            "chat_template_sha256": object_sha256(tokenizer.chat_template)}


def _source():
    return source_identity("llm_bias/investment_dial", "llm_bias/core")


def _parent(bundle, model_path, tokenizer, model, device):
    data, digest = bundle
    protocol = data["prepare/protocol.json"]
    result = data["analyze/result.json"]
    if protocol.get("protocol_version") != VERSION:
        raise ValueError("source protocol version mismatch")
    if protocol["model_identity"] != _identity(model_path, tokenizer):
        raise ValueError("source model/tokenizer mismatch")
    current_runtime = _runtime(model, tokenizer, device)
    if any(protocol["runtime"][key] != current_runtime[key]
           for key in ("torch", "transformers", "dtype", "chat_template_sha256")):
        raise ValueError("source inference runtime mismatch")
    if result["protocol_sha256"] != object_sha256(protocol):
        raise ValueError("result protocol mismatch")
    rows = data["prepare/trials.json"]
    if encode_trials(rows, tokenizer) != rows:
        raise ValueError("source tokenization mismatch")
    return protocol, rows, result, digest


def _generate(model, tokenizer, device, row, max_new_tokens):
    adapter = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    ids = row["prompt_ids"]
    sequence = generate_tokens(adapter, torch.tensor([ids], device=device), GenerationConfig(
        max_new_tokens=max_new_tokens, pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)))
    generated = sequence[0, len(ids):].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {k: row[k] for k in ("id", "ticker", "split", "positive_count", "reverse_options")} | {
        "text": text, "generated_ids": generated,
        "finish_reason": finish_reason(generated, eos_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens),
    } | parse_response(text)


def _decisions(model, tokenizer, device, rows, coordinate, delta, budget):
    context = mlp_addition(model, coordinate[0], coordinate[1], delta) if coordinate is not None else nullcontext()
    with context:
        return [_generate(model, tokenizer, device, row, budget) |
                {"layer": coordinate[0] if coordinate else None, "neuron": coordinate[1] if coordinate else None,
                 "delta": delta} for row in rows]


def _tagged(rows, **tags):
    return [row | tags for row in rows]


def run_check(input_path, model_path, run_id, *, artifact_root="artifacts", max_new_tokens=128,
              cpu_bf16=False, loaded=None):
    """One-prompt engineering check; never an investment-bias result."""
    data = json.loads(Path(input_path).read_text())
    trials = build_trials(data, repeats=1)
    trial = next(row for row in trials if row["split"] == "screen")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if cpu_bf16 and torch.cuda.is_available():
        raise ValueError("cpu-bf16 requires CUDA_VISIBLE_DEVICES='' before launch")
    model, tokenizer, device = (loaded if loaded is not None else
        load_model(model_path, dtype=torch.bfloat16 if cpu_bf16 else None))
    row = encode_trials([trial], tokenizer)[0]
    protocol = {"schema_version": 1, "protocol_version": VERSION,
                "purpose": "engineering_only_not_baseline_reproduction", "input_source": data["source"],
                "input_sha256": file_sha256(input_path), "model_identity": _identity(model_path, tokenizer),
                "source_identity": _source(), "runtime": _runtime(model, tokenizer, device),
                "max_new_tokens": max_new_tokens, "epsilons": [.01, .1],
                "thinking": False, "decision_prefix": PREFIX}
    with run_context(model_path, "investment-dial-check", run_id, artifact_root=artifact_root) as run, frozen_eval(model):
        with run.stage("prepare") as stage:
            write(run, "prepare/protocol.json", protocol)
            write(run, "prepare/trials.json", [row])
            stage.count(1)
        with run.stage("forward") as stage:
            margin, vectors = coordinate_derivatives(model, row["decision_ids"], *row["answer_ids"], device,
                                                       list(range(len(model.layers))))
            candidates = []
            for layer, vector in vectors.items():
                neuron = int(vector.abs().argmax())
                candidates.append((abs(float(vector[neuron])), layer, neuron, float(vector[neuron])))
            _, layer, neuron, derivative = max(candidates)
            del vectors
            numeric = [coordinate_finite_difference(model, row["decision_ids"], *row["answer_ids"], device,
                                                   layer, neuron, epsilon) for epsilon in protocol["epsilons"]]
            baseline = _decisions(model, tokenizer, device, [row], None, 0., max_new_tokens)
            zero = _decisions(model, tokenizer, device, [row], (layer, neuron), 0., max_new_tokens)
            shifted = _decisions(model, tokenizer, device, [row], (layer, neuron), .1, max_new_tokens)
            if baseline[0]["generated_ids"] != zero[0]["generated_ids"]:
                raise RuntimeError("zero intervention generation mismatch")
            write(run, "forward/check.json", {"margin": margin, "layer": layer, "neuron": neuron,
                  "signed_sensitivity": derivative, "numeric": numeric,
                  "zero_identical": True, "outputs": baseline + zero + shifted})
            stage.count(3)
        with run.stage("analyze") as stage:
            errors = [abs(record["finite_difference"] - derivative) for record in numeric]
            tolerance = max(.05, .25 * abs(derivative))
            write(run, "analyze/result.json", {"protocol_sha256": object_sha256(protocol),
                  "certified": False, "zero_identical": True, "numeric_absolute_errors": errors,
                  "numeric_tolerance": tolerance, "numeric_agreement": min(errors) <= tolerance,
                  "baseline_output": summary(baseline), "selected_layer": layer, "selected_neuron": neuron,
                  "scope": "single_prompt_engineering_only"})
            stage.count(1)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory


def run_screen(input_path, model_path, run_id, *, artifact_root="artifacts", top_k=3,
               repeats=2, seed=42, layers=None, loaded=None):
    # Validate missing/invalid data before loading expensive weights.
    data = json.loads(Path(input_path).read_text())
    trials = build_trials(data, seed=seed, repeats=repeats, graded=True)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model, tokenizer, device = _load(model_path, loaded)
    rows = encode_trials(trials, tokenizer)
    layers = list(range(len(model.layers))) if layers is None else layers
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= l < len(model.layers) for l in layers):
        raise ValueError("invalid layers")
    protocol = {"schema_version": 1, "protocol_version": VERSION, "status": "method_reproduction",
                "input_path": str(Path(input_path).resolve()), "input_sha256": file_sha256(input_path),
                "model_identity": _identity(model_path, tokenizer), "source_identity": _source(),
                "runtime": _runtime(model, tokenizer, device),
                "seed": seed, "repeats": repeats, "top_k": top_k, "layers": layers,
                "thinking": False, "decision_prefix": PREFIX, "batch_size": 1,
                "screen_objective": "next_token_buy_minus_sell_at_fixed_json_prefix",
                "screen_aggregation": "sum_positions_then_mean_trials_then_mean_tickers_then_abs"}
    screening = [row for row in rows if row["split"] == "screen" and row["positive_count"] == 2]
    with run_context(model_path, "investment-dial-screen", run_id, artifact_root=artifact_root) as run:
        with run.stage("prepare") as stage:
            write(run, "prepare/input.json", data)
            write(run, "prepare/protocol.json", protocol)
            write(run, "prepare/trials.json", rows)
            stage.count(len(rows))
        totals = {}
        margins = []
        ticker_count = 0
        with run.stage("forward") as stage:
            by_ticker = defaultdict(list)
            for row in screening:
                by_ticker[row["ticker"]].append(row)
            for ticker, ticker_rows in by_ticker.items():
                ticker_totals = {}
                for row in ticker_rows:
                    margin, vectors = coordinate_derivatives(model, row["decision_ids"], *row["answer_ids"], device, layers)
                    margins.append({"id": row["id"], "ticker": ticker, "margin": margin})
                    for layer, vector in vectors.items():
                        ticker_totals[layer] = ticker_totals.get(layer, 0) + vector.double() / len(ticker_rows)
                for layer, vector in ticker_totals.items():
                    totals[layer] = totals.get(layer, 0) + vector
                ticker_count += 1
            write(run, "forward/margins.jsonl", margins)
            stage.count(len(margins))
        with run.stage("analyze") as stage:
            candidates = []
            for layer, total in totals.items():
                mean = total / ticker_count
                # Stable ties preserve coordinate order, independent of torch topk ties.
                indices = torch.argsort(mean.abs(), descending=True, stable=True)[:top_k].tolist()
                for neuron in indices:
                    value = float(mean[neuron])
                    if math.isfinite(value) and value != 0:
                        candidates.append({"layer": layer, "neuron": neuron, "signed_sensitivity": value,
                                           "sensitivity": abs(value), "width": len(mean)})
            candidates.sort(key=lambda c: (-c["sensitivity"], c["layer"], c["neuron"]))
            candidates = candidates[:top_k]
            write(run, "analyze/result.json", {"protocol_sha256": object_sha256(protocol),
                  "candidates": candidates, "screen_tickers": ticker_count, "certified": False})
            stage.count(len(candidates))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory


def run_calibration(source_run, model_path, run_id, *, artifact_root="artifacts",
                    deltas=(-8., -4., 0., 4., 8.), targets=(-.3, 0., .3), minimum_rate=.9,
                    max_new_tokens=256, loaded=None):
    deltas, targets = list(deltas), list(targets)
    if (len(deltas) < 3 or deltas != sorted(set(deltas)) or 0 not in deltas or
        any(not math.isfinite(d) for d in deltas) or not targets or
        len(set(targets)) != len(targets) or any(not math.isfinite(t) or not -1 <= t <= 1 for t in targets) or
        not 0 < minimum_rate <= 1 or max_new_tokens < 1):
        raise ValueError("invalid calibration settings")
    bundle = verified_run(source_run, "investment-dial-screen", REQUIRED)
    if not bundle[0]["analyze/result.json"]["candidates"]:
        raise ValueError("no nonzero gradient candidates")
    model, tokenizer, device = _load(model_path, loaded)
    parent, rows, screen, digest = _parent(bundle, model_path, tokenizer, model, device)
    protocol = {"schema_version": 1, "protocol_version": VERSION,
                "model_identity": parent["model_identity"], "source_identity": _source(),
                "parent_manifest_sha256": digest, "parent_run": str(Path(source_run).resolve()),
                "runtime": _runtime(model, tokenizer, device),
                "screen_protocol": parent, "deltas": deltas, "targets": targets,
                "minimum_rate": minimum_rate, "max_new_tokens": max_new_tokens,
                "candidates": screen["candidates"], "coefficient_source": "A_only_no_full_universe_refit"}
    a = [r for r in rows if r["split"] == "A" and r["positive_count"] == 2]
    b = [r for r in rows if r["split"] == "B" and r["positive_count"] == 2]
    with run_context(model_path, "investment-dial-calibration", run_id, artifact_root=artifact_root) as run, frozen_eval(model):
        with run.stage("prepare") as stage:
            write(run, "prepare/protocol.json", protocol)
            write(run, "prepare/trials.json", rows)
            stage.count(len(a) + len(b))
        effects, results = [], []
        with run.stage("forward") as stage:
            baseline = _decisions(model, tokenizer, device, a, None, 0., max_new_tokens)
            effects.extend(_tagged(baseline, phase="A_baseline"))
            for candidate in screen["candidates"]:
                coordinate = candidate["layer"], candidate["neuron"]
                curves = []
                for delta in deltas:
                    records = _decisions(model, tokenizer, device, a, coordinate, delta, max_new_tokens)
                    if delta == 0 and [r["generated_ids"] for r in records] != [r["generated_ids"] for r in baseline]:
                        raise RuntimeError("zero intervention generation mismatch")
                    effects.extend(_tagged(records, phase="A_curve"))
                    curves.append(summary(records))
                result = candidate | {"A_curve": curves, "feasible": False, "rmse": None}
                if not feasible(curves, minimum_rate):
                    result["reason"] = "A_output_validity"
                else:
                    try:
                        coefficients = inverse_curve(deltas, [c["pi"] for c in curves], targets)
                    except ValueError as error:
                        result["reason"] = str(error)
                    else:
                        b_stats = []
                        for target, delta in zip(targets, coefficients):
                            records = _decisions(model, tokenizer, device, b, coordinate, delta, max_new_tokens)
                            effects.extend(_tagged(records, phase="B_selection", target=target))
                            b_stats.append(summary(records))
                        result.update(coefficients=coefficients, B_stats=b_stats)
                        if feasible(b_stats, minimum_rate):
                            result.update(feasible=True, reason=None,
                                          rmse=math.sqrt(sum((s["pi"] - t)**2 for s, t in zip(b_stats, targets)) / len(targets)))
                        else:
                            result["reason"] = "B_output_validity"
                results.append(result)
                print(f"calibration {coordinate}: feasible={result['feasible']} rmse={result['rmse']}", flush=True)
            write(run, "forward/effects.jsonl", effects)
            stage.count(len(effects))
        with run.stage("analyze") as stage:
            eligible = sorted([r for r in results if r["feasible"]], key=lambda r: (r["rmse"], r["layer"], r["neuron"]))
            write(run, "analyze/result.json", {"protocol_sha256": object_sha256(protocol), "candidates": results,
                  "selected": eligible[0] if eligible else None, "success": bool(eligible), "certified": False})
            stage.count(len(results))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory


def run_evaluation(source_run, model_path, run_id, *, artifact_root="artifacts", loaded=None):
    bundle = verified_run(source_run, "investment-dial-calibration", REQUIRED)
    if bundle[0]["analyze/result.json"]["selected"] is None:
        raise ValueError("calibration has no feasible candidate; independent test not run")
    model, tokenizer, device = _load(model_path, loaded)
    parent, rows, calibration, digest = _parent(bundle, model_path, tokenizer, model, device)
    selected = calibration["selected"]
    layer, neuron = selected["layer"], selected["neuron"]
    width = dense_down_projection(model.layers[layer]).in_features
    excluded = {c["neuron"] for c in parent["candidates"] if c["layer"] == layer}
    controls = [n for n in range(width) if n not in excluded]
    if not controls:
        raise ValueError("no same-layer noncandidate control")
    control = random.Random(parent["screen_protocol"]["seed"]).choice(controls)
    protocol = {"schema_version": 1, "protocol_version": VERSION,
                "model_identity": parent["model_identity"], "source_identity": _source(),
                "parent_manifest_sha256": digest, "parent_run": str(Path(source_run).resolve()),
                "runtime": _runtime(model, tokenizer, device),
                "selected": selected, "control_neuron": control, "targets": parent["targets"],
                "max_new_tokens": parent["max_new_tokens"]}
    test = [row for row in rows if row["split"] == "test"]
    with run_context(model_path, "investment-dial-evaluation", run_id, artifact_root=artifact_root) as run, frozen_eval(model):
        with run.stage("prepare") as stage:
            write(run, "prepare/protocol.json", protocol)
            write(run, "prepare/trials.json", test)
            stage.count(len(test))
        effects = []
        with run.stage("forward") as stage:
            effects.extend(_tagged(_decisions(model, tokenizer, device, test, None, 0., parent["max_new_tokens"]), arm="baseline", target=None))
            for target, delta in zip(parent["targets"], selected["coefficients"]):
                for arm, channel in (("selected", neuron), ("random", control)):
                    effects.extend(_tagged(_decisions(model, tokenizer, device, test, (layer, channel), delta,
                                                     parent["max_new_tokens"]), arm=arm, target=target))
            write(run, "forward/effects.jsonl", effects)
            stage.count(len(effects))
        with run.stage("analyze") as stage:
            groups = defaultdict(list)
            for record in effects:
                groups[(record["arm"], record["target"], record["positive_count"])].append(record)
            summaries = [{"arm": arm, "target": target, "positive_count": count} | summary(records)
                         for (arm, target, count), records in groups.items()]
            write(run, "analyze/result.json", {"protocol_sha256": object_sha256(protocol),
                  "summaries": summaries, "certified": False, "interpretation": "descriptive_independent_company_test"})
            stage.count(len(summaries))
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
