"""Investment-dial V2 fine-grid calibration and B reevaluation operator."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from llm_bias.core.artifact_paths import file_sha256, run_root
from llm_bias.core.artifacts.io import read_jsonl
from llm_bias.investment_dial import pipeline as p
from llm_bias.investment_dial.analysis import inverse_curve, summary

COORDINATE = (15, 8490)
PROTOCOL_VERSION = "investment-dial-calibration-v2"
DATASET = "investment-dial-calibration-v2"
V1_DATASET = "investment-dial-calibration"
FINE_A_DATASET = "investment-dial-fine-a"
TARGETS = (-0.3, 0.0, 0.3)
NEW_A_DELTAS = (-0.5, -0.25)
FINE_A_DELTAS = tuple(i / 4 for i in range(9))
V1_DELTAS = (-8.0, -4.0, 4.0, 8.0)
ALL_DELTAS = tuple(sorted((*V1_DELTAS, *NEW_A_DELTAS, *FINE_A_DELTAS)))
MINIMUM_RATE = 0.9
MAX_NEW_TOKENS = 256
RMSE_GATE = 0.15
MAX_RESIDUAL_GATE = 0.25
EXPECTED_ROWS = 340
EXPECTED_TICKERS = 85
EXPECTED_A_GENERATIONS = 680
EXPECTED_B_GENERATIONS = 1360
V1_MANIFEST_SHA256 = "173cc0cf29c6cd4c2980a87ea95bb0eb72bb467824644687c33c6dddcb6d5ff3"
FINE_A_MANIFEST_SHA256 = "ab8a6cb19a594584381218793c245d856e1f4ed7c601f4d6ede5f8983bc61957"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, help="Identical local checkpoint directory")
    result.add_argument("--v1-run", required=True, help="Complete validated V1 calibration run")
    result.add_argument("--fine-a-run", required=True, help="Complete validated fine-a diagnostic run")
    result.add_argument("--run-id", required=True, help="New V2 run ID; automatic resume is not supported")
    result.add_argument("--artifact-root", default="artifacts")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--smoke", action="store_true", help="Run the no-output real-model preflight")
    return result


def _manifest_digest(path: str | Path) -> str:
    return file_sha256(Path(path) / "manifest.json")


def _read_parent(
    path: str | Path,
    dataset: str,
    expected_manifest_sha256: str,
    required: set[str],
) -> tuple[dict, dict, str]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = _manifest_digest(root)
    if digest != expected_manifest_sha256:
        raise ValueError(
            f"fixed parent manifest digest mismatch: expected {expected_manifest_sha256}, got {digest}"
        )
    data, verified_digest = p.verified_run(root, dataset, required)
    if verified_digest != expected_manifest_sha256:
        raise ValueError("parent manifest digest changed during verification")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name in ("forward/effects.jsonl", *(f"forward/delta-{index}.jsonl" for index in range(len(FINE_A_DELTAS)))):
        path = root / name
        if path.is_file():
            data[name] = read_jsonl(path)
    return data, manifest, digest


def _row_key(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _validate_population(rows: list[dict], split: str) -> list[dict]:
    selected = [row for row in rows if row.get("split") == split and row.get("positive_count") == 2]
    if len(selected) != EXPECTED_ROWS or len({row.get("ticker") for row in selected}) != EXPECTED_TICKERS:
        raise ValueError(f"expected 340 {split} rows from 85 companies")
    ids = [row.get("id") for row in selected]
    if any(not isinstance(identifier, str) for identifier in ids) or len(set(ids)) != EXPECTED_ROWS:
        raise ValueError(f"{split} row identities are not exact and unique")
    return sorted(selected, key=lambda row: row["id"])


def _validate_parent_protocol(protocol: dict, expected_version: str) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("protocol_version") != expected_version:
        raise ValueError("parent protocol version or schema mismatch")
    if protocol.get("max_new_tokens") != MAX_NEW_TOKENS:
        raise ValueError("parent max_new_tokens mismatch")
    if expected_version == "investment-dial-fine-a-v1" and (protocol.get("layer"), protocol.get("neuron")) != COORDINATE:
        raise ValueError("fine-a coordinate mismatch")


def _validate_v1_and_fine(v1_path: str | Path, fine_path: str | Path) -> dict:
    v1_required = {"prepare/protocol.json", "prepare/trials.json", "analyze/result.json"}
    fine_required = {"prepare/protocol.json", "prepare/trials.json", "analyze/result.json"}
    v1_data, v1_manifest, v1_digest = _read_parent(v1_path, V1_DATASET, V1_MANIFEST_SHA256, v1_required)
    fine_data, fine_manifest, fine_digest = _read_parent(
        fine_path, FINE_A_DATASET, FINE_A_MANIFEST_SHA256, fine_required
    )
    v1_protocol = v1_data["prepare/protocol.json"]
    fine_protocol = fine_data["prepare/protocol.json"]
    _validate_parent_protocol(v1_protocol, "investment-dial-local-v1")
    _validate_parent_protocol(fine_protocol, "investment-dial-fine-a-v1")
    if fine_protocol.get("deltas") != list(FINE_A_DELTAS):
        raise ValueError("fine-a delta grid mismatch")
    if v1_protocol.get("model_identity") != fine_protocol.get("model_identity"):
        raise ValueError("parent model identity mismatch")
    if v1_protocol.get("runtime", {}).get("device") is None or fine_protocol.get("runtime", {}).get("device") is None:
        raise ValueError("parent runtime is incomplete")
    for key in ("torch", "transformers", "dtype", "chat_template_sha256"):
        if v1_protocol.get("runtime", {}).get(key) != fine_protocol.get("runtime", {}).get(key):
            raise ValueError(f"parent runtime mismatch: {key}")
    selected = v1_data["analyze/result.json"].get("selected") or {}
    if (selected.get("layer"), selected.get("neuron")) != COORDINATE:
        raise ValueError("V1 selected coordinate mismatch")

    v1_rows = v1_data["prepare/trials.json"]
    fine_rows = fine_data["prepare/trials.json"]
    v1_a = _validate_population(v1_rows, "A")
    v1_b = _validate_population(v1_rows, "B")
    fine_a = _validate_population(fine_rows, "A")
    if {_row_key(row) for row in fine_a} != {_row_key(row) for row in v1_a}:
        raise ValueError("fine-a and V1 A row identity sets differ")

    effects = v1_data["forward/effects.jsonl"]
    coarse = {}
    for delta in V1_DELTAS:
        records = [
            record for record in effects
            if record.get("phase") == "A_curve"
            and record.get("layer") == COORDINATE[0]
            and record.get("neuron") == COORDINATE[1]
            and record.get("delta") == delta
        ]
        _validate_records(records, v1_a, delta, require_phase="A_curve")
        coarse[delta] = records

    fine_records = {}
    for index, delta in enumerate(FINE_A_DELTAS):
        records = fine_data[f"forward/delta-{index}.jsonl"]
        _validate_records(records, fine_a, delta)
        fine_records[delta] = records

    return {
        "v1_protocol": v1_protocol,
        "fine_protocol": fine_protocol,
        "v1_manifest": v1_manifest,
        "fine_manifest": fine_manifest,
        "v1_digest": v1_digest,
        "fine_digest": fine_digest,
        "v1_a": v1_a,
        "v1_b": v1_b,
        "fine_a": fine_a,
        "coarse": coarse,
        "fine": fine_records,
    }


def _validate_records(records: list[dict], expected_rows: list[dict], delta: float,
                       require_phase: str | None = None, coordinate: tuple[int, int] | None = COORDINATE) -> None:
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"expected 340 records at delta {delta}")
    expected_by_id = {row["id"]: row for row in expected_rows}
    actual_ids = [record.get("id") for record in records]
    if set(actual_ids) != set(expected_by_id) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"record identity set mismatch at delta {delta}")
    for record in records:
        expected_coordinate = (None, None) if coordinate is None else coordinate
        if (record.get("layer"), record.get("neuron")) != expected_coordinate or record.get("delta") != delta:
            raise ValueError(f"record coordinate/delta mismatch at delta {delta}")
        expected = expected_by_id[record["id"]]
        if any(record.get(key) != expected.get(key) for key in ("ticker", "split", "positive_count")):
            raise ValueError(f"record row identity mismatch at delta {delta}")
        if record.get("split") != expected_rows[0]["split"] or record.get("positive_count") != 2:
            raise ValueError(f"record population mismatch at delta {delta}")
        if require_phase is not None and record.get("phase") != require_phase:
            raise ValueError(f"record phase mismatch at delta {delta}")
        for key in ("json_object", "schema_valid"):
            if not isinstance(record.get(key), bool):
                raise ValueError(f"record validity field missing at delta {delta}")


def _verify_loaded_identity(bundle: dict, model_path: str | Path, model, tokenizer, device) -> None:
    identity = p._identity(model_path, tokenizer)
    for protocol in (bundle["v1_protocol"], bundle["fine_protocol"]):
        if protocol.get("model_identity") != identity:
            raise ValueError("model/tokenizer identity mismatch")
    current_runtime = p._runtime(model, tokenizer, device)
    for protocol in (bundle["v1_protocol"], bundle["fine_protocol"]):
        runtime = protocol.get("runtime", {})
        for key in ("torch", "transformers", "dtype", "chat_template_sha256"):
            if runtime.get(key) != current_runtime.get(key):
                raise ValueError(f"source inference runtime mismatch: {key}")
    v1_rows = bundle["v1_a"] + bundle["v1_b"]
    if p.encode_trials(v1_rows, tokenizer) != v1_rows:
        raise ValueError("V1 tokenization mismatch")
    if p.encode_trials(bundle["fine_a"], tokenizer) != bundle["fine_a"]:
        raise ValueError("fine-a tokenization mismatch")


def _summary(delta: float, source: str, records: list[dict]) -> dict:
    result = {"delta": delta, "source": source} | summary(records)
    if result["pi"] is None or not all(math.isfinite(float(result[key])) for key in ("delta", "valid_decision_rate", "parse_rate", "schema_rate")):
        raise ValueError(f"invalid A summary at delta {delta}")
    return result


def assemble_curve(bundle: dict, new_records: dict[float, list[dict]]) -> dict:
    points = []
    for delta in V1_DELTAS:
        points.append(_summary(delta, "v1-raw", bundle["coarse"][delta]))
    for delta in NEW_A_DELTAS:
        _validate_records(new_records[delta], bundle["v1_a"], delta, require_phase="A_new")
        points.append(_summary(delta, "v2-new", new_records[delta]))
    for delta in FINE_A_DELTAS:
        points.append(_summary(delta, "fine-a", bundle["fine"][delta]))
    points.sort(key=lambda point: point["delta"])
    if [point["delta"] for point in points] != list(ALL_DELTAS):
        raise ValueError("assembled A curve has the wrong delta identity set")
    if any(point["valid_decision_rate"] < MINIMUM_RATE or point["schema_rate"] < MINIMUM_RATE for point in points):
        raise ValueError("A curve output validity below frozen minimum rate")
    values = [point["pi"] for point in points]
    if any(not math.isfinite(value) for value in values) or any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("assembled A curve is not finite non-decreasing")
    return {
        "schema_version": 1,
        "coordinate": {"layer": COORDINATE[0], "neuron": COORDINATE[1]},
        "points": points,
        "monotonicity": {"checked": True, "passed": True, "relation": "non-decreasing"},
    }


def _check_device(device: str) -> None:
    if device == "cpu" and torch.cuda.is_available():
        raise ValueError("CPU execution requires CUDA_VISIBLE_DEVICES='' before launch")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")


def _assert_new_run(model_path: str | Path, run_id: str, artifact_root: str | Path) -> None:
    destination = run_root(model_path, DATASET, run_id, artifact_root=artifact_root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {destination}")


def _protocol(bundle: dict, model_path: str | Path, model, tokenizer, device, args) -> dict:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "implemented_unrun",
        "purpose": "fine-grid A calibration followed by B reevaluation",
        "parent_runs": {
            "v1": {"path": str(Path(args.v1_run).resolve()), "manifest_sha256": bundle["v1_digest"]},
            "fine_a": {"path": str(Path(args.fine_a_run).resolve()), "manifest_sha256": bundle["fine_digest"]},
        },
        "model_identity": p._identity(model_path, tokenizer),
        "runtime": p._runtime(model, tokenizer, device),
        "parent_runtime": bundle["v1_protocol"]["runtime"],
        "parent_runtime_comparison": "torch/transformers/dtype/chat_template_sha256 exact; device may differ and is recorded",
        "source_identity": p._source(),
        "operator_sha256": file_sha256(__file__),
        "coordinate": {"layer": COORDINATE[0], "neuron": COORDINATE[1]},
        "targets": list(TARGETS),
        "a_curve_deltas": list(ALL_DELTAS),
        "a_curve_sources": {str(delta): ("v1-raw" if delta in V1_DELTAS else "fine-a" if delta in FINE_A_DELTAS else "v2-new") for delta in ALL_DELTAS},
        "inversion": {"method": "llm_bias.investment_dial.analysis.inverse_curve", "extrapolation": False},
        "minimum_rate": MINIMUM_RATE,
        "gates": {"rmse": RMSE_GATE, "max_error": MAX_RESIDUAL_GATE},
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_budget": {"a_new": EXPECTED_A_GENERATIONS, "b": EXPECTED_B_GENERATIONS},
        "smoke": bool(args.smoke),
    }


def _smoke(bundle: dict, model, tokenizer, device) -> None:
    preview = [_summary(delta, "v1-raw", bundle["coarse"][delta]) for delta in V1_DELTAS]
    preview += [_summary(delta, "fine-a", bundle["fine"][delta]) for delta in FINE_A_DELTAS]
    print(json.dumps({"a_curve_preview": preview, "missing_new_deltas": list(NEW_A_DELTAS)}, ensure_ascii=False), flush=True)
    a_row = bundle["v1_a"][0]
    b_row = bundle["v1_b"][0]
    a_records = p._decisions(model, tokenizer, device, [a_row], COORDINATE, -0.25, MAX_NEW_TOKENS)
    b_records = p._decisions(model, tokenizer, device, [b_row], None, 0.0, MAX_NEW_TOKENS)
    for records, label in ((a_records, "A"), (b_records, "B")):
        if len(records) != 1 or not records[0].get("json_object") or not records[0].get("schema_valid"):
            raise ValueError(f"smoke {label} response failed JSON/schema validation")
    print("SMOKE PASS: parent guards and A/B response parsing passed; no run directory written", flush=True)


def run(args):
    _check_device(args.device)
    _assert_new_run(args.model, args.run_id, args.artifact_root)
    bundle = _validate_v1_and_fine(args.v1_run, args.fine_a_run)
    model, tokenizer, device = p.load_model(args.model, dtype=torch.bfloat16)
    _verify_loaded_identity(bundle, args.model, model, tokenizer, device)
    if args.smoke:
        with p.frozen_eval(model):
            _smoke(bundle, model, tokenizer, device)
        return None

    protocol = _protocol(bundle, args.model, model, tokenizer, device, args)
    with p.run_context(args.model, DATASET, args.run_id, artifact_root=args.artifact_root) as output, p.frozen_eval(model):
        print("RUN", output.run_directory, flush=True)
        with output.stage("prepare") as stage:
            p.write(output, "prepare/protocol.json", protocol)
            p.write(output, "prepare/trials.json", bundle["v1_b"])
            stage.count(EXPECTED_ROWS)

        new_a_records: dict[float, list[dict]] = {}
        with output.stage("forward") as stage:
            for delta in NEW_A_DELTAS:
                records = [
                    record | {"phase": "A_new"}
                    for record in p._decisions(model, tokenizer, device, bundle["v1_a"], COORDINATE, delta, MAX_NEW_TOKENS)
                ]
                _validate_records(records, bundle["v1_a"], delta, require_phase="A_new")
                new_a_records[delta] = records
            p.write(output, "forward/a-negative.jsonl", [record for delta in NEW_A_DELTAS for record in new_a_records[delta]])
            curve = assemble_curve(bundle, new_a_records)
            curve["inversion"] = {
                "method": "llm_bias.investment_dial.analysis.inverse_curve",
                "targets": list(TARGETS),
                "delta_hats": inverse_curve(
                    [point["delta"] for point in curve["points"]],
                    [point["pi"] for point in curve["points"]],
                    list(TARGETS),
                ),
            }
            curve["inversion"]["curve_sha256"] = p.object_sha256(curve["points"])
            p.write(output, "forward/a_curve.json", curve)
            delta_hats = curve["inversion"]["delta_hats"]
            baseline = [record | {"phase": "B_baseline", "target": None} for record in
                        p._decisions(model, tokenizer, device, bundle["v1_b"], None, 0.0, MAX_NEW_TOKENS)]
            _validate_records(baseline, bundle["v1_b"], 0.0, require_phase="B_baseline", coordinate=None)
            p.write(output, "forward/b-baseline.jsonl", baseline)
            names = ("minus-0-3", "0", "plus-0-3")
            for target, delta_hat, name in zip(TARGETS, delta_hats, names):
                records = [record | {"phase": "B_reevaluation", "target": target, "delta_hat": delta_hat} for record in
                           p._decisions(model, tokenizer, device, bundle["v1_b"], COORDINATE, delta_hat, MAX_NEW_TOKENS)]
                _validate_records(records, bundle["v1_b"], delta_hat, require_phase="B_reevaluation")
                p.write(output, f"forward/b-target-{name}.jsonl", records)
            stage.count(EXPECTED_A_GENERATIONS + EXPECTED_B_GENERATIONS)

        with output.stage("analyze") as stage:
            b_stats = [{"target": None, "delta_hat": 0.0, **summary(baseline)}]
            target_records = []
            for target, delta_hat, name in zip(TARGETS, delta_hats, names):
                records = read_jsonl(output.run_directory / f"forward/b-target-{name}.jsonl")
                stats = {"target": target, "delta_hat": delta_hat, **summary(records)}
                stats["degraded"] = stats["valid_decision_rate"] < MINIMUM_RATE or stats["schema_rate"] < MINIMUM_RATE
                b_stats.append(stats)
                target_records.append(stats)
            residuals = [stats["pi"] - stats["target"] if stats["pi"] is not None else None for stats in target_records]
            degraded = any(stats["degraded"] or stats["pi"] is None for stats in target_records)
            if degraded:
                rmse = None
                max_residual = None
                verdict = "not_evaluable"
            else:
                rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
                max_residual = max(abs(residual) for residual in residuals)
                verdict = "pass" if rmse <= RMSE_GATE and max_residual <= MAX_RESIDUAL_GATE else "fail"
            result = {
                "schema_version": 1,
                "protocol_sha256": p.object_sha256(protocol),
                "coordinate": {"layer": COORDINATE[0], "neuron": COORDINATE[1]},
                "delta_hats": [{"target": target, "delta_hat": delta_hat} for target, delta_hat in zip(TARGETS, delta_hats)],
                "B_baseline": b_stats[0],
                "B_targets": target_records,
                "target_errors": residuals,
                "rmse": rmse,
                "max_error": max_residual,
                "gate_verdict": verdict,
                "degraded": degraded,
                "certified": verdict == "pass",
                "monotonicity": curve["monotonicity"],
                "certified_definition": "B reevaluation target RMSE and max residual under frozen tolerances",
            }
            p.write(output, "analyze/result.json", result)
            stage.count(1)
        output.finalize(required_stages={"prepare", "forward", "analyze"})
        return output.run_directory


if __name__ == "__main__":
    run(parser().parse_args())
