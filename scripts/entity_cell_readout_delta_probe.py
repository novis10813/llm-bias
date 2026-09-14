"""Entity-cell E4 residual-stream suppression readout probe (proposed; not a frozen protocol).

See docs/entity-cell-localization/details/proposal-e4.md for the full definition.

For each frozen V3 entity cell, compare the layer-wise transported
representation readout of the residual stream at the fact-frame readout
position, clean vs suppressed (dose -3.0, all_positions), with
matched-random and wrong-entity controls.  The lens is the canonical pinned
HF pretrained lens; scoring uses the v2 FP32 tail
(fp32_next_token_log_probs), the same path as the E2 selected-component
readout.  Only compact scalars and top-k tokens are persisted; no raw
activations, residuals, or transported vectors are saved.

Gold binding (fail-closed): for each (entity, frame) the probe recomputes
the clean greedy first-3-token gold and requires it to match the gold
recorded in the source formal V3 run (fact_amnesia.jsonl) AND to be marked
verified in that run's fact_gold_verifications.json.  Any mismatch aborts.

Usage (from the repo root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/entity_cell_readout_delta_probe.py \
        --output artifacts/qwen3.5-4b/entity-cell-localization/readout_delta_probe_v1.json \
        [--model .cache/models/qwen3.5-4b] \
        [--target JNJ --target BAC] \
        [--expected-lens-sha256 <64-hex>]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from jlens import ActivationRecorder

from llm_bias.core.analysis.transport import transport_residual_delta
from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.continuation_scoring import (
    fp32_next_token_log_probs,
    score_token_ids,
)
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import decode_token, input_ids
from llm_bias.entity_cell.fact_amnesia import (
    FACT_CONTROL_THRESHOLD,
    FACT_END_DOSE,
    FACT_EFFECT_THRESHOLD,
    fact_frame_prompt,
    greedy_gold_token_ids,
)
from llm_bias.entity_cell.mlp_cells import (
    MLPHookSession,
    OnlineVectorStats,
    select_matched_random_neuron,
)

CALIBRATION_RUN = (
    "artifacts/qwen3.5-4b/entity-cell-localization/runs/"
    "entity-cell-e1-v3-hfm2-calibration-v1"
)
HOLDOUT_RUN = (
    "artifacts/qwen3.5-4b/entity-cell-localization/runs/"
    "entity-cell-e1-v3-holdout-v1"
)

# Frozen V3 entity cells (calibration + hold-out). wrong_cell is the paired
# other frozen cell (entity-specificity control).
DEFAULT_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "ticker": "JNJ",
        "name": "Johnson & Johnson",
        "cell": (4, 7676),
        "frames": ("F0", "F2", "F3"),
        "source_run": CALIBRATION_RUN,
        "wrong_cell": (2, 5003),
    },
    {
        "ticker": "PLTR",
        "name": "Palantir Technologies",
        "cell": (2, 5003),
        "frames": ("F0", "F2", "F3"),
        "source_run": CALIBRATION_RUN,
        "wrong_cell": (4, 7676),
    },
    {
        "ticker": "BAC",
        "name": "Bank of America",
        "cell": (0, 7801),
        "frames": ("F0", "F2", "F3"),
        "source_run": HOLDOUT_RUN,
        "wrong_cell": (2, 7997),
    },
    {
        "ticker": "CAT",
        "name": "Caterpillar",
        "cell": (2, 7997),
        "frames": ("F2",),
        "source_run": HOLDOUT_RUN,
        "wrong_cell": (0, 7801),
    },
)


def _load_source_run_gold(source_run: Path, ticker: str) -> dict[str, dict[str, Any]]:
    """Verified own-frame golds recorded by a formal V3 fact-gate run."""
    rows_path = source_run / "e1" / "fact_amnesia.jsonl"
    verify_path = source_run / "e1" / "fact_gold_verifications.json"
    if not rows_path.is_file() or not verify_path.is_file():
        raise FileNotFoundError(f"source run {source_run} is missing fact-stage outputs")
    decisions = json.loads(verify_path.read_text(encoding="utf-8"))["decisions"]
    golds: dict[str, dict[str, Any]] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("check") != "own" or row.get("ticker") != ticker:
            continue
        if row.get("condition") != "clean":
            continue
        frame_id = str(row["frame_id"])
        decision = decisions.get(f"{ticker}:{frame_id}")
        if not isinstance(decision, dict) or decision.get("verified") is not True:
            continue  # fail-closed: unverified or rejected golds are excluded
        golds[frame_id] = {
            "gold_token_ids": [int(value) for value in row["gold_token_ids"]],
            "gold_text": str(row["gold_text"]),
            "prompt_id": str(row.get("prompt_id", "")),
            "decision_note": str(decision.get("note", "")),
        }
    return golds


def _load_baseline_stats(source_run: Path) -> dict[int, OnlineVectorStats]:
    path = source_run / "e1" / "baseline_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"source run {source_run} is missing e1/baseline_stats.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError(f"{path} is missing its layers object")
    return {int(layer): OnlineVectorStats.from_compact(value) for layer, value in layers.items()}


def _capture_layer_logps(
    model: Any,
    lens: Any,
    ids: torch.Tensor,
    *,
    final_layer: int,
    intervention: tuple[int, int] | None,
    dose: float,
) -> dict[int, torch.Tensor]:
    """One forward pass; per-layer transported-readout log-probabilities."""
    layers = tuple(range(final_layer + 1))
    if intervention is None:
        session: Any = nullcontext_session()
    else:
        session = MLPHookSession(
            model,
            [int(intervention[0])],
            channel_scales={int(intervention[0]): {int(intervention[1]): float(dose)}},
            scope="all_positions",
            scope_positions=None,
        )
    with session:
        with ActivationRecorder(model.layers, at=layers) as recorder:
            with torch.no_grad():
                model.forward(ids)
    position = int(ids.shape[1]) - 1
    result: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = recorder.activations[layer][0, position].detach().float()
        zeros = torch.zeros_like(residual)
        transported = transport_residual_delta(
            residual, zeros, layer=layer, final_layer=final_layer, lens=lens
        )
        logp = fp32_next_token_log_probs(model, transported.unsqueeze(0))[0]
        if not torch.isfinite(logp).all():
            raise ValueError(f"non-finite readout at layer {layer}")
        result[layer] = logp.cpu()
    return result


class _NullSession:
    def __enter__(self) -> "_NullSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def nullcontext_session() -> _NullSession:
    return _NullSession()


def _endpoint_gold_logp(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    gold_ids: list[int],
    intervention: tuple[int, int] | None,
    dose: float,
) -> float:
    if intervention is None:
        return float(score_token_ids(model, tokenizer, prompt, gold_ids, device=device).log_probability)
    with MLPHookSession(
        model,
        [int(intervention[0])],
        channel_scales={int(intervention[0]): {int(intervention[1]): float(dose)}},
        scope="all_positions",
        scope_positions=None,
    ):
        return float(score_token_ids(model, tokenizer, prompt, gold_ids, device=device).log_probability)


def run_probe(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    loaded_lens: Any,
    targets: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Run all (target, frame, condition, layer) rows plus per-frame summaries."""
    vocab_size = int(model._lm_head.weight.shape[0])
    final_layer = int(model.n_layers) - 1
    lens = loaded_lens.lens
    target_blocks: list[dict[str, Any]] = []
    for target in targets:
        ticker = str(target["ticker"])
        cell = (int(target["cell"][0]), int(target["cell"][1]))
        wrong_cell = (int(target["wrong_cell"][0]), int(target["wrong_cell"][1]))
        source_run = Path(str(target["source_run"]))
        recorded = _load_source_run_gold(source_run, ticker)
        stats = _load_baseline_stats(source_run)
        random_neuron = select_matched_random_neuron(
            stats, layer=cell[0], target_neuron=cell[1]
        )
        frames_out: list[dict[str, Any]] = []
        for frame_id in target["frames"]:
            prompt = fact_frame_prompt(frame_id, str(target["name"]))
            gold_ids = greedy_gold_token_ids(model, tokenizer, device, prompt)
            if frame_id not in recorded:
                raise ValueError(
                    f"{ticker}:{frame_id} has no verified gold in {source_run}; aborting (fail-closed)"
                )
            if gold_ids != recorded[frame_id]["gold_token_ids"]:
                raise ValueError(
                    f"{ticker}:{frame_id} recomputed gold {gold_ids} != recorded "
                    f"{recorded[frame_id]['gold_token_ids']}; aborting (fail-closed)"
                )
            ids = torch.tensor(
                [input_ids(tokenizer, prompt, add_special_tokens=True)],
                dtype=torch.long,
                device=device,
            )
            gold0 = int(gold_ids[0])
            conditions: list[tuple[str, tuple[int, int] | None]] = [
                ("clean", None),
                ("target", cell),
                ("matched_random", (cell[0], int(random_neuron))),
                ("wrong_entity", wrong_cell),
            ]
            condition_logps: dict[str, dict[int, torch.Tensor]] = {}
            endpoints: dict[str, float] = {}
            for condition, intervention in conditions:
                condition_logps[condition] = _capture_layer_logps(
                    model, lens, ids, final_layer=final_layer,
                    intervention=intervention, dose=FACT_END_DOSE,
                )
                endpoints[condition] = _endpoint_gold_logp(
                    model, tokenizer, device, prompt, gold_ids, intervention, FACT_END_DOSE
                )
            clean = condition_logps["clean"]
            rows: list[dict[str, Any]] = []
            per_condition_top10: dict[str, dict[int, list[int]]] = {}
            for condition, _ in conditions:
                for layer in range(final_layer + 1):
                    logp = condition_logps[condition][layer]
                    top_values, top_indices = torch.topk(logp, top_k)
                    per_condition_top10.setdefault(condition, {})[layer] = [
                        int(value) for value in top_indices.tolist()
                    ]
                    gold_logp = float(logp[gold0])
                    base = {
                        "schema_version": 1,
                        "artifact_type": "entity_cell_e4_readout_delta",
                        "ticker": ticker,
                        "frame_id": frame_id,
                        "prompt": prompt,
                        "prompt_id": recorded[frame_id]["prompt_id"],
                        "condition": condition,
                        "intervention": (
                            None if _ is None
                            else {"cell": [int(_[0]), int(_[1])], "dose": FACT_END_DOSE, "scope": "all_positions"}
                        ),
                        "layer": layer,
                        "is_transport": layer != final_layer,
                        "gold_token_id": gold0,
                        "gold_token": decode_token(tokenizer, gold0),
                        "gold_logp": gold_logp,
                        "delta_gold_logp": None if condition == "clean" else round(gold_logp - float(clean[layer][gold0]), 6),
                        "top10": [
                            {"token": decode_token(tokenizer, int(idx)), "logp": round(float(value), 6)}
                            for value, idx in zip(top_values.tolist(), top_indices.tolist())
                        ],
                    }
                    if condition == "matched_random":
                        base["matched_random_neuron"] = int(random_neuron)
                    rows.append(base)
            # Per-frame summary quantities.
            target_deltas = [abs(float(r["delta_gold_logp"])) for r in rows if r["condition"] == "target" and r["delta_gold_logp"] is not None]
            first_divergence = next(
                (int(r["layer"]) for r in rows if r["condition"] == "target" and abs(float(r["delta_gold_logp"])) >= FACT_EFFECT_THRESHOLD),
                None,
            )
            max_abs_layer, max_abs_value = max(
                (
                    (int(r["layer"]), abs(float(r["delta_gold_logp"])))
                    for r in rows
                    if r["condition"] == "target" and r["delta_gold_logp"] is not None
                ),
                key=lambda pair: pair[1],
            )
            control_max = {}
            for condition in ("matched_random", "wrong_entity"):
                values = [abs(float(r["delta_gold_logp"])) for r in rows if r["condition"] == condition and r["delta_gold_logp"] is not None]
                control_max[condition] = round(max(values), 6) if values else None
            overlaps = [
                len(set(per_condition_top10["clean"][layer]) & set(per_condition_top10["target"][layer])) / top_k
                for layer in range(final_layer + 1)
            ]
            clean_endpoint = endpoints["clean"]
            summary = {
                "ticker": ticker,
                "frame_id": frame_id,
                "cell": [cell[0], cell[1]],
                "matched_random_neuron": [cell[0], int(random_neuron)],
                "wrong_entity_cell": [wrong_cell[0], wrong_cell[1]],
                "gold_token_ids": gold_ids,
                "gold_text": recorded[frame_id]["gold_text"],
                "gold_verification_note": recorded[frame_id]["decision_note"],
                "endpoint_gold_joint_logp": {name: round(value, 6) for name, value in endpoints.items()},
                "endpoint_collapse": {
                    name: round(endpoints[name] - clean_endpoint, 6)
                    for name in endpoints
                    if name != "clean"
                },
                "first_divergence_layer": first_divergence,
                "max_abs_delta": {"layer": max_abs_layer, "value": round(max_abs_value, 6)},
                "control_max_abs_delta": control_max,
                "control_below_threshold": all(
                    value is not None and value <= FACT_CONTROL_THRESHOLD
                    for value in control_max.values()
                ),
                "clean_vs_target_top10_overlap": {
                    "min": round(min(overlaps), 6),
                    "max": round(max(overlaps), 6),
                },
            }
            frames_out.append({"summary": summary, "rows": rows})
        target_blocks.append(
            {
                "ticker": ticker,
                "name": str(target["name"]),
                "cell": [cell[0], cell[1]],
                "source_run": str(source_run),
                "frames": frames_out,
            }
        )
    return target_blocks


def _lens_provenance(loaded_lens: Any) -> dict[str, Any]:
    metadata = loaded_lens.metadata
    provenance = metadata.get("provenance", {})
    return {
        "path": str(loaded_lens.path),
        "sha256": sha256_file(loaded_lens.path),
        "source": str(provenance.get("source", "unknown")),
        "repo_id": str(provenance.get("repo_id", "")),
        "revision": str(provenance.get("revision", "")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--target", action="append", dest="targets", default=None,
                        help="ticker to include (repeatable); default: all four frozen cells")
    parser.add_argument("--expected-lens-sha256", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    selected = DEFAULT_TARGETS
    if args.targets:
        wanted = {str(value).upper() for value in args.targets}
        selected = [target for target in DEFAULT_TARGETS if str(target["ticker"]).upper() in wanted]
        if not selected:
            raise SystemExit(f"no default targets match {sorted(wanted)}")

    started = time.time()
    model, tokenizer, fallback_device = load_model(args.model)
    device = torch.device(getattr(model, "input_device", fallback_device))
    loaded_lens = load_validated_lens(
        model=model,
        model_name=args.model,
        require_complete=True,
    )
    if args.expected_lens_sha256 is not None:
        actual = sha256_file(loaded_lens.path)
        if actual != args.expected_lens_sha256:
            raise SystemExit(f"canonical lens sha {actual} != expected {args.expected_lens_sha256}")
    manifest_shas = {
        str(target["source_run"]): sha256_file(Path(str(target["source_run"])) / "manifest.json")
        for target in selected
    }
    target_blocks = run_probe(
        model=model,
        tokenizer=tokenizer,
        device=device,
        loaded_lens=loaded_lens,
        targets=selected,
        top_k=args.top_k,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "entity_cell_e4_readout_delta_probe",
        "protocol": "proposed",
        "provenance": {
            "model": args.model,
            "device": str(device),
            "dtype": "bfloat16" if str(device).startswith("cuda") else "float32",
            "instrument": "v2 fp32 tail (fp32_next_token_log_probs)",
            "lens": _lens_provenance(loaded_lens),
            "dose": FACT_END_DOSE,
            "effect_threshold": FACT_EFFECT_THRESHOLD,
            "control_threshold": FACT_CONTROL_THRESHOLD,
            "readout_position": "last prompt token",
            "top_k": args.top_k,
            "final_layer_identity_readout": True,
            "source_runs": manifest_shas,
            "raw_runtime_payloads": False,
        },
        "targets": target_blocks,
        "wall_seconds": round(time.time() - started, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "targets": [block["ticker"] for block in target_blocks],
        "lens_sha256": result["provenance"]["lens"]["sha256"][:16] + "...",
        "wall_seconds": result["wall_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
