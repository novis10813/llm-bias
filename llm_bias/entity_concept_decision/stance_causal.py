"""Bounded causal probe: does the L15 stance axis control the buy/sell margin?

Development only, no gates.  After the selected block (default L15), a scaled
copy of the stance direction — or of a seeded random unit direction — is added
to the instruction-span positions of the block output, and the buy/sell margin
is re-scored at the final position.  The stance-axis dose response is compared
against matched random directions at the same doses.  The stance direction is
refit here from the same 6 round-2 evaluation pairs and the same prompts as the
stance characterization run, so both runs use an identical direction formula.
No raw states are persisted: only compact derived statistics and direction
hashes.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifacts.io import write_json, write_jsonl
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.entity_concept_decision.development import (
    _answer_ids,
    _json_number,
    _margin,
    _register_reference,
)

EVAL_KINDS = ("evaluation_at_positive_concept", "evaluation_at_negative_concept")


def _tensor_sha256(vector: torch.Tensor) -> str:
    return hashlib.sha256(vector.detach().cpu().float().numpy().tobytes()).hexdigest()


def _unit(vector: torch.Tensor) -> torch.Tensor:
    norm = float(vector.norm())
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("degenerate direction (zero or non-finite norm)")
    return vector / norm


def _fit_stance_direction(
    states: Mapping[str, torch.Tensor],
    comparisons: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, list[str]]:
    """Mean of the evaluation-pair state differences, unit-normalized."""
    diffs: list[torch.Tensor] = []
    pair_ids: list[str] = []
    for c in comparisons:
        if c["kind"] not in EVAL_KINDS:
            continue
        positive = states.get(str(c["positive_id"]))
        negative = states.get(str(c["negative_id"]))
        if positive is None or negative is None:
            raise KeyError(f"missing state for evaluation pair {c['positive_id']}/{c['negative_id']}")
        diffs.append(positive.float() - negative.float())
        pair_ids.append(f"{c['positive_id']}>{c['negative_id']}")
    if not diffs:
        raise ValueError("no evaluation pairs found in comparisons")
    return _unit(torch.stack(diffs).mean(dim=0)), pair_ids


def _additive_transform(direction: torch.Tensor, dose: float, start: int, end: int, device: torch.device):
    """Block-output transform that adds `dose * direction` on span [start, end)."""
    add = (float(dose) * direction.float()).to(device)

    def fn(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, start:end, :] = (out[:, start:end, :].float() + add).to(x.dtype)
        return out

    return fn


def _score_margin(
    *,
    model: Any,
    tensor: torch.Tensor,
    layer: int,
    last: int,
    buy_id: int,
    sell_id: int,
    transform: Any | None = None,
) -> dict[str, float]:
    if transform is None:
        captured = record_residuals(model, tensor, [last])
    else:
        with residual_interventions(model, {layer: transform}):
            captured = record_residuals(model, tensor, [last])
    final = captured[last][:, -1, :].float()
    margin = _margin(model, final, buy_id, sell_id)
    del captured
    return {"margin": float(margin), "p_buy_2way": 1.0 / (1.0 + math.exp(-margin))}


def _analyze(records: list[dict[str, Any]], company_ids: list[str]) -> dict[str, Any]:
    """Aggregate the dose response for the stance axis vs matched randoms."""
    signed_doses = sorted({r["dose"] for r in records if r["direction"] == "stance"}, reverse=True)
    dose_response: list[dict[str, Any]] = []
    for dose in signed_doses:
        stance_d = [r["delta_margin"] for r in records if r["direction"] == "stance" and r["dose"] == dose]
        random_d = [r["delta_margin"] for r in records if r["direction"] != "stance" and r["dose"] == dose]
        if not stance_d or not random_d:
            raise ValueError(f"missing deltas for dose {dose}")
        stance_mean = sum(stance_d) / len(stance_d)
        random_mean = sum(random_d) / len(random_d)
        dose_response.append({
            "dose": _json_number(dose, name="dose"),
            "stance_mean_delta": _json_number(stance_mean, name="stance_mean_delta"),
            "stance_sd": _json_number(_sd(stance_d), name="stance_sd"),
            "random_mean_delta": _json_number(random_mean, name="random_mean_delta"),
            "random_sd": _json_number(_sd(random_d), name="random_sd"),
            "stance_minus_random": _json_number(stance_mean - random_mean, name="stance_minus_random"),
            "n_companies": len(stance_d),
            "n_random_draws": len(random_d),
        })
    company_deltas: dict[str, dict[str, Any]] = {}
    for cid in company_ids:
        company_deltas[cid] = {
            str(r["dose"]): _json_number(r["delta_margin"], name="delta_margin")
            for r in records if r["id"] == cid and r["direction"] == "stance"
        }
    max_dose = signed_doses[0]
    plus = [r for r in records if r["direction"] == "stance" and r["dose"] == max_dose]
    flipped = [r["id"] for r in plus if r["p_buy_2way"] > 0.5]
    slopes = []
    for dose in [d for d in signed_doses if d > 0]:
        mean = next(x["stance_mean_delta"] for x in dose_response if x["dose"] == dose)
        slopes.append({"dose": _json_number(dose, name="dose"), "mean_delta_per_unit": _json_number(mean / dose, name="slope")})
    return {
        "claim_status": "descriptive_probe",
        "scientific_status": "not_evaluated",
        "purpose": "development",
        "dose_response": dose_response,
        "stance_slopes_positive_doses": slopes,
        "company_stance_deltas": company_deltas,
        "flips_at_max_plus_dose": {"n_flipped": len(flipped), "companies": flipped},
    }


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def run_stance_causal(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    layer: int,
    run_id: str,
    model_name: str,
    artifact_root: str | Path,
    provenance: Mapping[str, Any],
    doses: Sequence[float] = (0.02, 0.05, 0.10),
    n_random: int = 4,
    random_seed: int = 1729,
) -> Path:
    """prepare -> forward (fit stance, baseline + dose x direction interventions) -> analyze."""
    last = int(model.n_layers) - 1
    if not (0 <= int(layer) < int(model.n_layers)):
        raise ValueError(f"layer {layer} out of range for {model.n_layers} layers")
    dose_list = sorted({abs(float(d)) for d in doses})
    if not dose_list or any(d <= 0 for d in dose_list):
        raise ValueError("doses must be positive magnitudes")
    if n_random < 1:
        raise ValueError("n_random must be >= 1")
    material = [dict(p) for p in prompts if p.get("kind") != "company"]
    companies = [dict(p) for p in prompts if p.get("kind") == "company"]
    if len(material) < 2 or len(companies) < 3:
        raise ValueError("need material rows and at least 3 companies")

    run = ArtifactRun.create(model_name, "entity-concept-stance-causal", run_id, artifact_root=artifact_root)
    try:
        with run.stage("prepare") as stage:
            prepare_dir = run.run_directory / "prepare"
            write_jsonl(prepare_dir / "prompts.jsonl", [*material, *companies], overwrite=False)
            metadata = {
                "purpose": "development",
                "claim_status": "descriptive_probe",
                "provenance": dict(provenance),
                "layer": int(layer),
                "final_layer": last,
                "doses": dose_list,
                "n_random_directions": n_random,
                "random_seed": random_seed,
            }
            write_json(prepare_dir / "metadata.json", metadata, overwrite=False)
            _register_reference(run, provenance["upstream"], role="upstream", stage="prepare")
            if provenance.get("lens"):
                _register_reference(run, provenance["lens"], role="lens", stage="prepare")
            stage.count(len(material) + len(companies))

        with run.stage("forward") as stage:
            d_model = int(model.d_model)
            # 1) Fit the stance direction from the material rows (same formula as the characterization run).
            states: dict[str, torch.Tensor] = {}
            for item in material:
                tensor = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                captured = record_residuals(model, tensor, [layer])
                span = item["instruction_span"]
                block = captured[layer][:, span[0]:span[1], :]
                states[item["id"]] = block.float().mean(dim=1).detach().cpu().reshape(-1)
                del captured, block
            stance, pair_ids = _fit_stance_direction(states, comparisons)
            del states
            generator = torch.Generator(device="cpu").manual_seed(random_seed)
            random_directions = [_unit(torch.randn(d_model, generator=generator)) for _ in range(n_random)]
            directions = [("stance", stance)] + [(f"random_{i}", d) for i, d in enumerate(random_directions)]

            records: list[dict[str, Any]] = []
            state_norms: list[float] = []
            total = len(companies) * (1 + len(dose_list) * 2 * len(directions))
            done = 0
            for item in companies:
                tensor = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
                span = item["instruction_span"]
                ids, buy_id, sell_id = _answer_ids(tokenizer, item["formatted"])
                started = time.perf_counter()
                captured = record_residuals(model, tensor, [layer, last])
                state_norms.append(float(captured[layer][:, span[0]:span[1], :].float().norm(dim=2).mean()))
                baseline = _margin(model, captured[last][:, -1, :].float(), buy_id, sell_id)
                del captured
                done += 1
                records.append({
                    "id": item["id"], "role": "baseline", "dose": 0.0, "direction": "none",
                    "margin": _json_number(float(baseline), name="margin"),
                    "p_buy_2way": _json_number(1.0 / (1.0 + math.exp(-float(baseline))), name="p_buy_2way"),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                })
                for dose in dose_list:
                    for sign in (1.0, -1.0):
                        for kind, direction in directions:
                            transform = _additive_transform(direction, sign * dose, span[0], span[1], device)
                            t0 = time.perf_counter()
                            scored = _score_margin(
                                model=model, tensor=tensor, layer=layer, last=last,
                                buy_id=buy_id, sell_id=sell_id, transform=transform,
                            )
                            done += 1
                            records.append({
                                "id": item["id"], "role": "intervention",
                                "dose": _json_number(float(sign * dose), name="dose"),
                                "direction": kind,
                                "margin": _json_number(scored["margin"], name="margin"),
                                "delta_margin": _json_number(scored["margin"] - float(baseline), name="delta_margin"),
                                "p_buy_2way": _json_number(scored["p_buy_2way"], name="p_buy_2way"),
                                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                            })
                            del transform
                print(f"company {done}/{total + len(material)}: {item['id']}", flush=True)
                del tensor
            company_ids = [c["id"] for c in companies]
            records_path = run.run_directory / "forward" / "records.jsonl"
            write_jsonl(records_path, records, overwrite=False)
            forward_meta = {
                "n_records": len(records),
                "directions": {
                    kind: _tensor_sha256(direction) for kind, direction in directions
                },
                "stance_n_pairs": len(pair_ids),
                "stance_pairs": pair_ids,
                "random_seed": random_seed,
                "baseline_state_norm_instr_span_mean": _json_number(sum(state_norms) / len(state_norms), name="state_norm"),
                "baseline_state_norm_instr_span_min": _json_number(min(state_norms), name="state_norm"),
                "baseline_state_norm_instr_span_max": _json_number(max(state_norms), name="state_norm"),
                "total_forward_seconds": round(sum(r["elapsed_seconds"] for r in records), 1),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0,
            }
            write_json(run.run_directory / "forward" / "metadata.json", forward_meta, overwrite=False)
            run.manifest.register_artifact(records_path, artifact_type="stance_causal_forward_records", stage="forward", record_count=len(records))
            run.manifest.register_artifact(run.run_directory / "forward" / "metadata.json", artifact_type="stance_causal_forward_metadata", stage="forward")
            stage.count(len(records))

        with run.stage("analyze") as stage:
            summary = _analyze(records, company_ids)
            summary.update({
                "schema_version": 1,
                "layer": int(layer),
                "doses": dose_list,
                "n_companies": len(company_ids),
                "stance_direction_sha256": _tensor_sha256(stance),
                "random_direction_sha256s": [_tensor_sha256(d) for d in random_directions],
                "baseline_margins": {
                    r["id"]: next(x["margin"] for x in records if x["id"] == r["id"] and x["role"] == "baseline")
                    for r in companies
                },
            })
            summary_path = run.run_directory / "analyze" / "summary.json"
            write_json(summary_path, summary, overwrite=False)
            run.manifest.register_artifact(summary_path, artifact_type="stance_causal_summary", stage="analyze")
            stage.count(1)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = ["run_stance_causal", "EVAL_KINDS"]
