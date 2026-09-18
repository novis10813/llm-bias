"""M6 frozen-operator external-population pipeline.

Protocol: docs/selective-intervention/details/proposal-m6.md.
The pipeline consumes a frozen V1 basis/center definition and a preselected
12-company manifest. It never fits or tunes the operator on external rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.generation import GenerationConfig, finish_reason, generate_tokens
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.model import load_model, load_tokenizer_for_inference
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids

from . import template as T
from .analysis import decision_flips, iqr, per_ticker_margins
from .m6_analysis import (
    M6_BOOTSTRAP_SAMPLES,
    M6_BOOTSTRAP_SEED,
    evaluate_m6,
    generation_flip_stats,
    validate_external_manifest,
)
from .scoring import answer_token_ids, margin_forward, scoring_ids
from .spans import anonymous_prompt, instruction_token_span, _anonymous_row
from .subspace import (
    align_grid,
    calibration_centers,
    load_e01_basis,
    random_orthonormal_basis,
    subspace_removal_transform,
    tensor_sha256,
)

M6_SCHEMA_VERSION = "selective-intervention-m6-v2"
M6_DATASET = "selective-intervention-m6-v2"
M6_PROTOCOL = "docs/selective-intervention/details/proposal-m6-v2.md"
M6_PROTOCOL_REV = 1
M6_GENERATION_GREEDY = True
M6_MAIN_ARM = "dose_100"
M6_RANDOM_ARM = "ctrl_random"
M6_CLEAN_ARM = "clean"
M6_ANON_CLEAN_ARM = "anon_clean"
M6_ANON_INT_ARM = "anon_int"
M6_MAX_NEW_TOKENS = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forward_device(model: Any) -> Any:
    return (
        model.input_device
        if hasattr(model, "input_device")
        else "cuda" if torch.cuda.is_available() else "cpu"
    )


def _verify_complete(run_root: Path) -> dict[str, Any]:
    path = run_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"upstream manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"upstream run {run_root.name} is not complete")
    return {"run_id": run_root.name, "root": str(run_root), "manifest_sha256": _sha256(path)}


def _decision(margin: float) -> str:
    return "buy" if margin > 0 else "sell"


def _record(
    row: Mapping[str, Any],
    *,
    arm: str,
    alpha: float,
    margin: float,
    generated_decision: str | None = None,
    generation_parse_success: bool | None = None,
    generation_finish_reason: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "prompt_id": row["id"],
        "ticker": row["ticker"],
        "sector": row["sector"],
        "reverse": int(bool(row["reverse"])),
        "order": int(row["order"]),
        "arm": arm,
        "alpha": float(alpha),
        "centering": "cloud" if arm not in {M6_CLEAN_ARM, M6_ANON_CLEAN_ARM} else None,
        "layer": T.INTERVENTION_LAYER if arm not in {M6_CLEAN_ARM, M6_ANON_CLEAN_ARM} else None,
        "position_scope": "instruction" if arm not in {M6_CLEAN_ARM, M6_ANON_CLEAN_ARM} else None,
        "margin": float(margin),
        "decision": _decision(float(margin)),
    }
    if generated_decision is not None or generation_parse_success is not None:
        record.update(
            {
                "generated_decision": generated_decision,
                "generation_parse_success": bool(generation_parse_success),
                "generation_finish_reason": generation_finish_reason,
            }
        )
    return record


def _replace_header(source_prompt: str, source_ticker: str, source_name: str, ticker: str, name: str) -> str:
    source_header = f"{T.TICKER_LINE_PREFIX}{source_ticker}]\n\n{T.NAME_LINE_PREFIX}{source_name}]"
    target_header = f"{T.TICKER_LINE_PREFIX}{ticker}]\n\n{T.NAME_LINE_PREFIX}{name}]"
    if source_prompt.count(source_header) != 1:
        raise ValueError(f"source prompt header for {source_ticker} does not occur exactly once")
    return source_prompt.replace(source_header, target_header)


def _load_template_rows(
    phase2a_run: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _verify_complete(phase2a_run)
    rows_path = phase2a_run / "prepare" / "prompts.jsonl"
    rows = read_jsonl(rows_path)
    if len(rows) != T.N_PROMPTS:
        raise ValueError(f"Phase 2A template population must contain {T.N_PROMPTS} rows")
    by_variant: dict[tuple[bool, int], dict[str, Any]] = {}
    source_tickers: set[str] = set()
    for row in rows:
        source_tickers.add(str(row["ticker"]))
        key = (bool(row["reverse"]), int(row["order"]))
        if key in by_variant:
            continue
        by_variant[key] = row
    expected = {(False, 0), (False, 1), (True, 0), (True, 1)}
    if set(by_variant) != expected:
        raise ValueError(f"Phase 2A template variants mismatch: {sorted(by_variant)}")
    canonical = [
        row for row in rows if not bool(row["reverse"]) and int(row["order"]) == 0
    ]
    if len(canonical) != T.N_COMPANIES:
        raise ValueError(f"Phase 2A canonical source rows must contain {T.N_COMPANIES} companies")
    return (
        list(by_variant.values()),
        canonical,
        {**manifest, "source_tickers": sorted(source_tickers), "rows_sha256": _sha256(rows_path)},
    )


def _external_rows(tokenizer: Any, template_rows: list[dict[str, Any]], companies: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for company in companies:
        for source in template_rows:
            prompt = _replace_header(
                source["prompt"], str(source["ticker"]), str(source["name"]),
                company["ticker"], company["name"],
            )
            formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
            prompt_ids = input_ids(tokenizer, formatted, add_special_tokens=True)
            span = instruction_token_span(tokenizer, formatted, prompt)
            rows.append(
                {
                    "id": f"{company['ticker']}:rev{int(bool(source['reverse']))}:ord{int(source['order'])}",
                    "prompt_type": "named",
                    "prompt": prompt,
                    "formatted": formatted,
                    "prompt_ids": prompt_ids,
                    "ticker": company["ticker"],
                    "name": company["name"],
                    "sector": company["sector"],
                    "reverse": bool(source["reverse"]),
                    "order": int(source["order"]),
                    "instruction_span": list(span),
                }
            )
    if len(rows) != 48:
        raise ValueError(f"external prompt construction produced {len(rows)} rows, expected 48")
    span_lengths = {
        int(row["instruction_span"][1]) - int(row["instruction_span"][0]) for row in rows
    }
    if len(span_lengths) != 1 or not next(iter(span_lengths)) > 0:
        raise ValueError(f"external instruction spans are not equal and non-empty: {sorted(span_lengths)}")
    for ticker in {row["ticker"] for row in rows}:
        company_spans = {
            tuple(row["instruction_span"])
            for row in rows
            if row["ticker"] == ticker
        }
        if len(company_spans) != 1:
            raise ValueError(f"{ticker}: variants do not share the V1 instruction span")
    return rows


def _generation_target(model: Any) -> InjectedModelAdapter:
    return InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))


def _parse_generated_decision(tokenizer: Any, sequence: torch.Tensor, prompt_length: int) -> tuple[str | None, bool, str]:
    generated_ids = sequence[0, prompt_length:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    match = re.search(r'"decision"\s*:\s*"\s*(buy|sell)\s*"', text, flags=re.IGNORECASE)
    if match is None:
        return None, False, "unparsed"
    return match.group(1).lower(), True, "parsed"


def _generation_safe_transform(transform: Any) -> Any:
    """Keep cached one-token generation steps outside prompt interventions."""
    def apply(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor
        return transform(tensor)

    return apply


def _generate_decision(
    model: Any,
    tokenizer: Any,
    device: Any,
    row: Mapping[str, Any],
    transforms: Mapping[int, Any] | None,
) -> tuple[str | None, bool, str]:
    ids = torch.tensor([list(row["prompt_ids"])], dtype=torch.long, device=device)
    target = _generation_target(model)
    config = GenerationConfig(
        max_new_tokens=M6_MAX_NEW_TOKENS,
        temperature=0.0,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    )
    # HF cached decoding sends only the newly generated token through later
    # residual hooks. The instruction-span transform belongs to the prefill
    # sequence; later one-token calls must pass through unchanged.
    generation_transforms = {
        layer: _generation_safe_transform(transform)
        for layer, transform in (transforms or {}).items()
    }
    with residual_interventions(model, generation_transforms):
        sequence = generate_tokens(target, ids, config)
    decision, parsed, _ = _parse_generated_decision(tokenizer, sequence, ids.shape[1])
    generated_ids = sequence[0, ids.shape[1]:].tolist()
    reason = finish_reason(
        generated_ids,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        max_new_tokens=M6_MAX_NEW_TOKENS,
    ) if parsed else "unparsed"
    return decision, parsed, reason


def _prepare(
    run: ArtifactRun,
    tokenizer: Any,
    *,
    external_manifest: Path,
    phase2a_run: Path,
    e01_run: Path,
    v1_run: Path,
) -> dict[str, Any]:
    out_dir = run.run_directory / "prepare"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare"):
        manifest_raw = json.loads(external_manifest.read_text(encoding="utf-8"))
        manifest = validate_external_manifest(manifest_raw)
        template_rows, source_rows, template_prov = _load_template_rows(phase2a_run)
        source_tickers = set(template_prov["source_tickers"])
        external_tickers = {row["ticker"] for row in manifest["companies"]}
        if source_tickers & external_tickers:
            raise ValueError("external manifest overlaps the Phase 2A source population")
        v1_manifest = _verify_complete(v1_run)
        v1_forward_meta_path = v1_run / "forward" / "metadata.json"
        if not v1_forward_meta_path.is_file():
            raise FileNotFoundError(f"V1 forward metadata not found: {v1_forward_meta_path}")
        v1_forward_meta = json.loads(v1_forward_meta_path.read_text(encoding="utf-8"))
        expected_center_digest = v1_forward_meta.get("centers", {}).get(str(T.INTERVENTION_LAYER))
        if not isinstance(expected_center_digest, str) or len(expected_center_digest) != 64:
            raise ValueError("V1 metadata is missing the frozen L15 center digest")
        basis16, singular = load_e01_basis(e01_run / "analyze" / "summary.json", expected_k=T.E01_PCA_DIM)
        basis = basis16[:, : T.K_PRIMARY]
        random_basis = random_orthonormal_basis(basis.shape[0], basis.shape[1], T.RANDOM_SEED)
        rows = _external_rows(tokenizer, template_rows, manifest["companies"])
        anchor = rows[0]
        anonymous = _anonymous_row(tokenizer, anchor)
        rows_path = out_dir / "rows.jsonl"
        write_count = write_jsonl(rows_path, rows + [anonymous], overwrite=True)
        provenance = {
            "schema_version": M6_SCHEMA_VERSION,
            "artifact_type": "selective_intervention_m6_prepare_provenance",
            "protocol": M6_PROTOCOL,
            "protocol_rev": M6_PROTOCOL_REV,
            "raw_runtime_payloads": False,
            "external_manifest_sha256": _sha256(external_manifest),
            "external_manifest": manifest,
            "phase2a": template_prov,
            "e01": {**_verify_complete(e01_run), "basis_sha256": tensor_sha256(basis), "singular_range": [float(singular[0]), float(singular[-1])]},
            "v1": {**v1_manifest, "center_digest_l15": expected_center_digest},
            "random_control": {"seed": T.RANDOM_SEED, "basis_sha256": tensor_sha256(random_basis)},
            "n_external_companies": len(manifest["companies"]),
            "n_external_prompts": len(rows),
            "n_variants": T.N_2A_VARIANTS,
        }
        write_json(out_dir / "provenance.json", provenance, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": M6_SCHEMA_VERSION,
                "artifact_type": "selective_intervention_m6_prepare",
                "n_rows": write_count,
                "n_external_companies": len(manifest["companies"]),
                "center_reconstruction": "source_16_in_memory_digest_checked",
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(rows_path, artifact_type="selective_intervention_m6_rows", stage="prepare", role="output", record_count=write_count)
        run.manifest.register_artifact(out_dir / "provenance.json", artifact_type="selective_intervention_m6_provenance", stage="prepare", role="output")
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="selective_intervention_m6_prepare_metadata", stage="prepare", role="output")
    return {
        "rows": rows,
        "anonymous": anonymous,
        "template_rows": template_rows,
        "source_rows": source_rows,
        "manifest": manifest,
        "phase2a_provenance": template_prov,
        "basis": basis,
        "random_basis": random_basis,
        "expected_center_digest": expected_center_digest,
        "v1_run": v1_run,
    }


def _forward(run: ArtifactRun, model_path: str, ctx: dict[str, Any]) -> None:
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records: list[dict[str, Any]] = []
    with run.stage("forward"):
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        rows = ctx["rows"]
        anonymous = ctx["anonymous"]
        basis = ctx["basis"].to(device=device, dtype=torch.float32)
        random_basis = ctx["random_basis"].to(device=device, dtype=torch.float32)
        source_rows = ctx["source_rows"]
        source_ref = next(row for row in source_rows if not row["reverse"] and int(row["order"]) == 0)
        source_anon = _anonymous_row(tokenizer, source_ref)
        source_ref_len = len(scoring_ids(tokenizer, source_ref["formatted"]))
        calibration = calibration_centers(
            model,
            tokenizer,
            source_rows,
            source_ref,
            source_anon,
            layers=[T.INTERVENTION_LAYER],
            anon_layer=T.INTERVENTION_LAYER,
            ref_seq_len=source_ref_len,
            device=device,
        )
        center_digest = tensor_sha256(calibration["mu_bar"][T.INTERVENTION_LAYER])
        center_digest_matches_v1 = center_digest == ctx["expected_center_digest"]
        centers: dict[str, dict[int, torch.Tensor]] = {}
        for row in rows:
            span = (int(row["instruction_span"][0]), int(row["instruction_span"][1]))
            centers[row["ticker"]] = align_grid(
                calibration["mu_bar"][T.INTERVENTION_LAYER], calibration["ref_span"], span
            )
        anon_span = (int(anonymous["instruction_span"][0]), int(anonymous["instruction_span"][1]))
        anon_center = align_grid(calibration["mu_bar"][T.INTERVENTION_LAYER], calibration["ref_span"], anon_span)
        seqs: dict[str, tuple[torch.Tensor, tuple[int, int]]] = {}
        for row in rows + [anonymous]:
            ids = scoring_ids(tokenizer, row["formatted"])
            seqs[row["id"]] = (
                torch.tensor([ids], dtype=torch.long, device=device),
                answer_token_ids(tokenizer, row["formatted"] + T.DECISION_PREFIX),
            )

        def margin(row: Mapping[str, Any], transform: Any | None = None) -> float:
            tensor, (buy_id, sell_id) = seqs[row["id"]]
            value = margin_forward(model, tensor, {T.INTERVENTION_LAYER: transform} if transform is not None else None, buy_id, sell_id)
            if not math.isfinite(value):
                raise ValueError(f"non-finite M6 margin for {row['id']}")
            return value

        def intervention(row: Mapping[str, Any], arm: str, alpha: float, arm_basis: torch.Tensor, *, generate: bool = False) -> None:
            positions = tuple(range(int(row["instruction_span"][0]), int(row["instruction_span"][1])))
            transform = subspace_removal_transform(arm_basis, alpha, centers[row["ticker"]], positions)
            value = margin(row, transform)
            generated = (None, None, None)
            if generate:
                generated = _generate_decision(model, tokenizer, device, row, {T.INTERVENTION_LAYER: transform})
            records.append(_record(row, arm=arm, alpha=alpha, margin=value, generated_decision=generated[0], generation_parse_success=generated[1], generation_finish_reason=generated[2]))

        # Clean and generation reference.
        for row in rows:
            value = margin(row)
            generated = _generate_decision(model, tokenizer, device, row, None)
            records.append(_record(row, arm=M6_CLEAN_ARM, alpha=0.0, margin=value, generated_decision=generated[0], generation_parse_success=generated[1], generation_finish_reason=generated[2]))

        for alpha in (0.25, 0.5, 0.75, 1.0):
            arm = f"dose_{round(alpha * 100)}"
            for row in rows:
                intervention(row, arm, alpha, basis, generate=(alpha == 1.0))

        for row in rows:
            intervention(row, M6_RANDOM_ARM, 1.0, random_basis, generate=True)

        anon_value = margin(anonymous)
        records.append(_record(anonymous, arm=M6_ANON_CLEAN_ARM, alpha=0.0, margin=anon_value))
        anon_transform = subspace_removal_transform(basis, 1.0, anon_center, tuple(range(anon_span[0], anon_span[1])))
        anon_int_value = margin(anonymous, anon_transform)
        records.append(_record(anonymous, arm=M6_ANON_INT_ARM, alpha=1.0, margin=anon_int_value))

        path = out_dir / "records.jsonl"
        write_jsonl(path, records, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {
                "schema_version": M6_SCHEMA_VERSION,
                "artifact_type": "selective_intervention_m6_forward",
                "n_records": len(records),
                "n_external_prompts": len(rows),
                "center_digest_l15": center_digest,
                "expected_center_digest_l15": ctx["expected_center_digest"],
                "center_digest_matches_v1": center_digest_matches_v1,
                "center_mode": "current_runtime_deterministic_reconstruction",
                "random_basis_seed": T.RANDOM_SEED,
                "generation_config": {"do_sample": False, "temperature": 0.0},
                "generation_max_new_tokens": M6_MAX_NEW_TOKENS,
                "runtime_seconds": round(time.time() - started, 1),
                "raw_runtime_payloads": False,
            },
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="selective_intervention_m6_forward", stage="forward", role="output", record_count=len(records))
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="selective_intervention_m6_forward_metadata", stage="forward", role="output")


def _smoke_forward(run: ArtifactRun, model_path: str, ctx: dict[str, Any]) -> None:
    """Run one external prompt through clean/main/random arms only.

    This is a real-weight preflight, not an M6 estimate: it checks model
    loading, center reconstruction, intervention hooks and greedy generation
    while deliberately avoiding a formal summary or full-population result.
    """
    out_dir = run.run_directory / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    with run.stage("forward"):
        model, tokenizer, _ = load_model(model_path, dtype=None)
        device = _forward_device(model)
        basis = ctx["basis"].to(device=device, dtype=torch.float32)
        random_basis = ctx["random_basis"].to(device=device, dtype=torch.float32)
        source_rows = ctx["source_rows"]
        source_ref = next(row for row in source_rows if not row["reverse"] and int(row["order"]) == 0)
        source_anon = _anonymous_row(tokenizer, source_ref)
        source_ref_len = len(scoring_ids(tokenizer, source_ref["formatted"]))
        calibration = calibration_centers(
            model,
            tokenizer,
            source_rows,
            source_ref,
            source_anon,
            layers=[T.INTERVENTION_LAYER],
            anon_layer=T.INTERVENTION_LAYER,
            ref_seq_len=source_ref_len,
            device=device,
        )
        center_digest = tensor_sha256(calibration["mu_bar"][T.INTERVENTION_LAYER])
        center_digest_matches_v1 = center_digest == ctx["expected_center_digest"]
        row = ctx["rows"][0]
        span = (int(row["instruction_span"][0]), int(row["instruction_span"][1]))
        center = align_grid(calibration["mu_bar"][T.INTERVENTION_LAYER], calibration["ref_span"], span)
        ids = torch.tensor([scoring_ids(tokenizer, row["formatted"])], dtype=torch.long, device=device)
        buy_id, sell_id = answer_token_ids(tokenizer, row["formatted"] + T.DECISION_PREFIX)

        def score(transform: Any | None = None) -> float:
            value = margin_forward(
                model,
                ids,
                {T.INTERVENTION_LAYER: transform} if transform is not None else None,
                buy_id,
                sell_id,
            )
            if not math.isfinite(value):
                raise ValueError("non-finite M6 smoke margin")
            return float(value)

        transforms = {
            "main": subspace_removal_transform(basis, 1.0, center, tuple(range(span[0], span[1]))),
            "random": subspace_removal_transform(random_basis, 1.0, center, tuple(range(span[0], span[1]))),
        }
        clean_margin = score()
        main_margin = score(transforms["main"])
        random_margin = score(transforms["random"])
        clean_decision, clean_parsed, clean_finish = _generate_decision(model, tokenizer, device, row, None)
        main_decision, main_parsed, main_finish = _generate_decision(model, tokenizer, device, row, {T.INTERVENTION_LAYER: transforms["main"]})
        random_decision, random_parsed, random_finish = _generate_decision(model, tokenizer, device, row, {T.INTERVENTION_LAYER: transforms["random"]})
        smoke = {
            "schema_version": M6_SCHEMA_VERSION,
            "artifact_type": "selective_intervention_m6_real_model_smoke",
            "protocol": M6_PROTOCOL,
            "raw_runtime_payloads": False,
            "prompt_id": row["id"],
            "ticker": row["ticker"],
            "center_digest_l15": center_digest,
            "expected_center_digest_l15": ctx["expected_center_digest"],
            "center_digest_matches_v1": center_digest_matches_v1,
            "center_mode": "current_runtime_deterministic_reconstruction",
            "generation_config": {"do_sample": False, "temperature": 0.0},
            "margins": {"clean": clean_margin, "main": main_margin, "random": random_margin},
            "generation": {
                "clean": {"decision": clean_decision, "parsed": clean_parsed, "finish": clean_finish},
                "main": {"decision": main_decision, "parsed": main_parsed, "finish": main_finish},
                "random": {"decision": random_decision, "parsed": random_parsed, "finish": random_finish},
            },
        }
        path = out_dir / "smoke.json"
        write_json(path, smoke, overwrite=True)
        write_metadata(
            out_dir / "metadata.json",
            {"schema_version": M6_SCHEMA_VERSION, "artifact_type": "selective_intervention_m6_smoke_metadata", "n_prompts": 1, "raw_runtime_payloads": False},
            overwrite=True,
        )
        run.manifest.register_artifact(path, artifact_type="selective_intervention_m6_smoke", stage="forward", role="output")
        run.manifest.register_artifact(out_dir / "metadata.json", artifact_type="selective_intervention_m6_smoke_metadata", stage="forward", role="output")


def _arm(records: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    result = [record for record in records if record["arm"] == arm]
    if not result:
        raise ValueError(f"M6 arm missing: {arm}")
    return result


def _analyze(run: ArtifactRun) -> dict[str, Any]:
    out_dir = run.run_directory / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(run.run_directory / "forward" / "records.jsonl")
    meta = json.loads((run.run_directory / "forward" / "metadata.json").read_text(encoding="utf-8"))
    with run.stage("analyze"):
        clean = _arm(records, M6_CLEAN_ARM)
        main = _arm(records, M6_MAIN_ARM)
        random_arm = _arm(records, M6_RANDOM_ARM)
        anon_clean = float(next(row["margin"] for row in records if row["arm"] == M6_ANON_CLEAN_ARM))
        anon_int = float(next(row["margin"] for row in records if row["arm"] == M6_ANON_INT_ARM))
        summary = evaluate_m6(
            clean,
            main,
            random_arm,
            anon_clean=anon_clean,
            anon_intervention=anon_int,
            bootstrap_samples=M6_BOOTSTRAP_SAMPLES,
            bootstrap_seed=M6_BOOTSTRAP_SEED,
        )
        summary["dose_response"] = []
        for alpha in (0.25, 0.5, 0.75, 1.0):
            arm = _arm(records, f"dose_{round(alpha * 100)}")
            per_company = per_ticker_margins(
                {ticker: [float(row["margin"]) for row in arm if row["ticker"] == ticker]
                 for ticker in sorted({row["ticker"] for row in arm})}
            )
            clean_per_company = summary["clean"]["per_company"]
            summary["dose_response"].append(
                {
                    "alpha": alpha,
                    "spread": iqr(list(per_company.values())),
                    "spread_ratio": iqr(list(per_company.values())) / summary["clean"]["spread"],
                    "mean_margin": sum(float(row["margin"]) for row in arm) / len(arm),
                    "generation": generation_flip_stats(clean, arm) if alpha == 1.0 else None,
                    "per_company": per_company,
                    "clean_per_company": clean_per_company,
                }
            )
        summary["generation"] = {
            "main": generation_flip_stats(clean, main),
            "random": generation_flip_stats(clean, random_arm),
        }
        summary["anon_probe"] = {"clean": anon_clean, "intervention": anon_int, "delta": anon_int - anon_clean}
        summary["protocol"] = M6_PROTOCOL
        summary["protocol_rev"] = M6_PROTOCOL_REV
        summary["metadata"] = {
            "center_digest_l15": meta["center_digest_l15"],
            "expected_center_digest_l15": meta["expected_center_digest_l15"],
            "center_digest_matches_v1": meta["center_digest_matches_v1"],
            "center_mode": meta["center_mode"],
            "n_records": meta["n_records"],
        }
        summary["raw_runtime_payloads"] = False
        path = out_dir / "summary.json"
        write_json(path, summary, overwrite=True)
        run.manifest.register_artifact(path, artifact_type="selective_intervention_m6_summary", stage="analyze", role="output")
    return summary


def run_selective_intervention_m6_smoke(
    *,
    model_path: str,
    run_id: str,
    external_manifest: str | Path,
    phase2a_run: str | Path,
    e01_run: str | Path,
    v1_run: str | Path,
    artifact_root: str | Path = "artifacts",
) -> Path:
    """Run a one-prompt real-model M6 preflight without formal analysis."""
    external_manifest = Path(external_manifest)
    phase2a_run = Path(phase2a_run)
    e01_run = Path(e01_run)
    v1_run = Path(v1_run)
    tokenizer = load_tokenizer_for_inference(model_path)
    run = ArtifactRun.create(Path(model_path).name, f"{M6_DATASET}-smoke", run_id, artifact_root=artifact_root)
    try:
        ctx = _prepare(
            run,
            tokenizer,
            external_manifest=external_manifest,
            phase2a_run=phase2a_run,
            e01_run=e01_run,
            v1_run=v1_run,
        )
        _smoke_forward(run, model_path, ctx)
        run.finalize(required_stages={"prepare", "forward"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


def run_selective_intervention_m6(
    *,
    model_path: str,
    run_id: str,
    external_manifest: str | Path,
    phase2a_run: str | Path,
    e01_run: str | Path,
    v1_run: str | Path,
    artifact_root: str | Path = "artifacts",
) -> Path:
    """Run M6 prepare → forward → analyze with a frozen source operator."""
    external_manifest = Path(external_manifest)
    phase2a_run = Path(phase2a_run)
    e01_run = Path(e01_run)
    v1_run = Path(v1_run)
    tokenizer = load_tokenizer_for_inference(model_path)
    run = ArtifactRun.create(Path(model_path).name, M6_DATASET, run_id, artifact_root=artifact_root)
    try:
        ctx = _prepare(
            run,
            tokenizer,
            external_manifest=external_manifest,
            phase2a_run=phase2a_run,
            e01_run=e01_run,
            v1_run=v1_run,
        )
        _forward(run, model_path, ctx)
        _analyze(run)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_directory


__all__ = ["run_selective_intervention_m6", "run_selective_intervention_m6_smoke"]
