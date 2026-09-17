"""Phase 1 prepare, forward, and analyze lifecycle."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.artifact_paths import sha256_file
from llm_bias.core.inference.generation import GenerationConfig
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt

from .analysis import descriptive_stats, evaluate_gates, group_rows
from .population import DEFAULT_DATA_PATH, SEED, _stratified_sample, load_population, make_splits, population_digest, split_counts
from .screening import screen_prompt
from .template import CONDITIONS, build_prompt_record

DATASET = "evidence-insensitivity"
MODEL_SLUG = "qwen3.5-4b"


def _model_identity(model_path: str) -> str:
    return Path(model_path).name


def _prepare_rows(population: list[dict[str, str]], tokenizer: Any, *, pilot: bool = False, use_chat_template: bool = True, model_slug: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments, order_swap = make_splits(population)
    selected = population
    if pilot:
        chosen_tickers = sorted(_stratified_sample(population, 50, SEED))
        selected = [row for row in population if row["ticker"] in chosen_tickers]
    rows: list[dict[str, Any]] = []
    for company in selected:
        ticker = company["ticker"]
        condition_list = ("zero", "N6", "P6", "N15", "P15") if pilot else CONDITIONS
        for condition in condition_list:
            rows.append(build_prompt_record(tokenizer, ticker, company["company_name"], company["gics_sector"], condition, split=assignments[ticker], format_fn=format_prompt if use_chat_template else None))
    if not pilot:
        for company in population:
            if company["ticker"] not in order_swap: continue
            for condition in CONDITIONS[1:]:
                rows.append(build_prompt_record(tokenizer, company["ticker"], company["company_name"], company["gics_sector"], condition, split=assignments[company["ticker"]], arm="order_swap", reverse=True, format_fn=format_prompt if use_chat_template else None))
        for condition in CONDITIONS:
            rows.append(build_prompt_record(tokenizer, "TICKER", "Company X", "anonymous", condition, split="control", arm="anon", anonymous=True, format_fn=format_prompt if use_chat_template else None))
    else:
        anon_conditions = CONDITIONS
        for condition in anon_conditions:
            rows.append(build_prompt_record(tokenizer, "TICKER", "Company X", "anonymous", condition, split="control", arm="anon", anonymous=True, format_fn=format_prompt if use_chat_template else None))
    expected = 50 * 5 + 9 if pilot else 503 * 9 + 100 * 8 + 9
    if len(rows) != expected:
        raise ValueError(f"prepared {len(rows)} prompts, expected {expected}")
    provenance = {"population_digest": population_digest(population), "evidence_family": "phase1-frozen", "seed": SEED, "split_counts": split_counts(assignments), "order_swap_count": len(order_swap), "model": model_slug or MODEL_SLUG, "pilot": pilot, "status": "pilot" if pilot else "formal"}
    return rows, provenance


def prepare_stage(run: ArtifactRun, *, data_path: str | Path = DEFAULT_DATA_PATH, model_path: str = MODEL_SLUG, pilot: bool = False, tokenizer: Any | None = None, use_chat_template: bool = True, model_slug: str | None = None) -> Path:
    population = load_population(data_path)
    tokenizer = tokenizer or load_tokenizer(model_path)
    rows, provenance = _prepare_rows(population, tokenizer, pilot=pilot, use_chat_template=use_chat_template, model_slug=model_slug)
    output = run.run_directory / "prepare"
    output.mkdir(parents=True, exist_ok=True)
    with run.stage("prepare") as stage:
        prompts_path = output / "prompts.jsonl"
        write_jsonl(prompts_path, rows, overwrite=True)
        provenance["evidence_sha256"] = hashlib.sha256("".join(sorted(row["evidence_sha"] for row in rows)).encode()).hexdigest()
        provenance_path = write_json(output / "provenance.json", provenance, overwrite=True)
        run.manifest.register_artifact(prompts_path, artifact_type="evidence_insensitivity_prompts", stage="prepare", record_count=len(rows))
        run.manifest.register_artifact(provenance_path, artifact_type="evidence_insensitivity_provenance", stage="prepare")
        stage.count(len(rows))
    return prompts_path


def forward_stage(run: ArtifactRun, *, model_path: str = MODEL_SLUG, model: Any | None = None, tokenizer: Any | None = None, device: Any = "cpu") -> Path:
    prompts = read_jsonl(run.run_directory / "prepare" / "prompts.jsonl")
    if model is None or tokenizer is None:
        model, tokenizer, device = load_model(model_path, dtype=None)
    output = run.run_directory / "forward"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt in prompts:
        result = screen_prompt(model, tokenizer, prompt["prompt_text"], device=device)
        rows.append({"prompt_id": prompt["prompt_id"], "arm": prompt["arm"], "condition": prompt["condition"], "margin": result["margin"], "decision": result["decision"], "generated_text": result["generated_text"], "n_new_tokens": result["n_new_tokens"]})
    path = output / "records.jsonl"
    write_jsonl(path, rows, overwrite=True)
    rng = random.Random(SEED)
    indices = rng.sample(range(len(prompts)), min(20, len(prompts)))
    rerun = [screen_prompt(model, tokenizer, prompts[index]["prompt_text"], device=device) for index in indices]
    mismatches = sum(rows[index]["generated_text"] != check["generated_text"] for index, check in zip(indices, rerun))
    max_delta = max((abs(float(rows[index]["margin"]) - float(check["margin"])) for index, check in zip(indices, rerun)), default=0.0)
    metadata = {"generation": {"strategy": "greedy", "max_new_tokens": 128, "temperature": 0.0}, "model": _model_identity(model_path), "tokenizer": _model_identity(model_path), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_forwards": len(rows), "scoring_modes": sorted({check["scoring_mode"] for check in rerun}), "determinism_check": {"n_prompts": len(indices), "max_abs_delta_margin": float(max_delta), "text_mismatch_count": mismatches}, "status": "pilot" if len(prompts) == 259 else "formal"}
    metadata_path = write_metadata(output / "metadata.json", metadata, overwrite=True)
    with run.stage("forward") as stage:
        run.manifest.register_artifact(path, artifact_type="evidence_insensitivity_records", stage="forward", record_count=len(rows))
        run.manifest.register_artifact(metadata_path, artifact_type="evidence_insensitivity_forward_metadata", stage="forward")
        stage.count(len(rows))
    return path


def analyze_stage(run: ArtifactRun) -> Path:
    records = read_jsonl(run.run_directory / "forward" / "records.jsonl")
    prepared = {row["prompt_id"]: row for row in read_jsonl(run.run_directory / "prepare" / "prompts.jsonl")}
    enriched = [{**prepared[row["prompt_id"]], **row} for row in records]
    groups = group_rows(enriched)
    metadata = json.loads((run.run_directory / "forward" / "metadata.json").read_text(encoding="utf-8"))
    gate = evaluate_gates(enriched, groups, metadata["determinism_check"])
    summary = {"schema_version": "evidence-insensitivity-phase1-v1", "status": metadata.get("status", "formal"), "n_prompts": len(records), "gates": gate["gates"], "fallback": gate["fallback"], "groups": groups, "descriptive_stats": descriptive_stats(enriched, groups), "raw_runtime_payloads": False}
    output = run.run_directory / "analyze"
    output.mkdir(parents=True, exist_ok=True)
    path = write_json(output / "summary.json", summary, overwrite=True)
    with run.stage("analyze") as stage:
        run.manifest.register_artifact(path, artifact_type="evidence_insensitivity_summary", stage="analyze")
        stage.count(len(groups))
    return path


def create_run(run_id: str, *, artifact_root: str | Path = "artifacts", model_slug: str | None = None) -> ArtifactRun:
    return ArtifactRun.create(model_slug or MODEL_SLUG, DATASET, run_id, artifact_root=artifact_root)


def run_prepare(run_id: str, *, artifact_root: str | Path = "artifacts", data_path: str | Path = DEFAULT_DATA_PATH, model_path: str = MODEL_SLUG, pilot: bool = False, tokenizer: Any | None = None, use_chat_template: bool = True, model_slug: str | None = None) -> Path:
    run = create_run(run_id, artifact_root=artifact_root, model_slug=model_slug)
    try:
        prepare_stage(run, data_path=data_path, model_path=model_path, pilot=pilot, tokenizer=tokenizer, use_chat_template=use_chat_template, model_slug=model_slug)
        return run.run_directory
    except BaseException as exc: run.fail(exc); raise


def run_forward(run_id: str, *, artifact_root: str | Path = "artifacts", model_path: str = MODEL_SLUG, model: Any | None = None, tokenizer: Any | None = None, device: Any = "cpu", model_slug: str | None = None) -> Path:
    run = ArtifactRun.open(Path(artifact_root) / (model_slug or MODEL_SLUG) / DATASET / "runs" / run_id / "manifest.json")
    try: return forward_stage(run, model_path=model_path, model=model, tokenizer=tokenizer, device=device)
    except BaseException as exc: run.fail(exc); raise


def run_pilot(run_id: str, *, artifact_root: str | Path = "artifacts", model_path: str = MODEL_SLUG, data_path: str | Path = DEFAULT_DATA_PATH, model: Any | None = None, tokenizer: Any | None = None, device: Any = "cpu", use_chat_template: bool = True, model_slug: str | None = None) -> Path:
    run = create_run(run_id, artifact_root=artifact_root, model_slug=model_slug)
    try:
        prepare_stage(run, data_path=data_path, model_path=model_path, pilot=True, tokenizer=tokenizer, use_chat_template=use_chat_template, model_slug=model_slug)
        forward_stage(run, model_path=model_path, model=model, tokenizer=tokenizer, device=device)
        analyze_stage(run)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def run_analyze(run_id: str, *, artifact_root: str | Path = "artifacts", model_slug: str | None = None) -> Path:
    run = ArtifactRun.open(Path(artifact_root) / (model_slug or MODEL_SLUG) / DATASET / "runs" / run_id / "manifest.json")
    try:
        result = analyze_stage(run)
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return result
    except BaseException as exc: run.fail(exc); raise
