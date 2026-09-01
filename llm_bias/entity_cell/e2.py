"""E2 attribution helpers and compact stage runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from llm_bias.core.artifact_paths import count_jsonl_records, sha256_file
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.prompt_input.encoding import format_prompt

from .attention_attribution import (
    FULL_ATTENTION_LAYERS,
    PRIMARY_ATTENTION_LAYERS,
    compact_attribution_record,
    direct_logit_attribution,
    frozen_margin_direction,
    rank_attention_heads,
    reconstruct_captured_attention,
    resolve_single_token_pair,
    routing_label,
    validate_attention_layers,
    capture_attention_forward,
    remove_hooks,
    patch_head_output,
)

E2_STAGES = ("e2-attribution", "e2-patching", "analyze")
E2_READOUT_STAGE = "e2-readout"
E2_ALL_STAGES = (*E2_STAGES[:1], E2_READOUT_STAGE, *E2_STAGES[1:])
DECISION_PREFIX = '{\n  "decision": "'


def _model_layers(model: Any) -> Any:
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("E2 model must expose decoder layers")
    return layers


def _attention(model: Any, layer: int) -> Any:
    from .attention_attribution import _attention_module
    return _attention_module(_model_layers(model)[int(layer)])


def _forward(model: Any, ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> Any:
    kwargs = {} if attention_mask is None else {"attention_mask": attention_mask}
    try:
        return model.forward(ids, **kwargs)
    except TypeError:
        return model.forward(ids)


def _residual_final(model: Any, ids: torch.Tensor) -> torch.Tensor:
    final_layer = int(getattr(model, "n_layers", len(_model_layers(model))) - 1)
    return record_residuals(model, ids, [final_layer])[final_layer][:, -1, :][0]


def _clean_attention_output(model: Any, ids: torch.Tensor, layer: int) -> tuple[Any, Any]:
    attention = _attention(model, layer)
    capture, handles = capture_attention_forward(attention)
    try:
        with torch.no_grad():
            _forward(model, ids)
        if capture.output is None:
            raise ValueError("attention output capture is empty")
        return capture, handles
    except BaseException:
        remove_hooks(handles)
        raise


def run_e2(
    *,
    prepared_dir: str | Path,
    model_name: str,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    stages: Iterable[str] = E2_STAGES,
    layers: Iterable[int] = FULL_ATTENTION_LAYERS,
    max_tickers: int | None = None,
    device: Any | None = None,
    lens_path: str | Path | None = None,
    expected_lens_sha256: str | None = None,
    expected_model_revision: str | None = None,
    expected_tokenizer_identity: str | None = None,
) -> Path:
    """Run E2 from T1 prepared prompts and optional E1 records."""
    enabled = tuple(dict.fromkeys(stages))
    unknown = set(enabled) - set(E2_ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown E2 stages: {sorted(unknown)}")
    selected_layers = validate_attention_layers(layers)
    prepared = Path(prepared_dir)
    financial_path = prepared / "financial_prompts.jsonl"
    metadata_path = prepared / "metadata.json"
    if not financial_path.is_file() or not metadata_path.is_file():
        raise ValueError("prepared_dir must contain T1 financial_prompts.jsonl and metadata.json")
    financial = read_jsonl(financial_path)
    tickers = sorted({str(row["ticker"]) for row in financial})
    if max_tickers is not None:
        if max_tickers < 1:
            raise ValueError("max_tickers must be positive")
        tickers = tickers[:max_tickers]
    financial = [row for row in financial if str(row["ticker"]) in tickers]
    run = ArtifactRun.create(model_name, "entity-cell-localization", run_id, artifact_root=artifact_root)
    output_dir = run.run_directory / "e2"
    try:
        run.manifest.register_artifact(metadata_path, artifact_type="entity_cell_prepare_metadata", stage="prepare", role="input")
        run.manifest.register_artifact(financial_path, artifact_type="entity_cell_financial_prompt", stage="prepare", role="input")
        from llm_bias.core.model import load_model
        model, tokenizer, fallback_device = load_model(model_name)
        target = torch.device(device or getattr(model, "input_device", fallback_device))
        component_vectors: dict[tuple[str, str, int, int], torch.Tensor] = {}
        rows: list[dict[str, Any]] = []
        if "e2-attribution" in enabled:
            with run.stage("e2-attribution") as stage:
                positive, negative = None, None
                for row in financial:
                    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=target)
                    for layer in selected_layers:
                        capture, handles = _clean_attention_output(model, ids, layer)
                        try:
                            reconstruction = reconstruct_captured_attention(_model_layers(model)[layer], capture, row["source_groups"], query_position=int(row["final_query_position"]))
                            if not reconstruction.additive:
                                raise ValueError(f"attention additivity failed at layer {layer}: {reconstruction.max_abs_error:g}")
                            residual = _residual_final(model, ids)
                            if positive is None:
                                positive, negative = resolve_single_token_pair(tokenizer, format_prompt(tokenizer, row["prompt"], use_chat_template=True, enable_thinking=False) + DECISION_PREFIX)
                            direction = frozen_margin_direction(residual, model._final_norm, model._lm_head, positive, negative)
                            dla = direct_logit_attribution(reconstruction, direction, final_norm=model._final_norm, lm_head=model._lm_head, positive_token_id=positive, negative_token_id=negative)
                            for head in range(len(dla)):
                                identity = dla[head]["identity_header"]["frozen_scale_margin"]
                                instruction = dla[head]["instruction_context"]["frozen_scale_margin"]
                                rows.append(compact_attribution_record(ticker=str(row["ticker"]), prompt_id=str(row["prompt_id"]), layer=layer, head=head, dla={head: dla[head]}, additivity=reconstruction, routing=routing_label(identity, instruction)))
                                key = (str(row["ticker"]), str(row["prompt_id"]), int(layer), int(head))
                                component_vectors[key] = torch.stack(tuple(reconstruction.head_group_vectors[head].values())).sum(dim=0).detach()
                        finally:
                            remove_hooks(handles)
                path = output_dir / "head_attribution.jsonl"
                count = write_jsonl(path, rows, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_e2_attribution", stage="e2-attribution", role="output", record_count=count)
                stage.count(count)
        if "e2-readout" in enabled:
            if "e2-attribution" not in enabled:
                raise ValueError("e2-readout requires e2-attribution in the same run")
            from .attention_attribution import rank_attention_heads
            from .readout import readout_selected_component, unavailable_component_readout
            selected = {(int(row["layer"]), int(row["head"])): row for row in rank_attention_heads(rows) if row.get("selection_eligible")}
            with run.stage("e2-readout") as stage:
                loaded_lens = load_validated_lens(model=model, model_name=model_name, lens_path=lens_path, artifact_root=artifact_root, require_complete=True)
                run.manifest.register_artifact(loaded_lens.path, artifact_type="jacobian_lens", stage="e2-readout", role="lens", metadata={"source": loaded_lens.source, "binary_sha256": sha256_file(loaded_lens.path)})
                readout_rows: list[dict[str, Any]] = []
                for key, vector in component_vectors.items():
                    ticker, prompt_id, layer, head = key
                    selection = selected.get((layer, head))
                    if selection is None:
                        continue
                    if layer not in {int(value) for value in loaded_lens.lens.source_layers}:
                        readout_rows.append(unavailable_component_readout(
                            layer=layer,
                            component_identity={"layer": layer, "head": head},
                            prompt_id=prompt_id,
                            ticker=ticker,
                            selection_status=selection,
                            reason=f"canonical lens does not cover source layer L{layer}",
                        ))
                    else:
                        readout_rows.append(readout_selected_component(
                            model=model, tokenizer=tokenizer, model_name=model_name,
                            component_vector=vector, source_layer=layer,
                            component_identity={"layer": layer, "head": head},
                            prompt_id=prompt_id, ticker=ticker,
                            selection_status=selection, loaded_lens=loaded_lens,
                            expected_lens_sha256=expected_lens_sha256,
                            expected_model_revision=expected_model_revision,
                            expected_tokenizer_identity=expected_tokenizer_identity,
                            component_kind="aggregate_head",
                            component_provenance={"selection_rank": selection.get("rank"), "selection_rule": "E2 frozen top-five plus sign consistency"},
                        ))
                readout_path = output_dir / "readout.jsonl"
                readout_count = write_jsonl(readout_path, readout_rows, overwrite=True)
                readout_metadata_path = output_dir / "readout_metadata.json"
                write_metadata(readout_metadata_path, {
                    "schema_version": 1,
                    "artifact_type": "entity_cell_e2_selected_component_readout_metadata",
                    "model": model_name,
                    "lens_path": str(loaded_lens.path),
                    "lens_sha256": sha256_file(loaded_lens.path),
                    "selected_component_kind": "aggregate_head",
                    "top_k": 10,
                    "readout_count": readout_count,
                    "component_vectors_persisted": False,
                    "transported_vectors_persisted": False,
                    "full_vocabulary_persisted": False,
                }, overwrite=True)
                run.manifest.register_artifact(readout_path, artifact_type="entity_cell_e2_selected_component_readout", stage="e2-readout", role="output", record_count=readout_count)
                run.manifest.register_artifact(readout_metadata_path, artifact_type="entity_cell_e2_selected_component_readout_metadata", stage="e2-readout", role="output")
                stage.count(readout_count)
        if "e2-patching" in enabled:
            with run.stage("e2-patching") as stage:
                path = output_dir / "patching.jsonl"
                # Patching is intentionally exposed as a separate API. A full
                # population patch run requires matched identity contracts that
                # T1 does not materialize, so this stage emits no fabricated rows.
                count = write_jsonl(path, [], overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_e2_head_patch", stage="e2-patching", role="output", record_count=count)
                stage.count(count)
        if "analyze" in enabled:
            with run.stage("analyze") as stage:
                source = output_dir / "head_attribution.jsonl"
                if not source.is_file():
                    raise ValueError("analyze requires e2-attribution output")
                records = read_jsonl(source)
                summary = {"schema_version": 1, "artifact_type": "entity_cell_e2_analysis", "layers": list(selected_layers), "primary_layers": list(PRIMARY_ATTENTION_LAYERS), "head_ranking": rank_attention_heads(records), "readout": {"path": "e2/readout.jsonl", "record_count": count_jsonl_records(output_dir / "readout.jsonl") if (output_dir / "readout.jsonl").is_file() else 0, "component_kind": "aggregate_head", "raw_runtime_payloads": False}, "raw_runtime_payloads": False}
                path = run.run_directory / "analyze" / "summary.json"
                write_json(path, summary, overwrite=True)
                run.manifest.register_artifact(path, artifact_type="entity_cell_e2_analysis", stage="analyze", role="output")
                stage.count(len(summary["head_ranking"]))
        run.finalize(required_stages=set(enabled))
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


def analyze_e2(run_root: str | Path) -> Path:
    root = Path(run_root)
    records = read_jsonl(root / "e2" / "head_attribution.jsonl")
    readout_path = root / "e2" / "readout.jsonl"
    summary = {"schema_version": 1, "artifact_type": "entity_cell_e2_analysis", "head_ranking": rank_attention_heads(records), "readout": {"path": "e2/readout.jsonl", "record_count": count_jsonl_records(readout_path) if readout_path.is_file() else 0, "component_kind": "aggregate_head", "raw_runtime_payloads": False}, "raw_runtime_payloads": False}
    path = root / "analyze" / "summary.json"
    write_json(path, summary, overwrite=True)
    return path


__all__ = ["E2_STAGES", "E2_READOUT_STAGE", "E2_ALL_STAGES", "FULL_ATTENTION_LAYERS", "PRIMARY_ATTENTION_LAYERS", "run_e2", "analyze_e2", "patch_head_output"]
