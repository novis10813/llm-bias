"""Baseline trial-plan prompt execution.

Runs the converted baseline CSV through the existing prompt-analysis pipeline
(``prepare -> forward -> analyze -> finalize``) with a baseline-specific
extension: a ``lens-forward`` stage that persists per-layer Jacobian-lens
next-token readouts over the generated (prompt + generated) sequence.

The forward stage is ``prompt-analysis generate``; the lens-forward stage
consumes the exact persisted forward artifact; the backward stage is
``prompt-analysis attribute-generated``.  All stage artifacts are compact
records (top-k, ranks, probabilities, token IDs/text, provenance); raw
activations are never persisted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.model import WrappedModel

from llm_bias.core.artifact_manifest import RunManifest
from llm_bias.core.artifact_paths import (
    dataset_slug,
    model_slug,
    run_root,
    sha256_file,
    stable_record_id,
)
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model as load_lens_model
from llm_bias.core.prompt_input import decode_token
from llm_bias.core.inference.generation import finish_reason as _core_finish_reason
from llm_bias.prompt_analysis.artifact_io import (
    ARTIFACT_SCHEMA_VERSION,
    read_jsonl as read_jsonl_records,
    sidecar_metadata,
)

DEFAULT_INPUT = "data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv"
DEFAULT_MODEL = ".cache/models/qwen3.6-27b"
LENS_FORWARD_ARTIFACT_TYPE = "generated_sequence_lens_readout"
STAGES = ("readout", "forward", "lens-forward", "backward", "validate")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json_line(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


@dataclass
class BaselineRunConfig:
    """Resolved, validated inputs for one baseline trial run."""

    input: Path
    model: str
    lens: Path
    dataset: str
    run_id: str
    artifact_root: Path
    run_root: Path
    stages: tuple[str, ...]
    max_seq_len: int
    batch_size: int
    top_k: int
    max_rows: int | None
    prompt_columns: tuple[str, ...] | None
    generate_full: bool
    sample_per_condition: int
    max_new_tokens: int
    backward_input_top_k: int | None
    backward_output_token_top_k: int | None
    forward_artifact: Path | None
    validate_seed: int
    lens_forward_layers: list[int] | None = None
    device_map: str | None = None

    @classmethod
    def resolve(
        cls,
        *,
        input_path: Path,
        model: str,
        lens: Path | None,
        dataset: str | None,
        run_id: str,
        artifact_root: Path,
        stages: tuple[str, ...],
        max_seq_len: int,
        batch_size: int,
        top_k: int,
        max_rows: int | None,
        prompt_columns: Iterable[str] | None,
        generate_full: bool,
        sample_per_condition: int,
        max_new_tokens: int,
        backward_input_top_k: int | None,
        backward_output_token_top_k: int | None,
        forward_artifact: Path | None,
        validate_seed: int,
        lens_forward_layers: list[int] | None = None,
        device_map: str | None = None,
    ) -> "BaselineRunConfig":
        if not input_path.is_file():
            raise FileNotFoundError(f"baseline trial-plan CSV not found: {input_path}")
        if not model:
            raise ValueError("model is required")
        if not run_id.strip() or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be one non-empty directory name")
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            raise ValueError(f"unknown baseline stages: {', '.join(unknown)}")
        if not stages:
            raise ValueError("at least one stage is required")
        if "forward" in stages and "lens-forward" in stages and not generate_full:
            raise ValueError(
                "lens-forward requires the full forward artifact; pass "
                "--generate-full (baseline trial plans have no per-date sampling)"
            )
        if "backward" in stages and "lens-forward" not in stages:
            raise ValueError("backward requires lens-forward (it persists the forward artifact)")
        if "validate" in stages and "backward" not in stages:
            raise ValueError("validate requires backward")
        if lens is None and any(stage in stages for stage in ("readout", "lens-forward")):
            raise ValueError("lens is required for readout and lens-forward stages")
        if "forward" not in stages:
            if forward_artifact is None:
                raise FileNotFoundError(
                    "forward artifact not found; pass --forward-artifact or run the forward stage"
                )
            if not forward_artifact.is_file():
                raise FileNotFoundError(f"forward artifact not found: {forward_artifact}")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be positive when provided")
        if not generate_full and sample_per_condition < 1:
            raise ValueError("sample_per_condition must be positive when provided")
        if backward_input_top_k is not None and backward_input_top_k < 1:
            raise ValueError("backward_input_top_k must be positive when provided")
        if backward_output_token_top_k is not None and backward_output_token_top_k < 1:
            raise ValueError("backward_output_token_top_k must be positive when provided")
        if validate_seed < 0:
            raise ValueError("validate_seed must be non-negative")

        dataset_name = dataset or input_path.stem
        return cls(
            input=input_path,
            model=model,
            lens=Path(lens) if lens is not None else Path("lens/required.pt"),
            dataset=dataset_name,
            run_id=run_id,
            artifact_root=artifact_root,
            run_root=run_root(model, dataset_name, run_id, artifact_root=artifact_root),
            stages=stages,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            top_k=top_k,
            max_rows=max_rows,
            prompt_columns=tuple(prompt_columns) if prompt_columns else None,
            generate_full=generate_full,
            sample_per_condition=sample_per_condition,
            max_new_tokens=max_new_tokens,
            backward_input_top_k=backward_input_top_k,
            backward_output_token_top_k=backward_output_token_top_k,
            forward_artifact=forward_artifact,
            validate_seed=validate_seed,
            lens_forward_layers=lens_forward_layers,
            device_map=device_map,
        )


def _stage_outputs(stage: str, run_root: Path) -> list[Path]:
    if stage == "readout":
        readout_dir = run_root / "readout"
        return [
            readout_dir / "prompt_layer_topk.jsonl",
            readout_dir / "prompt_layer_uncertainty.jsonl",
            readout_dir / "average_layer_topk.jsonl",
            readout_dir / "average_layer_topk.csv",
            readout_dir / "output_topk_distribution.png",
            readout_dir / "metadata.json",
        ]
    if stage == "forward":
        return [
            run_root / "forward" / "generated_outputs.jsonl",
            run_root / "forward" / "metadata.json",
        ]
    if stage == "lens-forward":
        return [
            run_root / "lens-forward" / "lens_readout.jsonl",
            run_root / "lens-forward" / "metadata.json",
        ]
    if stage == "backward":
        return [
            run_root / "backward" / "generated_token_attribution.jsonl",
            run_root / "backward" / "metadata.json",
        ]
    if stage == "validate":
        return [run_root / "attribution_validation" / "metadata.json"]
    raise ValueError(f"unknown stage: {stage}")


def _artifact_type_for(path: Path, stage: str) -> str:
    if path.suffix != ".jsonl":
        return f"{stage}_metadata"
    names = {
        "prompt_layer_topk.jsonl": "prompt_layer_topk",
        "prompt_layer_uncertainty.jsonl": "prompt_layer_uncertainty",
        "average_layer_topk.jsonl": "average_layer_topk",
        "generated_outputs.jsonl": "generated_outputs",
        "lens_readout.jsonl": LENS_FORWARD_ARTIFACT_TYPE,
        "generated_token_attribution.jsonl": "generated_token_attribution",
    }
    if path.name not in names:
        raise ValueError(f"no artifact type registered for {path.name} in stage {stage}")
    return names[path.name]


def _register_stage_outputs(
    manifest: RunManifest, stage: str, run_root: Path
) -> int | None:
    records = 0
    for path in _stage_outputs(stage, run_root):
        if not path.is_file():
            continue
        manifest.register_artifact(
            path,
            artifact_type=_artifact_type_for(path, stage),
            stage=stage,
            role="output",
            metadata={"provenance": f"baseline-trial {stage} stage output"},
        )
        records += 1
    return records or None


def _manifest_stages(config: BaselineRunConfig) -> dict[str, str]:
    """Map enabled stage names onto the manifest stage set."""
    return dict.fromkeys(config.stages)


def _load_model_and_lens(config: BaselineRunConfig):
    lens_model, tokenizer, _device = load_lens_model(config.model, device_map=config.device_map)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    loaded_lens = None
    lens_source = None
    if config.lens is not None and config.lens.is_file():
        loaded_lens = load_validated_lens(
            model=model,
            model_name=config.model,
            lens_path=config.lens,
            require_complete=True,
        )
        lens_source = loaded_lens.path
    return lens_model, model, tokenizer, loaded_lens, lens_source


def _run_forward(config: BaselineRunConfig, model: WrappedModel | None = None) -> Path:
    from llm_bias.prompt_analysis.generation import generate_prompt_outputs

    if model is not None:
        return _forward_with_model(config, model)
    output = generate_prompt_outputs(
        input_path=str(config.input),
        model_name=config.model,
        output_path=str(config.run_root / "forward" / "generated_outputs.jsonl"),
        sample_per_condition=None if config.generate_full else config.sample_per_condition,
        full_generation=config.generate_full,
        max_new_tokens=config.max_new_tokens,
        max_seq_len=config.max_seq_len,
        dataset_format="auto",
    )
    return Path(output)


def _run_backward(config: BaselineRunConfig, forward_artifact: Path) -> Path:
    if config.device_map is not None:
        raise ValueError(
            "backward stage does not support sharded loading; run --device-map "
            "qwen27b_two_gpu with --stage lens-forward --forward-artifact <forward.jsonl> "
            "after producing the forward artifact (sharded or single-GPU)"
        )
    from llm_bias.prompt_analysis.generated_attribution import attribute_generated_outputs

    return Path(
        attribute_generated_outputs(
            forward_artifact=str(forward_artifact),
            model_name=config.model,
            output_path=str(config.run_root / "backward" / "generated_token_attribution.jsonl"),
            input_top_k=config.backward_input_top_k,
            output_token_top_k=config.backward_output_token_top_k,
            max_seq_len=config.max_seq_len,
        )
    )


def _run_validation(config: BaselineRunConfig) -> Path:
    if config.device_map is not None:
        raise ValueError(
            "validate stage does not support sharded loading; run --device-map "
            "qwen27b_two_gpu with --stage lens-forward --forward-artifact <forward.jsonl> "
            "after producing the forward artifact (sharded or single-GPU)"
        )
    from llm_bias.prompt_analysis.validation import evaluate_semantic_scope

    return Path(
        evaluate_semantic_scope(
            attribution_path=str(config.run_root / "backward" / "generated_token_attribution.jsonl"),
            model_name=config.model,
            output_dir=str(config.run_root / "attribution_validation"),
            seed=config.validate_seed,
            max_seq_len=config.max_seq_len,
        )
    )


def _forward_record_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Authoritative per-record identity of a forward artifact.

    Combines the row position (records may legitimately share a ``record_id``:
    the same date/ticker/prompt-column repeats across trial occurrences) with
    the persisted ``record_id`` or a compact identity digest fallback.
    """
    ids: list[str] = []
    for row_index, row in enumerate(rows):
        record_id = row.get("record_id")
        if not (isinstance(record_id, str) and record_id.strip()):
            record_id = stable_record_id(
                {
                    "prompt_token_ids": row.get("prompt_token_ids"),
                    "generated_token_ids": row.get("generated_token_ids"),
                    "prompt_column": row.get("prompt_column"),
                    "date": row.get("date"),
                }
            )
        ids.append(stable_record_id(row_index, record_id))
    if len(ids) != len(set(ids)):
        raise ValueError("forward artifact contains duplicate record identities")
    return ids


@torch.no_grad()
def _lens_forward(
    config: BaselineRunConfig,
    forward_artifact: Path,
    model: WrappedModel,
    lens: Any,
    lens_source: Path,
) -> Path:
    """Per-layer Jacobian-lens readout over the generated sequence.

    For each forward record, runs one model forward over the full
    (prompt + generated) sequence, then at every generated position reads the
    residual at each lens source layer, transports it to the final-layer
    basis with the lens Jacobian, and records the compact next-token
    distribution (top-k, entropy, effective temperature).
    """
    from llm_bias.core.artifacts.io import read_jsonl, write_jsonl, write_metadata

    rows = read_jsonl(forward_artifact)
    if not rows:
        raise ValueError(f"forward artifact has no records: {forward_artifact}")
    parent_hash = sha256_file(forward_artifact)
    forward_metadata = sidecar_metadata(forward_artifact)
    if forward_metadata is None:
        raise ValueError("forward artifact metadata is required")
    forward_model = forward_metadata.get("model")
    if not isinstance(forward_model, str) or not forward_model.strip():
        raise ValueError("forward artifact metadata is missing model identity")
    if model_slug(forward_model) != model_slug(config.model):
        raise ValueError(
            "forward artifact model does not match requested model: "
            f"{forward_model!r} != {config.model!r}"
        )

    destination = config.run_root / "lens-forward"
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "lens_readout.jsonl"

    final_layer = model.n_layers - 1
    requested = (
        config.lens_forward_layers
        if config.lens_forward_layers is not None
        else list(lens.source_layers) + [final_layer]
    )
    layers = sorted(set(requested))
    invalid = [layer for layer in layers if layer < 0 or layer > final_layer]
    if invalid:
        raise ValueError(f"lens-forward layers out of range 0..{final_layer}: {invalid}")
    tokenizer = model.tokenizer
    top_k = min(config.top_k, model._lm_head.weight.shape[0])
    log_vocab = math.log(max(model._lm_head.weight.shape[0], 2))
    # Keep the Jacobians resident on the unembed (lm_head) device: otherwise
    # transport() copies the d x d matrix for every (position, layer) readout.
    # Single-GPU runs: same device as before. Sharded runs: the lm_head sits
    # on the last shard, so the lens follows it (27B: GPU 1).
    head_device = model._lm_head.weight.device
    lens.jacobians = {
        layer: jacobian.to(head_device)
        for layer, jacobian in lens.jacobians.items()
    }
    token_cache: dict[int, str] = {}
    records: list[dict[str, Any]] = []
    total_positions = 0
    for parent_index, parent in enumerate(rows, start=1):
        prompt_ids = parent.get("prompt_token_ids")
        generated_ids = parent.get("generated_token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(generated_ids, list):
            raise ValueError("forward record is missing prompt/generated token IDs")
        sequence_ids = [int(token) for token in [*prompt_ids, *generated_ids]]
        if not sequence_ids:
            raise ValueError("forward record has an empty sequence")
        if len(sequence_ids) > config.max_seq_len:
            raise ValueError(
                f"forward record sequence length {len(sequence_ids)} exceeds max_seq_len {config.max_seq_len}"
            )
        input_ids = torch.tensor([sequence_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)

        with ActivationRecorder(model.layers, at=layers) as recorder:
            model._decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            activations = {layer: recorder.activations[layer].detach() for layer in layers}

        generated_token_count = len(generated_ids)
        prompt_length = len(prompt_ids)
        positions = _lens_batch_readout(
            activations=activations,
            layer_indices=layers,
            residual_start=prompt_length,
            residual_count=generated_token_count,
            lens=lens,
            model=model,
            final_layer=final_layer,
            tokenizer=tokenizer,
            top_k=top_k,
            log_vocab=log_vocab,
            token_cache=token_cache,
        )
        total_positions += generated_token_count
        record = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": LENS_FORWARD_ARTIFACT_TYPE,
            "record_id": parent.get("record_id"),
            "date": parent.get("date", ""),
            "ticker": parent.get("ticker"),
            "prompt_column": parent.get("prompt_column"),
            "index": parent.get("index"),
            "context": parent.get("context"),
            "condition": parent.get("condition"),
            "generated_text": parent.get("generated_text"),
            "generated_token_ids": [int(token) for token in generated_ids],
            "sequence_length": len(sequence_ids),
            "parent_forward_sha256": parent_hash,
            "positions": positions,
        }
        records.append(record)
        if len(records) % 25 == 0 or parent_index == len(rows):
            print(f"lens-forward: {len(records)}/{len(rows)} records", flush=True)
        del activations

    write_jsonl(output_path, records, overwrite=True)
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": LENS_FORWARD_ARTIFACT_TYPE,
        "model": config.model,
        "model_slug": model_slug(config.model),
        "lens": str(lens_source),
        "layers": layers,
        "top_k": top_k,
        "max_seq_len": config.max_seq_len,
        "output_position": "every_generated_position",
        "readout_method": (
            "jacobian-lens transported final-normalized residual, full-vocabulary "
            "softmax, per-position top-k, entropy, and effective temperature"
        ),
        "parent_forward_path": str(forward_artifact),
        "parent_forward_sha256": parent_hash,
        "output_sha256": sha256_file(output_path),
        "records_written": len(records),
        "lens_positions_read": total_positions,
        "generated_token_count": total_positions,
        "backpropagation": False,
    }
    write_metadata(destination / "metadata.json", metadata, overwrite=True)
    print(f"Wrote lens-forward J-space readout to {destination}", flush=True)
    return output_path


def _lens_batch_readout(
    *,
    activations: dict[int, torch.Tensor],
    layer_indices: list[int],
    residual_start: int,
    residual_count: int,
    lens: Any,
    model: WrappedModel,
    final_layer: int,
    tokenizer: Any,
    top_k: int,
    log_vocab: float,
    token_cache: dict[int, str],
) -> list[dict[str, Any]]:
    """Vectorized per-layer J-space readout over a contiguous residual span.

    One batched transport/unembed/softmax/topk per layer replaces the
    per-position scalar path; produces the same per-position record layout.
    """
    per_layer: dict[int, tuple[list, list, list, list]] = {}
    head_device = model._lm_head.weight.device
    for layer_index in layer_indices:
        residual = activations[layer_index][
            0, residual_start : residual_start + residual_count
        ].float()
        if layer_index != final_layer and layer_index in lens.jacobians:
            # Hop the small residual batch (not the Jacobian) to the lens/
            # head device so sharded runs transport on the lm_head GPU.
            residual = lens.transport(residual.to(head_device), layer_index)
        else:
            residual = residual.to(head_device)
        logits = model.unembed(residual).float()
        probabilities = logits.softmax(dim=-1)
        entropy = (-(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)).cpu().tolist()
        normalized_hidden = model._final_norm(
            residual.to(model._lm_head.weight.dtype)
        ).float()
        inverse_temperature = normalized_hidden.norm(dim=-1).cpu().tolist()
        top = probabilities.topk(top_k, dim=-1)
        token_ids = top.indices.cpu().tolist()
        token_values = top.values.cpu().tolist()
        per_layer[layer_index] = (token_ids, token_values, entropy, inverse_temperature)
    positions: list[dict[str, Any]] = []
    for position in range(residual_count):
        positions.append(
            {
                "position": position,
                "layers": [
                    {
                        "layer": layer_index,
                        "is_output": layer_index == final_layer,
                        "top_tokens": [
                            {
                                "rank": rank,
                                "token_id": int(token_id),
                                "token": token_cache.setdefault(
                                    int(token_id), decode_token(tokenizer, int(token_id))
                                ),
                                "probability": float(probability),
                            }
                            for rank, (token_id, probability) in enumerate(
                                zip(ids[position], values[position], strict=True), start=1
                            )
                        ],
                        "entropy_nats": ent[position],
                        "normalized_entropy": ent[position] / log_vocab,
                        "effective_inverse_temperature": inv[position],
                        "effective_temperature": 1.0 / max(inv[position], 1e-12),
                    }
                    for layer_index, (ids, values, ent, inv) in per_layer.items()
                ],
            }
        )
    return positions


def run_baseline_trial(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    model: str = DEFAULT_MODEL,
    lens: str | Path | None = None,
    dataset: str | None = None,
    run_id: str,
    artifact_root: str | Path = "artifacts",
    stages: tuple[str, ...] = STAGES,
    max_seq_len: int = 1024,
    batch_size: int = 8,
    top_k: int = 15,
    max_rows: int | None = None,
    prompt_columns: Iterable[str] | None = None,
    generate_full: bool = True,
    sample_per_condition: int = 32,
    max_new_tokens: int = 256,
    backward_input_top_k: int | None = None,
    backward_output_token_top_k: int | None = None,
    forward_artifact: str | Path | None = None,
    validate_seed: int = 0,
    lens_forward_layers: list[int] | None = None,
    device_map: str | None = None,
) -> Path:
    """Execute the enabled baseline stages and persist one canonical run tree.

    Returns the run root: ``artifacts/<model-slug>/<dataset-slug>/runs/<run-id>``.
    """
    config = BaselineRunConfig.resolve(
        input_path=Path(input_path),
        model=model,
        lens=Path(lens) if lens is not None else None,
        dataset=dataset,
        run_id=run_id,
        artifact_root=Path(artifact_root),
        stages=stages,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        top_k=top_k,
        max_rows=max_rows,
        prompt_columns=prompt_columns,
        generate_full=generate_full,
        sample_per_condition=sample_per_condition,
        max_new_tokens=max_new_tokens,
        backward_input_top_k=backward_input_top_k,
        backward_output_token_top_k=backward_output_token_top_k,
        forward_artifact=Path(forward_artifact) if forward_artifact is not None else None,
        validate_seed=validate_seed,
        lens_forward_layers=lens_forward_layers,
        device_map=device_map,
    )

    manifest = RunManifest(
        model=config.model,
        dataset=config.dataset,
        run_id=config.run_id,
        run_directory=config.run_root,
    )
    manifest.register_artifact(
        config.input,
        artifact_type="prompt_input",
        stage="prepare",
        role="input",
        metadata={
            "dataset_format": "legacy-wide",
            "provenance": "converted baseline trial-plan CSV",
        },
    )
    if config.lens is not None and config.lens.is_file():
        manifest.register_artifact(
            config.lens,
            artifact_type="jacobian_lens",
            stage="prepare",
            role="lens",
            metadata={"provenance": "configured canonical Jacobian lens"},
        )
    for name in _manifest_stages(config):
        manifest.stages[name] = {"status": "created"}
    config.run_root.mkdir(parents=True, exist_ok=True)
    manifest.save()
    if config.lens is not None and config.lens.is_file():
        manifest.stages["prepare"] = {
            **manifest.stages.get("prepare", {}),
            "lens_sha256": sha256_file(config.lens),
            "started_at": _utc_now(),
        }
        manifest.save()

    _lens_model, model, tokenizer, loaded_lens, lens_source = _load_model_and_lens(config)
    lens = loaded_lens.lens if loaded_lens is not None else None
    forward_path: Path | None = None
    current_stage: str | None = None
    try:
        manifest.start().save()
        if "readout" in config.stages:
            current_stage = "readout"
            manifest.start_stage("readout").save()
            _run_readout(config)
            manifest.finish_stage("readout", record_count=_register_stage_outputs(manifest, "readout", config.run_root)).save()
        if "forward" in config.stages:
            current_stage = "forward"
            manifest.start_stage("forward").save()
            forward_path = _run_forward(config, model=model)
            manifest.finish_stage("forward", record_count=_register_stage_outputs(manifest, "forward", config.run_root)).save()
        if "lens-forward" in config.stages:
            if forward_path is None:
                forward_path = config.forward_artifact
            record_ids = _forward_record_ids(read_jsonl_records(forward_path))
            current_stage = "lens_forward"
            manifest.start_stage("lens_forward").save()
            _lens_forward(config, forward_path, model, lens, lens_source)
            manifest.finish_stage(
                "lens_forward",
                record_count=_register_stage_outputs(manifest, "lens-forward", config.run_root),
            ).save()
            manifest.stages["lens_forward"]["forward_record_ids"] = record_ids
        if "backward" in config.stages:
            current_stage = "attribution"
            manifest.start_stage("attribution").save()
            _run_backward(config, forward_path)
            manifest.finish_stage("attribution", record_count=_register_stage_outputs(manifest, "backward", config.run_root)).save()
        if "validate" in config.stages:
            current_stage = "validation"
            manifest.start_stage("validation").save()
            _run_validation(config)
            manifest.finish_stage("validation", record_count=_register_stage_outputs(manifest, "validate", config.run_root)).save()
        manifest.complete().save()
    except Exception:
        if current_stage is not None:
            manifest.finish_stage(current_stage, status="failed")
        manifest.fail(f"baseline-trial run {config.run_id} failed").save()
        raise
    print(f"Baseline trial run complete: {config.run_root}", flush=True)
    return config.run_root


def _forward_with_model(config: BaselineRunConfig, model: WrappedModel) -> Path:
    """Single-process forward generation, reusing an already loaded model."""
    from llm_bias.core.artifact_paths import atomic_write_json
    from llm_bias.core.prompt_input import find_token_subsequence
    from llm_bias.prompt_analysis.generation import (
        _generated_part,
        _prepare_prompt,
        _record_identity,
        _select_rows,
        _write_json_line,
        generate_tokens,
    )
    from llm_bias.prompt_analysis.input_data import load_prompt_table

    source = config.input
    table = load_prompt_table(source, config.prompt_columns, dataset_format="auto")
    candidates_by_column, effective_dates, effective_pairs, is_return_pairs = _select_rows(
        table,
        sample_per_condition=None if config.generate_full else config.sample_per_condition,
        dates=None,
        selection="default",
        return_pairs_full=False,
        full_generation=config.generate_full,
    )
    tokenizer = model.tokenizer
    forward_destination = config.run_root / "forward"
    forward_destination.mkdir(parents=True, exist_ok=True)
    output_path = forward_destination / "generated_outputs.jsonl"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    generation_config = {
        "strategy": "greedy",
        "do_sample": False,
        "max_new_tokens": config.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "base_seed": None,
        "seed_policy": None,
        "run_index": 0,
        "run_seed": None,
    }
    records_written = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for column in table.columns:
            for sample_index, row in enumerate(candidates_by_column[column.name]):
                prompt = (row.get(column.name) or row.get("prompt", "")).strip()
                formatted = _prepare_prompt(tokenizer, row, prompt)
                encoded = tokenizer(
                    formatted, return_tensors="pt", truncation=True, max_length=config.max_seq_len
                )
                prompt_ids = encoded.input_ids.to(model.device)
                raw_encoded = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=config.max_seq_len
                )
                input_span = find_token_subsequence(
                    prompt_ids[0].tolist(), raw_encoded.input_ids[0].tolist()
                )
                sequence = generate_tokens(
                    model,
                    prompt_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                )
                generated = _generated_part(sequence, prompt_ids, config.max_new_tokens)
                generated_token_ids = [int(token_id) for token_id in generated[0].tolist()]
                identity = _record_identity(
                    row=row,
                    column=column,
                    sample_index=sample_index,
                    is_return_pairs=is_return_pairs,
                )
                record = {
                    "schema_version": 1,
                    "artifact_type": "generated_outputs",
                    "record_id": stable_record_id(identity),
                    "record_identity": identity,
                    "run_index": 0,
                    "sample_index": sample_index,
                    "date": row.get("Date", row.get("filing_date", "")),
                    "prompt_column": column.name,
                    "index": column.index,
                    "context": column.context,
                    "prompt": prompt,
                    "prompt_token_ids": [int(token_id) for token_id in prompt_ids[0].tolist()],
                    "input_span": [int(input_span[0]), int(input_span[1])],
                    "generated_token_ids": generated_token_ids,
                    "generated_text": tokenizer.decode(generated_token_ids, skip_special_tokens=False),
                    "generation_config": generation_config,
                    "finish_reason": _core_finish_reason(
                        generated_token_ids,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=config.max_new_tokens,
                    ),
                }
                for key in ("ticker", "name", "sector", "marketcap", "condition", "input_schema", "row_index"):
                    if key in row:
                        record[key] = row[key]
                _write_json_line(handle, record)
                records_written += 1
                if records_written % 50 == 0:
                    print(f"forward: {records_written} records", flush=True)
    temporary.replace(output_path)
    artifact_sha256 = sha256_file(output_path)
    metadata = {
        "schema_version": 1,
        "artifact_type": "generated_outputs",
        "artifact": str(output_path),
        "artifact_sha256": artifact_sha256,
        "generated_outputs_sha256": artifact_sha256,
        "input": str(source),
        "input_sha256": sha256_file(source),
        "model": config.model,
        "model_slug": model_slug(config.model),
        "dataset_format": table.dataset_format,
        "selection": "full",
        "sample_per_condition": None,
        "selected_dates": sorted(effective_dates),
        "selected_pairs": sorted(effective_pairs),
        "prompt_columns": [column.name for column in table.columns],
        "records_written": records_written,
        "max_seq_len": config.max_seq_len,
        "input_top_k": None,
        "generation_config": generation_config,
        "backpropagation": False,
    }
    atomic_write_json(forward_destination / "metadata.json", metadata)
    print(f"Wrote forward generated outputs to {forward_destination}", flush=True)
    return output_path


def _run_readout(config: BaselineRunConfig) -> None:
    from llm_bias.prompt_analysis.readout import analyze_prompt_outputs

    readout_kwargs: dict[str, Any] = dict(
        input_path=str(config.input),
        model_name=config.model,
        lens_path=str(config.lens),
        output_dir=str(config.run_root / "readout"),
        top_k=config.top_k,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        dataset_format="auto",
    )
    if config.max_rows is not None:
        readout_kwargs["max_rows"] = config.max_rows
    if config.prompt_columns is not None:
        readout_kwargs["prompt_columns"] = list(config.prompt_columns)
    analyze_prompt_outputs(**readout_kwargs)


__all__ = [
    "BaselineRunConfig",
    "DEFAULT_INPUT",
    "DEFAULT_MODEL",
    "LENS_FORWARD_ARTIFACT_TYPE",
    "STAGES",
    "run_baseline_trial",
]
