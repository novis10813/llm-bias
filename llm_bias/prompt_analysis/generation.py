"""Forward-only generation and reusable generated-output artifacts.

This module intentionally owns no attribution logic.  It performs model
``generate`` calls under ``torch.no_grad`` and stores the generated token IDs
so a later attribution stage can consume exactly the same output without
running generation again.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, TextIO

import torch

from llm_bias.core.artifact_paths import atomic_write_json, model_slug, stable_record_id
from llm_bias.core.model import DEFAULT_MODEL, load_model as load_lens_model
from llm_bias.core.prompting import find_token_subsequence, format_messages, format_prompt
from llm_bias.prompt_analysis.readout import load_prompt_table
from jspace_viz.model import WrappedModel


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "generated_outputs"
DEFAULT_SAMPLE_PER_CONDITION = 32
Selection = Literal["default", "sampled", "full"]


def _sample_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Select a deterministic spread of rows, retaining the legacy policy."""
    if count < 1:
        raise ValueError("sample count must be positive")
    if len(rows) <= count:
        return rows
    indices = (
        [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
        if count > 1
        else [0]
    )
    return [rows[index] for index in indices]


def _stable_record_id(identity: dict[str, Any]) -> str:
    """Return the shared stable ID for one logical prompt condition."""
    return stable_record_id(identity)


def _prepare_prompt(tokenizer: Any, row: dict[str, str], prompt: str) -> str:
    system_prompt = row.get("system_prompt") or None
    if system_prompt:
        return format_messages(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            use_chat_template=True,
            enable_thinking=False,
        )
    return format_prompt(
        tokenizer,
        prompt,
        use_chat_template=True,
        enable_thinking=False,
    )


def _generation_kwargs(
    model: WrappedModel,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    """Build the exact generation controls passed to Hugging Face."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
        "pad_token_id": model.tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    return kwargs


@torch.no_grad()
def generate_tokens(
    model: WrappedModel,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """Generate one sequence without constructing a gradient graph.

    The returned tensor is the model's complete sequence (prompt followed by
    generated tokens), matching the contract of ``PreTrainedModel.generate``.
    ``GenerateDecoderOnlyOutput`` and plain tensor return values are both
    accepted to keep deterministic fake models useful in tests.
    """
    result = model.hf_model.generate(
        prompt_ids,
        **_generation_kwargs(
            model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
    )
    sequences = getattr(result, "sequences", result)
    if isinstance(sequences, (tuple, list)):
        if not sequences:
            raise ValueError("model.generate returned no sequences")
        sequences = sequences[0]
    if not isinstance(sequences, torch.Tensor):
        raise TypeError("model.generate must return a tensor or an output with sequences")
    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        raise ValueError(
            "generation-only artifact currently requires exactly one generated sequence"
        )
    return sequences


def _generated_part(sequence: torch.Tensor, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Extract generated IDs while checking the authoritative prompt prefix."""
    prompt_length = prompt_ids.shape[1]
    if sequence.shape[1] >= prompt_length and torch.equal(
        sequence[:, :prompt_length], prompt_ids
    ):
        return sequence[:, prompt_length:]
    # A small fake model may return only newly generated IDs.  Accept that
    # shape, but never silently truncate a complete model sequence.
    if sequence.shape[1] <= max_new_tokens:
        return sequence
    raise ValueError("model.generate output does not contain the prompt token prefix")


def _finish_reason(
    generated_ids: list[int],
    *,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
    max_new_tokens: int,
) -> str:
    if not generated_ids:
        return "empty"
    eos_ids = (
        set(eos_token_id)
        if isinstance(eos_token_id, (list, tuple, set))
        else ({eos_token_id} if eos_token_id is not None else set())
    )
    if eos_ids.intersection(generated_ids):
        return "eos_token"
    if len(generated_ids) >= max_new_tokens:
        return "max_new_tokens"
    return "model_stop"


def _select_rows(
    columns: list[Any],
    rows: list[dict[str, str]],
    *,
    sample_per_condition: int | None,
    dates: Iterable[str] | None,
    selection: Selection,
    return_pairs_full: bool,
    full_generation: bool,
) -> tuple[dict[str, list[dict[str, str]]], set[str], set[str], bool]:
    """Apply the legacy-wide sampling or explicit return-pairs full selection."""
    if selection not in {"default", "sampled", "full"}:
        raise ValueError("selection must be default, sampled, or full")
    is_return_pairs = bool(rows and rows[0].get("input_schema") == "return-pairs")
    selected_dates = set(dates or ())
    effective_dates = set(selected_dates)
    effective_pairs: set[str] = set()

    if is_return_pairs:
        if selected_dates:
            effective_pairs = {
                row["pair_id"]
                for row in rows
                if row.get("filing_date", row.get("Date", "")) in selected_dates
            }
        elif full_generation or selection == "full" or return_pairs_full or sample_per_condition is None:
            effective_pairs = {row["pair_id"] for row in rows}
        else:
            pair_ids = sorted({row["pair_id"] for row in rows})
            effective_pairs = {
                row["pair_id"]
                for row in _sample_rows(
                    [{"pair_id": pair_id} for pair_id in pair_ids],
                    sample_per_condition,
                )
            }
    elif not selected_dates:
        if full_generation or selection == "full":
            effective_dates = {
                row.get("Date", "")
                for row in rows
                if row.get("Date", "")
                and any((row.get(column.name) or "").strip() for column in columns)
            }
        else:
            if sample_per_condition is None:
                raise ValueError("sample_per_condition is required for legacy-wide inputs")
            date_sets = [
                {
                    row.get("Date", "")
                    for row in rows
                    if (row.get(column.name) or "").strip() and row.get("Date", "")
                }
                for column in columns
            ]
            common_dates = set.intersection(*date_sets) if date_sets else set()
            if not common_dates:
                raise ValueError("prompt columns have no common non-empty dates to sample")
            effective_dates = {
                row["Date"]
                for row in _sample_rows(
                    [{"Date": date} for date in sorted(common_dates)],
                    sample_per_condition,
                )
            }

    candidates_by_column = {
        column.name: [
            row
            for row in rows
            if (row.get(column.name) or row.get("prompt", "")).strip()
            and (not effective_pairs or row.get("pair_id") in effective_pairs)
            and (
                not effective_dates
                or row.get("Date", row.get("filing_date", "")) in effective_dates
            )
            and row.get("condition") in (None, column.condition or column.context)
        ]
        for column in columns
    }
    return candidates_by_column, effective_dates, effective_pairs, is_return_pairs


def _record_identity(
    *,
    row: dict[str, str],
    column: Any,
    sample_index: int,
    is_return_pairs: bool,
) -> dict[str, Any]:
    if is_return_pairs:
        return {
            "input_schema": "return-pairs",
            "pair_id": row["pair_id"],
            "condition": row["condition"],
        }
    return {
        "input_schema": "legacy-wide",
        "date": row.get("Date", ""),
        "prompt_column": column.name,
        "index": column.index,
        "context": column.context,
    }


def _write_json_line(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _validate_controls(
    *,
    sample_per_condition: int | None,
    max_new_tokens: int,
    runs: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
) -> None:
    if sample_per_condition is not None and sample_per_condition < 1:
        raise ValueError("sample_per_condition must be positive when provided")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if runs < 1:
        raise ValueError("runs must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    if runs > 1 and temperature == 0:
        raise ValueError("runs greater than one require temperature greater than zero")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and between zero and one")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative when provided")


def _seed_run(base_seed: int | None, run_index: int) -> int | None:
    if base_seed is None:
        return None
    run_seed = base_seed + run_index
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    return run_seed


def generate_prompt_outputs(
    *,
    input_path: str,
    model_name: str = DEFAULT_MODEL,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    sample_per_condition: int | None = DEFAULT_SAMPLE_PER_CONDITION,
    full_generation: bool = False,
    max_new_tokens: int = 64,
    input_top_k: int | None = None,
    max_seq_len: int = 256,
    prompt_columns: Iterable[str] | None = None,
    dates: Iterable[str] | None = None,
    runs: int = 1,
    temperature: float = 0.0,
    seed: int | None = None,
    top_p: float = 1.0,
    top_k: int = 0,
    dataset_format: str = "auto",
    selection: Selection = "default",
    return_pairs_full: bool | None = None,
) -> Path:
    """Generate and save reusable output records under ``<output_dir>/forward``.

    Legacy-wide inputs preserve the historical deterministic spread of 32 dates
    by default. Return-pairs inputs can opt into all rows with
    ``selection="full"`` (or ``return_pairs_full=True`` / ``sample_per_condition=None``)
    without changing the generic sampling default.
    """
    _validate_controls(
        sample_per_condition=sample_per_condition,
        max_new_tokens=max_new_tokens,
        runs=runs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if input_top_k is not None and input_top_k < 1:
        raise ValueError("input_top_k must be positive when provided")
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    columns, rows = load_prompt_table(
        source,
        prompt_columns,
        dataset_format=dataset_format,
    )
    candidates_by_column, effective_dates, effective_pairs, is_return_pairs = _select_rows(
        columns,
        rows,
        sample_per_condition=sample_per_condition,
        dates=dates,
        selection=selection,
        return_pairs_full=bool(return_pairs_full),
        full_generation=full_generation,
    )
    lens_model, tokenizer, _device = load_lens_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generation_config = {
        "strategy": "sampling" if temperature > 0 else "greedy",
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "base_seed": seed,
        "seed_policy": "base_seed_plus_run_index" if seed is not None else None,
    }
    explicit_output_path: Path | None = None
    if output_path is not None:
        requested_output = Path(output_path)
        if requested_output.exists() and requested_output.is_dir():
            output_dir = str(requested_output)
        elif requested_output.name == "forward" and requested_output.suffix == "":
            output_dir = str(requested_output.parent)
        elif requested_output.suffix.lower() == ".jsonl":
            if runs != 1:
                raise ValueError("output_path file targets require runs=1")
            explicit_output_path = requested_output
            output_dir = str(requested_output.parent)
        else:
            raise ValueError(
                "output_path must be an existing directory, a forward directory, "
                "or a .jsonl file"
            )
    if output_dir is None:
        raise ValueError("output_dir or output_path is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if runs > 1 and any(destination.iterdir()):
        raise FileExistsError(f"multi-run output directory must be empty: {destination}")
    run_directories = (
        [destination / f"run_{run_index:03d}" for run_index in range(runs)]
        if runs > 1
        else [destination]
    )
    run_manifest: list[dict[str, Any]] = []

    for run_index, run_destination in enumerate(run_directories):
        run_seed = _seed_run(seed, run_index)
        forward_destination = (
            explicit_output_path.parent
            if explicit_output_path is not None
            else run_destination / "forward"
        )
        forward_destination.mkdir(parents=True, exist_ok=True)
        output_path = (
            explicit_output_path
            if explicit_output_path is not None
            else forward_destination / "generated_outputs.jsonl"
        )
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        run_config = {**generation_config, "run_index": run_index, "run_seed": run_seed}
        records_written = 0
        with temporary.open("w", encoding="utf-8") as handle:
            for column in columns:
                for sample_index, row in enumerate(candidates_by_column[column.name]):
                    prompt = (row.get(column.name) or row.get("prompt", "")).strip()
                    formatted = _prepare_prompt(tokenizer, row, prompt)
                    encoded = tokenizer(
                        formatted,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_seq_len,
                    )
                    prompt_ids = encoded.input_ids.to(model.device)
                    raw_encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_seq_len,
                    )
                    input_span = find_token_subsequence(
                        prompt_ids[0].tolist(), raw_encoded.input_ids[0].tolist()
                    )
                    sequence = generate_tokens(
                        model,
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    generated = _generated_part(sequence, prompt_ids, max_new_tokens)
                    generated_token_ids = [int(token_id) for token_id in generated[0].tolist()]
                    identity = _record_identity(
                        row=row,
                        column=column,
                        sample_index=sample_index,
                        is_return_pairs=is_return_pairs,
                    )
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": ARTIFACT_TYPE,
                        "record_id": _stable_record_id(identity),
                        "record_identity": identity,
                        "run_index": run_index,
                        "sample_index": sample_index,
                        "date": row.get("Date", row.get("filing_date", "")),
                        "prompt_column": column.name,
                        "index": column.index,
                        "context": column.context,
                        "prompt": prompt,
                        "prompt_token_ids": [int(token_id) for token_id in prompt_ids[0].tolist()],
                        "input_span": [int(input_span[0]), int(input_span[1])],
                        "generated_token_ids": generated_token_ids,
                        "generated_text": tokenizer.decode(
                            generated_token_ids,
                            skip_special_tokens=False,
                        ),
                        "generation_config": run_config,
                        "finish_reason": _finish_reason(
                            generated_token_ids,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=max_new_tokens,
                        ),
                    }
                    for key in (
                        "input_schema", "pair_id", "filing_date", "ticker", "peer_ticker",
                        "condition", "target_label", "fwd_return_1d", "system_prompt",
                    ):
                        if key in row:
                            record[key] = row[key]
                    _write_json_line(handle, record)
                    records_written += 1
        temporary.replace(output_path)
        artifact_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "artifact": str(output_path),
            "artifact_sha256": artifact_sha256,
            "generated_outputs_sha256": artifact_sha256,
            "input": str(source),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model": model_name,
            "model_slug": model_slug(model_name),
            "dataset_format": dataset_format,
            "selection": "full" if (full_generation or selection == "full" or return_pairs_full or sample_per_condition is None) else "sampled",
            "sample_per_condition": sample_per_condition,
            "selected_dates": sorted(effective_dates),
            "selected_pairs": sorted(effective_pairs),
            "prompt_columns": [column.name for column in columns],
            "records_written": records_written,
            "max_seq_len": max_seq_len,
            "input_top_k": input_top_k,
            "generation_config": run_config,
            "backpropagation": False,
        }
        atomic_write_json(forward_destination / "metadata.json", metadata)
        run_manifest.append({"run_index": run_index, "records_written": records_written})

    if runs > 1:
        atomic_write_json(
            destination / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "runs": runs,
                "run_directories": run_manifest,
            },
        )
    return (
        explicit_output_path
        if explicit_output_path is not None
        else run_directories[0] / "forward" / "generated_outputs.jsonl"
    )


# Descriptive aliases for callers that prefer the stage name.
generate_outputs = generate_prompt_outputs
generate_forward_artifact = generate_prompt_outputs

__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_SAMPLE_PER_CONDITION",
    "SCHEMA_VERSION",
    "generate_forward_artifact",
    "generate_outputs",
    "generate_prompt_outputs",
    "generate_tokens",
]
