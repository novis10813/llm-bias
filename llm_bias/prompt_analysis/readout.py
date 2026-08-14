"""Average J-space next-token distributions for prompt columns in a CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch

from jspace_viz.hooks import ActivationRecorder
from jspace_viz.lens import JacobianLens
from jspace_viz.model import WrappedModel

from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import DEFAULT_MODEL, load_model as load_lens_model
from llm_bias.core.prompt_input import decode_token, format_messages, format_prompt
from llm_bias.core.readout import last_unmasked_positions
from llm_bias.core.analysis.records import top_k_token_records as _core_top_k_token_records
from llm_bias.prompt_analysis.input_data import PromptColumn, load_prompt_table

DEFAULT_INPUT = "sp500_r1k_r2k_entityBiasPrompt.csv"
INDEX_LABELS = {
    "sp500": "S&P 500",
    "russell1000": "Russell 1000",
    "russell2000": "Russell 2000",
}


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """Backward-compatible local alias for the shared decoder."""
    return decode_token(tokenizer, token_id)


def topk_token_records(
    probabilities: torch.Tensor,
    *,
    top_k: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for the shared compact token contract."""
    return _core_top_k_token_records(probabilities, top_k=top_k, tokenizer=tokenizer)


_last_unmasked_positions = last_unmasked_positions

def _write_json_line(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _dependency_revisions(root: Path) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for name in ("jacobian-lens", "jspace-viz"):
        checkout = root / "third_party" / name
        try:
            revisions[name] = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revisions[name] = "unknown"
    return revisions


def _prepare_prompt(
    tokenizer: Any,
    prompt: str,
    use_chat_template: bool,
    enable_thinking: bool = False,
    *,
    system_prompt: str | None = None,
) -> str:
    """Format a prompt, using a genuine system turn for return pairs."""
    if system_prompt is None:
        return format_prompt(tokenizer, prompt, use_chat_template=use_chat_template, enable_thinking=enable_thinking)
    return format_messages(
        tokenizer,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )


@torch.no_grad()
def _analyze_column(
    *,
    model: WrappedModel,
    lens: JacobianLens,
    column: PromptColumn,
    rows: list[dict[str, str]],
    layers: list[int],
    top_k: int,
    batch_size: int,
    max_seq_len: int,
    prompt_output: TextIO | None,
    uncertainty_output: TextIO | None,
    use_chat_template: bool,
    enable_thinking: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """Analyze one condition, returning exact mean-distribution top-k by layer."""
    examples = [
        (row_index, row.get("Date", ""), prompt, row)
        for row_index, row in enumerate(rows)
        if (prompt := (row.get(column.name) or row.get("prompt", "")).strip())
        and (row.get("condition") in (None, column.condition or column.context))
    ]
    eligible_rows = [row for row in rows if row.get("condition") in (None, column.condition or column.context)]
    skipped = len(eligible_rows) - len(examples)
    if not examples:
        raise ValueError(f"{column.name} contains no non-empty prompts")

    probability_sums: dict[int, torch.Tensor] = {}
    final_layer = model.n_layers - 1
    tokenizer = model.tokenizer
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            [
                _prepare_prompt(
                    tokenizer, prompt, use_chat_template, enable_thinking,
                    system_prompt=(row.get("system_prompt") or None),
                )
                for _row_index, _date, prompt, row in batch
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)
        final_positions = _last_unmasked_positions(attention_mask)
        batch_indices = torch.arange(len(batch), device=model.device)

        with ActivationRecorder(model.layers, at=layers) as recorder:
            model._decoder(  # Mirrors jspace_viz.analysis.read_grid, with padding support.
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            activations = {
                layer: recorder.activations[layer].detach() for layer in layers
            }

        prompt_readouts: list[list[dict[str, Any]]] = [[] for _ in batch]
        uncertainty_readouts: list[list[dict[str, Any]]] = [[] for _ in batch]
        for layer in layers:
            residual = activations[layer][batch_indices, final_positions].float()
            if layer != final_layer and layer in lens.jacobians:
                residual = lens.transport(residual, layer)
            normalized_hidden = model._final_norm(
                residual.to(model._lm_head.weight.dtype)
            ).float()
            effective_inverse_temperature = normalized_hidden.norm(dim=-1)
            effective_temperature = effective_inverse_temperature.reciprocal().clamp_min(1e-12)
            logits = model.unembed(residual).float()
            probabilities = logits.softmax(dim=-1)
            batch_sum = probabilities.sum(dim=0).double().cpu()
            if layer in probability_sums:
                probability_sums[layer].add_(batch_sum)
            else:
                probability_sums[layer] = batch_sum

            if prompt_output is not None:
                top = probabilities.topk(top_k, dim=-1)
                ids = top.indices.cpu().tolist()
                values = top.values.cpu().tolist()
                for batch_index, (token_ids, token_probabilities) in enumerate(
                    zip(ids, values, strict=True)
                ):
                    prompt_readouts[batch_index].append(
                        {
                            "layer": layer,
                            "is_output": layer == final_layer,
                            "top_tokens": [
                                {
                                    "rank": rank,
                                    "token_id": int(token_id),
                                    "token": _decode_token(tokenizer, int(token_id)),
                                    "probability": float(probability),
                                }
                                for rank, (token_id, probability) in enumerate(
                                    zip(
                                        token_ids,
                                        token_probabilities,
                                        strict=True,
                                    ),
                                    start=1,
                                )
                            ],
                        }
                    )
            if uncertainty_output is not None:
                entropy = -(
                    probabilities.clamp_min(1e-12).log() * probabilities
                ).sum(dim=-1)
                topk_mass = probabilities.topk(top_k, dim=-1).values.sum(dim=-1)
                for batch_index in range(len(batch)):
                    entropy_value = float(entropy[batch_index])
                    uncertainty_readouts[batch_index].append(
                        {
                            "layer": layer,
                            "is_output": layer == final_layer,
                            "entropy_nats": entropy_value,
                            "normalized_entropy": entropy_value
                            / float(torch.log(torch.tensor(probabilities.shape[-1]))),
                            "perplexity": float(torch.exp(entropy[batch_index])),
                            "top1_probability": float(probabilities[batch_index].max()),
                            "topk_mass": float(topk_mass[batch_index]),
                            "effective_inverse_temperature": float(
                                effective_inverse_temperature[batch_index]
                            ),
                            "effective_temperature": float(
                                effective_temperature[batch_index]
                            ),
                        }
                    )
            del (
                residual,
                normalized_hidden,
                effective_inverse_temperature,
                effective_temperature,
                logits,
                probabilities,
            )

        if prompt_output is not None:
            for (row_index, date, _prompt, row), readouts in zip(
                batch, prompt_readouts, strict=True
            ):
                _write_json_line(
                    prompt_output,
                    {
                        "row_index": row_index,
                        "date": date,
                        "prompt_column": column.name,
                        "index": column.index,
                        "context": column.context,
                        "layers": readouts,
                        **({key: row[key] for key in ("input_schema", "pair_id", "filing_date", "ticker", "peer_ticker", "condition", "target_label", "fwd_return_1d") if key in row}),
                    },
                )
        if uncertainty_output is not None:
            for (row_index, date, _prompt, row), readouts in zip(
                batch, uncertainty_readouts, strict=True
            ):
                _write_json_line(
                    uncertainty_output,
                    {
                        "row_index": row_index,
                        "date": date,
                        "prompt_column": column.name,
                        "index": column.index,
                        "context": column.context,
                        "layers": readouts,
                        **({key: row[key] for key in ("input_schema", "pair_id", "filing_date", "ticker", "peer_ticker", "condition", "target_label", "fwd_return_1d") if key in row}),
                    },
                )
        processed = min(batch_start + len(batch), len(examples))
        if processed == len(examples) or processed % max(batch_size, 256) < len(batch):
            print(f"{column.name}: {processed}/{len(examples)} prompts", flush=True)
        del activations

    averages = []
    for layer in layers:
        mean_probabilities = probability_sums[layer] / len(examples)
        averages.append(
            {
                "prompt_column": column.name,
                "index": column.index,
                "context": column.context,
                "n_prompts": len(examples),
                "layer": layer,
                "is_output": layer == final_layer,
                "top_tokens": topk_token_records(
                    mean_probabilities,
                    top_k=top_k,
                    tokenizer=tokenizer,
                ),
            }
        )
    return averages, len(examples), skipped


def _write_average_csv(records: list[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_column",
                "index",
                "context",
                "n_prompts",
                "layer",
                "is_output",
                "rank",
                "token_id",
                "token",
                "probability",
            ],
        )
        writer.writeheader()
        for record in records:
            base = {key: value for key, value in record.items() if key != "top_tokens"}
            for token in record["top_tokens"]:
                writer.writerow(base | token)


def _batched_output_gradients(
    scores: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Return one embedding gradient per output score using a batched VJP."""
    identity = torch.eye(scores.numel(), dtype=scores.dtype, device=scores.device)
    return torch.autograd.grad(
        outputs=scores,
        inputs=embedding,
        grad_outputs=identity,
        is_grads_batched=True,
    )[0]


def _top_input_token_records(
    contributions: torch.Tensor,
    *,
    top_k: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    top = contributions.topk(min(top_k, contributions.numel()))
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": _decode_token(tokenizer, int(token_id)),
            "mean_attribution": float(attribution),
        }
        for rank, (token_id, attribution) in enumerate(
            zip(top.indices.tolist(), top.values.tolist(), strict=True),
            start=1,
        )
        if attribution > 0
    ]


@torch.enable_grad()
def _attribute_column(
    *,
    model: WrappedModel,
    column: PromptColumn,
    rows: list[dict[str, str]],
    output_record: dict[str, Any],
    output_top_k: int | None,
    input_top_k: int,
    batch_size: int,
    max_seq_len: int,
    use_chat_template: bool,
    enable_thinking: bool,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    """Aggregate output-token gradient × input attribution by ID and position."""
    examples = [
        row
        for row in rows
        if (row.get(column.name) or row.get("prompt", "")).strip()
        and (row.get("condition") in (None, column.condition or column.context))
    ]
    if max_examples is not None and len(examples) > max_examples:
        # Deterministic spread across the date-sorted CSV rather than a
        # contiguous prefix, so attribution covers the whole time range.
        indices = [
            round(index * (len(examples) - 1) / (max_examples - 1))
            for index in range(max_examples)
        ] if max_examples > 1 else [0]
        examples = [examples[index] for index in indices]
    output_tokens = output_record["top_tokens"][:output_top_k]
    output_ids = torch.tensor(
        [token["token_id"] for token in output_tokens],
        dtype=torch.long,
        device=model.device,
    )
    n_outputs = len(output_tokens)
    vocab_size = int(model._lm_head.weight.shape[0])
    by_token_id = torch.zeros((n_outputs, vocab_size), dtype=torch.float64)
    by_right_aligned_position = torch.zeros(
        (n_outputs, max_seq_len), dtype=torch.float64
    )
    tokens_at_position = [Counter() for _ in range(max_seq_len)]
    tokenizer = model.tokenizer
    final_layer = model.n_layers - 1

    for batch_start in range(0, len(examples), batch_size):
        prompt_rows = examples[batch_start : batch_start + batch_size]
        prompts = [(row.get(column.name) or row.get("prompt", "")).strip() for row in prompt_rows]
        encoded = tokenizer(
            [
                _prepare_prompt(
                    tokenizer, prompt, use_chat_template, enable_thinking,
                    system_prompt=(row.get("system_prompt") or None),
                )
                for prompt, row in zip(prompts, prompt_rows, strict=True)
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)
        final_positions = _last_unmasked_positions(attention_mask)
        batch_indices = torch.arange(len(prompts), device=model.device)
        embedding_box: dict[str, torch.Tensor] = {}

        def root_embedding(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            rooted = output.detach().requires_grad_(True)
            embedding_box["value"] = rooted
            return rooted

        handle = model._embed.register_forward_hook(root_embedding)
        try:
            with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
                model._decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                residual = recorder.activations[final_layer][
                    batch_indices, final_positions
                ].float()
            logits = model.unembed(residual).float()
            selected_log_prob_sums = (
                logits.log_softmax(dim=-1)[:, output_ids].sum(dim=0)
            )
            embedding = embedding_box["value"]
            gradients = _batched_output_gradients(
                selected_log_prob_sums,
                embedding,
            )
        finally:
            handle.remove()

        attribution = (
            gradients.float() * embedding.detach().float().unsqueeze(0)
        ).abs().sum(dim=-1)
        attribution.mul_(attention_mask.unsqueeze(0))
        attribution.div_(
            attribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        )

        flat_ids = input_ids.flatten().cpu()
        flat_attribution = attribution.flatten(start_dim=1).double().cpu()
        for output_index in range(n_outputs):
            by_token_id[output_index].scatter_add_(
                0,
                flat_ids,
                flat_attribution[output_index],
            )

        lengths = attention_mask.sum(dim=1).cpu().tolist()
        batch_ids = input_ids.cpu()
        batch_attribution = attribution.double().cpu()
        for example_index, length_value in enumerate(lengths):
            length = int(length_value)
            right_start = max_seq_len - length
            by_right_aligned_position[:, right_start:].add_(
                batch_attribution[:, example_index, :length]
            )
            for position, token_id in enumerate(
                batch_ids[example_index, :length].tolist()
            ):
                tokens_at_position[right_start + position][int(token_id)] += 1

        processed = min(batch_start + len(prompts), len(examples))
        if processed == len(examples) or processed % max(batch_size, 256) < len(prompts):
            print(
                f"{column.name} attribution: {processed}/{len(examples)} prompts",
                flush=True,
            )
        del (
            residual,
            logits,
            gradients,
            embedding,
            attribution,
        )

    by_token_id.div_(len(examples))
    by_right_aligned_position.div_(len(examples))
    results: list[dict[str, Any]] = []
    for output_index, output_token in enumerate(output_tokens):
        input_positions = []
        for right_position, counter in enumerate(tokens_at_position):
            if not counter:
                continue
            attribution = float(
                by_right_aligned_position[output_index, right_position]
            )
            representative_id, count = tokens_at_position[
                right_position
            ].most_common(1)[0]
            position_examples = sum(tokens_at_position[right_position].values())
            input_positions.append(
                {
                    "position_from_end": right_position - (max_seq_len - 1),
                    "representative_token_id": representative_id,
                    "representative_token": _decode_token(
                        tokenizer, representative_id
                    ),
                    "representative_fraction": count / position_examples,
                    "mean_attribution": attribution,
                }
            )
        top_positions = [
            {"rank": rank, **position}
            for rank, position in enumerate(
                sorted(
                    input_positions,
                    key=lambda item: item["mean_attribution"],
                    reverse=True,
                )[:input_top_k],
                start=1,
            )
        ]
        results.append(
            {
                "prompt_column": column.name,
                "index": column.index,
                "context": column.context,
                "n_prompts": len(examples),
                "output_rank": output_token["rank"],
                "output_token_id": output_token["token_id"],
                "output_token": output_token["token"],
                "output_mean_probability": output_token["probability"],
                "method": "gradient_attribution",
                "operation": "absolute_gradient_x_input_embedding_of_log_probability",
                "normalization": "sum_to_one_over_input_positions_per_prompt",
                "top_input_tokens": _top_input_token_records(
                    by_token_id[output_index],
                    top_k=input_top_k,
                    tokenizer=tokenizer,
                ),
                "top_input_positions": top_positions,
                "input_positions": input_positions,
            }
        )
    return results


def _plot_output_distributions(
    records: list[dict[str, Any]],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_records = [record for record in records if record["is_output"]]
    indices = list(dict.fromkeys(record["index"] for record in output_records))
    contexts = ("without", "with")
    figure, axes = plt.subplots(
        len(indices),
        len(contexts),
        figsize=(15, max(5, 4.8 * len(indices))),
        squeeze=False,
    )
    colors = {"without": "#6c8ebf", "with": "#d79b00"}
    by_condition = {
        (record["index"], record["context"]): record for record in output_records
    }
    for row_index, index in enumerate(indices):
        for column_index, context in enumerate(contexts):
            axis = axes[row_index][column_index]
            record = by_condition.get((index, context))
            if record is None:
                axis.set_visible(False)
                continue
            tokens = record["top_tokens"][::-1]
            labels = [
                f"{token['rank']:>2}. {token['token']!r}  [{token['token_id']}]"
                for token in tokens
            ]
            probabilities = [token["probability"] for token in tokens]
            axis.barh(labels, probabilities, color=colors[context])
            axis.set_title(
                f"{INDEX_LABELS.get(index, index)} · {context} context "
                f"(n={record['n_prompts']})"
            )
            axis.set_xlabel("Mean next-token probability")
            axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        "Mean output-layer top-k distribution at the final prompt position",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_input_attributions(
    records: list[dict[str, Any]],
    destination: Path,
    *,
    input_top_k: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    indices = list(dict.fromkeys(record["index"] for record in records))
    contexts = ("without", "with")
    figure, axes = plt.subplots(
        len(indices),
        len(contexts),
        figsize=(16, max(6, 5.5 * len(indices))),
        squeeze=False,
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["index"], record["context"]), []).append(record)
    common_max = max(
        position["mean_attribution"]
        for record in records
        for position in record["input_positions"]
    )

    for row_index, index in enumerate(indices):
        for column_index, context in enumerate(contexts):
            axis = axes[row_index][column_index]
            group = sorted(
                grouped.get((index, context), []),
                key=lambda record: record["output_rank"],
            )
            if not group:
                axis.set_visible(False)
                continue
            position_scores: dict[int, float] = {}
            position_labels: dict[int, str] = {}
            for record in group:
                for position in record["input_positions"]:
                    offset = position["position_from_end"]
                    position_scores[offset] = (
                        position_scores.get(offset, 0.0)
                        + position["mean_attribution"]
                    )
                    position_labels[offset] = (
                        f"{offset}: {position['representative_token']!r}"
                    )
            selected_positions = sorted(
                sorted(
                    position_scores,
                    key=position_scores.__getitem__,
                    reverse=True,
                )[:input_top_k]
            )
            matrix = np.zeros((len(group), len(selected_positions)), dtype=float)
            for output_index, record in enumerate(group):
                values = {
                    position["position_from_end"]: position["mean_attribution"]
                    for position in record["input_positions"]
                }
                matrix[output_index] = [
                    values.get(position, 0.0) for position in selected_positions
                ]
            image = axis.imshow(
                matrix,
                aspect="auto",
                cmap="magma",
                vmin=0.0,
                vmax=common_max,
            )
            axis.set_xticks(
                range(len(selected_positions)),
                [position_labels[position] for position in selected_positions],
                rotation=60,
                ha="right",
                fontsize=7,
            )
            axis.set_yticks(
                range(len(group)),
                [
                    f"{record['output_rank']}. {record['output_token']!r}"
                    for record in group
                ],
                fontsize=8,
            )
            axis.set_title(
                f"{INDEX_LABELS.get(index, index)} · {context} context"
            )
            axis.set_xlabel("Input position from end (0 = final prompt token)")
            axis.set_ylabel("Mean output-distribution token")
            figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.suptitle(
        "Output-token sensitivity to input positions · |gradient × embedding|",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_prompt_outputs(
    *,
    input_path: str = DEFAULT_INPUT,
    model_name: str = DEFAULT_MODEL,
    lens_path: str | Path | None = None,
    output_dir: str | Path,
    top_k: int = 15,
    batch_size: int = 32,
    max_seq_len: int = 256,
    max_rows: int | None = None,
    prompt_columns: Iterable[str] | None = None,
    save_prompt_topk: bool = True,
    save_prompt_uncertainty: bool = True,
    compute_input_attribution: bool = False,
    attribution_batch_size: int = 8,
    input_top_k: int = 15,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    attribution_output_top_k: int | None = None,
    attribution_max_rows: int | None = None,
    dataset_format: str = "auto",
) -> Path:
    """Run batched final-position J-space readout and save compact artifacts."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")
    if attribution_batch_size < 1:
        raise ValueError("attribution_batch_size must be positive")
    if input_top_k < 1:
        raise ValueError("input_top_k must be positive")
    if attribution_output_top_k is not None and attribution_output_top_k < 1:
        raise ValueError("attribution_output_top_k must be positive when provided")
    if attribution_max_rows is not None and attribution_max_rows < 1:
        raise ValueError("attribution_max_rows must be positive when provided")

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    table = load_prompt_table(
        source, prompt_columns, max_rows, dataset_format=dataset_format
    )
    columns = table.columns
    rows = table.rows

    lens_model, tokenizer, _device = load_lens_model(model_name)
    model = WrappedModel(lens_model._hf_model, tokenizer)
    loaded_lens = load_validated_lens(
        model=model,
        model_name=model_name,
        lens_path=lens_path,
        require_complete=True,
    )
    lens = loaded_lens.lens
    lens_source = loaded_lens.path
    final_layer = model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final_layer})
    if top_k > model._lm_head.weight.shape[0]:
        raise ValueError("top_k exceeds the model vocabulary size")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prompt_path = destination / "prompt_layer_topk.jsonl"
    prompt_temporary = prompt_path.with_suffix(prompt_path.suffix + ".tmp")
    prompt_handle = (
        prompt_temporary.open("w", encoding="utf-8") if save_prompt_topk else None
    )
    uncertainty_path = destination / "prompt_layer_uncertainty.jsonl"
    uncertainty_temporary = uncertainty_path.with_suffix(
        uncertainty_path.suffix + ".tmp"
    )
    uncertainty_handle = (
        uncertainty_temporary.open("w", encoding="utf-8")
        if save_prompt_uncertainty
        else None
    )
    average_records: list[dict[str, Any]] = []
    attribution_records: list[dict[str, Any]] = []
    condition_counts: dict[str, dict[str, int]] = {}
    try:
        for column in columns:
            records, processed, skipped = _analyze_column(
                model=model,
                lens=lens,
                column=column,
                rows=rows,
                layers=layers,
                top_k=top_k,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                prompt_output=prompt_handle,
                uncertainty_output=uncertainty_handle,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            average_records.extend(records)
            condition_counts[column.name] = {
                "processed_prompts": processed,
                "skipped_empty_prompts": skipped,
            }
            if compute_input_attribution:
                output_record = next(
                    record for record in records if record["is_output"]
                )
                attribution_records.extend(
                    _attribute_column(
                        model=model,
                        column=column,
                        rows=rows,
                        output_record=output_record,
                        output_top_k=attribution_output_top_k,
                        input_top_k=input_top_k,
                        batch_size=attribution_batch_size,
                        max_seq_len=max_seq_len,
                        use_chat_template=use_chat_template,
                        enable_thinking=enable_thinking,
                        max_examples=attribution_max_rows,
                    )
                )
    finally:
        if prompt_handle is not None:
            prompt_handle.close()
        if uncertainty_handle is not None:
            uncertainty_handle.close()
    if save_prompt_topk:
        prompt_temporary.replace(prompt_path)
    if save_prompt_uncertainty:
        uncertainty_temporary.replace(uncertainty_path)

    average_jsonl = destination / "average_layer_topk.jsonl"
    with average_jsonl.open("w", encoding="utf-8") as handle:
        for record in average_records:
            _write_json_line(handle, record)
    _write_average_csv(average_records, destination / "average_layer_topk.csv")
    _plot_output_distributions(
        average_records,
        destination / "output_topk_distribution.png",
    )
    if compute_input_attribution:
        attribution_path = destination / "input_token_attribution.jsonl"
        with attribution_path.open("w", encoding="utf-8") as handle:
            for record in attribution_records:
                _write_json_line(handle, record)
        _plot_input_attributions(
            attribution_records,
            destination / "input_token_attribution.png",
            input_top_k=input_top_k,
        )

    repository_root = Path(__file__).resolve().parents[2]
    metadata = {
        "input": str(source),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": model_name,
        "lens": str(lens_source),
        "top_k": top_k,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "max_rows": max_rows,
        "dataset_format": dataset_format,
        "prompt_columns": [column.name for column in columns],
        "layers": layers,
        "missing_layers": [],
        "output_position": "last_non_padding_prompt_token",
        "use_chat_template": use_chat_template,
        "enable_thinking": enable_thinking,
        "distribution_aggregation": "arithmetic_mean_of_full_vocabulary_softmax",
        "uncertainty_measurement": {
            "primary": "temperature_scope",
            "effective_inverse_temperature": "l2_norm_of_final_norm_hidden_state",
            "effective_temperature": "reciprocal_of_effective_inverse_temperature",
            "entropy": "retained_as_categorical_distribution_comparison",
        },
        "condition_counts": condition_counts,
        "saved_prompt_layer_topk": save_prompt_topk,
        "saved_prompt_layer_uncertainty": save_prompt_uncertainty,
        "backpropagation": compute_input_attribution,
        "input_attribution": {
            "enabled": compute_input_attribution,
            "batch_size": attribution_batch_size,
            "top_k": input_top_k,
            "output_top_k": attribution_output_top_k or top_k,
            "max_rows": attribution_max_rows,
            "method": "gradient_attribution",
            "operation": "absolute_gradient_x_input_embedding_of_log_probability",
            "normalization": "sum_to_one_over_input_positions_per_prompt",
            "position_alignment": "right_aligned_to_final_prompt_token",
            "position_zero": "final_non_padding_prompt_token",
        },
        "dependency_revisions": _dependency_revisions(repository_root),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote J-space prompt output analysis to {destination}", flush=True)
    return destination
