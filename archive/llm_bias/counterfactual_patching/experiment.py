"""Run lens readouts and patch experiments, then write tabular artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from llm_bias.core.model import DEFAULT_MODEL, load_model
from llm_bias.counterfactual_patching.data import (
    Pair,
    load_pairs,
    load_saved_pairs,
    save_pairs,
)
from llm_bias.counterfactual_patching.interventions import (
    next_logits,
    normalized_span_mapping,
    patched_next_logits,
    record_residuals,
)


def normalized_transfer(
    patched_margin: float,
    source_margin: float,
    target_margin: float,
) -> float | None:
    """Normalize patch movement so source=0 and target=1."""
    denominator = target_margin - source_margin
    if abs(denominator) <= 1e-6:
        return None
    return (patched_margin - source_margin) / denominator


def expected_outcome_margin(margin: float, expected_outcome: str | None) -> float | None:
    """Orient the fixed positive-minus-negative margin toward the gold outcome."""
    if expected_outcome == "positive":
        return margin
    if expected_outcome == "negative":
        return -margin
    return None


def _tokenizer_input(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt")
    return encoded.input_ids.to(device)


def _last_non_entity_position(
    *, entity_start: int, entity_end: int, sequence_length: int
) -> int:
    """Choose a non-entity control position, preferring the final token."""
    for position in range(sequence_length - 1, -1, -1):
        if not entity_start <= position < entity_end:
            return position
    raise ValueError("entity span covers the entire prompt")


def _span_readout_difference(
    model: Any,
    lens: Any,
    source_residual: torch.Tensor,
    target_residual: torch.Tensor,
    layer: int,
    pair: Pair,
) -> float:
    """Compare lens logits on the entity token ids across full spans."""
    source_logits = model.unembed(
        lens.transport(source_residual.float(), layer)
    ).float().cpu()
    target_logits = model.unembed(
        lens.transport(target_residual.float(), layer)
    ).float().cpu()
    source_ids = pair.source_entity_token_ids or [pair.source_entity_token]
    target_ids = pair.target_entity_token_ids or [pair.target_entity_token]
    source_score = sum(
        float(source_logits[index, token_id])
        for index, token_id in enumerate(source_ids)
    ) / len(source_ids)
    target_score = sum(
        float(target_logits[index, token_id])
        for index, token_id in enumerate(target_ids)
    ) / len(target_ids)
    return target_score - source_score


def prepare_data(
    model_name: str = DEFAULT_MODEL,
    output: str = "artifacts/counterfactual_patching/pairs.jsonl",
    max_pairs: int | None = None,
) -> Path:
    _model, tokenizer, _device = load_model(model_name)
    pairs = load_pairs(tokenizer, max_pairs=max_pairs)
    if not pairs:
        raise RuntimeError("No aligned entity-span pairs survived tokenization")
    save_pairs(pairs, output)
    print(f"Prepared {len(pairs)} aligned pairs at {output}")
    return Path(output)


def run_patch(
    model_name: str = DEFAULT_MODEL,
    pairs_path: str = "artifacts/counterfactual_patching/pairs.jsonl",
    lens_path: str | None = None,
    output: str = "artifacts/counterfactual_patching/patch_results.jsonl",
    max_pairs: int | None = None,
) -> Path:
    model, tokenizer, device = load_model(model_name)
    pairs = load_pairs(tokenizer) if not Path(pairs_path).exists() else load_saved_pairs(pairs_path)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise RuntimeError("No aligned pairs available; run prepare-data first")

    lens = None
    if lens_path:
        from llm_bias.core.lens_loader import load_validated_lens

        lens = load_validated_lens(
            model=model,
            model_name=model_name,
            lens_path=lens_path,
        ).lens
    layers = list(range(model.n_layers))
    results: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs, start=1):
        source_ids = _tokenizer_input(tokenizer, pair.source_prompt, device)
        target_ids = _tokenizer_input(tokenizer, pair.target_prompt, device)
        source_residuals = record_residuals(model, source_ids, layers)
        target_residuals = record_residuals(model, target_ids, layers)
        source_logits = next_logits(model, source_ids)
        target_logits = next_logits(model, target_ids)
        source_margin = float(source_logits[pair.answer_target_token] - source_logits[pair.answer_source_token])
        target_margin = float(target_logits[pair.answer_target_token] - target_logits[pair.answer_source_token])
        source_span = (pair.source_entity_start, pair.source_entity_end)
        target_span = (pair.target_entity_start, pair.target_entity_end)
        source_control_position = _last_non_entity_position(
            entity_start=pair.source_entity_start,
            entity_end=pair.source_entity_end,
            sequence_length=source_ids.shape[1],
        )
        target_control_position = _last_non_entity_position(
            entity_start=pair.target_entity_start,
            entity_end=pair.target_entity_end,
            sequence_length=target_ids.shape[1],
        )
        for layer in layers[:-1]:
            patched_logits = patched_next_logits(
                model,
                source_ids,
                layer=layer,
                source_span=source_span,
                replacement=target_residuals[layer][0, target_span[0] : target_span[1], :],
            )
            control_logits = patched_next_logits(
                model,
                source_ids,
                layer=layer,
                position=source_control_position,
                replacement=target_residuals[layer][0, target_control_position, :],
            )
            patched_margin = float(
                patched_logits[pair.answer_target_token]
                - patched_logits[pair.answer_source_token]
            )
            control_margin = float(
                control_logits[pair.answer_target_token]
                - control_logits[pair.answer_source_token]
            )
            row: dict[str, Any] = {
                "pair_id": pair.pair_id,
                "task_type": pair.task_type,
                "category": pair.category,
                "function": pair.function,
                "source_entity": pair.source_entity,
                "target_entity": pair.target_entity,
                "source_answer": pair.source_answer,
                "target_answer": pair.target_answer,
                "layer": layer,
                "source_margin": source_margin,
                "target_margin": target_margin,
                "patched_margin": patched_margin,
                "transfer": (
                    None
                    if pair.task_type == "entity_bias"
                    else normalized_transfer(patched_margin, source_margin, target_margin)
                ),
                "control_margin": control_margin,
                "control_transfer": (
                    None
                    if pair.task_type == "entity_bias"
                    else normalized_transfer(control_margin, source_margin, target_margin)
                ),
                "source_entity_span": list(source_span),
                "target_entity_span": list(target_span),
                "span_mapping": normalized_span_mapping(
                    source_span[1] - source_span[0], target_span[1] - target_span[0]
                ),
                "span_mapping_strategy": "normalized_nearest",
                "source_control_position": source_control_position,
                "target_control_position": target_control_position,
                "source_answer_rank": int((source_logits > source_logits[pair.answer_source_token]).sum()) + 1,
                "target_answer_rank": int((target_logits > target_logits[pair.answer_target_token]).sum()) + 1,
                "source_target_answer_rank": int((source_logits > source_logits[pair.answer_target_token]).sum()) + 1,
                "target_source_answer_rank": int((target_logits > target_logits[pair.answer_source_token]).sum()) + 1,
            }
            if pair.task_type == "entity_bias":
                row.update(
                    {
                        "task_type": pair.task_type,
                        "content_id": pair.content_id,
                        "contrast_id": pair.contrast_id,
                        "condition": pair.condition,
                        "pairing_strategy": pair.pairing_strategy,
                        "direction": pair.direction,
                        "expected_outcome": pair.expected_outcome,
                        "margin_definition": "logit(positive)-logit(negative)",
                        "direct_entity_effect": target_margin - source_margin,
                        "causal_patch_effect": patched_margin - source_margin,
                        "control_patch_effect": control_margin - source_margin,
                        "source_expected_margin": expected_outcome_margin(
                            source_margin, pair.expected_outcome
                        ),
                        "target_expected_margin": expected_outcome_margin(
                            target_margin, pair.expected_outcome
                        ),
                        "patched_expected_margin": expected_outcome_margin(
                            patched_margin, pair.expected_outcome
                        ),
                    }
                )
            if lens is not None and layer in lens.jacobians:
                row["entity_target_minus_source_readout"] = float(
                    _span_readout_difference(
                        model,
                        lens,
                        source_residuals[layer][0, source_span[0] : source_span[1]],
                        target_residuals[layer][0, target_span[0] : target_span[1]],
                        layer,
                        pair,
                    )
                )
                source_answer_logits = model.unembed(
                    lens.transport(source_residuals[layer][0, -1].float(), layer)
                ).float().cpu()
                target_answer_logits = model.unembed(
                    lens.transport(target_residuals[layer][0, -1].float(), layer)
                ).float().cpu()
                row["answer_target_minus_source_readout"] = float(
                    target_answer_logits[pair.answer_target_token]
                    - source_answer_logits[pair.answer_source_token]
                )
            results.append(row)
        if pair_index % 10 == 0 or pair_index == len(pairs):
            print(f"Processed {pair_index}/{len(pairs)} pairs")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(results)} layer rows to {destination}")
    return destination


def summarize(
    input_path: str = "artifacts/counterfactual_patching/patch_results.jsonl",
    output_dir: str = "artifacts/counterfactual_patching",
) -> Path:
    rows = pd.read_json(input_path, lines=True)
    rng = np.random.default_rng(0)
    summary_rows: list[dict[str, Any]] = []
    for (category, layer), group in rows.groupby(["category", "layer"]):
        values = group["transfer"].dropna().to_numpy(dtype=float)
        if len(values):
            samples = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        else:
            ci_low = ci_high = float("nan")
        summary_rows.append(
            {
                "category": category,
                "layer": layer,
                "pairs": group["pair_id"].nunique(),
                "mean_transfer": group["transfer"].mean(),
                "median_transfer": group["transfer"].median(),
                "transfer_ci_low": ci_low,
                "transfer_ci_high": ci_high,
                "mean_control_transfer": group["control_transfer"].mean(),
                "mean_patched_margin": group["patched_margin"].mean(),
                "mean_direct_entity_effect": (
                    group["direct_entity_effect"].mean()
                    if "direct_entity_effect" in group
                    else float("nan")
                ),
                "mean_causal_patch_effect": (
                    group["causal_patch_effect"].mean()
                    if "causal_patch_effect" in group
                    else float("nan")
                ),
                "mean_source_expected_margin": (
                    group["source_expected_margin"].mean()
                    if "source_expected_margin" in group
                    else float("nan")
                ),
                "mean_target_expected_margin": (
                    group["target_expected_margin"].mean()
                    if "target_expected_margin" in group
                    else float("nan")
                ),
                "mean_patched_expected_margin": (
                    group["patched_expected_margin"].mean()
                    if "patched_expected_margin" in group
                    else float("nan")
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "layer_summary.csv"
    summary.to_csv(summary_path, index=False)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 5))
        for category, group in summary.groupby("category"):
            group = group.sort_values("layer")
            axis.plot(group["layer"], group["mean_transfer"], marker="o", label=category)
            axis.fill_between(
                group["layer"].to_numpy(),
                group["transfer_ci_low"].to_numpy(),
                group["transfer_ci_high"].to_numpy(),
                alpha=0.12,
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Patch layer", ylabel="Mean target transfer", title="Entity activation patch transfer")
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / "transfer_by_layer.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    print(f"Wrote summary to {summary_path}")
    return summary_path


def visualize(
    input_path: str = "artifacts/counterfactual_patching/patch_results.jsonl",
    output_dir: str = "artifacts/counterfactual_patching",
) -> Path:
    """Create patch and Jacobian-lens representation heatmaps."""
    summary_path = summarize(input_path, output_dir)
    rows = pd.read_json(input_path, lines=True)
    summary = pd.read_csv(summary_path)
    destination = Path(output_dir)

    import matplotlib.pyplot as plt

    patch_matrix = summary.pivot(
        index="category", columns="layer", values="mean_transfer"
    ).sort_index()
    figure, axis = plt.subplots(figsize=(11, 4.5))
    image = axis.imshow(
        patch_matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axis.set(
        xticks=range(len(patch_matrix.columns)),
        xticklabels=patch_matrix.columns,
        yticks=range(len(patch_matrix.index)),
        yticklabels=patch_matrix.index,
        xlabel="Patch layer",
        ylabel="Entity category",
        title="Corrected entity patch transfer",
    )
    figure.colorbar(image, ax=axis, label="Normalized transfer")
    figure.tight_layout()
    figure.savefig(destination / "patch_transfer_heatmap.png", dpi=160)
    plt.close(figure)

    readout_columns = [
        "entity_target_minus_source_readout",
        "answer_target_minus_source_readout",
    ]
    available = [column for column in readout_columns if column in rows]
    if available:
        readout = rows.copy()
        readout[available] = readout[available].abs()
        readout = readout.groupby("layer", as_index=True)[available].mean().T
        readout = readout.dropna(axis=1, how="all")
        figure, axis = plt.subplots(figsize=(10, 3.5))
        image = axis.imshow(readout.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        axis.set(
            xticks=range(len(readout.columns)),
            xticklabels=readout.columns,
            yticks=range(len(readout.index)),
            yticklabels=["Entity readout", "Answer readout"],
            xlabel="Lens layer",
            title="Jacobian-lens representation contrast",
        )
        figure.colorbar(image, ax=axis, label="Mean absolute target/source logit contrast")
        figure.tight_layout()
        figure.savefig(destination / "jacobian_readout_heatmap.png", dpi=160)
        plt.close(figure)

    print(f"Wrote visualizations to {destination}")
    return destination
