"""Bilingual holdout evaluation for model-specific Jacobian-lens candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

import jlens
from jlens.hooks import ActivationRecorder

from llm_bias.core.lens_artifacts import (
    expected_source_layers,
    load_lens_metadata,
    model_slug,
    validate_lens_metadata,
)
from llm_bias.core.model import load_model
from llm_bias.core.prompting import format_prompt


@dataclass(frozen=True)
class BilingualEvalItem:
    item_id: str
    pair_id: str
    language: str
    prompt: str
    native_intermediate: str
    crosslingual_intermediate: str
    target: str


def load_bilingual_holdout(path: str | Path) -> list[BilingualEvalItem]:
    rows: list[BilingualEvalItem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                rows.append(
                    BilingualEvalItem(
                        item_id=str(value["id"]),
                        pair_id=str(value["pair_id"]),
                        language=str(value["language"]),
                        prompt=str(value["prompt"]),
                        native_intermediate=str(value["native_intermediate"]),
                        crosslingual_intermediate=str(
                            value["crosslingual_intermediate"]
                        ),
                        target=str(value["target"]),
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid bilingual holdout row at {path}:{line_number}"
                ) from exc
    if not rows:
        raise ValueError(f"bilingual holdout is empty: {path}")
    if len({row.item_id for row in rows}) != len(rows):
        raise ValueError("bilingual holdout contains duplicate IDs")
    return rows


def single_token_variants(tokenizer: Any, text: str) -> list[int]:
    """Return distinct single-token IDs for raw and leading-space spellings."""
    variants: list[int] = []
    for candidate in (text, f" {text}"):
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1 and int(ids[0]) not in variants:
            variants.append(int(ids[0]))
    return variants


def canonical_concept_token_id(
    tokenizer: Any,
    text: str,
    *,
    language: str,
) -> int:
    """Choose one language-appropriate concept token without variant advantage."""
    candidates = (
        (f" {text}", text) if language == "en" else (text, f" {text}")
    )
    for candidate in candidates:
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(
        f"concept has no canonical single-token spelling: {language}/{text!r}"
    )


def _metric_summary(ranks: torch.Tensor) -> dict[str, float]:
    values = ranks.to(torch.float64)
    return {
        "count": int(values.numel()),
        "mean_log10_rank": float(values.log10().mean().item()),
        "median_rank": float(values.median().item()),
        "mrr": float(values.reciprocal().mean().item()),
        **{
            f"pass_at_{threshold}": float((values <= threshold).float().mean().item())
            for threshold in (1, 5, 10, 25, 100)
        },
    }


def summarize_candidate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("candidate evaluation contains no rows")
    by_language: dict[str, Any] = {}
    for language in sorted({str(row["language"]) for row in values}):
        subset = [row for row in values if row["language"] == language]
        by_language[language] = {
            "native": _metric_summary(
                torch.tensor([row["native_min_rank"] for row in subset])
            ),
            "bilingual": _metric_summary(
                torch.tensor([row["bilingual_min_rank"] for row in subset])
            ),
        }
    balanced_native_mean_log10_rank = sum(
        metrics["native"]["mean_log10_rank"] for metrics in by_language.values()
    ) / len(by_language)
    balanced_bilingual_mean_log10_rank = sum(
        metrics["bilingual"]["mean_log10_rank"]
        for metrics in by_language.values()
    ) / len(by_language)
    return {
        "overall": {
            "native": _metric_summary(
                torch.tensor([row["native_min_rank"] for row in values])
            ),
            "bilingual": _metric_summary(
                torch.tensor([row["bilingual_min_rank"] for row in values])
            ),
        },
        "by_language": by_language,
        "balanced_native_mean_log10_rank": balanced_native_mean_log10_rank,
        "balanced_bilingual_mean_log10_rank": (
            balanced_bilingual_mean_log10_rank
        ),
        "selection_score": -balanced_native_mean_log10_rank,
    }


def _pair_native_log10_ranks(
    rows: Iterable[dict[str, Any]],
) -> dict[str, float]:
    """Reduce the two language observations to one score per semantic pair."""
    by_pair: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        rank = int(row["native_min_rank"])
        if rank < 1:
            raise ValueError("native token ranks must be positive")
        by_pair.setdefault(str(row["pair_id"]), []).append(
            (str(row["language"]), math.log10(rank))
        )
    if not by_pair:
        raise ValueError("candidate evaluation contains no paired rows")
    scores: dict[str, float] = {}
    for pair_id, values in by_pair.items():
        languages = [language for language, _score in values]
        if len(values) != 2 or set(languages) != {"en", "zh-CN"}:
            raise ValueError(
                f"pair {pair_id!r} must contain exactly one en and one "
                "zh-CN observation"
            )
        scores[pair_id] = sum(score for _language, score in values) / 2
    return scores


def paired_candidate_uncertainty(
    *,
    selected: str,
    candidate_rows: dict[str, Iterable[dict[str, Any]]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Quantify paired score differences without changing candidate selection.

    Negative selected-minus-competitor differences favor the selected lens.
    The sign-flip p-value is one-sided for that preregistered direction. Since
    the same holdout selected the winner, these statistics are descriptive.
    """
    if selected not in candidate_rows:
        raise ValueError(f"selected candidate is missing: {selected}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    pair_scores = {
        name: _pair_native_log10_ranks(rows)
        for name, rows in candidate_rows.items()
    }
    selected_scores = pair_scores[selected]
    pair_ids = sorted(selected_scores)
    comparisons: dict[str, Any] = {}
    for offset, competitor in enumerate(sorted(pair_scores)):
        if competitor == selected:
            continue
        if set(pair_scores[competitor]) != set(pair_ids):
            raise ValueError(
                f"{competitor} does not contain the same semantic pairs as "
                f"{selected}"
            )
        differences = torch.tensor(
            [
                selected_scores[pair_id]
                - pair_scores[competitor][pair_id]
                for pair_id in pair_ids
            ],
            dtype=torch.float64,
        )
        observed = differences.mean()
        generator = torch.Generator().manual_seed(seed + offset)
        sample_indices = torch.randint(
            len(pair_ids),
            (n_resamples, len(pair_ids)),
            generator=generator,
        )
        bootstrap = differences[sample_indices].mean(1)
        lower, upper = torch.quantile(
            bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        signs = (
            torch.randint(
                0,
                2,
                (n_resamples, len(pair_ids)),
                generator=generator,
            )
            .mul(2)
            .sub(1)
        )
        null_means = (differences * signs).mean(1)
        one_sided_p = (
            int((null_means <= observed).sum().item()) + 1
        ) / (n_resamples + 1)
        comparisons[competitor] = {
            "pair_count": len(pair_ids),
            "selected_minus_competitor_mean_log10_rank": float(
                observed.item()
            ),
            "paired_bootstrap_95_ci": [
                float(lower.item()),
                float(upper.item()),
            ],
            "sign_flip_one_sided_p": one_sided_p,
        }
    return {
        "unit": "semantic_pair_mean_across_en_and_zh-CN",
        "difference_direction": (
            "selected_minus_competitor; negative favors selected"
        ),
        "bootstrap_resamples": n_resamples,
        "sign_flip_resamples": n_resamples,
        "random_seed": seed,
        "interpretation": (
            "descriptive only: candidate selection and uncertainty both use "
            "the same holdout"
        ),
        "comparisons": comparisons,
    }


def validate_candidate_calibration(
    *,
    name: str,
    lens: Any,
    lens_path: str | Path,
    expected_n_prompts: int,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Reject incomplete or mislabeled candidate calibration artifacts."""
    if int(lens.n_prompts) != expected_n_prompts:
        raise ValueError(
            f"{name} lens was fitted on {lens.n_prompts} prompts, expected "
            f"{expected_n_prompts}"
        )
    metadata = load_lens_metadata(lens_path)
    if metadata is None:
        raise ValueError(f"{name} lens is missing reproducibility metadata")
    validate_lens_metadata(metadata=metadata, lens_path=lens_path)
    metadata_model = metadata.get("model")
    if model_name is not None and (
        not isinstance(metadata_model, str)
        or model_slug(metadata_model) != model_slug(model_name)
    ):
        raise ValueError(f"{name} lens metadata model does not match {model_name!r}")
    if int(metadata.get("calibration_count", -1)) != expected_n_prompts:
        raise ValueError(
            f"{name} metadata calibration_count does not match "
            f"{expected_n_prompts}"
        )
    source = metadata.get("calibration_source")
    if not isinstance(source, str) or Path(source).stem != name:
        raise ValueError(
            f"{name} metadata calibration_source is mislabeled: {source!r}"
        )
    if metadata.get("use_chat_template") is not True:
        raise ValueError(f"{name} candidate must use the Qwen chat template")
    if metadata.get("enable_thinking") is not False:
        raise ValueError(f"{name} candidate must disable thinking")
    return metadata


@torch.no_grad()
def _record_final_position_residuals(
    model: Any,
    tokenizer: Any,
    items: list[BilingualEvalItem],
    layers: list[int],
    *,
    max_seq_len: int,
    use_chat_template: bool,
) -> dict[int, torch.Tensor]:
    """Keep compact final-position residuals in memory only."""
    values: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for item in items:
        prompt = format_prompt(
            tokenizer,
            item.prompt,
            use_chat_template=use_chat_template,
            enable_thinking=False,
        )
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            for layer in layers:
                values[layer].append(
                    recorder.activations[layer][0, -1].detach().float().cpu()
                )
    return {layer: torch.stack(rows) for layer, rows in values.items()}


def _padded_expected_ids(
    tokenizer: Any,
    items: list[BilingualEvalItem],
) -> tuple[torch.Tensor, torch.Tensor]:
    per_item: list[list[int]] = []
    for item in items:
        cross_language = "zh-CN" if item.language == "en" else "en"
        native = canonical_concept_token_id(
            tokenizer,
            item.native_intermediate,
            language=item.language,
        )
        cross = canonical_concept_token_id(
            tokenizer,
            item.crosslingual_intermediate,
            language=cross_language,
        )
        per_item.append([native] + ([] if cross == native else [cross]))
    width = max(map(len, per_item))
    ids = torch.zeros((len(items), width), dtype=torch.long)
    native_mask = torch.zeros_like(ids, dtype=torch.bool)
    bilingual_mask = torch.zeros_like(ids, dtype=torch.bool)
    for row_index, row in enumerate(per_item):
        ids[row_index, : len(row)] = torch.tensor(row)
        native_mask[row_index, 0] = True
        bilingual_mask[row_index, : len(row)] = True
    return ids, torch.stack((native_mask, bilingual_mask))


@torch.no_grad()
def _score_lens(
    *,
    model: Any,
    lens: jlens.JacobianLens,
    residuals: dict[int, torch.Tensor],
    items: list[BilingualEvalItem],
    tokenizer: Any,
    band_layers: list[int],
) -> list[dict[str, Any]]:
    expected_ids, masks = _padded_expected_ids(tokenizer, items)
    expected_ids = expected_ids.to(model.input_device)
    masks = masks.to(model.input_device)
    native_best = torch.full(
        (len(items),), torch.iinfo(torch.int64).max, device=model.input_device
    )
    bilingual_best = native_best.clone()
    for layer in band_layers:
        hidden = residuals[layer].to(model.input_device)
        logits = model.unembed(lens.transport(hidden, layer)).float()
        target_logits = logits.gather(1, expected_ids)
        ranks = (logits.unsqueeze(-1) > target_logits.unsqueeze(1)).sum(1) + 1
        invalid_rank = logits.shape[-1] + 1
        native_at_layer = ranks.masked_fill(~masks[0], invalid_rank).min(1).values
        bilingual_at_layer = (
            ranks.masked_fill(~masks[1], invalid_rank).min(1).values
        )
        native_best = torch.minimum(native_best, native_at_layer)
        bilingual_best = torch.minimum(bilingual_best, bilingual_at_layer)
        del hidden, logits, target_logits, ranks
    native_values = native_best.cpu().tolist()
    bilingual_values = bilingual_best.cpu().tolist()
    return [
        {
            "id": item.item_id,
            "pair_id": item.pair_id,
            "language": item.language,
            "native_intermediate": item.native_intermediate,
            "crosslingual_intermediate": item.crosslingual_intermediate,
            "native_min_rank": int(native_rank),
            "bilingual_min_rank": int(bilingual_rank),
        }
        for item, native_rank, bilingual_rank in zip(
            items, native_values, bilingual_values, strict=True
        )
    ]


def evaluate_candidates(
    *,
    model_name: str,
    candidate_paths: dict[str, str | Path],
    holdout_path: str | Path,
    output_path: str | Path,
    max_seq_len: int = 128,
    use_chat_template: bool = True,
    band_start: int | None = None,
    band_end: int | None = None,
    expected_n_prompts: int = 128,
) -> dict[str, Any]:
    """Evaluate candidates on a preregistered balanced bilingual rank metric."""
    if len(candidate_paths) < 2:
        raise ValueError("candidate selection requires at least two lenses")
    items = load_bilingual_holdout(holdout_path)
    model, tokenizer, _device = load_model(model_name)
    expected_layers = expected_source_layers(model.n_layers)
    start = band_start if band_start is not None else math.ceil(model.n_layers / 3)
    end = band_end if band_end is not None else math.ceil(model.n_layers * 3 / 4)
    band_layers = [layer for layer in expected_layers if start <= layer < end]
    if not band_layers:
        raise ValueError(f"empty evaluation layer band [{start}, {end})")

    lenses: dict[str, jlens.JacobianLens] = {}
    lens_metadata_values: dict[str, dict[str, Any]] = {}
    for name, path in candidate_paths.items():
        lens = jlens.JacobianLens.load(str(path))
        if lens.d_model != model.d_model:
            raise ValueError(f"{name} lens d_model does not match the model")
        if lens.source_layers != expected_layers:
            raise ValueError(f"{name} lens does not have complete layer coverage")
        metadata = validate_candidate_calibration(
            name=name,
            lens=lens,
            lens_path=path,
            expected_n_prompts=expected_n_prompts,
            model_name=model_name,
        )
        lenses[name] = lens
        lens_metadata_values[name] = metadata

    residuals = _record_final_position_residuals(
        model,
        tokenizer,
        items,
        expected_layers,
        max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
    )
    candidates: dict[str, Any] = {}
    for name, lens in lenses.items():
        rows = _score_lens(
            model=model,
            lens=lens,
            residuals=residuals,
            items=items,
            tokenizer=tokenizer,
            band_layers=band_layers,
        )
        candidates[name] = {
            "lens_path": str(candidate_paths[name]),
            "n_prompts": lens.n_prompts,
            "lens_binary_sha256": lens_metadata_values[name]["binary_sha256"],
            "lens_metadata_sha256": lens_metadata_values[name]["metadata_sha256"],
            "summary": summarize_candidate_rows(rows),
            "rows": rows,
        }
    selected = max(
        candidates,
        key=lambda name: (
            candidates[name]["summary"]["selection_score"],
            -candidates[name]["summary"][
                "balanced_bilingual_mean_log10_rank"
            ],
            name,
        ),
    )
    selection_uncertainty = paired_candidate_uncertainty(
        selected=selected,
        candidate_rows={
            name: candidate["rows"]
            for name, candidate in candidates.items()
        },
    )
    result = {
        "schema_version": 1,
        "model": model_name,
        "holdout": str(holdout_path),
        "holdout_count": len(items),
        "use_chat_template": use_chat_template,
        "layer_band": band_layers,
        "selection_rule": (
            "maximize negative balanced mean log10 native-token rank across "
            "English and Simplified-Chinese holdout prompts; bilingual rank "
            "is the first tie-break"
        ),
        "selected_candidate": selected,
        "selection_uncertainty": selection_uncertainty,
        "candidates": candidates,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
