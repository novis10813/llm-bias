"""Hierarchical activation patching for causal tracing of Buy/Sell decisions.

This module implements the patching protocol described in
docs/activation-patching-causal-tracing/proposal.md. It caches full-sequence residuals
from a source forward pass and patches them into a target forward pass at
specified (layer, position) subsets, then measures the margin change.

No raw activations are persisted to disk. All cached residuals exist only in
GPU memory for the duration of one patching sweep, and only compact scalar
outputs with provenance are emitted.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import read_jsonl, write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.continuation_scoring import (
    CandidateMargin,
    continuation_token_ids,
    score_single_token_margin_fp32,
)
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, token_span

# Semantic span IDs matching the experiment document.
ARTIFACT_SCHEMA_VERSION = 1
PROMPT_TEMPLATE_VERSION = "canonical-valence-v1"
DEFAULT_DECISION_PREFIX = '{\n  "decision": "'
DEFAULT_POSITIVE_CANDIDATE = "buy"
DEFAULT_NEGATIVE_CANDIDATE = "sell"

SPAN_IDS = (
    "all_positions",
    "all_evidence",
    "evidence_qual",
    "evidence_quant",
    "header",
    "instruction_context",
    "instruction",
    "final_position",
)


def _input_tensor(tokenizer: Any, text: str, device: Any) -> torch.Tensor:
    ids = input_ids(tokenizer, text, add_special_tokens=True)
    return torch.tensor([ids], dtype=torch.long, device=device)


def _char_to_token_positions(
    tokenizer: Any, text: str, char_start: int, char_end: int
) -> tuple[int, int] | None:
    """Map a character range to token positions (start inclusive, end exclusive)."""
    return token_span(tokenizer, text, char_start, char_end)


def resolve_prompt_spans(
    tokenizer: Any,
    prompt: str,
    evidence_char_spans: dict[str, list[int]],
) -> dict[str, tuple[int, int]]:
    """Resolve semantic character spans to token position ranges.

    Returns a dict mapping span_id → (token_start, token_end) for the header,
    evidence_qual, evidence_quant, instruction, and derived union spans.
    """
    ids = input_ids(tokenizer, prompt, add_special_tokens=True)
    seq_len = len(ids)

    qual_chars = evidence_char_spans["qual"]
    quant_chars = evidence_char_spans["quant"]

    qual_tok = _char_to_token_positions(
        tokenizer, prompt, qual_chars[0], qual_chars[1]
    )
    quant_tok = _char_to_token_positions(
        tokenizer, prompt, quant_chars[0], quant_chars[1]
    )
    if qual_tok is None or quant_tok is None:
        raise ValueError("could not map evidence character spans to token positions")

    # Header: from start of sequence to the first evidence token.
    header_end = min(qual_tok[0], quant_tok[0])
    # Instruction: from after the last evidence token to the end.
    instruction_start = max(qual_tok[1], quant_tok[1])

    return {
        "header": (0, header_end),
        "evidence_qual": qual_tok,
        "evidence_quant": quant_tok,
        "all_evidence": (min(qual_tok[0], quant_tok[0]), max(qual_tok[1], quant_tok[1])),
        "instruction_context": (instruction_start, max(instruction_start, seq_len - 1)),
        "instruction": (instruction_start, seq_len),
        "all_positions": (0, seq_len),
        "final_position": (seq_len - 1, seq_len),
    }


def _nearest_position_mapping(
    source_span: tuple[int, int],
    target_span: tuple[int, int],
) -> dict[int, int]:
    """Map every target position to its nearest normalized source position.

    The returned mapping is ``target_position -> source_position``. Target-side
    coverage matters because a span patch must replace every requested target
    token even when the source span is shorter. Equal-length spans reduce to an
    identity-by-offset mapping.
    """
    src_start, src_end = source_span
    tgt_start, tgt_end = target_span
    n_src = src_end - src_start
    n_tgt = tgt_end - tgt_start
    if n_src <= 0 or n_tgt <= 0:
        return {}
    mapping = {}
    for target_offset in range(n_tgt):
        if n_tgt == 1:
            source_offset = 0
        else:
            source_offset = round(target_offset * (n_src - 1) / (n_tgt - 1))
        mapping[tgt_start + target_offset] = src_start + source_offset
    return mapping


def _build_position_mapping(
    source_spans: dict[str, tuple[int, int]],
    target_spans: dict[str, tuple[int, int]],
    span_condition: str,
    source_seq_len: int,
    target_seq_len: int,
) -> dict[int, int]:
    """Build a target→source position mapping for one span condition.

    For identical-text spans, positions align by offset. Evidence spans use a
    nearest-normalized mapping that covers every target position.
    """
    if span_condition == "all_positions":
        source_qual = source_spans["evidence_qual"]
        source_quant = source_spans["evidence_quant"]
        target_qual = target_spans["evidence_qual"]
        target_quant = target_spans["evidence_quant"]
        source_segments = (
            (0, source_qual[0]),
            source_qual,
            (source_qual[1], source_quant[0]),
            source_quant,
            (source_quant[1], source_seq_len),
        )
        target_segments = (
            (0, target_qual[0]),
            target_qual,
            (target_qual[1], target_quant[0]),
            target_quant,
            (target_quant[1], target_seq_len),
        )
        mapping: dict[int, int] = {}
        for source_segment, target_segment in zip(
            source_segments, target_segments, strict=True
        ):
            mapping.update(
                _nearest_position_mapping(source_segment, target_segment)
            )
        return mapping

    if span_condition == "final_position":
        return {target_seq_len - 1: source_seq_len - 1}

    if span_condition == "all_evidence":
        mapping = {}
        for span_id in ("evidence_qual", "evidence_quant"):
            sub = _nearest_position_mapping(
                source_spans[span_id], target_spans[span_id]
            )
            mapping.update(sub)
        return mapping

    if span_condition in ("evidence_qual", "evidence_quant"):
        return _nearest_position_mapping(
            source_spans[span_condition], target_spans[span_condition]
        )

    if span_condition == "header":
        # Header text can differ in ticker/name length. Cover every target
        # position with the nearest normalized source position.
        return _nearest_position_mapping(
            source_spans[span_condition], target_spans[span_condition]
        )

    if span_condition in ("instruction_context", "instruction"):
        # Identical-text spans: 1:1 mapping. These states can still differ
        # because the preceding evidence differs.
        src_span = source_spans[span_condition]
        tgt_span = target_spans[span_condition]
        n = min(src_span[1] - src_span[0], tgt_span[1] - tgt_span[0])
        return {tgt_span[0] + i: src_span[0] + i for i in range(n)}

    raise ValueError(f"unknown span condition: {span_condition!r}")


def cache_source_residuals(
    model: Any,
    tokenizer: Any,
    source_prompt: str,
    layers: Sequence[int],
    device: Any,
) -> dict[int, torch.Tensor]:
    """Run the source prompt forward and return full-sequence residuals.

    Each returned tensor has shape [1, seq_len, d_model] and stays on the
    model's device. Callers must not persist these to disk.
    """
    input_tensor = _input_tensor(tokenizer, source_prompt, device)
    return record_residuals(model, input_tensor, layers)


def run_patched_margin(
    *,
    model: Any,
    tokenizer: Any,
    target_prompt: str,
    source_residuals: dict[int, torch.Tensor],
    position_mapping: dict[int, int],
    layers: Sequence[int],
    positive_candidate: str = " Buy",
    negative_candidate: str = " Sell",
    device: Any,
) -> CandidateMargin:
    """Score the target prompt with source residuals patched at specified positions.

    For each layer in ``layers``, a forward hook replaces the target's residual
    at each target position with the source residual selected by the
    ``target_position -> source_position`` mapping.
    """
    layer_set = sorted(set(int(layer) for layer in layers))
    if not position_mapping:
        # No positions to patch — return clean margin.
        return score_single_token_margin_fp32(
            model, tokenizer, target_prompt, positive_candidate, negative_candidate,
            device=device,
        )

    transforms = {}
    for layer in layer_set:
        src_residual = source_residuals[layer]

        def make_transform(
            src_res: torch.Tensor = src_residual,
            pos_map: dict[int, int] = position_mapping,
        ):
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                patched = tensor.clone()
                for tgt_pos, src_pos in pos_map.items():
                    if tgt_pos < tensor.shape[1] and src_pos < src_res.shape[1]:
                        patched[:, tgt_pos, :] = src_res[:, src_pos, :].to(
                            device=tensor.device, dtype=tensor.dtype
                        )
                return patched
            return transform

        transforms[layer] = make_transform()

    with residual_interventions(model, transforms):
        margin = score_single_token_margin_fp32(
            model, tokenizer, target_prompt, positive_candidate, negative_candidate,
            device=device,
        )
    return margin


def run_activation_patching_record(
    *,
    model: Any,
    tokenizer: Any,
    source_prompt: str,
    target_prompt: str,
    source_evidence_char_spans: dict[str, list[int]],
    target_evidence_char_spans: dict[str, list[int]],
    layers: Sequence[int],
    span_condition: str,
    positive_candidate: str = " Buy",
    negative_candidate: str = " Sell",
    device: Any,
    source_residuals: dict[int, torch.Tensor] | None = None,
    source_clean_margin: CandidateMargin | None = None,
    target_clean_margin: CandidateMargin | None = None,
) -> dict[str, Any]:
    """Run one activation patching condition and return a compact result dict.

    If ``source_residuals`` is provided, reuses the cached residuals instead of
    re-running the source forward pass. Likewise for clean margins.
    """
    if span_condition not in SPAN_IDS:
        raise ValueError(
            f"unknown span_condition {span_condition!r}; expected one of {SPAN_IDS}"
        )

    layer_list = sorted(set(int(layer) for layer in layers))

    # Resolve token-level spans.
    source_spans = resolve_prompt_spans(
        tokenizer, source_prompt, source_evidence_char_spans
    )
    target_spans = resolve_prompt_spans(
        tokenizer, target_prompt, target_evidence_char_spans
    )

    source_ids = input_ids(tokenizer, source_prompt, add_special_tokens=True)
    target_ids = input_ids(tokenizer, target_prompt, add_special_tokens=True)
    source_seq_len = len(source_ids)
    target_seq_len = len(target_ids)

    # Build position mapping for this span condition.
    pos_mapping = _build_position_mapping(
        source_spans, target_spans, span_condition,
        source_seq_len, target_seq_len,
    )

    # Cache source residuals if not provided.
    if source_residuals is None:
        source_residuals = cache_source_residuals(
            model, tokenizer, source_prompt, layer_list, device
        )

    # Clean margins.
    if source_clean_margin is None:
        source_clean_margin = score_single_token_margin_fp32(
            model, tokenizer, source_prompt,
            positive_candidate, negative_candidate, device=device,
        )
    if target_clean_margin is None:
        target_clean_margin = score_single_token_margin_fp32(
            model, tokenizer, target_prompt,
            positive_candidate, negative_candidate, device=device,
        )

    # Run patched forward pass.
    patched_margin = run_patched_margin(
        model=model,
        tokenizer=tokenizer,
        target_prompt=target_prompt,
        source_residuals=source_residuals,
        position_mapping=pos_mapping,
        layers=layer_list,
        positive_candidate=positive_candidate,
        negative_candidate=negative_candidate,
        device=device,
    )

    target_clean_value = target_clean_margin.value
    delta_margin = patched_margin.value - target_clean_value
    import math
    flip = (
        math.copysign(1, patched_margin.value) != math.copysign(1, target_clean_value)
        if target_clean_value != 0
        else False
    )

    target_positions = sorted(pos_mapping)

    return {
        "layers": layer_list,
        "span_condition": span_condition,
        "positions_patched": target_positions,
        "position_count": len(target_positions),
        "source_seq_len": source_seq_len,
        "target_seq_len": target_seq_len,
        "source_clean_margin": source_clean_margin.value,
        "target_clean_margin": target_clean_value,
        "patched_margin": patched_margin.value,
        "delta_margin": delta_margin,
        "flip": flip,
        "score": {
            "positive": patched_margin.positive.to_dict(),
            "negative": patched_margin.negative.to_dict(),
        },
    }


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    """Load and validate a prepared valence-pair JSONL artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"valence pair artifact not found: {path}")
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{number}") from exc
            if pair.get("artifact_type") != "valence_pairs":
                raise ValueError(f"{path}:{number} is not a valence_pairs record")
            prompts = pair.get("prompts")
            spans = pair.get("evidence_char_spans")
            if not isinstance(prompts, Mapping) or not isinstance(spans, Mapping):
                raise ValueError(f"{path}:{number} lacks prompts or evidence_char_spans")
            for condition in ("positive", "negative"):
                if not isinstance(prompts.get(condition), str):
                    raise ValueError(f"{path}:{number} lacks {condition} prompt")
                condition_spans = spans.get(condition)
                if not isinstance(condition_spans, Mapping):
                    raise ValueError(f"{path}:{number} lacks {condition} spans")
                for kind in ("qual", "quant"):
                    value = condition_spans.get(kind)
                    if (
                        not isinstance(value, list)
                        or len(value) != 2
                        or not all(isinstance(item, int) for item in value)
                    ):
                        raise ValueError(
                            f"{path}:{number} has invalid {condition}/{kind} span"
                        )
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"valence pair artifact is empty: {path}")
    return pairs


def _prepare_scoring_condition(
    tokenizer: Any,
    raw_prompt: str,
    evidence_char_spans: Mapping[str, Sequence[int]],
    *,
    decision_prefix: str,
) -> tuple[str, dict[str, list[int]]]:
    """Apply the chat template and shift raw evidence spans into scored text."""
    formatted = format_prompt(
        tokenizer,
        raw_prompt,
        use_chat_template=True,
        enable_thinking=False,
    )
    raw_offset = formatted.find(raw_prompt)
    if raw_offset < 0:
        raise ValueError("formatted chat prompt does not contain the raw valence prompt")
    scoring_prompt = formatted + decision_prefix
    shifted = {
        kind: [raw_offset + int(span[0]), raw_offset + int(span[1])]
        for kind, span in evidence_char_spans.items()
    }
    return scoring_prompt, shifted


def _validated_layer_conditions(
    layer_conditions: Mapping[str, Sequence[int]],
    *,
    n_layers: int,
) -> dict[str, tuple[int, ...]]:
    if not layer_conditions:
        raise ValueError("at least one layer condition is required")
    result = {}
    for name, layers in layer_conditions.items():
        if not str(name).strip():
            raise ValueError("layer condition names must be non-empty")
        resolved = tuple(sorted({int(layer) for layer in layers}))
        if not resolved:
            raise ValueError(f"layer condition {name!r} is empty")
        if resolved[0] < 0 or resolved[-1] >= n_layers:
            raise ValueError(
                f"layer condition {name!r} is outside model range 0..{n_layers - 1}"
            )
        result[str(name)] = resolved
    return result


def default_layer_conditions(phase: str, *, n_layers: int) -> dict[str, tuple[int, ...]]:
    """Return the frozen Draft-1 default layer conditions for one phase."""
    if phase == "phase0":
        return {}
    last_intermediate = n_layers - 2
    if phase == "phase1":
        result = {
            f"single_L{layer}": (layer,)
            for layer in range(last_intermediate + 1)
        }
        fixed_ranges = {
            "early_L0-L13": (0, min(13, last_intermediate)),
            "full_L14-L26": (14, min(26, last_intermediate)),
            "late_L24-L30": (24, min(30, last_intermediate)),
        }
        for name, (start, end) in fixed_ranges.items():
            if start <= end:
                result[name] = tuple(range(start, end + 1))
        for end in (16, 18, 20, 22, 24, 26):
            if 14 <= min(end, last_intermediate):
                result[f"cumulative_L14-L{end}"] = tuple(
                    range(14, min(end, last_intermediate) + 1)
                )
        return result
    if phase == "phase2":
        if last_intermediate < 14:
            return {f"full_L0-L{last_intermediate}": tuple(range(n_layers - 1))}
        end = min(26, last_intermediate)
        return {f"full_L14-L{end}": tuple(range(14, end + 1))}
    raise ValueError("phase3 requires explicit layer conditions")


def default_span_conditions(phase: str) -> tuple[str, ...]:
    if phase == "phase0":
        return ()
    if phase == "phase1":
        return ("all_positions",)
    if phase == "phase2":
        return (
            "all_evidence",
            "evidence_qual",
            "evidence_quant",
            "header",
            "instruction",
            "final_position",
        )
    raise ValueError("phase3 requires explicit span conditions")


def _clean_gate_record(
    pair: Mapping[str, Any],
    positive_margin: CandidateMargin,
    negative_margin: CandidateMargin,
) -> dict[str, Any]:
    eligible = positive_margin.value > 0.0 and negative_margin.value < 0.0
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "causal_trace_baseline",
        "record_id": stable_record_id("causal-trace-baseline", str(pair["record_id"])),
        "pair_record_id": pair["record_id"],
        "ticker": pair["ticker"],
        "source_trial_key": pair["source_trial_key"],
        "positive_margin": positive_margin.value,
        "negative_margin": negative_margin.value,
        "clean_decision_match": eligible,
        "positive_score": positive_margin.to_dict(),
        "negative_score": negative_margin.to_dict(),
    }


def summarize_patching_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute equal-ticker means and flip rates per frozen condition."""
    grouped: dict[tuple[str, str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = (
            str(record["patching_direction"]),
            str(record["layer_condition"]),
            str(record["span_condition"]),
        )
        grouped[key][str(record["ticker"])].append(record)
    rows = []
    for (direction, layer_condition, span_condition), by_ticker in sorted(grouped.items()):
        ticker_delta = []
        ticker_flip = []
        for ticker_records in by_ticker.values():
            ticker_delta.append(
                sum(float(row["delta_margin"]) for row in ticker_records)
                / len(ticker_records)
            )
            ticker_flip.append(
                sum(bool(row["flip"]) for row in ticker_records)
                / len(ticker_records)
            )
        rows.append(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "causal_trace_summary",
                "patching_direction": direction,
                "layer_condition": layer_condition,
                "span_condition": span_condition,
                "ticker_count": len(by_ticker),
                "record_count": sum(len(value) for value in by_ticker.values()),
                "equal_ticker_mean_delta_margin": sum(ticker_delta) / len(ticker_delta),
                "equal_ticker_flip_rate": sum(ticker_flip) / len(ticker_flip),
            }
        )
    return rows


def _confirmation_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _confirmation_bootstrap_ci(
    values: Sequence[float], *, seed: int, samples: int, confidence: float
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty ticker collection")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid confirmation bootstrap configuration")
    rng = random.Random(seed)
    count = len(values)
    bootstrap_means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return [
        _confirmation_quantile(bootstrap_means, alpha),
        _confirmation_quantile(bootstrap_means, 1.0 - alpha),
    ]


def _confirmation_exact_sign_flip_p(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot sign-flip an empty ticker collection")
    observed = sum(values) / len(values)
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null_mean = sum(value * sign for value, sign in zip(values, signs, strict=True)) / len(values)
        if null_mean >= observed - 1e-15:
            count += 1
    return count / (2 ** len(values))


def _confirmation_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def evaluate_activation_patching_confirmation(
    records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Evaluate a frozen activation-patching confirmation matrix.

    The aggregation unit is a ticker: trials are averaged within each
    ticker/direction/condition, directions are averaged within ticker, and
    tickers receive equal weight. The function only consumes compact records.
    """
    if split not in {"calibration", "test"}:
        raise ValueError("split must be calibration or test")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    directions = tuple(str(value) for value in config.get("directions", ()))
    spans = tuple(str(value) for value in config.get("span_conditions", ()))
    primary_diagonal = config.get("primary_diagonal")
    if not directions or len(set(directions)) != len(directions):
        raise ValueError("config directions must be non-empty and unique")
    if not spans or len(set(spans)) != len(spans):
        raise ValueError("config span_conditions must be non-empty and unique")
    if not isinstance(primary_diagonal, Sequence) or not primary_diagonal:
        raise ValueError("config primary_diagonal must be non-empty")
    layer_order = tuple(
        str(entry["layer_condition"])
        for entry in primary_diagonal
        if isinstance(entry, Mapping) and "layer_condition" in entry
    )
    if len(layer_order) != len(primary_diagonal) or len(set(layer_order)) != len(layer_order):
        raise ValueError("primary_diagonal layer conditions must be unique")
    bootstrap_seed = int(config.get("bootstrap_seed", config.get("seed", 0)))
    bootstrap_samples = int(config.get("bootstrap_samples", config.get("samples", 0)))
    confidence = float(config.get("ci_level", 0.95))
    if bootstrap_samples < 1:
        raise ValueError("config bootstrap_samples must be positive")

    baseline_by_pair: dict[str, Mapping[str, Any]] = {}
    eligible_pairs = 0
    eligible_tickers: set[str] = set()
    for baseline in baseline_records:
        pair_id = str(baseline.get("pair_record_id", ""))
        if not pair_id or pair_id in baseline_by_pair:
            raise ValueError("baseline records must have unique pair_record_id values")
        baseline_by_pair[pair_id] = baseline
        if bool(baseline.get("clean_decision_match")):
            eligible_pairs += 1
            ticker = str(baseline.get("ticker", ""))
            if not ticker:
                raise ValueError("eligible baseline records must contain a ticker")
            eligible_tickers.add(ticker)

    trial_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seen_records: set[tuple[str, str, str, str]] = set()
    for record in records:
        pair_id = str(record.get("pair_record_id", ""))
        baseline = baseline_by_pair.get(pair_id)
        if baseline is None:
            raise ValueError(f"patching record references unknown pair_record_id: {pair_id!r}")
        if not bool(baseline.get("clean_decision_match")):
            continue
        direction = str(record.get("patching_direction", ""))
        layer = str(record.get("layer_condition", ""))
        span = str(record.get("span_condition", ""))
        if direction not in directions:
            raise ValueError(f"unknown patching direction: {direction!r}")
        if layer not in layer_order:
            raise ValueError(f"unknown layer condition: {layer!r}")
        if span not in spans:
            raise ValueError(f"unknown span condition: {span!r}")
        ticker = str(record.get("ticker") or "")
        baseline_ticker = str(baseline.get("ticker") or "")
        if not ticker or ticker != baseline_ticker:
            raise ValueError(f"ticker mismatch for pair_record_id {pair_id!r}")
        record_key = (pair_id, layer, span, direction)
        if record_key in seen_records:
            raise ValueError(f"duplicate confirmation record: {record_key}")
        seen_records.add(record_key)
        source_margin = _confirmation_float(
            record.get("source_clean_margin"), field="source_clean_margin"
        )
        target_margin = _confirmation_float(
            record.get("target_clean_margin"), field="target_clean_margin"
        )
        delta_margin = _confirmation_float(record.get("delta_margin"), field="delta_margin")
        denominator = source_margin - target_margin
        if denominator == 0.0:
            raise ValueError("source_clean_margin and target_clean_margin must differ")
        normalized_transfer = delta_margin / denominator
        if not math.isfinite(normalized_transfer):
            raise ValueError("normalized transfer must be finite")
        trial_values[(ticker, layer, span, direction)].append(normalized_transfer)

    if not trial_values:
        raise ValueError("no eligible activation-patching records")
    eligible_pair_ids = {
        pair_id
        for pair_id, baseline in baseline_by_pair.items()
        if bool(baseline.get("clean_decision_match"))
    }
    expected_records = {
        (pair_id, layer, span, direction)
        for pair_id in eligible_pair_ids
        for layer in layer_order
        for span in spans
        for direction in directions
    }
    missing_records = expected_records - seen_records
    if missing_records:
        example = sorted(missing_records)[0]
        raise ValueError(
            f"confirmation matrix is incomplete: {len(missing_records)} missing records; "
            f"first={example}"
        )

    ticker_direction_means: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for (ticker, layer, span, direction), values in sorted(trial_values.items()):
        ticker_direction_means[(ticker, layer, span)][direction] = sum(values) / len(values)

    ticker_condition_means: dict[tuple[str, str, str], float] = {}
    for key, direction_means in ticker_direction_means.items():
        missing = set(directions) - set(direction_means)
        if missing:
            raise ValueError(f"missing directions for ticker condition {key}: {sorted(missing)}")
        ticker_condition_means[key] = sum(direction_means.values()) / len(directions)

    unknown_layers = set(layer_order) - {
        key[1] for key in ticker_condition_means
    }
    if unknown_layers:
        raise ValueError(f"missing primary layer conditions: {sorted(unknown_layers)}")

    def ticker_values(layer: str, span: str) -> dict[str, float]:
        values = {
            ticker: value
            for (ticker, condition_layer, condition_span), value in ticker_condition_means.items()
            if condition_layer == layer and condition_span == span
        }
        if not values:
            raise ValueError(f"no eligible values for {layer}/{span}")
        return values

    matrix: list[dict[str, Any]] = []
    for layer in layer_order:
        for span in spans:
            values_by_ticker = ticker_values(layer, span)
            values = [values_by_ticker[ticker] for ticker in sorted(values_by_ticker)]
            direction_means = {
                direction: sum(
                    ticker_direction_means[(ticker, layer, span)][direction]
                    for ticker in sorted(values_by_ticker)
                )
                / len(values_by_ticker)
                for direction in directions
            }
            matrix.append(
                {
                    "layer_condition": layer,
                    "span_condition": span,
                    "equal_ticker_mean_transfer": sum(values) / len(values),
                    "ticker_bootstrap_95_ci": _confirmation_bootstrap_ci(
                        values,
                        seed=bootstrap_seed + sum(map(ord, layer + span)),
                        samples=bootstrap_samples,
                        confidence=confidence,
                    ),
                    "ticker_count": len(values),
                    "direction_means": direction_means,
                }
            )

    contrasts: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for index, diagonal in enumerate(primary_diagonal):
        layer = str(diagonal["layer_condition"])
        primary_span = str(diagonal["span_condition"])
        control_spans = [span for span in spans if span != primary_span]
        if len(control_spans) != 2:
            raise ValueError("each primary diagonal row must have exactly two control spans")
        primary_values = ticker_values(layer, primary_span)
        control_values = [ticker_values(layer, span) for span in control_spans]
        tickers = sorted(set(primary_values) & set(control_values[0]) & set(control_values[1]))
        if not tickers:
            raise ValueError(f"no common eligible tickers for contrast {layer}/{primary_span}")
        contrast_values = [
            primary_values[ticker]
            - (control_values[0][ticker] + control_values[1][ticker]) / 2.0
            for ticker in tickers
        ]
        raw_p = _confirmation_exact_sign_flip_p(contrast_values)
        raw_p_values.append(raw_p)
        contrasts.append(
            {
                "layer_condition": layer,
                "primary_span": primary_span,
                "control_spans": control_spans,
                "equal_ticker_mean_row_dominance": sum(contrast_values) / len(contrast_values),
                "ticker_bootstrap_95_ci": _confirmation_bootstrap_ci(
                    contrast_values,
                    seed=bootstrap_seed + 1000 + index,
                    samples=bootstrap_samples,
                    confidence=confidence,
                ),
                "one_sided_exact_sign_flip_p": raw_p,
                "positive_ticker_fraction": sum(value > 0.0 for value in contrast_values)
                / len(contrast_values),
                "ticker_values": {
                    ticker: value for ticker, value in zip(tickers, contrast_values, strict=True)
                },
            }
        )

    from llm_bias.core.analysis.statistics import holm_bonferroni

    for contrast, adjusted in zip(
        contrasts, holm_bonferroni(raw_p_values), strict=True
    ):
        contrast["holm_adjusted_p"] = adjusted

    checks: list[dict[str, Any]] = [
        {
            "gate": "eligible_sample",
            "pass": eligible_pairs >= int(config["minimum_eligible_pairs"])
            and len(eligible_tickers) >= int(config["minimum_eligible_tickers"]),
            "eligible_pairs": eligible_pairs,
            "eligible_tickers": len(eligible_tickers),
        }
    ]
    matrix_by_condition = {
        (row["layer_condition"], row["span_condition"]): row for row in matrix
    }
    for diagonal in primary_diagonal:
        layer = str(diagonal["layer_condition"])
        span = str(diagonal["span_condition"])
        row = matrix_by_condition[(layer, span)]
        checks.extend(
            [
                {
                    "gate": "diagonal_minimum",
                    "layer_condition": layer,
                    "span_condition": span,
                    "threshold": float(diagonal["minimum_mean_transfer"]),
                    "observed": row["equal_ticker_mean_transfer"],
                    "pass": row["equal_ticker_mean_transfer"]
                    >= float(diagonal["minimum_mean_transfer"]),
                },
                {
                    "gate": "bidirectional_positive",
                    "layer_condition": layer,
                    "span_condition": span,
                    "observed": row["direction_means"],
                    "pass": all(value > 0.0 for value in row["direction_means"].values()),
                },
            ]
        )
    for contrast in contrasts:
        checks.extend(
            [
                {
                    "gate": "row_dominance_minimum",
                    "layer_condition": contrast["layer_condition"],
                    "threshold": float(config["row_dominance_minimum"]),
                    "observed": contrast["equal_ticker_mean_row_dominance"],
                    "pass": contrast["equal_ticker_mean_row_dominance"]
                    > float(config["row_dominance_minimum"]),
                },
                {
                    "gate": "row_dominance_ci",
                    "layer_condition": contrast["layer_condition"],
                    "observed": contrast["ticker_bootstrap_95_ci"],
                    "pass": contrast["ticker_bootstrap_95_ci"][0] > 0.0,
                },
                {
                    "gate": "holm_sign_flip",
                    "layer_condition": contrast["layer_condition"],
                    "observed": contrast["holm_adjusted_p"],
                    "pass": contrast["holm_adjusted_p"] < 0.05,
                },
            ]
        )

    observed_tickers = {ticker for ticker, _, _ in ticker_direction_means}
    checks[0]["eligible_tickers"] = len(observed_tickers)
    checks[0]["pass"] = eligible_pairs >= int(config["minimum_eligible_pairs"]) and len(
        observed_tickers
    ) >= int(config["minimum_eligible_tickers"])
    success = all(bool(check["pass"]) for check in checks)
    result: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "causal_trace_confirmation_evaluation",
        "split": split,
        "eligible_pairs": eligible_pairs,
        "eligible_tickers": len(observed_tickers),
        "matrix": matrix,
        "contrasts": contrasts,
        "gate_checks": checks,
        "success": success,
    }
    if split == "calibration":
        result["test_authorized"] = success
        result["interpretation_limit"] = (
            "calibration gate for frozen held-out position-transfer confirmation; no layer/span selection"
        )
    else:
        result["formal"] = True
        result["interpretation_limit"] = (
            "held-out resample-patching position-transfer confirmation for fixed Buy/Sell margin; "
            "sufficiency not necessity, mediation, or unique route"
        )
    return result


def evaluate_activation_patching_confirmation_artifacts(
    records_path: str | Path,
    baseline_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate compact JSONL artifacts and publish a new JSON result atomically."""
    records_path = Path(records_path)
    baseline_path = Path(baseline_path)
    config_path = Path(config_path)
    output_path = Path(output_path)
    records = read_jsonl(records_path)
    baseline_records = read_jsonl(baseline_path)
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_value, Mapping):
        raise ValueError("confirmation config must be a JSON object")
    result = evaluate_activation_patching_confirmation(
        records, baseline_records, config_value, split=split
    )
    result.update(
        {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "parent_records": str(records_path),
            "parent_sha256": sha256_file(records_path),
            "baseline_records": str(baseline_path),
            "baseline_sha256": sha256_file(baseline_path),
        }
    )
    write_json(output_path, result, overwrite=False)
    return result


def run_activation_patching_pipeline(
    *,
    pairs_path: str | Path,
    model_name: str,
    run_id: str,
    phase: str = "phase1",
    layer_conditions: Mapping[str, Sequence[int]] | None = None,
    span_conditions: Sequence[str] | None = None,
    directions: Sequence[str] = ("positive_to_negative", "negative_to_positive"),
    decision_prefix: str = DEFAULT_DECISION_PREFIX,
    positive_candidate: str = DEFAULT_POSITIVE_CANDIDATE,
    negative_candidate: str = DEFAULT_NEGATIVE_CANDIDATE,
    dataset: str = "jspace-causal-tracing",
    artifact_root: str | Path = "artifacts",
    max_records: int | None = None,
    max_seq_len: int = 1024,
) -> Path:
    """Run one hierarchical activation-patching phase on prepared valence pairs."""
    if phase not in {"phase0", "phase1", "phase2", "phase3"}:
        raise ValueError("phase must be one of phase0, phase1, phase2, phase3")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    allowed_directions = {"positive_to_negative", "negative_to_positive"}
    direction_list = tuple(dict.fromkeys(str(value) for value in directions))
    if not direction_list or set(direction_list) - allowed_directions:
        raise ValueError(f"directions must be drawn from {sorted(allowed_directions)}")

    pairs_path = Path(pairs_path)
    pairs = _load_pairs(pairs_path)
    if max_records is not None:
        pairs = pairs[:max_records]

    preflight_tokenizer = load_tokenizer(model_name)
    prepared_pairs = []
    for pair in pairs:
        prepared = {"pair": pair, "conditions": {}}
        for condition in ("positive", "negative"):
            scoring_prompt, shifted_spans = _prepare_scoring_condition(
                preflight_tokenizer,
                pair["prompts"][condition],
                pair["evidence_char_spans"][condition],
                decision_prefix=decision_prefix,
            )
            candidate_lengths = [
                len(continuation_token_ids(preflight_tokenizer, scoring_prompt, candidate)[1])
                for candidate in (positive_candidate, negative_candidate)
            ]
            if candidate_lengths != [1, 1]:
                raise ValueError(
                    "activation patching requires one-token positive and negative candidates"
                )
            if len(input_ids(preflight_tokenizer, scoring_prompt)) + 1 > max_seq_len:
                raise ValueError(
                    f"formatted {condition} prompt exceeds max_seq_len={max_seq_len}"
                )
            prepared["conditions"][condition] = {
                "prompt": scoring_prompt,
                "spans": shifted_spans,
            }
        prepared_pairs.append(prepared)
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(
        pairs_path,
        artifact_type="valence_pairs",
        stage="prepare",
        role="input",
    )
    run.manifest.save()
    try:
        model, tokenizer, fallback_device = load_model(model_name)
        device = getattr(model, "input_device", fallback_device)
        n_layers = int(model.n_layers)
        resolved_layer_conditions = (
            _validated_layer_conditions(layer_conditions, n_layers=n_layers)
            if layer_conditions is not None
            else default_layer_conditions(phase, n_layers=n_layers)
        )
        resolved_span_conditions = tuple(
            dict.fromkeys(span_conditions or default_span_conditions(phase))
        )
        if phase != "phase0":
            if not resolved_span_conditions:
                raise ValueError("at least one span condition is required")
            unknown_spans = set(resolved_span_conditions) - set(SPAN_IDS)
            if unknown_spans:
                raise ValueError(f"unknown span conditions: {sorted(unknown_spans)}")

        prepare_dir = run.run_directory / "prepare"
        prepare_metadata_path = prepare_dir / "metadata.json"
        with run.stage("prepare") as stage:
            prepare_metadata = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "causal_trace_prepare_metadata",
                "model": model_name,
                "phase": phase,
                "pairs_input": str(pairs_path),
                "pairs_input_sha256": sha256_file(pairs_path),
                "pair_count": len(prepared_pairs),
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "decision_prefix": decision_prefix,
                "positive_candidate": positive_candidate,
                "negative_candidate": negative_candidate,
                "layer_conditions": {
                    name: list(layers) for name, layers in resolved_layer_conditions.items()
                },
                "span_conditions": list(resolved_span_conditions),
                "directions": list(direction_list),
                "max_seq_len": max_seq_len,
                "state_cache_persisted": False,
            }
            write_metadata(prepare_metadata_path, prepare_metadata, overwrite=False)
            stage.count(len(prepared_pairs))
        run.manifest.register_artifact(
            prepare_metadata_path,
            artifact_type="causal_trace_prepare_metadata",
            stage="prepare",
            role="output",
        )
        run.manifest.save()

        forward_dir = run.run_directory / "forward"
        baseline_path = forward_dir / "phase0_baseline.jsonl"
        result_path = forward_dir / f"{phase}_records.jsonl"
        baseline_records = []
        patch_records = []
        with run.stage("forward") as stage:
            for prepared in prepared_pairs:
                pair = prepared["pair"]
                positive = prepared["conditions"]["positive"]
                negative = prepared["conditions"]["negative"]
                positive_margin = score_single_token_margin_fp32(
                    model,
                    tokenizer,
                    positive["prompt"],
                    positive_candidate,
                    negative_candidate,
                    device=device,
                )
                negative_margin = score_single_token_margin_fp32(
                    model,
                    tokenizer,
                    negative["prompt"],
                    positive_candidate,
                    negative_candidate,
                    device=device,
                )
                gate = _clean_gate_record(pair, positive_margin, negative_margin)
                baseline_records.append(gate)
                if not gate["clean_decision_match"] or phase == "phase0":
                    continue

                all_layers = sorted(
                    {layer for layers in resolved_layer_conditions.values() for layer in layers}
                )
                condition_data = {
                    "positive": (positive, positive_margin),
                    "negative": (negative, negative_margin),
                }
                source_cache = {}
                for direction in direction_list:
                    source_name, target_name = (
                        ("positive", "negative")
                        if direction == "positive_to_negative"
                        else ("negative", "positive")
                    )
                    source, source_margin = condition_data[source_name]
                    target, target_margin = condition_data[target_name]
                    if source_name not in source_cache:
                        source_cache[source_name] = cache_source_residuals(
                            model,
                            tokenizer,
                            source["prompt"],
                            all_layers,
                            device,
                        )
                    for layer_name, layers in resolved_layer_conditions.items():
                        for span_condition in resolved_span_conditions:
                            record = run_activation_patching_record(
                                model=model,
                                tokenizer=tokenizer,
                                source_prompt=source["prompt"],
                                target_prompt=target["prompt"],
                                source_evidence_char_spans=source["spans"],
                                target_evidence_char_spans=target["spans"],
                                layers=layers,
                                span_condition=span_condition,
                                positive_candidate=positive_candidate,
                                negative_candidate=negative_candidate,
                                device=device,
                                source_residuals={
                                    layer: source_cache[source_name][layer] for layer in layers
                                },
                                source_clean_margin=source_margin,
                                target_clean_margin=target_margin,
                            )
                            patch_records.append(
                                {
                                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                                    "artifact_type": "causal_trace_record",
                                    "record_id": stable_record_id(
                                        "causal-trace",
                                        str(pair["record_id"]),
                                        direction,
                                        layer_name,
                                        span_condition,
                                    ),
                                    "pair_record_id": pair["record_id"],
                                    "ticker": pair["ticker"],
                                    "name": pair.get("name", ""),
                                    "sector": pair.get("sector", ""),
                                    "source_trial_key": pair["source_trial_key"],
                                    "phase": phase,
                                    "patching_direction": direction,
                                    "source_condition": source_name,
                                    "target_condition": target_name,
                                    "layer_condition": layer_name,
                                    "prompt_template": PROMPT_TEMPLATE_VERSION,
                                    "evidence_item_hashes": pair.get(
                                        "evidence_item_hashes", {}
                                    ),
                                    **record,
                                }
                            )
            baseline_count = write_jsonl(
                baseline_path, baseline_records, overwrite=False
            )
            result_count = 0
            if phase != "phase0":
                result_count = write_jsonl(result_path, patch_records, overwrite=False)
            stage.count(baseline_count + result_count)
        run.manifest.register_artifact(
            baseline_path,
            artifact_type="causal_trace_baseline",
            stage="forward",
            role="output",
            record_count=baseline_count,
        )
        if phase != "phase0":
            run.manifest.register_artifact(
                result_path,
                artifact_type="causal_trace_records",
                stage="forward",
                role="output",
                record_count=result_count,
            )
        run.manifest.save()

        analyze_dir = run.run_directory / "analyze"
        summary_path = analyze_dir / "summary.json"
        with run.stage("analyze") as stage:
            summary = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": "causal_trace_analysis",
                "phase": phase,
                "pair_count": len(baseline_records),
                "eligible_pair_count": sum(
                    bool(record["clean_decision_match"])
                    for record in baseline_records
                ),
                "patch_record_count": len(patch_records),
                "condition_summaries": summarize_patching_records(patch_records),
                "interpretation_limit": (
                    "resample patching localizes sufficient state transfer for the fixed "
                    "Buy/Sell margin; it does not identify a unique reasoning path"
                ),
            }
            write_json(summary_path, summary, overwrite=False)
            stage.count(len(summary["condition_summaries"]))
        run.manifest.register_artifact(
            summary_path,
            artifact_type="causal_trace_analysis",
            stage="analyze",
            role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        if run.status != "failed":
            run.fail(exc)
        raise


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_DECISION_PREFIX",
    "DEFAULT_NEGATIVE_CANDIDATE",
    "DEFAULT_POSITIVE_CANDIDATE",
    "SPAN_IDS",
    "cache_source_residuals",
    "default_layer_conditions",
    "evaluate_activation_patching_confirmation",
    "evaluate_activation_patching_confirmation_artifacts",
    "default_span_conditions",
    "resolve_prompt_spans",
    "run_activation_patching_pipeline",
    "run_activation_patching_record",
    "run_patched_margin",
    "summarize_patching_records",
]
