"""Positive-vs-negative J-space vocabulary readout pipeline.

Captures the qualitative and quantitative evidence-item end residuals at the
requested layers plus the final model layer, transports non-final layers through the validated canonical
Jacobian lens, unembeds with the model final norm + LM head, and computes the
complete vocabulary softmax.  Complete probability sums are accumulated in
memory by condition/layer and ticker/condition/layer (float64); only compact
per-prompt evidence-position-mean top-k plus entropy/effective temperature are persisted. Analysis
averages the complete condition softmax vectors first and selects top-k only
afterwards.  The output nominates transported-representation candidates for
later signed steering, gain, or swap experiments; it is not causal evidence.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.artifact_paths import sha256_file, stable_record_id
from llm_bias.core.artifacts.io import write_json, write_jsonl, write_metadata
from llm_bias.core.artifacts.lifecycle import ArtifactRun
from llm_bias.core.inference.forward import encode_batch, record_residuals
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model, load_tokenizer
from llm_bias.core.prompt_input.encoding import decode_token, format_prompt, input_ids, token_span
from llm_bias.jspace_intervention.candidates import single_leading_space_token
from llm_bias.jspace_intervention.splits import SPLITS
from llm_bias.jspace_intervention.valence import (
    CONDITIONS,
    PROMPT_TEMPLATE_VERSION,
    build_valence_pair,
    resolve_valence_layers,
    select_valence_trials,
)

ARTIFACT_SCHEMA_VERSION = 2
PAIRS_ARTIFACT_TYPE = "valence_pairs"
PREPARE_METADATA_TYPE = "valence_prepare_metadata"
READOUT_ARTIFACT_TYPE = "valence_readout"
FORWARD_METADATA_TYPE = "valence_readout_metadata"
CONTRAST_ARTIFACT_TYPE = "valence_token_contrast"
ANALYSIS_METADATA_TYPE = "valence_analysis_metadata"
CANDIDATES_ARTIFACT_TYPE = "frozen_candidate_suggestions"

DEFAULT_BAND_LAYERS = tuple(range(14, 27))
LOG_RATIO_EPS = 1e-12
MIN_MEAN_PROBABILITY = 1e-5
TOP_K_PER_SIDE = 50
MAX_FROZEN_CANDIDATES = 12
MIN_TICKER_SIGN_FRACTION = 0.70
MIN_LAYER_SIGN_FRACTION = 0.75
ANSWER_EXCLUSIONS = frozenset({"buy", "sell"})
FORMAT_BOILERPLATE = frozenset({
    "a", "an", "the", "and", "of", "with", "in", "on", "to", "do", "not",
    "one", "only", "keys", "valid", "json", "object", "containing",
    "refer", "below", "make", "final", "investment", "decision", "stock",
    "ticker", "name", "evidence", "respond", "reason", "brief",
    "justification", "choose", "hold",
})
CANDIDATE_LABEL = (
    "transported-representation candidate for later signed steering/gain/swap; "
    "not causal evidence"
)
AGGREGATION_CONTRACT = (
    "complete vocabulary softmax vectors are averaged per condition (and per "
    "ticker) before any top-k selection; per-prompt artifacts persist only "
    "compact top-k, entropy, and effective temperature"
)


def _load_raw_trial_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"raw trial row {path}:{number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"raw trial input is empty: {path}")
    return rows


def _load_split_manifest(path: Path) -> tuple[dict[str, str], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in split manifest {path}") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "jspace_intervention_splits":
        raise ValueError("split manifest artifact_type must be jspace_intervention_splits")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("split manifest is missing ticker assignments")
    split_input_sha256 = payload.get("input_sha256")
    if not isinstance(split_input_sha256, str) or len(split_input_sha256) != 64:
        raise ValueError("split manifest is missing the original input_sha256")
    return {str(key): str(value) for key, value in assignments.items()}, split_input_sha256


def prepare_valence_pairs(
    raw_input: str | Path,
    split_manifest: str | Path,
    *,
    sector: str,
    split_name: str,
    trials_per_ticker: int,
    seed: int,
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Select, validate, render, and preflight all valence prompt pairs.

    Chat-template token lengths are checked for every pair before any run
    directory is created.  Returns the pair records and the bound hashes
    (raw input SHA, split SHA, split's original input hash).
    """
    raw_input = Path(raw_input)
    split_manifest = Path(split_manifest)
    rows = _load_raw_trial_rows(raw_input)
    assignments, split_input_sha256 = _load_split_manifest(split_manifest)
    selected = select_valence_trials(
        rows,
        sector=sector,
        split_assignments=assignments,
        split_name=split_name,
        trials_per_ticker=trials_per_ticker,
        seed=seed,
    )
    pairs = [build_valence_pair(row) for row in selected]
    for pair in pairs:
        pair["evidence_token_positions"] = {}
        for condition in CONDITIONS:
            raw_prompt = pair["prompts"][condition]
            formatted = format_prompt(
                tokenizer,
                raw_prompt,
                use_chat_template=True,
                enable_thinking=False,
            )
            ids = input_ids(tokenizer, formatted, add_special_tokens=True)
            if len(ids) > max_seq_len:
                raise ValueError(
                    f"valence prompt for {pair['ticker']}/{condition} has {len(ids)} tokens "
                    f"after chat templating, limit {max_seq_len}"
                )
            raw_offset = formatted.find(raw_prompt)
            if raw_offset < 0:
                raise ValueError("formatted chat prompt does not contain the raw valence prompt")
            positions = {}
            for kind, char_span in pair["evidence_char_spans"][condition].items():
                span = token_span(
                    tokenizer,
                    formatted,
                    raw_offset + int(char_span[0]),
                    raw_offset + int(char_span[1]),
                    add_special_tokens=True,
                )
                if span is None:
                    raise ValueError(
                        f"could not map {pair['ticker']}/{condition}/{kind} evidence span"
                    )
                positions[kind] = int(span[1] - 1)
            pair["evidence_token_positions"][condition] = positions
    return pairs, {
        "raw_input_sha256": sha256_file(raw_input),
        "split_manifest_sha256": sha256_file(split_manifest),
        "split_input_sha256": split_input_sha256,
    }


@torch.no_grad()
def accumulate_valence_readout(
    *,
    model: Any,
    tokenizer: Any,
    lens: Any,
    pairs: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    top_k: int,
    max_seq_len: int,
    device: Any,
) -> tuple[list[dict[str, Any]], dict, dict, dict, dict]:
    """Read both evidence-item end positions per pair and accumulate full sums.

    Returns ``(records, condition_sums, condition_counts, ticker_sums,
    ticker_counts)``.  Every sum is a float64 CPU tensor over the complete
    vocabulary; the persisted records hold only compact top-k, entropy, and
    effective temperature per prompt/layer.
    """
    if not pairs:
        raise ValueError("valence readout requires at least one prompt pair")
    final_layer = int(model.n_layers) - 1
    # Keep the Jacobians resident on the unembed (lm_head) device, as the
    # baseline lens-forward readout does.
    head_device = model._lm_head.weight.device
    lens.jacobians = {
        layer: jacobian.to(head_device) for layer, jacobian in lens.jacobians.items()
    }
    vocab_size = int(model._lm_head.weight.shape[0])
    k = min(int(top_k), vocab_size)
    log_vocab = math.log(max(vocab_size, 2))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    condition_sums: dict[tuple[str, int], torch.Tensor] = {}
    condition_counts: dict[tuple[str, int], int] = {}
    ticker_sums: dict[tuple[str, str, int], torch.Tensor] = {}
    ticker_counts: dict[tuple[str, str, int], int] = {}
    records: list[dict[str, Any]] = []
    token_cache: dict[int, str] = {}
    for pair in pairs:
        encoded_prompts: list[tuple[str, list[int]]] = []
        for condition in CONDITIONS:
            formatted = format_prompt(
                tokenizer,
                pair["prompts"][condition],
                use_chat_template=True,
                enable_thinking=False,
            )
            ids = input_ids(tokenizer, formatted, add_special_tokens=True)
            if len(ids) > max_seq_len:
                raise ValueError(
                    f"valence prompt for {pair['ticker']}/{condition} has {len(ids)} "
                    f"tokens, limit {max_seq_len}"
                )
            encoded_prompts.append((condition, ids))
        encoded = encode_batch(
            [ids for _, ids in encoded_prompts],
            device=device,
            pad_token_id=int(pad_token_id),
        )
        # Right-padding is after both selected evidence positions. Causal
        # attention prevents those future pad tokens from affecting either
        # readout, while record_residuals reuses the shared hook lifecycle.
        residuals = record_residuals(model, encoded.input_ids, layers)
        for row_index, (condition, ids) in enumerate(encoded_prompts):
            readout_layers = []
            position_map = pair["evidence_token_positions"][condition]
            position_kinds = ("qual", "quant")
            positions = [int(position_map[kind]) for kind in position_kinds]
            for layer in layers:
                residual = residuals[layer][row_index, positions, :].float()
                if layer != final_layer and layer in lens.jacobians:
                    residual = lens.transport(residual.to(head_device), layer)
                else:
                    residual = residual.to(head_device)
                logits = model.unembed(residual).float()
                probabilities = logits.softmax(dim=-1)
                mean_probabilities = probabilities.mean(dim=0)
                contribution = mean_probabilities.double().cpu()
                condition_key = (condition, int(layer))
                ticker_key = (str(pair["ticker"]), condition, int(layer))
                if condition_key in condition_sums:
                    condition_sums[condition_key] += contribution
                else:
                    condition_sums[condition_key] = contribution.clone()
                condition_counts[condition_key] = condition_counts.get(condition_key, 0) + 1
                if ticker_key in ticker_sums:
                    ticker_sums[ticker_key] += contribution
                else:
                    ticker_sums[ticker_key] = contribution.clone()
                ticker_counts[ticker_key] = ticker_counts.get(ticker_key, 0) + 1
                top = mean_probabilities.topk(k, dim=-1)
                token_ids = top.indices.cpu().tolist()
                token_values = top.values.cpu().tolist()
                entropy = float(
                    -(mean_probabilities.clamp_min(1e-12).log() * mean_probabilities).sum().cpu()
                )
                normalized = model._final_norm(residual.to(model._lm_head.weight.dtype)).float()
                inverse_temperature = float(normalized.norm(dim=-1).mean().cpu())
                if not math.isfinite(inverse_temperature) or inverse_temperature <= 0:
                    raise ValueError(
                        f"non-finite final-normalized norm at {pair['ticker']}/{condition}/L{layer}"
                    )
                readout_layers.append(
                    {
                        "layer": int(layer),
                        "is_output": layer == final_layer,
                        "readout_positions": [
                            {"kind": kind, "token_position": position}
                            for kind, position in zip(position_kinds, positions, strict=True)
                        ],
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
                                zip(token_ids, token_values, strict=True), start=1
                            )
                        ],
                        "entropy_nats": entropy,
                        "normalized_entropy": entropy / log_vocab,
                        "effective_temperature": 1.0 / max(inverse_temperature, 1e-12),
                    }
                )
            records.append(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": READOUT_ARTIFACT_TYPE,
                    "record_id": stable_record_id(pair["record_id"], condition),
                    "source_trial_key": pair["source_trial_key"],
                    "source_trial_index": pair["source_trial_index"],
                    "set_index": pair["set_index"],
                    "ticker": pair["ticker"],
                    "name": pair["name"],
                    "sector": pair["sector"],
                    "condition": condition,
                    "evidence_item_hashes": dict(pair["evidence_item_hashes"]),
                    "sequence_length": len(ids),
                    "readout_position": "mean_of_qual_and_quant_evidence_item_end_tokens",
                    "layers": readout_layers,
                }
            )
    return records, condition_sums, condition_counts, ticker_sums, ticker_counts


def average_condition_sums(
    condition_sums: Mapping[tuple[str, int], torch.Tensor],
    condition_counts: Mapping[tuple[str, int], int],
) -> dict[tuple[str, int], torch.Tensor]:
    """Average complete softmax sums per condition/layer (float64 means)."""
    means: dict[tuple[str, int], torch.Tensor] = {}
    for key, total in condition_sums.items():
        count = int(condition_counts.get(key, 0))
        if count <= 0:
            raise ValueError(f"condition sum {key} has no observations")
        means[key] = total / float(count)
    return means


def js_contribution(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Per-token Jensen-Shannon contribution (nats) between two distributions."""
    p = positive.float()
    q = negative.float()
    m = 0.5 * (p + q)

    def _term(a: torch.Tensor) -> torch.Tensor:
        term = torch.zeros_like(a)
        mask = a > 0
        term[mask] = a[mask] * torch.log(a[mask] / m[mask])
        return term

    return 0.5 * (_term(p) + _term(q))


def rank_valence_tokens(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    side: str,
    top_k: int = TOP_K_PER_SIDE,
    eps: float = LOG_RATIO_EPS,
) -> list[dict[str, Any]]:
    """Rank one side of an already-averaged condition contrast.

    ``positive`` and ``negative`` must be complete vocabulary probability
    vectors that are already the mean of per-prompt full-softmax vectors: the
    ranking is taken on the averaged vectors, never on per-prompt top-k lists.
    """
    if side not in CONDITIONS:
        raise ValueError(f"side must be one of {tuple(CONDITIONS)}")
    p = positive.float()
    q = negative.float()
    if p.ndim != 1 or p.shape != q.shape:
        raise ValueError("positive/negative must be equal one-dimensional full-vocabulary vectors")
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("condition means must be finite")
    k = min(int(top_k), p.numel())
    difference = p - q
    log_ratio = torch.log((p + eps) / (q + eps))
    js = js_contribution(p, q)
    order = torch.argsort(difference, descending=side == "positive")[:k]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "mean_positive": float(p[token_id]),
            "mean_negative": float(q[token_id]),
            "probability_diff": float(difference[token_id]),
            "smoothed_log_ratio": float(log_ratio[token_id]),
            "js_contribution": float(js[token_id]),
        }
        for rank, token_id in enumerate(order.tolist(), start=1)
    ]


def analyze_valence_contrast(
    *,
    condition_sums: Mapping[tuple[str, int], torch.Tensor],
    condition_counts: Mapping[tuple[str, int], int],
    layers: Sequence[int],
    band_layers: Sequence[int],
    top_k_per_side: int = TOP_K_PER_SIDE,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Rank both valence sides for every layer scope plus the band scope.

    The band scope is the equal-weight mean of the per-layer condition means
    over ``band_layers`` (the non-final readout layers).
    """
    means = average_condition_sums(condition_sums, condition_counts)
    requested = sorted({int(layer) for layer in layers})
    band = sorted({int(layer) for layer in band_layers})
    if not band:
        raise ValueError("band_layers must not be empty")
    for layer in [*requested, *band]:
        for condition in CONDITIONS:
            if (condition, layer) not in means:
                raise ValueError(f"missing condition sum for {condition} at layer {layer}")
    scopes: dict[str, dict[str, torch.Tensor]] = {
        f"layer_{layer}": {condition: means[(condition, layer)] for condition in CONDITIONS}
        for layer in requested
    }
    band_scope = f"band_{band[0]}-{band[-1]}"
    scopes[band_scope] = {
        condition: torch.stack([means[(condition, layer)] for layer in band]).mean(dim=0)
        for condition in CONDITIONS
    }
    return {
        scope: {
            side: rank_valence_tokens(
                vectors["positive"], vectors["negative"], side=side, top_k=top_k_per_side
            )
            for side in CONDITIONS
        }
        for scope, vectors in scopes.items()
    }


def build_frozen_candidates(
    *,
    ranked: Mapping[str, list[dict[str, Any]]],
    tokenizer: Any,
    diff_vectors: Sequence[torch.Tensor],
    n_tickers: int,
    n_layers: int,
    min_mean_probability: float = MIN_MEAN_PROBABILITY,
    max_candidates: int = MAX_FROZEN_CANDIDATES,
) -> list[dict[str, Any]]:
    """Filter band-scope candidates into robust representation suggestions.

    Eligibility: exactly one complete leading-space token (via
    ``single_leading_space_token``), not buy/sell, not a special,
    punctuation-only, or format-boilerplate token, adequate mean probability
    under the candidate's own condition, sign agreement across at least 70%
    of ticker-level band means and 75% of layer-level ticker means, plus a
    stable leave-one-ticker-out aggregate sign. Candidates are capped by
    |band probability difference|. ``diff_vectors`` must be ordered with all
    band layers for ticker 1, then all band layers for ticker 2, and so on.
    """
    if n_tickers < 1 or n_layers < 1:
        raise ValueError("ticker and layer counts must be positive")
    if not diff_vectors:
        raise ValueError("at least one ticker x band-layer difference vector is required")
    special = {
        str(token).strip().lower()
        for token in (getattr(tokenizer, "all_special_tokens", None) or ())
    }
    eligible: list[dict[str, Any]] = []
    for side in CONDITIONS:
        sign = 1.0 if side == "positive" else -1.0
        for row in ranked.get(side, []):
            token_id = int(row["token_id"])
            token = str(row.get("token") or "")
            concept = token.strip()
            if not concept:
                continue
            lowered = concept.lower()
            if lowered in ANSWER_EXCLUSIONS or lowered in FORMAT_BOILERPLATE or lowered in special:
                continue
            if not any(char.isalnum() for char in concept):
                continue
            resolved = single_leading_space_token(tokenizer, concept)
            if resolved is None or int(resolved[0]) != token_id:
                continue
            own_mean = row["mean_positive"] if side == "positive" else row["mean_negative"]
            if float(own_mean) < min_mean_probability:
                continue
            if len(diff_vectors) != n_tickers * n_layers:
                raise ValueError("difference vectors do not match ticker x layer dimensions")
            values = torch.tensor(
                [float(diff[token_id]) for diff in diff_vectors], dtype=torch.float64
            ).reshape(n_tickers, n_layers)
            ticker_values = values.mean(dim=1)
            layer_values = values.mean(dim=0)
            ticker_consistent = int((sign * ticker_values > 0).sum())
            layer_consistent = int((sign * layer_values > 0).sum())
            ticker_required = math.ceil(MIN_TICKER_SIGN_FRACTION * n_tickers)
            layer_required = math.ceil(MIN_LAYER_SIGN_FRACTION * n_layers)
            if ticker_consistent < ticker_required or layer_consistent < layer_required:
                continue
            leave_one_out_stable = True
            if n_tickers > 1:
                total = ticker_values.sum()
                leave_one_out = (total - ticker_values) / (n_tickers - 1)
                leave_one_out_stable = bool((sign * leave_one_out > 0).all())
            if not leave_one_out_stable:
                continue
            eligible.append(
                {
                    "token": token,
                    "concept": lowered,
                    "token_id": int(resolved[0]),
                    "side": side,
                    "mean_positive": float(row["mean_positive"]),
                    "mean_negative": float(row["mean_negative"]),
                    "band_probability_diff": float(row["probability_diff"]),
                    "band_smoothed_log_ratio": float(row["smoothed_log_ratio"]),
                    "band_js_contribution": float(row["js_contribution"]),
                    "sign_consistent_tickers": ticker_consistent,
                    "ticker_count": int(n_tickers),
                    "sign_consistent_layers": layer_consistent,
                    "layer_count": int(n_layers),
                    "leave_one_ticker_out_sign_stable": leave_one_out_stable,
                    "label": CANDIDATE_LABEL,
                }
            )
    eligible.sort(
        key=lambda item: (-abs(item["band_probability_diff"]), item["concept"], item["token_id"])
    )
    # Reserve half the shortlist for each side when both have enough eligible
    # tokens, then fill any unused slots by global contrast magnitude.
    per_side = max(1, int(max_candidates) // 2)
    selected = []
    for side in CONDITIONS:
        selected.extend([item for item in eligible if item["side"] == side][:per_side])
    selected_ids = {(item["side"], item["token_id"]) for item in selected}
    for item in eligible:
        if len(selected) >= int(max_candidates):
            break
        key = (item["side"], item["token_id"])
        if key not in selected_ids:
            selected.append(item)
            selected_ids.add(key)
    selected.sort(
        key=lambda item: (-abs(item["band_probability_diff"]), item["concept"], item["token_id"])
    )
    return selected


def run_valence_readout_pipeline(
    *,
    input_path: str | Path,
    split_manifest: str | Path,
    model_name: str,
    run_id: str,
    dataset: str = "jspace-valence-readout",
    artifact_root: str | Path = "artifacts",
    lens_path: str | Path | None = None,
    sector: str = "Technology",
    split_name: str = "discovery",
    trials_per_ticker: int = 3,
    layers: Sequence[int] = DEFAULT_BAND_LAYERS,
    top_k: int = 30,
    max_seq_len: int = 1024,
    seed: int = 0,
) -> Path:
    """Run the valence readout prepare/forward/analyze/finalize workflow."""
    input_path = Path(input_path)
    split_manifest = Path(split_manifest)
    if not input_path.is_file():
        raise FileNotFoundError(f"raw trial JSONL not found: {input_path}")
    if not split_manifest.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_manifest}")
    if split_name not in SPLITS:
        raise ValueError(f"unknown split {split_name!r}; expected one of {SPLITS}")
    if trials_per_ticker < 1:
        raise ValueError("trials_per_ticker must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")

    preflight_tokenizer = load_tokenizer(model_name)
    pairs, hashes = prepare_valence_pairs(
        input_path,
        split_manifest,
        sector=sector,
        split_name=split_name,
        trials_per_ticker=trials_per_ticker,
        seed=seed,
        tokenizer=preflight_tokenizer,
        max_seq_len=max_seq_len,
    )
    del preflight_tokenizer

    run = ArtifactRun.create(model_name, dataset, run_id, artifact_root=artifact_root)
    run.manifest.register_artifact(
        input_path, artifact_type="raw_trial_plan", stage="prepare", role="input"
    )
    run.manifest.register_artifact(
        split_manifest, artifact_type="jspace_intervention_splits", stage="prepare", role="input"
    )
    run.manifest.save()
    try:
        prepare_dir = run.run_directory / "prepare"
        pairs_path = prepare_dir / "valence_pairs.jsonl"
        with run.stage("prepare") as stage:
            pairs_count = write_jsonl(
                pairs_path,
                ({
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "artifact_type": PAIRS_ARTIFACT_TYPE,
                    **pair,
                } for pair in pairs),
                overwrite=False,
            )
            prepare_metadata = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": PREPARE_METADATA_TYPE,
                "model": model_name,
                "raw_input": str(input_path),
                "raw_input_sha256": hashes["raw_input_sha256"],
                "split_manifest": str(split_manifest),
                "split_manifest_sha256": hashes["split_manifest_sha256"],
                "split_input_sha256": hashes["split_input_sha256"],
                "sector": sector,
                "split": split_name,
                "trials_per_ticker": trials_per_ticker,
                "seed": seed,
                "layers_requested": [int(layer) for layer in layers],
                "top_k": top_k,
                "max_seq_len": max_seq_len,
                "prompt_template": PROMPT_TEMPLATE_VERSION,
                "evidence_selection": "positive=buy qual+buy quant; negative=sell qual+sell quant",
                "pair_count": pairs_count,
                "ticker_count": len({pair["ticker"] for pair in pairs}),
            }
            prepare_metadata_path = prepare_dir / "metadata.json"
            write_metadata(prepare_metadata_path, prepare_metadata, overwrite=False)
            stage.count(pairs_count)
        run.manifest.register_artifact(
            pairs_path, artifact_type=PAIRS_ARTIFACT_TYPE, stage="prepare",
            role="output", record_count=pairs_count,
        )
        run.manifest.register_artifact(
            prepare_metadata_path, artifact_type=PREPARE_METADATA_TYPE,
            stage="prepare", role="output",
        )
        run.manifest.save()

        model, tokenizer, fallback_device = load_model(model_name)
        loaded_lens = load_validated_lens(
            model=model,
            model_name=model_name,
            lens_path=lens_path,
            artifact_root=artifact_root,
        )
        run.manifest.register_artifact(
            loaded_lens.path,
            artifact_type="jacobian_lens",
            stage="prepare",
            role="lens",
            metadata={"source": loaded_lens.source},
        )
        run.manifest.save()
        final_layer = int(model.n_layers) - 1
        resolved_layers = resolve_valence_layers(layers, final_layer)
        band_layers = [layer for layer in resolved_layers if layer != final_layer]
        if not band_layers:
            raise ValueError("valence readout requires at least one non-final band layer")

        forward_dir = run.run_directory / "forward"
        readout_path = forward_dir / "valence_readout.jsonl"
        with run.stage("forward") as stage:
            records, condition_sums, condition_counts, ticker_sums, ticker_counts = (
                accumulate_valence_readout(
                    model=model,
                    tokenizer=tokenizer,
                    lens=loaded_lens.lens,
                    pairs=pairs,
                    layers=resolved_layers,
                    top_k=top_k,
                    max_seq_len=max_seq_len,
                    device=getattr(model, "input_device", fallback_device),
                )
            )
            count = write_jsonl(readout_path, records, overwrite=False)
            stage.count(count)
        forward_metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": FORWARD_METADATA_TYPE,
            "model": model_name,
            "canonical_lens": str(loaded_lens.path),
            "lens_source": loaded_lens.source,
            "layers": resolved_layers,
            "band_layers": band_layers,
            "final_layer": final_layer,
            "readout_position": "mean_of_qual_and_quant_evidence_item_end_tokens",
            "top_k": top_k,
            "max_seq_len": max_seq_len,
            "aggregation_contract": AGGREGATION_CONTRACT,
            "records_written": count,
            "backpropagation": False,
        }
        forward_metadata_path = forward_dir / "metadata.json"
        write_metadata(forward_metadata_path, forward_metadata, overwrite=False)
        run.manifest.register_artifact(
            readout_path, artifact_type=READOUT_ARTIFACT_TYPE, stage="forward",
            role="output", record_count=count,
        )
        run.manifest.register_artifact(
            forward_metadata_path, artifact_type=FORWARD_METADATA_TYPE,
            stage="forward", role="output",
        )
        run.manifest.save()

        analyze_dir = run.run_directory / "analyze"
        contrast_path = analyze_dir / "valence_token_contrast.jsonl"
        candidates_path = analyze_dir / "frozen_candidate_suggestions.json"
        analyze_metadata_path = analyze_dir / "metadata.json"
        with run.stage("analyze") as stage:
            contrast = analyze_valence_contrast(
                condition_sums=condition_sums,
                condition_counts=condition_counts,
                layers=resolved_layers,
                band_layers=band_layers,
            )
            token_cache: dict[int, str] = {}

            def _decode(token_id: int) -> str:
                return token_cache.setdefault(int(token_id), decode_token(tokenizer, int(token_id)))

            contrast_rows = []
            for scope, sides in contrast.items():
                for side, rows in sides.items():
                    for row in rows:
                        contrast_rows.append({
                            "schema_version": ARTIFACT_SCHEMA_VERSION,
                            "artifact_type": CONTRAST_ARTIFACT_TYPE,
                            "scope": scope,
                            "side": side,
                            "token": _decode(row["token_id"]),
                            **row,
                        })
            contrast_count = write_jsonl(contrast_path, contrast_rows, overwrite=False)
            band_scope = f"band_{band_layers[0]}-{band_layers[-1]}"
            ranked = {
                side: [
                    {**row, "token": _decode(row["token_id"])}
                    for row in contrast[band_scope][side]
                ]
                for side in CONDITIONS
            }
            tickers = sorted({str(pair["ticker"]) for pair in pairs})
            diff_vectors = []
            for ticker in tickers:
                for layer in band_layers:
                    positive = (
                        ticker_sums[(ticker, "positive", layer)]
                        / ticker_counts[(ticker, "positive", layer)]
                    )
                    negative = (
                        ticker_sums[(ticker, "negative", layer)]
                        / ticker_counts[(ticker, "negative", layer)]
                    )
                    diff_vectors.append(positive - negative)
            candidates = build_frozen_candidates(
                ranked=ranked,
                tokenizer=tokenizer,
                diff_vectors=diff_vectors,
                n_tickers=len(tickers),
                n_layers=len(band_layers),
            )
            candidates_document = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": CANDIDATES_ARTIFACT_TYPE,
                "candidates": candidates,
                "criteria": {
                    "single_leading_space_token": True,
                    "excluded_answer_words": sorted(ANSWER_EXCLUSIONS),
                    "excluded_format_boilerplate": sorted(FORMAT_BOILERPLATE),
                    "special_tokens": "excluded via tokenizer.all_special_tokens",
                    "punctuation_only": "excluded",
                    "min_mean_probability": MIN_MEAN_PROBABILITY,
                    "ticker_sign_fraction_min": MIN_TICKER_SIGN_FRACTION,
                    "layer_sign_fraction_min": MIN_LAYER_SIGN_FRACTION,
                    "leave_one_ticker_out_sign_stable": True,
                    "max_candidates": MAX_FROZEN_CANDIDATES,
                    "side_balance": "reserve half per side, then fill unused slots globally",
                },
                "label": CANDIDATE_LABEL,
            }
            write_json(candidates_path, candidates_document, overwrite=False)
            analyze_metadata = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": ANALYSIS_METADATA_TYPE,
                "model": model_name,
                "aggregation_contract": AGGREGATION_CONTRACT,
                "scopes": list(contrast),
                "band_layers": band_layers,
                "condition_counts": {
                    f"{condition}_L{layer}": int(counts)
                    for (condition, layer), counts in sorted(condition_counts.items())
                },
                "log_ratio_eps": LOG_RATIO_EPS,
                "top_k_per_side": TOP_K_PER_SIDE,
                "min_mean_probability": MIN_MEAN_PROBABILITY,
                "max_frozen_candidates": MAX_FROZEN_CANDIDATES,
                "candidate_label": CANDIDATE_LABEL,
                "contrast_rows": contrast_count,
            }
            write_metadata(analyze_metadata_path, analyze_metadata, overwrite=False)
            stage.count(contrast_count)
        run.manifest.register_artifact(
            contrast_path, artifact_type=CONTRAST_ARTIFACT_TYPE, stage="analyze",
            role="output", record_count=contrast_count,
        )
        run.manifest.register_artifact(
            candidates_path, artifact_type=CANDIDATES_ARTIFACT_TYPE,
            stage="analyze", role="output",
        )
        run.manifest.register_artifact(
            analyze_metadata_path, artifact_type=ANALYSIS_METADATA_TYPE,
            stage="analyze", role="output",
        )
        run.manifest.save()
        run.finalize(required_stages={"prepare", "forward", "analyze"})
        return run.run_directory
    except BaseException as exc:
        run.fail(exc)
        raise


__all__ = [
    "AGGREGATION_CONTRACT",
    "ANSWER_EXCLUSIONS",
    "CANDIDATE_LABEL",
    "DEFAULT_BAND_LAYERS",
    "FORMAT_BOILERPLATE",
    "LOG_RATIO_EPS",
    "MAX_FROZEN_CANDIDATES",
    "MIN_LAYER_SIGN_FRACTION",
    "MIN_MEAN_PROBABILITY",
    "MIN_TICKER_SIGN_FRACTION",
    "TOP_K_PER_SIDE",
    "accumulate_valence_readout",
    "analyze_valence_contrast",
    "average_condition_sums",
    "build_frozen_candidates",
    "js_contribution",
    "prepare_valence_pairs",
    "rank_valence_tokens",
    "run_valence_readout_pipeline",
]
