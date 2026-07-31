"""Static and interactive visualizations for prompt-analysis artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = "sp500_r1k_r2k_entityBiasPrompt.csv"
DEFAULT_ARTIFACT_ROOT = "artifacts/prompt_analysis"
DEFAULT_ATTRIBUTION = "artifacts/prompt_analysis/generated_attribution/generated_token_attribution.jsonl"
DEFAULT_VALIDATION = "artifacts/prompt_analysis/attribution_validation/semantic_scope_aopc.jsonl"
DEFAULT_OUTPUT_DIR = "artifacts/prompt_analysis/visualization"
DEFAULT_TOKENIZER = ".cache/models/llama-3.2-1b-instruct"

INDEX_LABELS = {
    "sp500": "S&P 500",
    "russell1000": "Russell 1000",
    "russell2000": "Russell 2000",
}
INDEX_ORDER = ("sp500", "russell1000", "russell2000")
CONTEXT_ORDER = ("without", "with")
DEFAULT_UNCERTAINTY_FILES = {
    ("sp500", "without"): "artifacts/prompt_analysis/readout/sp500_without/prompt_layer_uncertainty.jsonl",
    ("sp500", "with"): "artifacts/prompt_analysis/readout/sp500_with/prompt_layer_uncertainty.jsonl",
    ("russell1000", "without"): "artifacts/prompt_analysis/readout/russell1000_without/prompt_layer_uncertainty.jsonl",
    ("russell1000", "with"): "artifacts/prompt_analysis/readout/russell1000_with/prompt_layer_uncertainty.jsonl",
    ("russell2000", "without"): "artifacts/prompt_analysis/readout/russell2000_without/prompt_layer_uncertainty.jsonl",
    ("russell2000", "with"): "artifacts/prompt_analysis/readout/russell2000_with/prompt_layer_uncertainty.jsonl",
}


def uncertainty_paths_from_root(
    root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[tuple[str, str], Path]:
    """Resolve six uncertainty inputs, preferring one combined artifact.

    ``prompt-analysis readout`` writes a combined file directly below its
    output directory. Support both the historical named directory and the
    portable runner's ``per_date`` directory so callers do not need to create
    compatibility symlinks.
    """
    root_path = Path(root)
    combined_candidates = (
        root_path / "readout" / "prompt_layer_uncertainty.jsonl",
        root_path / "per_date" / "prompt_layer_uncertainty.jsonl",
        root_path / "prompt_layer_uncertainty.jsonl",
    )
    for combined in combined_candidates:
        if combined.is_file():
            return {key: combined for key in DEFAULT_UNCERTAINTY_FILES}
    return {
        key: root_path / Path(source).relative_to(DEFAULT_ARTIFACT_ROOT)
        for key, source in DEFAULT_UNCERTAINTY_FILES.items()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            rows.append(value)
    return rows


def load_final_layer_uncertainty(
    paths: dict[tuple[str, str], str | Path],
) -> list[dict[str, Any]]:
    """Load one final-layer uncertainty record per date and condition."""
    # A runner writes one combined JSONL for all prompt columns.  The historical
    # ``paths`` API represents that file six times (once per index/context), so
    # detect that shape and discover the conditions from the records instead of
    # filtering against the old fixed S&P/Russell condition set.
    unique_sources = {Path(source) for source in paths.values()}
    combined_source = next(iter(unique_sources)) if len(unique_sources) == 1 else None
    discover_conditions = combined_source is not None and len(paths) > 1
    result: list[dict[str, Any]] = []
    seen_dates_by_condition: dict[tuple[str, str], set[str]] = defaultdict(set)

    def append_row(
        row: dict[str, Any],
        *,
        index: str,
        context: str,
        source_path: Path,
    ) -> None:
        row_date = str(row.get("date", ""))
        layers = row.get("layers")
        if not row_date or not isinstance(layers, list):
            raise ValueError(f"uncertainty row is missing date/layers: {source_path}")
        output_layers = [layer for layer in layers if layer.get("is_output")]
        if len(output_layers) != 1:
            raise ValueError(
                f"expected exactly one output layer for {index}/{context} "
                f"date {row_date}, found {len(output_layers)}"
            )
        condition = (index, context)
        if row_date in seen_dates_by_condition[condition]:
            raise ValueError(
                f"duplicate uncertainty date {row_date} for {index}/{context} "
                f"in {source_path}"
            )
        seen_dates_by_condition[condition].add(row_date)
        output = output_layers[0]
        if "entropy_nats" not in output:
            raise ValueError(f"output layer has no entropy_nats: {source_path}")
        result.append(
            {
                "row_index": row.get("row_index"),
                "date": row_date,
                "index": index,
                "context": context,
                "layer": output.get("layer"),
                "entropy_nats": float(output["entropy_nats"]),
                "normalized_entropy": float(output.get("normalized_entropy", 0.0)),
                "perplexity": float(output.get("perplexity", 0.0)),
                "top1_probability": float(output.get("top1_probability", 0.0)),
                "topk_mass": float(output.get("topk_mass", 0.0)),
                "effective_inverse_temperature": float(
                    output.get("effective_inverse_temperature", 0.0)
                ),
                "effective_temperature": float(
                    output.get("effective_temperature", 0.0)
                ),
            }
        )

    if discover_conditions and combined_source is not None:
        for row in _read_jsonl(combined_source):
            index = str(row.get("index", ""))
            context = str(row.get("context", ""))
            if not index or not context:
                raise ValueError(
                    f"combined uncertainty row has no index/context: {combined_source}"
                )
            append_row(row, index=index, context=context, source_path=combined_source)
    else:
        for (index, context), source in paths.items():
            source_path = Path(source)
            for row in _read_jsonl(source_path):
                row_index_name = str(row.get("index", ""))
                row_context = str(row.get("context", ""))
                if (row_index_name or row_context) and (
                    row_index_name != index or row_context != context
                ):
                    continue
                append_row(row, index=index, context=context, source_path=source_path)
    return sorted(result, key=lambda row: (row["date"], row["index"], row["context"]))


def _load_prices(
    path: Path,
    indices: Iterable[str] = INDEX_ORDER,
) -> list[dict[str, str]]:
    """Load rows containing the price columns used by the experiment."""
    if not path.is_file():
        raise FileNotFoundError(path)
    price_indices = tuple(dict.fromkeys(str(index) for index in indices))
    if not price_indices:
        raise ValueError("at least one price column is required")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Date", *price_indices}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"price CSV must contain {sorted(required)}")
    valid_rows = []
    for row in rows:
        try:
            for index in price_indices:
                float(row[index])
        except (TypeError, ValueError):
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError(f"price CSV contains no complete price rows: {path}")
    return valid_rows


def _market_returns(
    prices: Iterable[dict[str, str]],
    indices: Iterable[str] = INDEX_ORDER,
) -> dict[str, dict[str, Any]]:
    rows = list(prices)
    price_indices = tuple(dict.fromkeys(str(index) for index in indices))
    result: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        values = {index: float(row[index]) for index in price_indices}
        previous = rows[row_index - 1] if row_index else None
        returns = (
            {
                index: (values[index] / float(previous[index]) - 1.0) * 100.0
                for index in price_indices
            }
            if previous is not None
            else None
        )
        result[row["Date"]] = {
            "date": row["Date"],
            "prices": values,
            "returns_pct": returns,
            "mean_return_pct": sum(returns.values()) / len(returns) if returns else None,
        }
    return result


def select_attribution_dates(
    attribution_rows: Iterable[dict[str, Any]],
    prices: Iterable[dict[str, str]],
    *,
    crash_count: int = 2,
    condition_order: Iterable[tuple[str, str]] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Select common crash and normal dates deterministically.

    ``condition_order`` keeps the original three-index default for backwards
    compatibility, while the visualization runner supplies conditions discovered
    from the input artifact (for example ``aapl`` through ``tsla``).
    """
    required_order = list(condition_order or _condition_order())
    required_conditions = set(required_order)
    by_date: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in attribution_rows:
        key = (str(row.get("index", "")), str(row.get("context", "")))
        date_value = str(row.get("date", ""))
        if key in required_conditions and date_value:
            by_date[date_value].add(key)

    indices = tuple(dict.fromkeys(index for index, _context in required_order))
    market = _market_returns(prices, indices)
    common = [
        day
        for day, conditions in by_date.items()
        if conditions == required_conditions and market.get(day, {}).get("mean_return_pct") is not None
    ]
    if len(common) < crash_count + 1:
        raise ValueError(
            f"need at least {crash_count + 1} common attribution dates with prices; "
            f"found {len(common)}"
        )
    crash_dates = sorted(
        common,
        key=lambda day: (market[day]["mean_return_pct"], day),
    )[:crash_count]
    normal_date = min(
        (day for day in common if day not in crash_dates),
        key=lambda day: (abs(market[day]["mean_return_pct"]), day),
    )
    selected = sorted(crash_dates + [normal_date])
    return selected, {day: market[day] for day in selected}


def _condition_order(
    records: Iterable[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Return deterministic index/context order.

    With no records this preserves the historical S&P/Russell order.  When
    records are supplied, conditions are discovered from their ``index`` and
    ``context`` fields while keeping each index's without/with ordering.
    """
    if records is None:
        return [(index, context) for index in INDEX_ORDER for context in CONTEXT_ORDER]
    rows = list(records)
    indices: list[str] = []
    contexts_by_index: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        index = str(row.get("index", ""))
        context = str(row.get("context", ""))
        if not index or not context:
            continue
        if index not in indices:
            indices.append(index)
        if context not in contexts_by_index[index]:
            contexts_by_index[index].append(context)
    result: list[tuple[str, str]] = []
    for index in indices:
        contexts = contexts_by_index[index]
        for context in CONTEXT_ORDER:
            if context in contexts:
                result.append((index, context))
        # Preserve non-standard context values without silently dropping them.
        for context in contexts:
            if (index, context) not in result:
                result.append((index, context))
    return result


def _index_label(index: str) -> str:
    """Human-readable label for a condition, including arbitrary tickers."""
    return INDEX_LABELS.get(index, index.upper())


def _context_label(context: str) -> str:
    return {"without": "without context", "with": "with context"}.get(
        context, context
    )


def _prompt_tokens(
    prompt: str,
    tokenizer: Any,
    *,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_len,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [
        {
            "position": position,
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
        for position, token_id in enumerate(token_ids)
    ]


def _attribution_panel(
    row: dict[str, Any],
    *,
    input_top_k: int,
    tokenizer: Any | None,
    max_seq_len: int,
    validation_by_position: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generated = row.get("generated_tokens")
    if not isinstance(generated, list) or not generated:
        raise ValueError("attribution record has no generated_tokens")

    by_output: list[dict[int, float]] = []
    input_metadata: dict[int, dict[str, Any]] = {}
    totals: dict[int, float] = defaultdict(float)
    for output in generated:
        contributions = output.get("top_input_tokens")
        if not isinstance(contributions, list):
            raise ValueError("generated token has no top_input_tokens")
        values: dict[int, float] = {}
        for item in contributions:
            position = int(item.get("prompt_position", item["position"]))
            attribution = float(item["attribution"])
            values[position] = attribution
            totals[position] += attribution
            input_metadata.setdefault(
                position,
                {
                    "position": position,
                    "token_id": int(item["token_id"]),
                    "token": str(item["token"]),
                },
            )
        by_output.append(values)

    if tokenizer is not None:
        prompt = str(row.get("prompt", ""))
        if not prompt:
            raise ValueError("attribution record has no prompt for full input display")
        full_input_tokens = _prompt_tokens(prompt, tokenizer, max_seq_len=max_seq_len)
        selected_positions = [item["position"] for item in full_input_tokens]
        input_metadata = {item["position"]: item for item in full_input_tokens}
    else:
        selected_positions = sorted(
            sorted(totals, key=lambda position: (-totals[position], position))[:input_top_k]
        )
    matrix = [
        [round(values.get(position, 0.0), 8) for position in selected_positions]
        for values in by_output
    ]
    input_attribution_complete = all(
        position in values for values in by_output for position in selected_positions
    )
    output_tokens = []
    for output in generated:
        output_record = {
            "position": int(output["position"]),
            "token_id": int(output["token_id"]),
            "token": str(output["token"]),
            "log_probability": float(output["log_probability"]),
        }
        if "logit" in output:
            output_record["target_logit"] = float(output["logit"])
        validation = (validation_by_position or {}).get(int(output["position"]))
        if validation is not None:
            semantic = validation.get("semantic_scope", {})
            random_baseline = validation.get("random", {})
            output_record["semantic_scope_aopc"] = float(semantic["aopc"])
            output_record["random_aopc"] = float(random_baseline["aopc"])
            output_record["semantic_scope_log_probability_delta"] = list(
                semantic.get("log_probability_delta", [])
            )
            output_record["random_log_probability_delta"] = list(
                random_baseline.get("log_probability_delta", [])
            )
        output_tokens.append(output_record)
    max_attribution = max((value for values in matrix for value in values), default=0.0)
    return {
        "index": str(row["index"]),
        "index_label": _index_label(str(row["index"])),
        "context": str(row["context"]),
        "context_label": _context_label(str(row["context"])),
        "prompt_column": str(row["prompt_column"]),
        "prompt": str(row.get("prompt", "")),
        "generated_text": str(row.get("generated_text", "")),
        "output_tokens": output_tokens,
        "input_tokens": [input_metadata[position] for position in selected_positions],
        "matrix": matrix,
        "max_attribution": max_attribution,
        "input_attribution_complete": input_attribution_complete,
        "validation_summary": row.get("validation_summary"),
    }


def build_attribution_data(
    attribution_rows: Iterable[dict[str, Any]],
    selected_dates: Iterable[str],
    *,
    input_top_k: int = 15,
    market: dict[str, dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    max_seq_len: int = 256,
    validation_rows: Iterable[dict[str, Any]] | None = None,
    condition_order: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the compact JSON payload used by the standalone dashboard."""
    if input_top_k < 1:
        raise ValueError("input_top_k must be positive")
    selected = list(selected_dates)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in attribution_rows:
        key = (str(row.get("date", "")), str(row.get("index", "")), str(row.get("context", "")))
        if key[0] in selected:
            if key in by_key:
                raise ValueError(f"duplicate attribution record for {key}")
            by_key[key] = row
    validation_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for validation in validation_rows or ():
        key = (
            str(validation.get("date", "")),
            str(validation.get("index", "")),
            str(validation.get("context", "")),
        )
        if key in validation_by_key:
            raise ValueError(f"duplicate validation record for {key}")
        validation_by_key[key] = validation

    required_order = list(condition_order or _condition_order())
    dates = []
    for day in selected:
        conditions = []
        for index, context in required_order:
            row = by_key.get((day, index, context))
            if row is None:
                raise ValueError(f"missing attribution record for {day}/{index}/{context}")
            validation = validation_by_key.get((day, index, context))
            validation_by_position = {
                int(token["position"]): token
                for token in (validation or {}).get("generated_tokens", [])
            }
            row_for_panel = dict(row)
            if validation is not None:
                row_for_panel["validation_summary"] = validation.get("summary")
            conditions.append(
                _attribution_panel(
                    row_for_panel,
                    input_top_k=input_top_k,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    validation_by_position=validation_by_position,
                )
            )
        dates.append(
            {
                "date": day,
                "market": (market or {}).get(day),
                "conditions": conditions,
            }
        )
    return {
        "metric": "semantic_scope_target_logit_gradient_l2_norm",
        "normalization": "none",
        "input_top_k": input_top_k,
        "validation": "semantic_scope_aopc_with_random_baseline"
        if validation_rows is not None
        else None,
        "dates": dates,
    }


def _load_template(name: str) -> str:
    return (
        Path(__file__).resolve().parents[1] / "static" / name
    ).read_text(encoding="utf-8")


def render_attribution_html(data: dict[str, Any]) -> str:
    """Embed compact attribution data into the standalone dashboard template."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    template = _load_template("attribution_dashboard.html")
    script = _load_template("attribution_dashboard.js")
    return template.replace("__ATTRIBUTION_DATA__", payload).replace("__ATTRIBUTION_SCRIPT__", script)


def plot_uncertainty(
    records: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    condition_order: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Write Temperature Scope curves and entropy comparison curves."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = list(records)
    required_order = list(condition_order or _condition_order(rows))
    indices = list(dict.fromkeys(index for index, _context in required_order))
    contexts = list(dict.fromkeys(context for _index, context in required_order))
    if not indices or not contexts:
        raise ValueError("uncertainty records contain no index/context conditions")
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "index", "context", "layer", "entropy_nats",
        "normalized_entropy", "perplexity", "top1_probability", "topk_mass",
        "effective_inverse_temperature", "effective_temperature",
    ]
    with (output_dir / "final_layer_uncertainty.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)
    # Keep the historical filename as a compatibility alias with fresh values.
    with (output_dir / "final_layer_entropy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    palette = (
        "#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c",
        "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#c026d3",
    )
    colors = {
        index: palette[position % len(palette)]
        for position, index in enumerate(indices)
    }
    for context in contexts:
        figure, axes = plt.subplots(
            len(indices),
            1,
            figsize=(15, 10),
            sharex=True,
            squeeze=False,
        )
        for index_position, index in enumerate(indices):
            axis = axes[index_position][0]
            series = sorted(
                (row for row in rows if row["index"] == index and row["context"] == context),
                key=lambda row: row["date"],
            )
            if not series:
                continue
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            values = [row["effective_temperature"] for row in series]
            axis.plot(dates, values, linewidth=1.25, color=colors[index])
            axis.set_title(_index_label(index), loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Effective temperature")
            axis.grid(alpha=0.25)
        axes[-1][0].set_xlabel("Date")
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        axes[-1][0].xaxis.set_major_locator(locator)
        axes[-1][0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        figure.suptitle(
            f"Temperature Scope uncertainty · {_context_label(context)}", fontsize=14
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"final_layer_effective_temperature_{context}_context.png",
            dpi=180,
        )
        plt.close(figure)

        # Retain an entropy comparison plot under the historical name.
        entropy_figure, entropy_axes = plt.subplots(
            len(indices), 1, figsize=(15, 10), sharex=True, squeeze=False
        )
        for index_position, index in enumerate(indices):
            axis = entropy_axes[index_position][0]
            series = sorted(
                (row for row in rows if row["index"] == index and row["context"] == context),
                key=lambda row: row["date"],
            )
            if not series:
                continue
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            axis.plot(
                dates,
                [row["entropy_nats"] for row in series],
                linewidth=1.25,
                color=colors[index],
            )
            axis.set_title(_index_label(index), loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Entropy (nats)")
            axis.grid(alpha=0.25)
        entropy_axes[-1][0].set_xlabel("Date")
        entropy_locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        entropy_axes[-1][0].xaxis.set_major_locator(entropy_locator)
        entropy_axes[-1][0].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(entropy_locator)
        )
        entropy_figure.suptitle(
            f"Final-layer entropy comparison · {_context_label(context)}", fontsize=14
        )
        entropy_figure.tight_layout()
        entropy_figure.savefig(
            output_dir / f"final_layer_entropy_{context}_context.png", dpi=180
        )
        plt.close(entropy_figure)


def visualize_prompt_results(
    *,
    uncertainty_paths: dict[tuple[str, str], str | Path] | None = None,
    attribution_path: str | Path = DEFAULT_ATTRIBUTION,
    validation_path: str | Path | None = DEFAULT_VALIDATION,
    prices_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_top_k: int = 15,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    max_seq_len: int = 256,
) -> Path:
    """Create uncertainty figures and the standalone attribution dashboard."""
    output = Path(output_dir)
    paths = uncertainty_paths or DEFAULT_UNCERTAINTY_FILES
    uncertainty = load_final_layer_uncertainty(paths)
    attribution_rows = _read_jsonl(Path(attribution_path))
    validation_rows = (
        _read_jsonl(Path(validation_path))
        if validation_path is not None and Path(validation_path).is_file()
        else None
    )
    condition_order = _condition_order(uncertainty)
    if not condition_order:
        condition_order = _condition_order(attribution_rows)
    indices = tuple(dict.fromkeys(index for index, _context in condition_order))
    prices = _load_prices(Path(prices_path), indices)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    selected_dates, market = select_attribution_dates(
        attribution_rows,
        prices,
        condition_order=condition_order,
    )
    data = build_attribution_data(
        attribution_rows,
        selected_dates,
        input_top_k=input_top_k,
        market=market,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        validation_rows=validation_rows,
        condition_order=condition_order,
    )
    plot_uncertainty(uncertainty, output, condition_order=condition_order)
    output.mkdir(parents=True, exist_ok=True)
    (output / "attribution_dashboard.html").write_text(
        render_attribution_html(data), encoding="utf-8"
    )
    (output / "attribution_selected_dates.json").write_text(
        json.dumps(
            {
                "selection": "two largest negative equal-weight index returns and one closest-to-zero return",
                "dates": selected_dates,
                "market": market,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
