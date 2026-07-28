"""Static and interactive visualizations for the Qwen prompt experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = "sp500_r1k_r2k_entityBiasPrompt.csv"
DEFAULT_ATTRIBUTION = "artifacts/qwen_generated_attribution_semantic_scope_full_selected/generated_token_attribution.jsonl"
DEFAULT_VALIDATION = "artifacts/qwen_semantic_scope_validation_selected/semantic_scope_aopc.jsonl"
DEFAULT_OUTPUT_DIR = "artifacts/qwen_result_visualization"
DEFAULT_TOKENIZER = ".cache/models/qwen3.5-4b"

INDEX_LABELS = {
    "sp500": "S&P 500",
    "russell1000": "Russell 1000",
    "russell2000": "Russell 2000",
}
INDEX_ORDER = ("sp500", "russell1000", "russell2000")
CONTEXT_ORDER = ("without", "with")
DEFAULT_UNCERTAINTY_FILES = {
    ("sp500", "without"): "artifacts/qwen3.5_sp500_no_context_per_prompt/prompt_layer_uncertainty.jsonl",
    ("sp500", "with"): "artifacts/qwen3.5_sp500_with_context_per_prompt/prompt_layer_uncertainty.jsonl",
    ("russell1000", "without"): "artifacts/qwen3.5_russell1000_without_context_per_prompt/prompt_layer_uncertainty.jsonl",
    ("russell1000", "with"): "artifacts/qwen3.5_russell1000_with_context_per_prompt/prompt_layer_uncertainty.jsonl",
    ("russell2000", "without"): "artifacts/qwen3.5_russell2000_without_context_per_prompt/prompt_layer_uncertainty.jsonl",
    ("russell2000", "with"): "artifacts/qwen3.5_russell2000_with_context_per_prompt/prompt_layer_uncertainty.jsonl",
}


def uncertainty_paths_from_root(root: str | Path = "artifacts") -> dict[tuple[str, str], Path]:
    """Resolve six uncertainty inputs, preferring one combined artifact.

    ``analyze-prompt-outputs`` writes a combined file directly below its
    output directory. Support both the historical named directory and the
    portable runner's ``per_date`` directory so callers do not need to create
    compatibility symlinks.
    """
    root_path = Path(root)
    combined_candidates = (
        root_path / "qwen3.5_temperature_scope_per_date" / "prompt_layer_uncertainty.jsonl",
        root_path / "per_date" / "prompt_layer_uncertainty.jsonl",
        root_path / "prompt_layer_uncertainty.jsonl",
    )
    for combined in combined_candidates:
        if combined.is_file():
            return {key: combined for key in DEFAULT_UNCERTAINTY_FILES}
    return {
        key: root_path / Path(source).relative_to("artifacts")
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
    result: list[dict[str, Any]] = []
    for (index, context), source in paths.items():
        rows = _read_jsonl(Path(source))
        seen_dates: set[str] = set()
        for row in rows:
            row_index_name = str(row.get("index", ""))
            row_context = str(row.get("context", ""))
            if (row_index_name or row_context) and (
                row_index_name != index or row_context != context
            ):
                continue
            row_index = row.get("row_index")
            row_date = str(row.get("date", ""))
            layers = row.get("layers")
            if not row_date or not isinstance(layers, list):
                raise ValueError(f"uncertainty row is missing date/layers: {source}")
            output_layers = [layer for layer in layers if layer.get("is_output")]
            if len(output_layers) != 1:
                raise ValueError(
                    f"expected exactly one output layer for {index}/{context} "
                    f"date {row_date}, found {len(output_layers)}"
                )
            if row_date in seen_dates:
                raise ValueError(f"duplicate uncertainty date {row_date} in {source}")
            seen_dates.add(row_date)
            output = output_layers[0]
            if "entropy_nats" not in output:
                raise ValueError(f"output layer has no entropy_nats: {source}")
            result.append(
                {
                    "row_index": row_index,
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
    return sorted(result, key=lambda row: (row["date"], row["index"], row["context"]))


def _load_prices(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Date", *INDEX_ORDER}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"price CSV must contain {sorted(required)}")
    valid_rows = []
    for row in rows:
        try:
            float(row["sp500"])
            float(row["russell1000"])
            float(row["russell2000"])
        except (TypeError, ValueError):
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError(f"price CSV contains no complete price rows: {path}")
    return valid_rows


def _market_returns(
    prices: Iterable[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    rows = list(prices)
    result: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        values = {index: float(row[index]) for index in INDEX_ORDER}
        previous = rows[row_index - 1] if row_index else None
        returns = (
            {
                index: (values[index] / float(previous[index]) - 1.0) * 100.0
                for index in INDEX_ORDER
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
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Select common crash and normal dates deterministically."""
    required_conditions = {(index, context) for index in INDEX_ORDER for context in CONTEXT_ORDER}
    by_date: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in attribution_rows:
        key = (str(row.get("index", "")), str(row.get("context", "")))
        date_value = str(row.get("date", ""))
        if key in required_conditions and date_value:
            by_date[date_value].add(key)

    market = _market_returns(prices)
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


def _condition_order() -> list[tuple[str, str]]:
    return [(index, context) for index in INDEX_ORDER for context in CONTEXT_ORDER]


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
        "context": str(row["context"]),
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

    dates = []
    for day in selected:
        conditions = []
        for index, context in _condition_order():
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
    return (Path(__file__).parent / "static" / name).read_text(encoding="utf-8")


def render_attribution_html(data: dict[str, Any]) -> str:
    """Embed compact attribution data into the standalone dashboard template."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    template = _load_template("attribution_dashboard.html")
    script = _load_template("attribution_dashboard.js")
    return template.replace("__ATTRIBUTION_DATA__", payload).replace("__ATTRIBUTION_SCRIPT__", script)


def plot_uncertainty(records: Iterable[dict[str, Any]], output_dir: Path) -> None:
    """Write Temperature Scope curves and entropy comparison curves."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = list(records)
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

    colors = {"sp500": "#2563eb", "russell1000": "#dc2626", "russell2000": "#059669"}
    for context in CONTEXT_ORDER:
        figure, axes = plt.subplots(
            len(INDEX_ORDER),
            1,
            figsize=(15, 10),
            sharex=True,
            squeeze=False,
        )
        for index in INDEX_ORDER:
            axis = axes[INDEX_ORDER.index(index)][0]
            series = sorted(
                (row for row in rows if row["index"] == index and row["context"] == context),
                key=lambda row: row["date"],
            )
            if not series:
                continue
            dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in series]
            values = [row["effective_temperature"] for row in series]
            axis.plot(dates, values, linewidth=1.25, color=colors[index])
            axis.set_title(INDEX_LABELS[index], loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Effective temperature")
            axis.grid(alpha=0.25)
        axes[-1][0].set_xlabel("Date")
        locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        axes[-1][0].xaxis.set_major_locator(locator)
        axes[-1][0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        figure.suptitle(f"Temperature Scope uncertainty · {context} context", fontsize=14)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"final_layer_effective_temperature_{context}_context.png",
            dpi=180,
        )
        plt.close(figure)

        # Retain an entropy comparison plot under the historical name.
        entropy_figure, entropy_axes = plt.subplots(
            len(INDEX_ORDER), 1, figsize=(15, 10), sharex=True, squeeze=False
        )
        for index in INDEX_ORDER:
            axis = entropy_axes[INDEX_ORDER.index(index)][0]
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
            axis.set_title(INDEX_LABELS[index], loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Entropy (nats)")
            axis.grid(alpha=0.25)
        entropy_axes[-1][0].set_xlabel("Date")
        entropy_locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
        entropy_axes[-1][0].xaxis.set_major_locator(entropy_locator)
        entropy_axes[-1][0].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(entropy_locator)
        )
        entropy_figure.suptitle(f"Final-layer entropy comparison · {context} context", fontsize=14)
        entropy_figure.tight_layout()
        entropy_figure.savefig(
            output_dir / f"final_layer_entropy_{context}_context.png", dpi=180
        )
        plt.close(entropy_figure)


def visualize_qwen_results(
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
    prices = _load_prices(Path(prices_path))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    selected_dates, market = select_attribution_dates(attribution_rows, prices)
    data = build_attribution_data(
        attribution_rows,
        selected_dates,
        input_top_k=input_top_k,
        market=market,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        validation_rows=validation_rows,
    )
    plot_uncertainty(uncertainty, output)
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
