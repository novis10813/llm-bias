import json

import pytest
import torch

from llm_bias.prompt_analysis.attribution import _semantic_scope_scores
from llm_bias.prompt_analysis.validation import _aopc

from llm_bias.prompt_analysis.visualization import (
    _parse_generated_answer,
    build_attribution_data,
    build_paired_uncertainty_deltas,
    build_uncertainty_distribution_rows,
    load_final_layer_uncertainty,
    load_price_distribution_samples,
    plot_price_distributions,
    plot_uncertainty_distribution_figures,
    render_attribution_html,
    select_attribution_dates,
    summarize_price_distributions,
    summarize_uncertainty_distributions,
    uncertainty_paths_from_root,
    visualize_price_distributions,
    visualize_uncertainty_distributions,
)


class _PromptTokenizer:
    def __call__(self, _prompt, **_kwargs):
        return {"input_ids": list(range(12))}

    def decode(self, token_ids, **_kwargs):
        return f"tok-{token_ids[0]}"


def test_forward_only_panel_marks_token_text_and_log_probability_unavailable():
    data = build_attribution_data(
        [{"date": "2020-01-01", "index": "sp500", "context": "without", "generated_token_ids": [7], "generated_text": "7"}],
        ["2020-01-01"],
        condition_order=[("sp500", "without")],
    )
    token = data["dates"][0]["conditions"][0]["output_tokens"][0]
    assert token["token_id"] == 7
    assert token["token"] is None
    assert token["log_probability"] is None


def _prices():
    return [
        {"Date": "2020-01-01", "sp500": "100", "russell1000": "100", "russell2000": "100"},
        {"Date": "2020-01-02", "sp500": "90", "russell1000": "90", "russell2000": "90"},
        {"Date": "2020-01-03", "sp500": "89.9", "russell1000": "90", "russell2000": "90"},
        {"Date": "2020-01-04", "sp500": "89", "russell1000": "89", "russell2000": "89"},
    ]


def test_semantic_scope_scores_use_gradient_l2_norm_per_input_token():
    gradient = torch.tensor([[[3.0, 4.0], [0.0, -5.0]]])

    assert torch.equal(_semantic_scope_scores(gradient), torch.tensor([[5.0, 5.0]]))


def test_aopc_uses_trapezoidal_log_probability_deltas():
    assert _aopc((0.0, 0.1, 0.2), (0.0, -1.0, -2.0)) == pytest.approx(-0.2)


def _attribution_rows(dates):
    rows = []
    for day in dates:
        for index in ("sp500", "russell1000", "russell2000"):
            for context in ("without", "with"):
                rows.append(
                    {
                        "date": day,
                        "prompt_column": f"prompt_{context}_context_{index}",
                        "index": index,
                        "context": context,
                        "generated_text": "ab",
                        "generated_tokens": [
                            {
                                "position": 0,
                                "token_id": 1,
                                "token": "a",
                                "log_probability": -0.1,
                                "top_input_tokens": [
                                    {"position": 10, "prompt_position": 8, "token_id": 5, "token": " x", "attribution": 0.8},
                                    {"position": 11, "prompt_position": 9, "token_id": 6, "token": " y", "attribution": 0.2},
                                ],
                            },
                            {
                                "position": 1,
                                "token_id": 2,
                                "token": "b",
                                "log_probability": -0.2,
                                "top_input_tokens": [
                                    {"position": 10, "prompt_position": 8, "token_id": 5, "token": " x", "attribution": 0.1},
                                ],
                            },
                        ],
                    }
                )
    return rows


def test_select_attribution_dates_chooses_crashes_and_normal_date():
    rows = _attribution_rows(["2020-01-02", "2020-01-03", "2020-01-04"])
    selected, market = select_attribution_dates(rows, _prices())

    assert selected == ["2020-01-02", "2020-01-03", "2020-01-04"]
    assert market["2020-01-02"]["mean_return_pct"] == pytest.approx(-10.0)


def test_build_attribution_data_keeps_output_input_alignment():
    rows = _attribution_rows(["2020-01-02"])
    data = build_attribution_data(
        rows,
        ["2020-01-02"],
        input_top_k=1,
        backward_rows=rows,
    )
    condition = data["dates"][0]["conditions"][0]

    assert condition["input_tokens"] == [
        {"position": 8, "token_id": 5, "token": " x"}
    ]
    assert condition["matrix"] == [[0.8], [0.1]]
    assert condition["output_tokens"][0]["token"] == "a"


def test_build_attribution_data_can_show_complete_prompt_tokens():
    rows = _attribution_rows(["2020-01-02"])
    for row in rows:
        row["prompt"] = "complete prompt"
    data = build_attribution_data(
        rows,
        ["2020-01-02"],
        tokenizer=_PromptTokenizer(),
        max_seq_len=256,
        backward_rows=rows,
    )
    condition = data["dates"][0]["conditions"][0]

    assert len(condition["input_tokens"]) == 12
    assert condition["input_tokens"][8]["token"] == "tok-8"
    assert condition["matrix"][0][8] == pytest.approx(0.8)
    assert condition["matrix"][1][8] == pytest.approx(0.1)
    assert condition["input_attribution_complete"] is False


def test_build_attribution_data_requires_all_six_conditions():
    rows = _attribution_rows(["2020-01-02"])
    with pytest.raises(ValueError, match="missing attribution record"):
        build_attribution_data(rows[:-1], ["2020-01-02"])


def test_load_final_layer_uncertainty_extracts_output_layer(tmp_path):
    path = tmp_path / "uncertainty.jsonl"
    path.write_text(
        json.dumps(
            {
                "date": "2020-01-01",
                "layers": [
                    {"layer": 0, "is_output": False, "entropy_nats": 2},
                    {
                        "layer": 1,
                        "is_output": True,
                        "entropy_nats": 0.75,
                        "top1_probability": 0.5,
                        "effective_inverse_temperature": 12.0,
                        "effective_temperature": 1 / 12,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_final_layer_uncertainty({("sp500", "without"): path})

    assert rows[0]["layer"] == 1
    assert rows[0]["entropy_nats"] == pytest.approx(0.75)
    assert rows[0]["effective_temperature"] == pytest.approx(1 / 12)


def test_load_final_layer_uncertainty_filters_combined_artifact(tmp_path):
    path = tmp_path / "uncertainty.jsonl"
    rows = [
        {
            "date": "2020-01-01",
            "index": "sp500",
            "context": "without",
            "layers": [{"layer": 1, "is_output": True, "entropy_nats": 1.0}],
        },
        {
            "date": "2020-01-01",
            "index": "sp500",
            "context": "with",
            "layers": [{"layer": 1, "is_output": True, "entropy_nats": 2.0}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_final_layer_uncertainty({("sp500", "without"): path})

    assert len(loaded) == 1
    assert loaded[0]["entropy_nats"] == pytest.approx(1.0)


def test_load_final_layer_uncertainty_discovers_arbitrary_conditions(tmp_path):
    path = tmp_path / "uncertainty.jsonl"
    rows = [
        {
            "date": "2020-01-01",
            "index": index,
            "context": context,
            "layers": [{"layer": 1, "is_output": True, "entropy_nats": 1.0}],
        }
        for index, context in (("aapl", "without"), ("aapl", "with"), ("msft", "without"))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_final_layer_uncertainty(
        {
            ("sp500", "without"): path,
            ("sp500", "with"): path,
        }
    )

    assert {(row["index"], row["context"]) for row in loaded} == {
        ("aapl", "without"),
        ("aapl", "with"),
        ("msft", "without"),
    }


def _uncertainty_distribution_records():
    rows = []
    for index in ("sp500", "russell1000", "russell2000"):
        for date_position, date in enumerate(
            ("2020-01-01", "2020-01-02", "2020-01-03")
        ):
            for context in ("without", "with"):
                if (
                    index == "russell1000"
                    and date == "2020-01-03"
                    and context == "with"
                ):
                    continue
                context_offset = 0.5 if context == "with" else 0.0
                rows.append(
                    {
                        "date": date,
                        "index": index,
                        "context": context,
                        "layer": 31,
                        "entropy_nats": 1.0 + date_position + context_offset,
                        "effective_temperature": 0.1
                        + date_position * 0.01
                        + context_offset * 0.02,
                    }
                )
    return rows


def test_uncertainty_distribution_rows_and_paired_deltas():
    records = _uncertainty_distribution_records()

    raw = build_uncertainty_distribution_rows(records)
    paired = build_paired_uncertainty_deltas(records)
    summary = summarize_uncertainty_distributions(raw, paired)

    assert len(raw) == len(records) * 2
    assert sum(row["index"] == "sp500" for row in paired) == 3
    assert sum(row["index"] == "russell1000" for row in paired) == 2
    assert sum(row["index"] == "russell2000" for row in paired) == 3
    first_pair = next(
        row
        for row in paired
        if row["index"] == "sp500" and row["date"] == "2020-01-01"
    )
    assert first_pair["entropy_delta_nats"] == pytest.approx(0.5)
    assert first_pair["effective_temperature_delta"] == pytest.approx(0.01)
    entropy_delta = next(
        row
        for row in summary
        if row["distribution"] == "paired_delta"
        and row["metric"] == "entropy_nats"
        and row["index"] == "sp500"
    )
    assert entropy_delta["n"] == 3
    assert entropy_delta["median"] == pytest.approx(0.5)


def test_uncertainty_distribution_rejects_duplicate_and_nonfinite_records():
    records = _uncertainty_distribution_records()
    with pytest.raises(ValueError, match="duplicate uncertainty condition"):
        build_uncertainty_distribution_rows([records[0], records[0]])

    invalid = {**records[0], "entropy_nats": float("nan")}
    with pytest.raises(ValueError, match="must be finite"):
        build_uncertainty_distribution_rows([invalid])

    mixed_layer = {**records[1], "layer": 30}
    with pytest.raises(ValueError, match="one consistent output layer"):
        build_uncertainty_distribution_rows([records[0], mixed_layer])


def test_uncertainty_delta_plot_rejects_indices_without_pairs(tmp_path):
    records = [
        row for row in _uncertainty_distribution_records()
        if row["index"] == "sp500" and row["context"] == "without"
    ]
    raw = build_uncertainty_distribution_rows(records)

    with pytest.raises(ValueError, match="no common dates"):
        plot_uncertainty_distribution_figures(raw, [], tmp_path)


def test_visualize_uncertainty_distributions_writes_research_outputs(tmp_path):
    source = tmp_path / "prompt_layer_uncertainty.jsonl"
    rows = []
    for record in _uncertainty_distribution_records():
        rows.append(
            {
                "date": record["date"],
                "index": record["index"],
                "context": record["context"],
                "layers": [
                    {
                        "layer": record["layer"],
                        "is_output": True,
                        "entropy_nats": record["entropy_nats"],
                        "effective_temperature": record["effective_temperature"],
                    }
                ],
            }
        )
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "output"

    result = visualize_uncertainty_distributions(
        uncertainty_paths={
            ("sp500", "without"): source,
            ("sp500", "with"): source,
        },
        output_dir=output,
    )

    assert result == output
    for name in (
        "final_layer_entropy_raw_ecdf.png",
        "final_layer_entropy_paired_delta_violin.png",
        "final_layer_effective_temperature_raw_ecdf.png",
        "final_layer_effective_temperature_paired_delta_violin.png",
    ):
        path = output / name
        assert path.is_file()
        assert path.stat().st_size > 0
    metadata = json.loads(
        (output / "final_layer_uncertainty_distribution_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["records"] == len(rows)
    assert metadata["raw_distribution_rows"] == len(rows) * 2
    assert metadata["paired"]["by_index"]["russell1000"] == {
        "n_pairs": 2,
        "unmatched_dates": 1,
        "date_rule": "intersection within index",
    }
    assert metadata["plots"]["dual_axis"] is False


def test_select_attribution_dates_accepts_arbitrary_price_columns():
    rows = []
    for day in ("2020-01-01", "2020-01-02", "2020-01-03"):
        for context in ("without", "with"):
            rows.append({"date": day, "index": "aapl", "context": context})
    prices = [
        {"Date": "2020-01-01", "aapl": "100"},
        {"Date": "2020-01-02", "aapl": "90"},
        {"Date": "2020-01-03", "aapl": "89.9"},
    ]

    selected, market = select_attribution_dates(
        rows,
        prices,
        crash_count=1,
        condition_order=[("aapl", "without"), ("aapl", "with")],
    )

    assert selected == ["2020-01-02", "2020-01-03"]
    assert market["2020-01-02"]["prices"] == {"aapl": 90.0}


def test_uncertainty_paths_from_root_discovers_runner_per_date_directory(tmp_path):
    path = tmp_path / "per_date" / "prompt_layer_uncertainty.jsonl"
    path.parent.mkdir()
    path.write_text("{}\n", encoding="utf-8")

    paths = uncertainty_paths_from_root(tmp_path)

    assert set(paths) == {
        (index, context)
        for index in ("sp500", "russell1000", "russell2000")
        for context in ("without", "with")
    }
    assert set(paths.values()) == {path}


def test_render_attribution_html_embeds_tokens_and_interaction_script():
    rows = _attribution_rows(["2020-01-02"])
    data = build_attribution_data(
        rows,
        ["2020-01-02"],
        input_top_k=1,
        backward_rows=rows,
    )
    html = render_attribution_html(data)

    assert "2020-01-02" in html
    assert "S&amp;P 500" not in html  # token data is embedded; labels are created with textContent
    assert "focusOutput" in html
    assert "input_tokens" in html
    assert ".input-line .token-track { flex: 1 1 auto; min-width: 0; max-width: 100%; flex-wrap: wrap; }" in html
    assert "scroll horizontally" not in html


def test_build_attribution_data_embeds_validation_scores():
    rows = _attribution_rows(["2020-01-02"])
    validation = [
        {
            "date": row["date"],
            "index": row["index"],
            "context": row["context"],
            "generated_tokens": [
                {
                    "position": 0,
                    "semantic_scope": {
                        "aopc": -0.3,
                        "log_probability_delta": [0.0, -0.1, -0.2, -0.3],
                    },
                    "random": {
                        "aopc": -0.1,
                        "log_probability_delta": [0.0, -0.03, -0.07, -0.1],
                    },
                }
            ],
            "summary": {
                "semantic_scope_aopc_mean": -0.3,
                "random_aopc_mean": -0.1,
                "n_output_tokens": 1,
            },
        }
        for row in rows
    ]

    data = build_attribution_data(rows, ["2020-01-02"], validation_rows=validation)

    output = data["dates"][0]["conditions"][0]["output_tokens"][0]
    assert output["semantic_scope_aopc"] == pytest.approx(-0.3)
    assert data["dates"][0]["conditions"][0]["validation_summary"]["random_aopc_mean"] == pytest.approx(-0.1)


def test_parse_generated_answer_accepts_json_with_wrappers():
    parsed = _parse_generated_answer(
        '```json\n{"answer": 123.5, "confidence": 90}\n```<|im_end|>'
    )

    assert parsed == {
        "parsed_answer": 123.5,
        "confidence": 90.0,
        "parse_status": "valid",
        "parse_reason": None,
    }


def test_parse_generated_answer_keeps_numeric_answer_when_confidence_is_non_numeric():
    parsed = _parse_generated_answer(
        'prefix {"answer": 101, "confidence": "high"} trailing text'
    )

    assert parsed == {
        "parsed_answer": 101.0,
        "confidence": None,
        "parse_status": "valid",
        "parse_reason": None,
    }


@pytest.mark.parametrize(
    ("generated_text", "reason"),
    [
        ('{"answer": null}', "answer_null"),
        ('{"answer": "123"}', "answer_not_numeric"),
        ('{"answer": true}', "answer_not_numeric"),
        ('{"answer": NaN}', "answer_not_finite"),
        ('{"confidence": 90}', "answer_missing"),
        ("not JSON", "json_object_not_found"),
        ("{broken", "malformed_json"),
        (None, "generated_text_not_string"),
    ],
)
def test_parse_generated_answer_rejects_invalid_values(generated_text, reason):
    parsed = _parse_generated_answer(generated_text)

    assert parsed["parse_status"] == "invalid"
    assert parsed["parse_reason"] == reason
    assert parsed["parsed_answer"] is None


def _write_price_sampling_fixture(tmp_path):
    indices = ("sp500", "russell1000", "russell2000")
    dates = ("2020-01-01", "2020-01-02")
    prices = {
        "2020-01-01": {"sp500": 100, "russell1000": 200, "russell2000": 300},
        "2020-01-02": {"sp500": 110, "russell1000": 210, "russell2000": 310},
    }
    price_path = tmp_path / "prices.csv"
    price_path.write_text(
        "Date,sp500,russell1000,russell2000\n"
        + "\n".join(
            f'{date},{values["sp500"]},{values["russell1000"]},{values["russell2000"]}'
            for date, values in prices.items()
        )
        + "\n",
        encoding="utf-8",
    )

    root = tmp_path / "sampling"
    root.mkdir()
    prompt_columns = [
        f"prompt_{context}_context_{index}"
        for index in indices
        for context in ("without", "with")
    ]
    records_per_run = len(dates) * len(prompt_columns)
    manifest = {
        "input": str(price_path),
        "model": "fake-model",
        "runs": 2,
        "run_indices": [0, 1],
        "generation": "sampling",
        "generation_config": {"temperature": 0.7},
        "selected_dates": list(dates),
        "condition_counts": {column: len(dates) for column in prompt_columns},
        "records_per_run": records_per_run,
        "run_directories": [
            {
                "run_index": run_index,
                "run_seed": 100 + run_index,
                "directory": f"run_{run_index:03d}",
                "records_written": records_per_run,
            }
            for run_index in range(2)
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    for run_index in range(2):
        run_dir = root / f"run_{run_index:03d}"
        run_dir.mkdir()
        rows = []
        for sample_index, date in enumerate(dates):
            for index in indices:
                for context in ("without", "with"):
                    actual = prices[date][index]
                    answer = actual + (-10 if run_index == 0 else 10)
                    generated_text = json.dumps({"answer": answer, "confidence": 90})
                    if (
                        run_index == 1
                        and date == "2020-01-01"
                        and index == "sp500"
                        and context == "without"
                    ):
                        generated_text = '{"answer": null, "confidence": 90}'
                    rows.append(
                        {
                            "run_index": run_index,
                            "sample_index": sample_index,
                            "date": date,
                            "prompt_column": f"prompt_{context}_context_{index}",
                            "index": index,
                            "context": context,
                            "generated_text": generated_text,
                        }
                    )
        (run_dir / "generated_token_attribution.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    return root, price_path


def test_load_and_summarize_price_distribution_samples(tmp_path):
    root, price_path = _write_price_sampling_fixture(tmp_path)

    samples, manifest = load_price_distribution_samples(root, price_path)
    summaries = summarize_price_distributions(samples)

    assert manifest["runs"] == 2
    assert len(samples) == 24
    assert len(summaries) == 12
    row = next(
        row
        for row in summaries
        if row["date"] == "2020-01-01"
        and row["index"] == "sp500"
        and row["context"] == "without"
    )
    assert row["n_valid"] == 1
    assert row["n_invalid"] == 1
    assert row["median"] == pytest.approx(90)
    assert row["median_absolute_percentage_error"] == pytest.approx(10)

    complete_row = next(
        row
        for row in summaries
        if row["date"] == "2020-01-01"
        and row["index"] == "sp500"
        and row["context"] == "with"
    )
    assert complete_row["q05"] == pytest.approx(91)
    assert complete_row["median"] == pytest.approx(100)
    assert complete_row["q95"] == pytest.approx(109)


def test_load_price_distribution_samples_rejects_incomplete_runs(tmp_path):
    root, price_path = _write_price_sampling_fixture(tmp_path)
    (root / "run_001" / "generated_token_attribution.jsonl").unlink()

    with pytest.raises(FileNotFoundError):
        load_price_distribution_samples(root, price_path)


def test_load_price_distribution_samples_validates_manifest_directories(tmp_path):
    root, price_path = _write_price_sampling_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_directories"][0]["directory"] = "run_001"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest directory"):
        load_price_distribution_samples(root, price_path)


def test_plot_price_distributions_includes_arbitrary_indices(tmp_path):
    rows = [
        {
            "date": "2020-01-01",
            "index": "aapl",
            "context": context,
            "actual_close": 100.0,
            "n_invalid": 0,
            "q05": 90.0,
            "q25": 95.0,
            "median": 100.0,
            "q75": 105.0,
            "q95": 110.0,
            "median_absolute_percentage_error": 5.0,
        }
        for context in ("without", "with")
    ]

    paths = plot_price_distributions(rows, tmp_path)

    assert paths == [tmp_path / "aapl_price_distribution.png"]
    assert paths[0].is_file()


def test_visualize_price_distributions_writes_research_outputs(tmp_path):
    root, price_path = _write_price_sampling_fixture(tmp_path)
    output = tmp_path / "output"

    result = visualize_price_distributions(
        sampling_root=root,
        prices_path=price_path,
        output_dir=output,
    )

    assert result == output
    for index in ("sp500", "russell1000", "russell2000"):
        figure = output / f"{index}_price_distribution.png"
        assert figure.is_file()
        assert figure.stat().st_size > 0
    samples = (output / "price_distribution_samples.csv").read_text(encoding="utf-8")
    summary = (output / "price_distribution_summary.csv").read_text(encoding="utf-8")
    metadata = json.loads(
        (output / "price_distribution_metadata.json").read_text(encoding="utf-8")
    )
    assert len(samples.splitlines()) == 25
    assert len(summary.splitlines()) == 13
    assert metadata["records_total"] == 24
    assert metadata["records_invalid"] == 1
    assert metadata["error_metric"]["name"] == "median_absolute_percentage_error"
