import json

import pytest
import torch

from llm_bias.prompt_analysis.attribution import _semantic_scope_scores
from llm_bias.prompt_analysis.validation import _aopc

from llm_bias.prompt_analysis.visualization import (
    build_attribution_data,
    load_final_layer_uncertainty,
    render_attribution_html,
    select_attribution_dates,
    uncertainty_paths_from_root,
)


class _PromptTokenizer:
    def __call__(self, _prompt, **_kwargs):
        return {"input_ids": list(range(12))}

    def decode(self, token_ids, **_kwargs):
        return f"tok-{token_ids[0]}"


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
    data = build_attribution_data(_attribution_rows(["2020-01-02"]), ["2020-01-02"], input_top_k=1)
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
    data = build_attribution_data(_attribution_rows(["2020-01-02"]), ["2020-01-02"], input_top_k=1)
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
