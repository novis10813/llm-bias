import json
from types import SimpleNamespace

import pytest
import torch

from llm_bias.prompt_analysis import readout
from llm_bias.prompt_analysis.readout import (
    _batched_output_gradients,
    _prepare_prompt,
    discover_prompt_columns,
    load_prompt_table,
    topk_token_records,
)


class _Tokenizer:
    chat_template = "available"

    def decode(self, token_ids, **_kwargs):
        return f"token-{token_ids[0]}"

    def apply_chat_template(self, messages, **kwargs):
        suffix = "<think>" if kwargs.get("enable_thinking") else ""
        return f"<user>{messages[0]['content']}</user><assistant>{suffix}"


def test_discover_prompt_columns_parses_context_and_index():
    columns = discover_prompt_columns(
        [
            "Date",
            "prompt_without_context_sp500",
            "prompt_with_context_russell2000",
        ]
    )

    assert [
        (column.name, column.context, column.index) for column in columns
    ] == [
        ("prompt_without_context_sp500", "without", "sp500"),
        ("prompt_with_context_russell2000", "with", "russell2000"),
    ]


def test_discover_prompt_columns_is_strict_and_ignores_extra_csv_fields():
    columns = discover_prompt_columns(
        [
            "Date",
            "prompt_without_context_sp500",
            "prompt_with_context_aapl",
            "notes",
            "prompt_with_context",
            "prompt_with_context_aapl_extra",
        ]
    )

    assert [column.name for column in columns] == [
        "prompt_without_context_sp500",
        "prompt_with_context_aapl",
        "prompt_with_context_aapl_extra",
    ]


def test_discover_prompt_columns_rejects_invalid_selected_column():
    with pytest.raises(ValueError, match="must match"):
        discover_prompt_columns(["prompt_sp500"], ["prompt_sp500"])


def test_load_prompt_table_supports_bom_quoted_multiline_empty_and_extra_fields(tmp_path):
    path = tmp_path / "prompts.csv"
    path.write_text(
        "﻿Date,prompt_without_context_aapl,prompt_with_context_sp500,notes\n"
        "2026-01-01,\"plain\",\"with\ncontext\",legacy\n"
        "2026-01-02,,\"  \",extra\n",
        encoding="utf-8",
    )

    columns, rows = load_prompt_table(path)

    assert [(column.index, column.context) for column in columns] == [
        ("aapl", "without"),
        ("sp500", "with"),
    ]
    assert rows == [
        {
            "Date": "2026-01-01",
            "prompt_without_context_aapl": "plain",
            "prompt_with_context_sp500": "with\ncontext",
            "notes": "legacy",
        },
        {
            "Date": "2026-01-02",
            "prompt_without_context_aapl": "",
            "prompt_with_context_sp500": "  ",
            "notes": "extra",
        },
    ]


def test_topk_token_records_ranks_mean_distribution():
    first = torch.tensor([0.60, 0.30, 0.10])
    second = torch.tensor([0.10, 0.30, 0.60])
    mean = torch.stack([first, second]).mean(dim=0)

    records = topk_token_records(mean, top_k=2, tokenizer=_Tokenizer())

    assert records == [
        {
            "rank": 1,
            "token_id": 0,
            "token": "token-0",
            "probability": pytest.approx(0.35),
        },
        {
            "rank": 2,
            "token_id": 2,
            "token": "token-2",
            "probability": pytest.approx(0.35),
        },
    ]


def test_batched_output_gradients_keeps_one_gradient_per_output():
    embedding = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        requires_grad=True,
    )
    scores = torch.stack(
        [
            embedding[..., 0].sum(),
            2 * embedding[..., 1].sum(),
        ]
    )

    gradients = _batched_output_gradients(scores, embedding)

    assert gradients.shape == (2, 1, 2, 2)
    assert torch.equal(
        gradients[0],
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    assert torch.equal(
        gradients[1],
        torch.tensor([[[0.0, 2.0], [0.0, 2.0]]]),
    )


def test_prepare_prompt_uses_user_chat_template_by_default():
    assert _prepare_prompt(_Tokenizer(), "hello", True) == (
        "<user>hello</user><assistant>"
    )
    assert _prepare_prompt(_Tokenizer(), "hello", False) == "hello"


def test_return_pairs_expand_by_pair_and_preserve_identity(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "cik,filename,item,filing_date,ticker,peer_ticker,system_prompt,prompt,counterfactual_prompt,return_label,fwd_return_1d\n"
        "1,a.txt,item_1,2026-01-01,AAA,BBB,sys,orig,counter,neutral,0.0\n"
        "2,a.txt,item_1,2026-01-01,CCC,DDD,sys,orig2,counter2,bullish,0.1\n",
        encoding="utf-8",
    )
    columns, rows = load_prompt_table(path, dataset_format="return-pairs", max_rows=2)
    assert [column.condition for column in columns] == ["original", "counterfactual"]
    assert len(rows) == 4
    assert {row["pair_id"] for row in rows} == {"1|a.txt|item_1", "2|a.txt|item_1"}
    assert [row["condition"] for row in rows[:2]] == ["original", "counterfactual"]
    assert [(row["ticker"], row["peer_ticker"]) for row in rows[:2]] == [
        ("AAA", "BBB"),
        ("AAA", "BBB"),
    ]
    assert [row["fwd_return_1d"] for row in rows[:2]] == [0.0, 0.0]


def test_auto_does_not_treat_partial_pair_schema_as_return_pairs(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text("cik,prompt_without_context_x\n1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires a Date column"):
        load_prompt_table(path, dataset_format="auto")


def _patch_deterministic_prompt_run(monkeypatch):
    class _FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"

    class _FakePromptModel:
        d_model = 4
        n_layers = 2
        device = torch.device("cpu")
        tokenizer = _FakeTokenizer()
        _lm_head = SimpleNamespace(weight=torch.zeros((8, 4)))

    model = _FakePromptModel()
    lens = SimpleNamespace(d_model=4, source_layers=[0], jacobians={})
    monkeypatch.setattr(
        readout,
        "load_lens_model",
        lambda _model_name: (SimpleNamespace(_hf_model=object()), model.tokenizer, "cpu"),
    )
    monkeypatch.setattr(readout, "WrappedModel", lambda *_args: model)
    monkeypatch.setattr(readout.JacobianLens, "load", lambda _path: lens)

    def fake_analyze_column(*, column, **_kwargs):
        return [
            {
                "prompt_column": column.name,
                "index": column.index,
                "context": column.context,
                "n_prompts": 1,
                "layer": 1,
                "is_output": True,
                "top_tokens": [
                    {
                        "rank": 1,
                        "token_id": 1,
                        "token": " answer",
                        "probability": 0.75,
                    }
                ],
            }
        ], 1, 0

    monkeypatch.setattr(readout, "_analyze_column", fake_analyze_column)
    monkeypatch.setattr(readout, "_plot_output_distributions", lambda *_args: None)
    monkeypatch.setattr(readout, "_plot_input_attributions", lambda *_args, **_kwargs: None)
    return model


def _write_prompt_analysis_inputs(tmp_path):
    input_path = tmp_path / "prompts.csv"
    input_path.write_text(
        "Date,prompt_without_context_aapl,prompt_with_context_aapl\n"
        "2026-01-01,plain,with context\n",
        encoding="utf-8",
    )
    lens_path = tmp_path / "lens.pt"
    lens_path.write_bytes(b"fake lens")
    return input_path, lens_path


def test_forward_only_readout_writes_readout_uncertainty_without_attribution(
    tmp_path, monkeypatch
):
    _patch_deterministic_prompt_run(monkeypatch)
    input_path, lens_path = _write_prompt_analysis_inputs(tmp_path)
    monkeypatch.setattr(
        readout,
        "_attribute_column",
        lambda **_kwargs: pytest.fail("forward-only readout must not backpropagate"),
    )

    output_dir = tmp_path / "forward-only"
    readout.analyze_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        lens_path=str(lens_path),
        output_dir=str(output_dir),
        top_k=1,
        compute_input_attribution=False,
    )

    assert (output_dir / "prompt_layer_topk.jsonl").is_file()
    assert (output_dir / "prompt_layer_uncertainty.jsonl").is_file()
    assert (output_dir / "average_layer_topk.jsonl").is_file()
    assert (output_dir / "average_layer_topk.csv").is_file()
    assert (output_dir / "metadata.json").is_file()
    assert not (output_dir / "input_token_attribution.jsonl").exists()
    assert not (output_dir / "input_token_attribution.png").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["backpropagation"] is False
    assert metadata["input_attribution"]["enabled"] is False


def test_backprop_readout_calls_attribution_and_records_provenance(tmp_path, monkeypatch):
    _patch_deterministic_prompt_run(monkeypatch)
    input_path, lens_path = _write_prompt_analysis_inputs(tmp_path)
    attribution_calls = []

    def fake_attribute_column(**kwargs):
        attribution_calls.append(kwargs)
        column = kwargs["column"]
        return [
            {
                "prompt_column": column.name,
                "index": column.index,
                "context": column.context,
                "n_prompts": 1,
                "output_rank": 1,
                "output_token_id": 1,
                "output_token": " answer",
                "output_mean_probability": 0.75,
                "input_positions": [],
                "top_input_tokens": [],
                "top_input_positions": [],
            }
        ]

    monkeypatch.setattr(readout, "_attribute_column", fake_attribute_column)
    output_dir = tmp_path / "with-backprop"
    readout.analyze_prompt_outputs(
        input_path=str(input_path),
        model_name="fake",
        lens_path=str(lens_path),
        output_dir=str(output_dir),
        top_k=1,
        compute_input_attribution=True,
    )

    assert len(attribution_calls) == 2
    assert (output_dir / "input_token_attribution.jsonl").is_file()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["backpropagation"] is True
    assert metadata["input_attribution"]["enabled"] is True
