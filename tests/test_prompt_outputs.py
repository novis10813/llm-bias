import pytest
import torch

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


def test_discover_prompt_columns_rejects_invalid_selected_column():
    with pytest.raises(ValueError, match="must match"):
        discover_prompt_columns(["prompt_sp500"], ["prompt_sp500"])


def test_load_prompt_table_supports_bom_and_generic_index_names(tmp_path):
    path = tmp_path / "prompts.csv"
    path.write_text(
        "﻿Date,prompt_without_context_aapl,prompt_with_context_sp500\n"
        "2026-01-01,plain,with context\n",
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
            "prompt_with_context_sp500": "with context",
        }
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


def test_auto_does_not_treat_partial_pair_schema_as_return_pairs(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text("cik,prompt_without_context_x\n1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no prompt_with_context"):
        load_prompt_table(path, dataset_format="auto")
