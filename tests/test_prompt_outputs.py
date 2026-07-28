import pytest
import torch

from llm_bias.prompt_analysis.readout import (
    _batched_output_gradients,
    _prepare_prompt,
    discover_prompt_columns,
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
