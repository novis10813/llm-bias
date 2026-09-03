import pytest
import torch

from llm_bias.core.continuation_scoring import (
    continuation_token_ids,
    fp32_next_token_logits,
    score_candidate,
    score_margin,
    score_single_token_margin_fp32,
)


class _Tokenizer:
    def __init__(self):
        self.vocab = {"P": 1, "A": 2, "B": 3, "C": 4}

    def __call__(self, text, *, add_special_tokens=True):
        ids = [self.vocab[char] for char in text]
        if add_special_tokens:
            ids = [99, *ids]
        return {"input_ids": ids}


class _Model:
    def forward(self, input_ids):
        # The next-token score is determined by the preceding token ID.
        logits = torch.zeros((*input_ids.shape, 100), dtype=torch.float32)
        for index in range(input_ids.shape[1]):
            previous = int(input_ids[0, index - 1]) if index else 0
            logits[0, index, 2] = float(previous)
            logits[0, index, 3] = float(previous) + 1
        return logits


def test_continuation_ids_use_complete_prompt_boundary():
    tokenizer = _Tokenizer()
    assert continuation_token_ids(tokenizer, "P", "AB") == ([99, 1], [2, 3])


def test_candidate_score_uses_teacher_forced_token_positions():
    score = score_candidate(_Model(), _Tokenizer(), "P", "AB")
    assert score.candidate == "AB"
    assert score.token_ids == [2, 3]
    assert score.token_count == 2
    assert score.log_probability < 0


def test_margin_is_positive_minus_negative():
    margin = score_margin(_Model(), _Tokenizer(), "P", "B", "A")
    assert margin.positive.candidate == "B"
    assert margin.negative.candidate == "A"
    assert margin.value > 0
    assert margin.definition == "logP(B)-logP(A)"


def test_single_token_fp32_margin_uses_final_residual_path():
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens=True):
            return {"input_ids": {"P": [0], "PB": [0, 1], "PA": [0, 2]}[text]}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.n_layers = 1
            self._final_norm = torch.nn.LayerNorm(2)
            self._lm_head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self._lm_head.weight.copy_(
                    torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                )

        def forward(self, input_ids):
            hidden = torch.tensor([[[1.0, 0.0]]], device=input_ids.device).expand(
                input_ids.shape[0], input_ids.shape[1], 2
            )
            hidden = self.layers[0](hidden)
            normalized = self._final_norm(hidden).to(torch.bfloat16)
            return self._lm_head(normalized)

    margin = score_single_token_margin_fp32(
        Model(), Tokenizer(), "P", "B", "A", device="cpu"
    )
    assert margin.positive.token_ids == [1]
    assert margin.negative.token_ids == [2]
    assert margin.value == pytest.approx(2.0, rel=1e-4)


class _StandardRMSNorm(torch.nn.Module):
    """Standard RMSNorm style: ``norm(x) * w``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = torch.nn.Parameter(torch.full((dim,), 2.0))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * output.to(x.dtype)


class _Llama3RMSNorm(torch.nn.Module):
    """Llama-3 / Qwen3.5 RMSNorm style: ``norm(x) * (1 + w)``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        output = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).float()
        return (output * (1.0 + self.weight.float())).type_as(x)


class _FP32Tokenizer:
    def __call__(self, text, *, add_special_tokens=True):
        return {"input_ids": {"P": [0], "PB": [0, 1], "PA": [0, 2]}[text]}


def _fp32_tail_model(norm: torch.nn.Module) -> torch.nn.Module:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.n_layers = 1
            self._final_norm = norm
            self._lm_head = torch.nn.Linear(2, 3, bias=False)
            with torch.no_grad():
                self._lm_head.weight.copy_(
                    torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                )

        def forward(self, input_ids):
            hidden = torch.tensor([[[1.0, 0.5]]], device=input_ids.device).expand(
                input_ids.shape[0], input_ids.shape[1], 2
            )
            return self._lm_head(self._final_norm(self.layers[0](hidden)))

    return Model()


@pytest.mark.parametrize("norm_cls", [_StandardRMSNorm, _Llama3RMSNorm])
def test_fp32_tail_matches_the_model_module_for_norm_style(norm_cls) -> None:
    model = _fp32_tail_model(norm_cls(2))
    residual = torch.tensor([[1.0, 0.5]])
    logits = fp32_next_token_logits(model, residual)
    expected = model._lm_head(model._final_norm(residual)).float()
    assert torch.allclose(logits, expected, atol=1e-6)


@pytest.mark.parametrize("norm_cls", [_StandardRMSNorm, _Llama3RMSNorm])
def test_fp32_margin_matches_the_model_true_logprobs(norm_cls) -> None:
    model = _fp32_tail_model(norm_cls(2))
    margin = score_single_token_margin_fp32(
        model, _FP32Tokenizer(), "P", "B", "A", device="cpu"
    )
    normalized = model._final_norm(torch.tensor([[1.0, 0.5]]))[0].float().detach()
    expected = float(normalized[0] - normalized[1])
    assert margin.value == pytest.approx(expected, rel=1e-4)
