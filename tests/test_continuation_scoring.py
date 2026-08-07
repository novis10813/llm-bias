import torch

from llm_bias.core.continuation_scoring import (
    continuation_token_ids,
    score_candidate,
    score_margin,
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
