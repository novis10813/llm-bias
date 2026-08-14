import pytest
import torch

from llm_bias.core.analysis import (
    JACOBIAN_TRANSPORT_METHOD,
    bootstrap_mean_ci,
    full_vocabulary_stats,
    holm_bonferroni,
    mean_full_vocabulary,
    restricted_softmax,
    sign_flip_pvalue,
    top_k_token_records,
    transport_residual_delta,
)


class Tokenizer:
    def decode(self, ids, **_kwargs):
        return f"tok-{ids[0]}"


def test_transport_identifies_jacobian_method_and_preserves_shape():
    result = transport_residual_delta(
        torch.tensor([[2.0, 1.0]]), torch.zeros(1, 2), layer=0, final_layer=1,
        jacobian_cache={0: torch.eye(2)},
    )
    assert result.tolist() == [[2.0, 1.0]]
    assert JACOBIAN_TRANSPORT_METHOD == "jacobian_transport"


def test_mean_full_vocabulary_precedes_top_k():
    mean = mean_full_vocabulary(torch.tensor([[.8, .2, 0.], [0., .2, .8]]))
    assert torch.allclose(mean, torch.tensor([.4, .2, .4]))
    assert top_k_token_records(mean, top_k=2, tokenizer=Tokenizer())[0]["probability"] == pytest.approx(.4)


def test_restricted_and_full_statistics_are_separate():
    probs = restricted_softmax(torch.tensor([0., 1., 2., 100.]), [0, 1, 2])
    assert sum(probs.tolist()) == pytest.approx(1)
    assert full_vocabulary_stats(torch.tensor([.5, .5]))["normalized_entropy"] == pytest.approx(1)


def test_bootstrap_is_deterministic_and_holm_preserves_order():
    assert bootstrap_mean_ci([1., 2., 3.], seed=4, n_resamples=100) == bootstrap_mean_ci([1., 2., 3.], seed=4, n_resamples=100)
    assert sign_flip_pvalue([1., 1.], seed=1, n_resamples=100) is not None
    assert holm_bonferroni([.01, .2, .03])[0] <= holm_bonferroni([.01, .2, .03])[1]
