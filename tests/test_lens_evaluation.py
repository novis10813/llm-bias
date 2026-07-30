import json
from types import SimpleNamespace

import pytest
import torch

from llm_bias.lens_fitting.evaluation import (
    BilingualEvalItem,
    _score_lens,
    canonical_concept_token_id,
    load_bilingual_holdout,
    paired_candidate_uncertainty,
    single_token_variants,
    summarize_candidate_rows,
    validate_candidate_calibration,
)


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        values = {
            "word": [10],
            " word": [11],
            "many pieces": [20, 21],
            " many pieces": [22, 23],
        }
        return SimpleNamespace(input_ids=values[text])


class _ScoringTokenizer:
    def __call__(self, text, **_kwargs):
        values = {
            " native": [1],
            "native": [3],
            "跨": [2],
            " 跨": [4],
        }
        return SimpleNamespace(input_ids=values[text])


def test_single_token_variants_keeps_raw_and_leading_space_ids():
    assert single_token_variants(_Tokenizer(), "word") == [10, 11]
    assert single_token_variants(_Tokenizer(), "many pieces") == []


def test_canonical_concept_token_uses_one_language_specific_variant():
    assert canonical_concept_token_id(
        _Tokenizer(), "word", language="en"
    ) == 11
    assert canonical_concept_token_id(
        _Tokenizer(), "word", language="zh-CN"
    ) == 10
    with pytest.raises(ValueError, match="no canonical"):
        canonical_concept_token_id(
            _Tokenizer(), "many pieces", language="en"
        )


def test_bilingual_summary_uses_balanced_language_metric():
    rows = [
        {
            "language": "en",
            "native_min_rank": 1,
            "bilingual_min_rank": 1,
        },
        {
            "language": "en",
            "native_min_rank": 100,
            "bilingual_min_rank": 10,
        },
        {
            "language": "zh-CN",
            "native_min_rank": 10,
            "bilingual_min_rank": 5,
        },
    ]

    result = summarize_candidate_rows(rows)

    assert result["by_language"]["en"]["native"]["count"] == 2
    assert result["by_language"]["zh-CN"]["native"]["count"] == 1
    assert result["balanced_native_mean_log10_rank"] == pytest.approx(1.0)
    assert result["selection_score"] == pytest.approx(-1.0)


def test_load_bilingual_holdout_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "holdout.jsonl"
    row = {
        "id": "same",
        "pair_id": "pair",
        "language": "en",
        "prompt": "prompt",
        "native_intermediate": "word",
        "crosslingual_intermediate": "词",
        "target": "answer",
    }
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_bilingual_holdout(path)


def _paired_rows(ranks):
    rows = []
    for index, (english_rank, chinese_rank) in enumerate(ranks):
        for language, rank in (
            ("en", english_rank),
            ("zh-CN", chinese_rank),
        ):
            rows.append(
                {
                    "pair_id": f"pair-{index}",
                    "language": language,
                    "native_min_rank": rank,
                }
            )
    return rows


def test_paired_uncertainty_is_deterministic_and_favors_better_candidate():
    rows = {
        "winner": _paired_rows([(1, 1), (1, 10), (10, 1), (1, 1)]),
        "loser": _paired_rows(
            [(100, 100), (100, 1000), (1000, 100), (100, 100)]
        ),
    }

    first = paired_candidate_uncertainty(
        selected="winner",
        candidate_rows=rows,
        n_resamples=2_000,
        seed=7,
    )
    second = paired_candidate_uncertainty(
        selected="winner",
        candidate_rows=rows,
        n_resamples=2_000,
        seed=7,
    )

    assert first == second
    comparison = first["comparisons"]["loser"]
    assert comparison["pair_count"] == 4
    assert comparison[
        "selected_minus_competitor_mean_log10_rank"
    ] == pytest.approx(-2.0)
    assert comparison["paired_bootstrap_95_ci"][1] < 0
    assert comparison["sign_flip_one_sided_p"] < 0.1


def test_paired_uncertainty_requires_complete_bilingual_pairs():
    incomplete = [
        {
            "pair_id": "pair-0",
            "language": "en",
            "native_min_rank": 1,
        }
    ]

    with pytest.raises(ValueError, match="exactly one en and one zh-CN"):
        paired_candidate_uncertainty(
            selected="winner",
            candidate_rows={"winner": incomplete, "other": incomplete},
            n_resamples=10,
        )


def test_candidate_calibration_validation_checks_count_and_label(tmp_path):
    lens_path = tmp_path / "english" / "jacobian_lens.pt"
    lens_path.parent.mkdir()
    lens_path.touch()
    lens_path.with_name("jacobian_lens.pt.metadata.json").write_text(
        json.dumps(
            {
                "calibration_count": 128,
                "calibration_source": "data/calibration/english.jsonl",
                "use_chat_template": True,
                "enable_thinking": False,
            }
        ),
        encoding="utf-8",
    )

    metadata = validate_candidate_calibration(
        name="english",
        lens=SimpleNamespace(n_prompts=128),
        lens_path=lens_path,
        expected_n_prompts=128,
    )

    assert metadata["calibration_count"] == 128
    with pytest.raises(ValueError, match="fitted on 127 prompts"):
        validate_candidate_calibration(
            name="english",
            lens=SimpleNamespace(n_prompts=127),
            lens_path=lens_path,
            expected_n_prompts=128,
        )


def test_candidate_calibration_validation_rejects_wrong_source(tmp_path):
    lens_path = tmp_path / "english" / "jacobian_lens.pt"
    lens_path.parent.mkdir()
    lens_path.touch()
    lens_path.with_name("jacobian_lens.pt.metadata.json").write_text(
        json.dumps(
            {
                "calibration_count": 128,
                "calibration_source": "data/calibration/mixed.jsonl",
                "use_chat_template": True,
                "enable_thinking": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mislabeled"):
        validate_candidate_calibration(
            name="english",
            lens=SimpleNamespace(n_prompts=128),
            lens_path=lens_path,
            expected_n_prompts=128,
        )


def test_score_lens_tracks_minimum_rank_across_layers():
    class Model:
        input_device = torch.device("cpu")

        @staticmethod
        def unembed(hidden):
            return hidden

    class Lens:
        @staticmethod
        def transport(hidden, layer):
            return hidden + layer

    item = BilingualEvalItem(
        item_id="item",
        pair_id="pair",
        language="en",
        prompt="prompt",
        native_intermediate="native",
        crosslingual_intermediate="跨",
        target="target",
    )
    residuals = {
        0: torch.tensor([[0.0, 4.0, 3.0, 2.0, 1.0]]),
        1: torch.tensor([[0.0, 1.0, 4.0, 3.0, 2.0]]),
    }

    rows = _score_lens(
        model=Model(),
        lens=Lens(),
        residuals=residuals,
        items=[item],
        tokenizer=_ScoringTokenizer(),
        band_layers=[0, 1],
    )

    assert rows[0]["native_min_rank"] == 1
    assert rows[0]["bilingual_min_rank"] == 1
