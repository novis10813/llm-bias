import json

import pytest

from llm_bias.prompt_analysis.validation import _resolve_max_seq_len


def test_resolve_max_seq_len_uses_attribution_metadata(tmp_path):
    attribution = tmp_path / "generated_token_attribution.jsonl"
    attribution.write_text("{}\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"max_seq_len": 512}),
        encoding="utf-8",
    )

    assert _resolve_max_seq_len(attribution, None) == 512
    assert _resolve_max_seq_len(attribution, 384) == 384


def test_resolve_max_seq_len_rejects_invalid_values(tmp_path):
    attribution = tmp_path / "generated_token_attribution.jsonl"
    attribution.write_text("{}\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"max_seq_len": 0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid max_seq_len"):
        _resolve_max_seq_len(attribution, None)
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_max_seq_len(attribution, 0)
