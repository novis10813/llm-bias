import json

import pytest

from llm_bias.core.artifact_paths import sha256_file
from llm_bias.prompt_analysis.validation import _load_backward, _resolve_max_seq_len


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_backward_parent_relative_path_resolves_from_metadata_sidecar(tmp_path):
    forward = tmp_path / "forward" / "generated_outputs.jsonl"
    backward = tmp_path / "nested" / "backward" / "generated_token_attribution.jsonl"
    row = {"record_id": "r1", "date": "2026-01-01", "generated_tokens": []}
    _write_jsonl(forward, [row])
    _write_jsonl(backward, [{**row, "artifact_type": "generated_token_attribution"}])
    (backward.parent / "metadata.json").write_text(json.dumps({
        "artifact_type": "generated_token_attribution",
        "parent_forward_path": "../../forward/generated_outputs.jsonl",
        "parent_forward_sha256": sha256_file(forward),
    }), encoding="utf-8")

    _rows, _kind, _metadata, resolved, _digest = _load_backward(backward)
    assert resolved == forward.resolve()


def test_backward_validation_rejects_missing_forward_coverage(tmp_path):
    forward = tmp_path / "forward.jsonl"
    backward = tmp_path / "backward.jsonl"
    _write_jsonl(forward, [{"record_id": "r1"}, {"record_id": "r2"}])
    _write_jsonl(backward, [{"record_id": "r1", "artifact_type": "generated_token_attribution"}])
    (tmp_path / "metadata.json").write_text(json.dumps({
        "artifact_type": "generated_token_attribution",
        "parent_forward_path": "forward.jsonl",
        "parent_forward_sha256": sha256_file(forward),
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="coverage"):
        _load_backward(backward)


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
