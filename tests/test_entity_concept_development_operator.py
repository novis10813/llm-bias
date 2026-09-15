"""Operator validation without a checkpoint or GPU."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


def _operator():
    spec = importlib.util.spec_from_file_location("concept_development_operator", Path("scripts/entity_concept_development.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def test_company_prefix_append_preserves_archive():
    module = _operator()
    row = {"id": "old", "ticker": "BDX", "formatted": "abcd", "prompt_ids": [97,98,99,100], "instruction_span": [1,3]}
    original = dict(row)
    result = module.prepare_companies(Tokenizer(), [row])[0]
    assert row == original
    assert result["formatted"] == "abcd" + module.ANSWER_PREFIX
    assert result["instruction_ids"] == [98,99]
    assert result["partition"]["instruction"] == [1,3]
    assert result["partition"]["after_instruction"][1] == len(result["input_ids"])
    assert result["final_position"] == len(result["input_ids"])-1
    assert result["source_formatted"] == "abcd"


def test_company_tokenizer_mismatch_refused():
    module = _operator()
    with pytest.raises(ValueError, match="archived tokenizer"):
        module.prepare_companies(Tokenizer(), [{"id":"bad", "ticker":"IT", "formatted":"abcd", "prompt_ids":[1], "instruction_span":[1,3]}])


def test_missing_upstream_fails_before_any_model_loading(tmp_path):
    module = _operator()
    with pytest.raises(FileNotFoundError):
        module.verified_sources(tmp_path, ".cache/models/qwen3.5-4b")
