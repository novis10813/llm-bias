"""Tests for the model-free Phase 1 V2 development material preparation."""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bias.entity_concept_decision.development_materials import (
    build_development_materials,
    prepare_development_prompt,
)


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "docs/entity-concept-decision/details/materials-phase1-v2-draft.md"
REVIEW = ROOT / "docs/entity-concept-decision/details/review-materials-phase1-v2.md"


class _CharacterTokenizer:
    chat_template = "test-template"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_special_tokens_mask: bool = False,
    ) -> SimpleNamespace:
        result: dict[str, object] = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        if return_special_tokens_mask:
            result["special_tokens_mask"] = [False] * len(text)
        return SimpleNamespace(**result)

    def apply_chat_template(self, messages, **kwargs) -> str:
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        return f"<user>{messages[0]['content']}</user><assistant>"


class _CrossBoundaryTokenizer(_CharacterTokenizer):
    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        result = super().__call__(text, **kwargs)
        if kwargs.get("return_offsets_mapping"):
            suffix = "DECIDE"
            start = text.find(suffix)
            offsets = list(result.offset_mapping)
            offsets[start] = (start - 1, start + 1)
            result.offset_mapping = offsets
        return result


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_real_sources_produce_the_exact_development_material_set() -> None:
    result = build_development_materials(SOURCE, REVIEW)

    assert result["schema_version"] == 1
    assert result["protocol"] == "phase1-v2-development"
    assert result["review_status"] == "ai_reviewed_development"
    assert len(result["rows"]) == 60
    assert len(result["comparisons"]) == 38
    assert len({row["id"] for row in result["rows"]}) == 60
    assert {row["id"] for row in result["rows"]}.isdisjoint(
        {"C-V02-P", "C-V02-N", "C-V04-P", "C-V04-N"}
    )
    assert sum(row["role"] == "primary" for row in result["rows"]) == 36
    assert sum(row["role"] == "evaluation" for row in result["rows"]) == 16
    assert sum(row["role"] in {"lexical", "competitor"} for row in result["rows"]) == 8
    assert [row["id"] for row in result["rows"]] == sorted(row["id"] for row in result["rows"])
    assert {row["group_id"] for row in result["rows"] if row["family_id"] == "C-E01"} == {"C-E01"}
    assert {row["source_group"] for row in result["rows"] if row["family_id"] == "C-E01"} == {"C2"}
    assert {row["group_id"] for row in result["rows"] if row["family_id"] == "C-F01"} == {"C1"}
    assert all("split" not in row and "approved" not in row.values() for row in result["rows"])


def test_material_conversion_rejects_duplicate_ids_empty_text_and_bad_replacements(
    tmp_path: Path,
) -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    review_text = REVIEW.read_text(encoding="utf-8")

    duplicate_source = source_text.replace("| C-F01-N |", "| C-F01-P |", 1)
    source_path = tmp_path / "duplicate.md"
    review_path = tmp_path / "review.md"
    _write(source_path, duplicate_source)
    _write(review_path, review_text)
    with pytest.raises(ValueError, match="duplicate material ID"):
        build_development_materials(source_path, review_path)

    empty_source = source_text.replace(
        "| C-F01-P | The company's three largest customers account for 75% of annual revenue. |",
        "| C-F01-P |  |",
        1,
    )
    _write(source_path, empty_source)
    with pytest.raises(ValueError, match="empty text"):
        build_development_materials(source_path, review_path)

    bad_review = review_text.replace(
        "| S-F01-P | Across past economic expansions and contractions, the company's revenue rose substantially during expansions and fell substantially during contractions. |",
        "| S-F01-P | Buy this company. |",
        1,
    )
    _write(source_path, source_text)
    _write(review_path, bad_review)
    with pytest.raises(ValueError, match="investment advice"):
        build_development_materials(source_path, review_path)

    same_review = review_text.replace(
        "| S-F01-P | Across past economic expansions and contractions, the company's revenue rose substantially during expansions and fell substantially during contractions. |",
        "| S-F01-P | Across past economic expansions and contractions, the company's revenue rose and fell substantially. |",
        1,
    )
    _write(review_path, same_review)
    with pytest.raises(ValueError, match="must differ"):
        build_development_materials(source_path, review_path)


def test_material_conversion_rejects_incomplete_source_and_replacement_set(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    review_path = tmp_path / "review.md"
    source_text = SOURCE.read_text(encoding="utf-8")
    review_text = REVIEW.read_text(encoding="utf-8")
    _write(source_path, source_text.replace("| C-F01-P |", "| C-F99-P |", 1))
    _write(review_path, review_text)
    with pytest.raises(ValueError, match="unexpected material ID"):
        build_development_materials(source_path, review_path)

    _write(source_path, source_text)
    _write(review_path, review_text.replace("| S-F02-P |", "| S-F99-P |", 1))
    with pytest.raises(ValueError, match="unexpected replacement ID"):
        build_development_materials(source_path, review_path)


def test_prepare_prompt_has_exact_token_partition_and_answer_suffix() -> None:
    suffix = "DECIDE using the evidence."
    answer_prefix = "Answer:"
    result = prepare_development_prompt(
        _CharacterTokenizer(),
        "Revenue is diversified.",
        instruction_suffix=suffix,
        answer_prefix=answer_prefix,
    )

    start, end = result["instruction_span"]
    assert result["raw_prompt"].endswith(suffix)
    assert result["formatted"].endswith(answer_prefix)
    assert result["formatted"].count(suffix) == 1
    assert result["instruction_ids"] == result["input_ids"][start:end]
    assert result["final_position"] == len(result["input_ids"]) - 1
    assert result["partition"] == {
        "before_instruction": [0, start],
        "instruction": [start, end],
        "after_instruction": [end, len(result["input_ids"])],
    }
    assert result["partition"]["after_instruction"][1] - end >= len(answer_prefix)
    assert result["formatted"].endswith(answer_prefix)
    assert result["formatted"][start:end] == suffix


def test_prepare_prompt_rejects_repeated_suffix_and_cross_boundary_token() -> None:
    tokenizer = _CharacterTokenizer()
    with pytest.raises(ValueError, match="exactly once"):
        prepare_development_prompt(
            tokenizer,
            "The note says DECIDE using the evidence.",
            instruction_suffix="DECIDE using the evidence.",
            answer_prefix="Answer:",
        )

    with pytest.raises(ValueError, match="crosses instruction boundary"):
        prepare_development_prompt(
            _CrossBoundaryTokenizer(),
            "Revenue is diversified.",
            instruction_suffix="DECIDE using the evidence.",
            answer_prefix="Answer:",
        )


def test_prepare_prompt_fails_closed_without_offsets_or_with_invalid_input() -> None:
    class _NoOffsets(_CharacterTokenizer):
        def __call__(self, text: str, **kwargs) -> SimpleNamespace:
            kwargs.pop("return_offsets_mapping", None)
            return super().__call__(text, **kwargs)

    with pytest.raises(ValueError, match="offset"):
        prepare_development_prompt(
            _NoOffsets(), "Description", instruction_suffix="DECIDE", answer_prefix="Answer:"
        )
    with pytest.raises(ValueError, match="non-empty"):
        prepare_development_prompt(
            _CharacterTokenizer(), "", instruction_suffix="DECIDE", answer_prefix="Answer:"
        )
    with pytest.raises(ValueError, match="non-empty"):
        prepare_development_prompt(
            _CharacterTokenizer(), "Description", instruction_suffix="", answer_prefix="Answer:"
        )
