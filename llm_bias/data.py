"""Build aligned factual counterfactual pairs for the entity experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pair:
    pair_id: str
    category: str
    function: str
    source_entity: str
    target_entity: str
    source_prompt: str
    target_prompt: str
    source_answer: str
    target_answer: str
    source_entity_start: int
    source_entity_end: int
    target_entity_start: int
    target_entity_end: int
    source_entity_token: int
    target_entity_token: int
    answer_source_token: int
    answer_target_token: int
    source_entity_token_ids: list[int] | None = None
    target_entity_token_ids: list[int] | None = None

    def __post_init__(self) -> None:
        """Normalize legacy single-token pairs to the span representation."""
        if not self.source_entity_token_ids:
            object.__setattr__(self, "source_entity_token_ids", [self.source_entity_token])
        else:
            object.__setattr__(
                self, "source_entity_token_ids", list(self.source_entity_token_ids)
            )
        if not self.target_entity_token_ids:
            object.__setattr__(self, "target_entity_token_ids", [self.target_entity_token])
        else:
            object.__setattr__(
                self, "target_entity_token_ids", list(self.target_entity_token_ids)
            )
        if len(self.source_entity_token_ids) != (
            self.source_entity_end - self.source_entity_start
        ):
            raise ValueError("source_entity_token_ids must match the source entity span")
        if len(self.target_entity_token_ids) != (
            self.target_entity_end - self.target_entity_start
        ):
            raise ValueError("target_entity_token_ids must match the target entity span")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / (
        "third_party/jacobian-lens/data/experiments/flexible-generalization.json"
    )


def _token_span(tokenizer: Any, text: str, start: int, end: int) -> tuple[int, int] | None:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = encoded["offset_mapping"]
    specials = encoded["special_tokens_mask"]
    spans = [
        index
        for index, ((token_start, token_end), special) in enumerate(
            zip(offsets, specials, strict=True)
        )
        if not special
        and token_end > token_start
        and token_start < end
        and token_end > start
    ]
    if not spans:
        return None
    return min(spans), max(spans) + 1


def _single_token_id(tokenizer: Any, word: str) -> int | None:
    # Answers occur immediately after a prompt ending in a word, so the
    # generated token normally includes its leading space.
    token_ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def load_pairs(
    tokenizer: Any,
    *,
    spec_path: str | Path | None = None,
    max_pairs: int | None = None,
) -> list[Pair]:
    """Load source/target pairs from the upstream factual spec.

    Pairs are retained when both entities occupy contiguous token spans and
    both expected answers are single-token continuations. Source and target
    prompts may have different token lengths; span patching maps the two
    entity spans separately.
    """
    path = Path(spec_path) if spec_path else default_spec_path()
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)

    pairs: list[Pair] = []
    for category in spec["categories"]:
        category_name = category["name"]
        entities = category["args"]
        for function in category["funcs"]:
            template = function["template"]
            answers = function["answers"]
            if template.count("{arg}") != 1:
                continue
            prefix, suffix = template.split("{arg}")
            for source in entities:
                for target in entities:
                    if source == target:
                        continue
                    source_prompt = prefix + source + suffix
                    target_prompt = prefix + target + suffix
                    source_span = _token_span(
                        tokenizer, source_prompt, len(prefix), len(prefix) + len(source)
                    )
                    target_span = _token_span(
                        tokenizer, target_prompt, len(prefix), len(prefix) + len(target)
                    )
                    source_answer = answers[source]
                    target_answer = answers[target]
                    source_answer_token = _single_token_id(tokenizer, source_answer)
                    target_answer_token = _single_token_id(tokenizer, target_answer)
                    source_input_ids = tokenizer(source_prompt, add_special_tokens=True).input_ids
                    target_input_ids = tokenizer(target_prompt, add_special_tokens=True).input_ids
                    source_entity_token = int(source_input_ids[source_span[0]]) if source_span else None
                    target_entity_token = int(target_input_ids[target_span[0]]) if target_span else None
                    if (
                        source_span is None
                        or target_span is None
                        or source_answer_token is None
                        or target_answer_token is None
                        or source_entity_token is None
                        or target_entity_token is None
                    ):
                        continue
                    source_entity_token_ids = [
                        int(token_id)
                        for token_id in source_input_ids[source_span[0] : source_span[1]]
                    ]
                    target_entity_token_ids = [
                        int(token_id)
                        for token_id in target_input_ids[target_span[0] : target_span[1]]
                    ]
                    pair = Pair(
                        pair_id=f"{category_name}-{function['name']}-{source}-to-{target}",
                        category=category_name,
                        function=function["name"],
                        source_entity=source,
                        target_entity=target,
                        source_prompt=source_prompt,
                        target_prompt=target_prompt,
                        source_answer=source_answer,
                        target_answer=target_answer,
                        source_entity_start=source_span[0],
                        source_entity_end=source_span[1],
                        target_entity_start=target_span[0],
                        target_entity_end=target_span[1],
                        source_entity_token=source_entity_token,
                        target_entity_token=target_entity_token,
                        answer_source_token=source_answer_token,
                        answer_target_token=target_answer_token,
                        source_entity_token_ids=source_entity_token_ids,
                        target_entity_token_ids=target_entity_token_ids,
                    )
                    pairs.append(pair)
                    if max_pairs is not None and len(pairs) >= max_pairs:
                        return pairs
    return pairs


def save_pairs(pairs: list[Pair], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def load_saved_pairs(path: str | Path) -> list[Pair]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Pair(**json.loads(line)) for line in handle if line.strip()]


def calibration_prompts(count: int = 16) -> list[str]:
    """Return deterministic, non-task prompts for fitting a model-specific lens."""
    seeds = [
        "A small village sits beside a river and depends on the water for travel and trade.",
        "Researchers collected careful notes before comparing the results of the experiment.",
        "The old library contains maps, letters, and books from many different periods.",
        "During the afternoon, clouds moved across the valley while the temperature slowly fell.",
        "A teacher asked the class to explain the pattern using a short example and clear evidence.",
        "The mechanic inspected the engine, replaced a worn part, and tested the vehicle again.",
        "Several birds gathered on the roof before the storm arrived from the western horizon.",
        "The museum displayed tools that showed how people solved practical problems in the past.",
    ]
    return [seeds[index % len(seeds)] + f" This is calibration passage {index}." for index in range(count)]
