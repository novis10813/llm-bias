"""Data preparation and compact records for the easy-bias binary task.

The task is intentionally separate from the reviewed financial
``entity_bias`` protocol.  It measures a model's association between a career
prompt and a binary continuation; it does not define a gold parental outcome.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from llm_bias.core.continuation_scoring import continuation_token_ids
from llm_bias.counterfactual_patching.interventions import normalized_span_mapping

DATASET_NAME = "easy-bias-zh-tw-binary-v1"
TASK_TYPE = "binary_association"
MARGIN_DEFINITION = "logP(媽媽 continuation|prompt)-logP(爸爸 continuation|prompt)"
CANDIDATES = ("媽媽", "爸爸")
ORDERS = ("dad_first", "mom_first")


@dataclass(frozen=True)
class EasyBiasTemplates:
    system_dad_first: str
    user_dad_first: str
    system_mom_first: str
    user_mom_first: str
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class SpanRecord:
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RenderedPrompt:
    career_id: str
    career: str
    split: str
    prompt_order: str
    system_prompt: str
    user_prompt: str
    formatted_prompt: str
    career_char_occurrences: list[list[int]]
    entity_spans: list[SpanRecord]
    candidate_token_ids: dict[str, list[int]]
    source_dataset: str = DATASET_NAME
    task_type: str = TASK_TYPE

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["entity_spans"] = [span.to_dict() for span in self.entity_spans]
        return result


@dataclass(frozen=True)
class BinaryAssociationPair:
    pair_id: str
    contrast_id: str
    direction: str
    prompt_order: str
    split: str
    source_career_id: str
    target_career_id: str
    source_career: str
    target_career: str
    source_prompt_id: str
    target_prompt_id: str
    source_prompt: str
    target_prompt: str
    source_entity_spans: list[dict[str, Any]]
    target_entity_spans: list[dict[str, Any]]
    source_entity_token_ids: list[list[int]]
    target_entity_token_ids: list[list[int]]
    candidate_spec: list[str]
    margin_definition: str = MARGIN_DEFINITION
    task_type: str = TASK_TYPE
    dataset: str = DATASET_NAME

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_easy_bias_root() -> Path:
    return Path(__file__).resolve().parents[2].parent / "easy-bias"


def default_careers_path() -> Path:
    return _default_easy_bias_root() / "expanded_careers.json"


def default_inference_path() -> Path:
    return _default_easy_bias_root() / "inference.py"


def default_order_path() -> Path:
    return _default_easy_bias_root() / "compare_option_order.py"


def _literal_assignments(path: str | Path) -> dict[str, Any]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def load_templates(
    inference_path: str | Path | None = None,
    order_path: str | Path | None = None,
) -> EasyBiasTemplates:
    """Extract prompt constants without importing easy-bias runtime dependencies."""
    inference = Path(inference_path or default_inference_path())
    order = Path(order_path or default_order_path())
    inference_values = _literal_assignments(inference)
    order_values = _literal_assignments(order)
    names = {
        "system_dad_first": order_values.get("SYSTEM_DAD_FIRST")
        or inference_values.get("SYSTEM_PROMPT_BINARY"),
        "user_dad_first": order_values.get("USER_DAD_FIRST")
        or inference_values.get("USER_PROMPT_BINARY"),
        "system_mom_first": order_values.get("SYSTEM_MOM_FIRST"),
        "user_mom_first": order_values.get("USER_MOM_FIRST"),
    }
    missing = [name for name, value in names.items() if not isinstance(value, str)]
    if missing:
        raise ValueError(f"easy-bias template constants missing: {missing}")
    return EasyBiasTemplates(
        **names,
        source_files=(str(inference), str(order)),
    )


def load_careers(path: str | Path | None = None) -> list[str]:
    source = Path(path or default_careers_path())
    values = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("careers input must be a JSON list of strings")
    careers = [value.strip() for value in values]
    if any(not value for value in careers):
        raise ValueError("careers input contains an empty career")
    if len(set(careers)) != len(careers):
        raise ValueError("careers input contains duplicate careers")
    return careers


def career_split(career: str, *, seed: int = 0) -> str:
    """Assign one career deterministically to train/calibration/confirmation."""
    digest = hashlib.sha256(f"{seed}\0{career}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "calibration"
    return "confirmation"


def _format_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(formatted, str):
            raise TypeError("tokenizer chat template must return text when tokenize=False")
        return formatted
    return f"{system_prompt}\n{user_prompt}\n"


def _input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _all_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    occurrences: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = text.find(value, offset)
        if start < 0:
            return occurrences
        occurrences.append((start, start + len(value)))
        offset = start + len(value)


def _span_records(tokenizer: Any, text: str, occurrences: Iterable[tuple[int, int]]) -> list[SpanRecord]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = encoded["offset_mapping"]
    specials = encoded.get("special_tokens_mask", [False] * len(offsets))
    input_ids = _input_ids(tokenizer, text)
    records: list[SpanRecord] = []
    for char_start, char_end in occurrences:
        token_indices = [
            index
            for index, ((token_start, token_end), special) in enumerate(
                zip(offsets, specials, strict=True)
            )
            if not special
            and token_end > token_start
            and token_start < char_end
            and token_end > char_start
        ]
        if not token_indices:
            raise ValueError(f"could not map character span {(char_start, char_end)}")
        token_start, token_end = min(token_indices), max(token_indices) + 1
        records.append(
            SpanRecord(
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
                token_ids=input_ids[token_start:token_end],
            )
        )
    return records


def render_prompt(
    tokenizer: Any,
    *,
    career_id: str,
    career: str,
    split: str,
    prompt_order: str,
    templates: EasyBiasTemplates,
) -> RenderedPrompt:
    if prompt_order not in ORDERS:
        raise ValueError(f"unsupported prompt order: {prompt_order}")
    if prompt_order == "dad_first":
        system_prompt, user_template = templates.system_dad_first, templates.user_dad_first
    else:
        system_prompt, user_template = templates.system_mom_first, templates.user_mom_first
    user_prompt = user_template.format(career=career)
    formatted = _format_prompt(tokenizer, system_prompt, user_prompt)
    occurrences = _all_occurrences(formatted, career)
    if len(occurrences) != 2:
        raise ValueError(
            f"expected two career occurrences in formatted prompt, found {len(occurrences)}"
        )
    spans = _span_records(tokenizer, formatted, occurrences)
    candidate_ids = {
        candidate: continuation_token_ids(tokenizer, formatted, candidate)[1]
        for candidate in CANDIDATES
    }
    return RenderedPrompt(
        career_id=career_id,
        career=career,
        split=split,
        prompt_order=prompt_order,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        formatted_prompt=formatted,
        career_char_occurrences=[list(item) for item in occurrences],
        entity_spans=spans,
        candidate_token_ids=candidate_ids,
    )


def _pair_records(rendered: list[RenderedPrompt]) -> list[BinaryAssociationPair]:
    by_key = {(item.career_id, item.prompt_order): item for item in rendered}
    by_split: dict[str, list[RenderedPrompt]] = {}
    for item in rendered:
        by_split.setdefault(item.split, []).append(item)
    order_by_career: dict[str, set[str]] = {}
    for career_id, prompt_order in by_key:
        order_by_career.setdefault(career_id, set()).add(prompt_order)
    complete_ids = {
        career_id
        for career_id, orders in order_by_career.items()
        if set(ORDERS).issubset(orders)
    }
    pairs: list[BinaryAssociationPair] = []
    for split, records in sorted(by_split.items()):
        records = sorted(
            [item for item in records if item.career_id in complete_ids],
            key=lambda item: item.career_id,
        )
        if len(records) < 2:
            continue
        seen_unordered: set[tuple[str, str]] = set()
        for index, source in enumerate(records):
            target = records[(index + 1) % len(records)]
            if source.career_id == target.career_id:
                continue
            unordered = tuple(sorted((source.career_id, target.career_id)))
            if unordered in seen_unordered:
                continue
            seen_unordered.add(unordered)
            for prompt_order in ORDERS:
                source_record = by_key[(source.career_id, prompt_order)]
                target_record = by_key[(target.career_id, prompt_order)]
                contrast_id = f"{split}-{source.career_id}-{target.career_id}-{prompt_order}"
                pairs.append(
                    BinaryAssociationPair(
                        pair_id=f"{contrast_id}-forward",
                        contrast_id=contrast_id,
                        direction="source_to_target",
                        prompt_order=prompt_order,
                        split=split,
                        source_career_id=source.career_id,
                        target_career_id=target.career_id,
                        source_career=source.career,
                        target_career=target.career,
                        source_prompt_id=f"{source.career_id}-{prompt_order}",
                        target_prompt_id=f"{target.career_id}-{prompt_order}",
                        source_prompt=source_record.formatted_prompt,
                        target_prompt=target_record.formatted_prompt,
                        source_entity_spans=[span.to_dict() for span in source_record.entity_spans],
                        target_entity_spans=[span.to_dict() for span in target_record.entity_spans],
                        source_entity_token_ids=[span.token_ids for span in source_record.entity_spans],
                        target_entity_token_ids=[span.token_ids for span in target_record.entity_spans],
                        candidate_spec=list(CANDIDATES),
                    )
                )
                pairs.append(
                    BinaryAssociationPair(
                        pair_id=f"{contrast_id}-reverse",
                        contrast_id=contrast_id,
                        direction="target_to_source",
                        prompt_order=prompt_order,
                        split=split,
                        source_career_id=target.career_id,
                        target_career_id=source.career_id,
                        source_career=target.career,
                        target_career=source.career,
                        source_prompt_id=f"{target.career_id}-{prompt_order}",
                        target_prompt_id=f"{source.career_id}-{prompt_order}",
                        source_prompt=target_record.formatted_prompt,
                        target_prompt=source_record.formatted_prompt,
                        source_entity_spans=[span.to_dict() for span in target_record.entity_spans],
                        target_entity_spans=[span.to_dict() for span in source_record.entity_spans],
                        source_entity_token_ids=[span.token_ids for span in target_record.entity_spans],
                        target_entity_token_ids=[span.token_ids for span in source_record.entity_spans],
                        candidate_spec=list(CANDIDATES),
                    )
                )
    return pairs


def prepare_binary_association(
    tokenizer: Any,
    *,
    output_dir: str | Path,
    careers_path: str | Path | None = None,
    inference_path: str | Path | None = None,
    order_path: str | Path | None = None,
    seed: int = 0,
    max_careers: int | None = None,
) -> dict[str, Any]:
    """Render careers, write omissions, and materialize deterministic pairs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    careers = load_careers(careers_path)
    if max_careers is not None:
        careers = careers[:max_careers]
    templates = load_templates(inference_path, order_path)
    rendered: list[RenderedPrompt] = []
    omissions: list[dict[str, Any]] = []
    for index, career in enumerate(careers):
        career_id = f"career-{index:04d}"
        split = career_split(career, seed=seed)
        for prompt_order in ORDERS:
            try:
                rendered.append(
                    render_prompt(
                        tokenizer,
                        career_id=career_id,
                        career=career,
                        split=split,
                        prompt_order=prompt_order,
                        templates=templates,
                    )
                )
            except Exception as error:
                omissions.append(
                    {
                        "career_id": career_id,
                        "career": career,
                        "split": split,
                        "prompt_order": prompt_order,
                        "reason": str(error),
                    }
                )
    pairs = _pair_records(rendered)
    _write_jsonl(output / "careers.jsonl", [
        {"career_id": f"career-{index:04d}", "career": career, "split": career_split(career, seed=seed)}
        for index, career in enumerate(careers)
    ])
    _write_jsonl(output / "rendered_prompts.jsonl", [item.to_dict() for item in rendered])
    _write_jsonl(output / "pairs.jsonl", [item.to_dict() for item in pairs])
    _write_jsonl(output / "omissions.jsonl", omissions)
    metadata = {
        "dataset": DATASET_NAME,
        "task_type": TASK_TYPE,
        "seed": seed,
        "career_count": len(careers),
        "rendered_prompt_count": len(rendered),
        "pair_count": len(pairs),
        "omission_count": len(omissions),
        "template_sources": list(templates.source_files),
        "margin_definition": MARGIN_DEFINITION,
    }
    (output / "prepare_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_rendered_prompts(path: str | Path) -> Iterator[RenderedPrompt]:
    """Yield rendered prompts without materializing the full artifact."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                value["entity_spans"] = [
                    SpanRecord(**span) for span in value["entity_spans"]
                ]
                yield RenderedPrompt(**value)


def iter_binary_pairs(path: str | Path) -> Iterator[BinaryAssociationPair]:
    """Yield binary pairs without materializing the full artifact."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield BinaryAssociationPair(**json.loads(line))


def validate_pair(pair: BinaryAssociationPair) -> None:
    """Fail closed on the structural invariants needed by intervention."""
    if pair.task_type != TASK_TYPE or pair.margin_definition != MARGIN_DEFINITION:
        raise ValueError("pair has incompatible binary-association semantics")
    if len(pair.source_entity_spans) != 2 or len(pair.target_entity_spans) != 2:
        raise ValueError("binary-association pair must contain two entity occurrences")
    for spans, token_ids in (
        (pair.source_entity_spans, pair.source_entity_token_ids),
        (pair.target_entity_spans, pair.target_entity_token_ids),
    ):
        if len(spans) != len(token_ids):
            raise ValueError("span and token-id occurrence counts differ")
        for span, ids in zip(spans, token_ids, strict=True):
            if span["token_end"] <= span["token_start"] or not ids:
                raise ValueError("entity span must be non-empty")
            if span["token_end"] - span["token_start"] != len(ids):
                raise ValueError("entity token IDs do not match span length")
    if pair.candidate_spec != list(CANDIDATES):
        raise ValueError("unexpected candidate specification")
    if pair.source_career_id == pair.target_career_id:
        raise ValueError("source and target careers must differ")
    if pair.source_prompt.replace(pair.source_career, pair.target_career) != pair.target_prompt:
        raise ValueError("source and target prompts differ beyond the career replacement")
    for source, target in zip(pair.source_entity_spans, pair.target_entity_spans, strict=True):
        normalized_span_mapping(
            source["token_end"] - source["token_start"],
            target["token_end"] - target["token_start"],
        )
