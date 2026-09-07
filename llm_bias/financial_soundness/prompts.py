"""Paired short sentences; no behavioral selection using held-out responses."""
from __future__ import annotations

from typing import Any
from llm_bias.core.continuation_scoring import continuation_token_ids

VERSION = "exploratory-v1"
SPLITS = ("discovery", "calibration", "held-out")
FAMILIES = ("topic", "company", "evidence", "comparison")
MODES = ("direct", "labels-forward", "labels-reversed")
CONCEPTS = {
    "stability": ("financial stability", "operational stability", " stable", " unstable"),
    "burden": ("debt burden", "workload burden", " low", " high"),
    "liquidity": ("liquidity", "productivity", " adequate", " inadequate"),
}
SHORT = (
    "The {attribute} of {subject} is",
    "Regarding {subject}, its {attribute} is",
    "For {subject}, the {attribute} is",
    "The reported {attribute} of {subject} is",
    "Considering {subject}, its {attribute} is",
    "As for {subject}, its {attribute} is",
)
QUESTIONS = (
    "Its {attribute} is:", "The {attribute} is:",
    "Based only on these amounts, its {attribute} is:",
    "Given these amounts, the {attribute} is:",
    "Using only these figures, its {attribute} is:",
    "On these figures alone, the {attribute} is:",
)
NUMBERS = (((20, 5), (30, 10)), ((40, 8), (60, 15)), ((70, 10), (90, 30)))


def _member(text: str, positive: str, negative: str, expected: int | None, mode: str) -> dict[str, Any]:
    if mode != "direct":
        a, b = (positive, negative) if mode == "labels-forward" else (negative, positive)
        text += f" Choose A ({a.strip()}) or B ({b.strip()}). Answer:"
        positive, negative = (" A", " B") if mode == "labels-forward" else (" B", " A")
    return {"text": text, "positive": positive, "negative": negative, "expected": expected}


def build_prompts(companies: list[str]) -> dict[str, Any]:
    if len(companies) < 2 or len(set(companies)) != len(companies) or any(not c.strip() or "\n" in c for c in companies):
        raise ValueError("provide at least two distinct, nonempty company names")
    pairs = []
    def append(family, concept, split, group, template, mode, a, b, pos, neg, truth):
        pairs.append({
            "pair_id": f"{family}-{concept}-{split}-{group}-t{template}-{mode}",
            "family": family, "concept": concept, "split": split, "group_id": group,
            "template_id": template, "answer_mode": mode,
            "a": _member(a, pos, neg, 1 if truth else None, mode),
            "b": _member(b, pos, neg, -1 if truth else None, mode),
        })
    for si, split in enumerate(SPLITS):
        for ti in (2 * si, 2 * si + 1):
            for mode in MODES:
                for concept, (attribute, control, pos, neg) in CONCEPTS.items():
                    text = SHORT[ti].format(attribute=attribute, subject="the company")
                    other = SHORT[ti].format(attribute=control, subject="the company")
                    append("topic", concept, split, concept, ti, mode, text, other, pos, neg, False)
                    for ci, company in enumerate(companies):
                        named = SHORT[ti].format(attribute=attribute, subject=company)
                        append("company", concept, split, f"company-{ci}", ti, mode, named, text, pos, neg, False)
                for ni, (high, low) in enumerate(NUMBERS[si]):
                    for family in ("evidence", "comparison"):
                        def sentence(x, y):
                            if family == "evidence":
                                evidence = f"The company has USD {x} million in cash and USD {y} million in debt due within one year."
                                attribute = "cash coverage of that debt"
                            else:
                                evidence = f"The warehouse has {x} units in stock and orders for {y} units."
                                attribute = "stock coverage of those orders"
                            return evidence + " " + QUESTIONS[ti].format(attribute=attribute)
                        append(family, "coverage", split, f"numbers-{si}-{ni}", ti, mode,
                               sentence(high, low), sentence(low, high), " adequate", " inadequate", True)
    result = {"schema_version": 1, "protocol_version": VERSION, "companies": companies, "pairs": pairs}
    validate_prompts(result)
    return result


def validate_prompts(data: dict[str, Any]) -> None:
    if data.get("schema_version") != 1 or data.get("protocol_version") != VERSION:
        raise ValueError("unsupported prompt protocol")
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("empty pairs")
    ids, coverage, templates, numbers = set(), set(), {}, {}
    group_splits = {}
    for pair in pairs:
        for key in ("pair_id", "family", "concept", "split", "group_id", "answer_mode"):
            if not isinstance(pair.get(key), str) or not pair[key].strip():
                raise ValueError(f"missing or invalid {key}")
        if pair["pair_id"] in ids:
            raise ValueError("duplicate pair ID")
        ids.add(pair["pair_id"])
        family, split, mode = pair["family"], pair["split"], pair["answer_mode"]
        if family not in FAMILIES or split not in SPLITS or mode not in MODES:
            raise ValueError("unknown family/split/answer mode")
        if not isinstance(pair.get("template_id"), int):
            raise ValueError("missing template ID")
        key = pair["template_id"]
        if key in templates and templates[key] != split:
            raise ValueError("template leaks across splits")
        templates[key] = split
        if family in {"evidence", "comparison"}:
            key = pair["group_id"]
            if key in numbers and numbers[key] != split:
                raise ValueError("numeric group leaks across splits")
            numbers[key] = split
        coverage.add((family, split, mode))
        group_splits.setdefault((family, pair["concept"], mode), set()).add(split)
        for name, expected in (("a", 1), ("b", -1)):
            row = pair.get(name)
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("text", "positive", "negative")):
                raise ValueError("invalid pair member")
            if row["positive"] == row["negative"]:
                raise ValueError("answers must differ")
            wanted = expected if family in {"evidence", "comparison"} else None
            if row.get("expected") != wanted:
                raise ValueError("invalid expected answer")
    required = {(f, s, m) for f in FAMILIES for s in SPLITS for m in MODES}
    if coverage != required:
        raise ValueError("missing required family, split or answer-mode controls")
    if any(splits != set(SPLITS) for splits in group_splits.values()):
        raise ValueError("each concept/mode group must occur in all splits")


def prepare_encoded(data: dict[str, Any], tokenizer: Any, max_tokens: int = 512) -> dict[str, Any]:
    validate_prompts(data)
    prepared = {k: v for k, v in data.items() if k != "pairs"}
    prepared["pairs"] = []
    for pair in data["pairs"]:
        item = dict(pair)
        for name in ("a", "b"):
            row = dict(pair[name])
            p, positive = continuation_token_ids(tokenizer, row["text"], row["positive"])
            q, negative = continuation_token_ids(tokenizer, row["text"], row["negative"])
            if not p or p != q or len(p) + max(len(positive), len(negative)) > max_tokens:
                raise ValueError("empty, inconsistent or overlong encoded prompt")
            row.update(prompt_ids=p, positive_ids=positive, negative_ids=negative, position=len(p) - 1)
            item[name] = row
        prepared["pairs"].append(item)
    return prepared
