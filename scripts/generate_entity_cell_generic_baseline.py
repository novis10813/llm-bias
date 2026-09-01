#!/usr/bin/env python3
"""Generate and validate the adapted entity-cell generic baseline.

The endpoint response is used only as an in-memory candidate source. This operator
writes the accepted JSONL and compact provenance; it never writes response bodies,
rejected text, or model reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

ADAPTED_IDENTITY = "adapted:qwen3.5-9b-generic-cloze-v1"
ENDPOINT = "http://127.0.0.1:11432/v1/chat/completions"
MODEL_ID = "qwen3.5-9b"
VALIDATION_VERSION = "generic-cloze-v1"
EXPECTED_COUNT = 399
CATEGORY_MINIMUM = 10
CATEGORY_MAXIMUM = 35
CATEGORIES = (
    "purpose", "process", "cause", "change", "comparison", "classification",
    "measurement", "sequence", "location", "material", "function", "pattern",
    "learning", "perception", "communication", "planning", "time", "environment",
    "abstract",
)

SYSTEM_PROMPT = """You create a corpus of generic English cloze-style prompt prefixes for a language-model activation baseline.
Return JSON only, with this exact top-level shape: {\"candidates\":[{\"category\":\"...\",\"prompt\":\"...\"}]}.
Return 50 objects. Use each category at least twice and use every listed category:
purpose, process, cause, change, comparison, classification, measurement, sequence, location, material, function, pattern, learning, perception, communication, planning, time, environment, abstract.
Each prompt must be one short, natural English prefix that ends before the answer. It must be generic, not about finance, companies, products, places, people, or the assistant. Vary grammar, vocabulary, and sentence openings. Do not complete a sentence. Prefer endings that invite a missing word or phrase, such as 'is to', 'depends on', 'can be', 'often leads to', or 'the next step is'. Do not use quotation marks, brackets, question marks, headings, labels, bullets, markdown, numbers, URLs, or code. Do not add explanations."""

USER_PROMPT = """Generate 50 varied generic cloze prefixes now. A prefix such as 'The primary purpose of this tool is to' shows the role, but do not copy it. Keep each item between 4 and 16 words and end with a plausible continuation point such as a verb, preposition, article, or linking word. Do not include the category name in the prompt unless it naturally belongs there. Return only the requested JSON."""

_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_URL_RE = re.compile(r"(?:https?://|www\.)|\b\S+@\S+\b", re.IGNORECASE)
_DIALOGUE_RE = re.compile(r"^(?:user|assistant|system|human|bot)\s*:", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
_FORBIDDEN_TERMS = {
    "company", "companies", "corporation", "corporate", "ticker", "stock", "stocks",
    "share", "shares", "invest", "investment", "investments", "finance", "financial",
    "market", "markets", "trading", "trade", "portfolio", "revenue", "profit", "earnings",
    "fund", "funds", "dividend", "bond", "bank", "banking", "quarter", "sec", "nasdaq",
    "nyse", "business", "issuer", "analyst", "equity", "asset", "assets",
    "assistant", "chatbot", "prompt", "model", "user", "system", "respond", "answer",
    "write", "generate", "list", "explain", "describe", "provide", "tell", "complete",
    "please", "instruction", "instructions",
}
_ENDING_WORDS = {
    "a", "an", "the", "to", "of", "for", "with", "by", "from", "in", "on", "at", "as",
    "into", "through", "before", "after", "during", "because", "when", "while", "that", "which",
    "is", "are", "was", "were", "be", "been", "being", "can", "could", "may", "might", "will",
    "would", "should", "often", "usually", "typically", "helps", "allows", "depends", "serves",
    "supports", "requires", "uses", "follows", "reveals", "shows", "means", "makes", "keeps",
    "becomes", "remains", "seems", "appears", "contains", "includes", "involves", "creates",
    "designed", "based", "shaped", "driven", "guided", "known", "found", "linked", "related",
    "important", "likely", "ready", "able", "common", "useful", "best", "most", "less", "more",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", prompt).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_count(prompt: str) -> int:
    return len(_WORD_RE.findall(prompt))


def _lexical_tokens(prompt: str) -> list[str]:
    return [word.casefold() for word in _WORD_RE.findall(prompt)]


def _has_proper_noun(prompt: str) -> bool:
    words = prompt.split()
    # The first token may be capitalized because it starts the prefix. Later
    # title-case tokens are a deterministic screen for named entities.
    return any(token[:1].isupper() and token[1:].islower() for token in words[1:])


def validate_candidate(
    candidate: Mapping[str, Any],
    *,
    forbidden_identities: Iterable[str] = (),
    seen_normalized: set[str] | None = None,
    category_counts: Mapping[str, int] | None = None,
    prefix_counts: Mapping[str, int] | None = None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Return (accepted, rejection reason, normalized compact candidate)."""
    if not isinstance(candidate, Mapping):
        return False, "candidate_not_object", None
    category = candidate.get("category")
    prompt = candidate.get("prompt")
    if not isinstance(category, str) or category not in CATEGORIES:
        return False, "unknown_category", None
    if not isinstance(prompt, str):
        return False, "prompt_not_string", None
    prompt = prompt.strip()
    if not prompt:
        return False, "empty_prompt", None
    if "\n" in prompt or "\r" in prompt:
        return False, "multi_paragraph_or_line", None
    if not prompt.isascii():
        return False, "non_english_or_non_ascii", None
    if _URL_RE.search(prompt):
        return False, "url_or_email", None
    if _PLACEHOLDER_RE.search(prompt):
        return False, "placeholder_or_brackets", None
    if "```" in prompt or "\t" in prompt:
        return False, "code_or_noisy_fragment", None
    if "?" in prompt or any(prompt.endswith(mark) for mark in ".!?;:"):
        return False, "question_or_completed_punctuation", None
    if _DIALOGUE_RE.search(prompt):
        return False, "dialogue_role", None
    if _has_proper_noun(prompt):
        return False, "named_entity_screen", None
    words = _lexical_tokens(prompt)
    if not 4 <= len(words) <= 16:
        return False, "length_bounds", None
    lower = set(words)
    if lower & _FORBIDDEN_TERMS:
        return False, "forbidden_domain_or_instruction_term", None
    normalized = normalize_prompt(prompt)
    identity_text = normalize_prompt(prompt.replace("[", "").replace("]", ""))
    for identity in forbidden_identities:
        identity_norm = normalize_prompt(identity)
        if not identity_norm:
            continue
        # Match whole ticker/name phrases. A bare substring would reject nearly
        # every prompt for one-letter tickers or common words in company names.
        pattern = rf"(?<!\w){re.escape(identity_norm)}(?!\w)"
        if re.search(pattern, normalized) or re.search(pattern, identity_text):
            return False, "baseline_identity", None
    if seen_normalized is not None and normalized in seen_normalized:
        return False, "normalized_duplicate", None
    if words[-1] not in _ENDING_WORDS:
        return False, "complete_declarative_sentence", None
    prefix = " ".join(words[:4])
    if prefix_counts is not None and prefix_counts.get(prefix, 0) >= 3:
        return False, "prefix_diversity_limit", None
    if category_counts is not None and category_counts.get(category, 0) >= CATEGORY_MAXIMUM:
        return False, "category_quota_full", None
    return True, None, {"category": category, "prompt": prompt, "normalized": normalized, "word_count": len(words), "prefix4": prefix}


def parse_response(content: str) -> list[dict[str, Any]]:
    """Parse only the requested JSON object; never persist the content."""
    value = json.loads(content)
    if not isinstance(value, Mapping) or not isinstance(value.get("candidates"), list):
        raise ValueError("response must be an object with a candidates list")
    return [dict(item) for item in value["candidates"] if isinstance(item, Mapping)]


def request_batch(
    *, url: str, model: str, timeout: float, temperature: float, top_p: float, seed: int, max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}],
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    content = body["choices"][0]["message"]["content"]
    return parse_response(content), {"response_id": body.get("id"), "model": body.get("model"), "usage": body.get("usage", {})}


def collect_candidates(
    *, baseline_csv: Path, url: str, model: str, batch_count: int, timeout: float, temperature: float,
    top_p: float, seed: int, max_tokens: int, retry_count: int, backoff: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import csv

    identities: list[str] = []
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identities.extend(value for value in (row.get("ticker"), row.get("name")) if value)
    identities = sorted(set(identities), key=lambda value: (-len(value), value.casefold()))
    reasons: Counter[str] = Counter()
    categories_seen: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()
    prefix_counts: Counter[str] = Counter()
    request_count = 0
    response_count = 0
    parse_failures = 0
    response_meta: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        for attempt in range(retry_count + 1):
            request_count += 1
            try:
                candidates, meta = request_batch(
                    url=url, model=model, timeout=timeout, temperature=temperature, top_p=top_p,
                    seed=seed + batch_index, max_tokens=max_tokens,
                )
                response_count += len(candidates)
                response_meta.append({"batch": batch_index, "attempt": attempt, **meta})
                break
            except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
                if attempt >= retry_count:
                    reasons["request_or_parse_failure"] += 1
                    parse_failures += 1
                    candidates = []
                    break
                time.sleep(backoff * (2**attempt))
        for candidate in candidates:
            ok, reason, normalized = validate_candidate(
                candidate, forbidden_identities=identities, seen_normalized=seen_normalized,
                category_counts=categories_seen, prefix_counts=prefix_counts,
            )
            if not ok:
                reasons[reason or "rejected"] += 1
                continue
            assert normalized is not None
            accepted.append(normalized)
            seen_normalized.add(normalized["normalized"])
            categories_seen[normalized["category"]] += 1
            prefix_counts[normalized["prefix4"]] += 1
    if len(accepted) < EXPECTED_COUNT or any(categories_seen[category] < CATEGORY_MINIMUM for category in CATEGORIES):
        raise RuntimeError(
            f"generation yielded {len(accepted)} accepted candidates; category counts={dict(categories_seen)}; "
            f"increase --batches or adjust endpoint availability"
        )
    # Reserve the same minimum number from every category, then fill the
    # remaining slots by a stable digest. This separates validation from the
    # final cardinality choice and makes the selected set reproducible.
    accepted_by_category = {category: sorted((row for row in accepted if row["category"] == category), key=lambda row: sha256_bytes(row["normalized"].encode("utf-8"))) for category in CATEGORIES}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in CATEGORIES:
        for row in accepted_by_category[category][:CATEGORY_MINIMUM]:
            selected.append(row)
            selected_ids.add(row["normalized"])
    remaining = sorted((row for row in accepted if row["normalized"] not in selected_ids), key=lambda row: sha256_bytes(f'{row["category"]}:{row["normalized"]}'.encode("utf-8")))
    selected.extend(remaining[: EXPECTED_COUNT - len(selected)])
    selected.sort(key=lambda item: (CATEGORIES.index(item["category"]), item["normalized"]))
    selected_categories = Counter(row["category"] for row in selected)
    stats = {
        "candidate_count": response_count,
        "accepted_count": len(selected),
        "validated_candidate_count": len(accepted),
        "rejected_count": sum(reasons.values()),
        "rejection_reasons": dict(sorted(reasons.items())),
        "request_count": request_count,
        "parse_failure_count": parse_failures,
        "category_counts": dict(sorted(selected_categories.items())),
        "candidate_category_counts": dict(sorted(categories_seen.items())),
        "response_metadata": response_meta,
    }
    return selected, stats


def write_dataset(rows: list[dict[str, Any]], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            prompt = row["prompt"]
            record = {
                "schema_version": 1,
                "artifact_type": "entity_cell_generic_baseline_prompt",
                "baseline_identity": ADAPTED_IDENTITY,
                "prompt_id": f"generic_cloze_{index:03d}",
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "prompt": prompt,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256_bytes(output.read_bytes())


def build_provenance(*, output: Path, provenance_path: Path, dataset_sha256: str, rows: list[dict[str, Any]], stats: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    words = [int(row["word_count"]) for row in rows]
    prefixes = [row["prefix4"] for row in rows]
    script_sha256 = sha256_bytes(Path(__file__).read_bytes())
    return {
        "schema_version": 1,
        "artifact_type": "entity_cell_generic_baseline_provenance",
        "adaptation_identity": ADAPTED_IDENTITY,
        "dataset_path": str(output),
        "provenance_path": str(provenance_path),
        "dataset_sha256": dataset_sha256,
        "record_count": len(rows),
        "endpoint": {"base_url": args.base_url, "chat_completions_url": args.url, "model_id": args.model},
        "generation": {
            "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "user_prompt_sha256": sha256_bytes(USER_PROMPT.encode("utf-8")),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "batch_size_requested": 50,
            "batch_count": args.batches,
            "retry_count": args.retries,
            "backoff_seconds": args.backoff,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "script": {"path": "scripts/generate_entity_cell_generic_baseline.py", "sha256": script_sha256, "validation_version": VALIDATION_VERSION},
        "created_at": datetime.now(UTC).isoformat(),
        "source_baseline_csv": {"path": str(args.baseline_csv), "sha256": sha256_bytes(args.baseline_csv.read_bytes())},
        "counts": dict(stats),
        "diversity": {
            "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
            "normalized_duplicate_check": {"unique": len({row["normalized"] for row in rows}), "records": len(rows), "duplicates": 0},
            "word_count": {"min": min(words), "max": max(words), "mean": round(sum(words) / len(words), 4), "counts": dict(sorted(Counter(words).items()))},
            "unique_prefix4": {"unique": len(set(prefixes)), "records": len(prefixes), "max_repeated": max(Counter(prefixes).values())},
        },
        "validation_rules": {
            "version": VALIDATION_VERSION,
            "length_words": [4, 16],
            "category_minimum": CATEGORY_MINIMUM,
            "category_maximum": CATEGORY_MAXIMUM,
            "max_same_prefix4": 3,
            "forbidden_identity_matching": "case-insensitive substring matching for ticker and full company name, including bracket-stripped prompt text",
            "raw_rejected_transcripts_written": False,
            "reasoning_written": False,
        },
        "raw_runtime_payloads": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=Path("data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl"))
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11432/v1")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not args.baseline_csv.is_file():
        parser.error(f"baseline CSV not found: {args.baseline_csv}")
    if args.batches <= 0 or args.retries < 0:
        parser.error("batches must be positive and retries cannot be negative")
    args.url = args.base_url.rstrip("/") + "/chat/completions"
    rows, stats = collect_candidates(
        baseline_csv=args.baseline_csv, url=args.url, model=args.model, batch_count=args.batches,
        timeout=args.timeout, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        max_tokens=args.max_tokens, retry_count=args.retries, backoff=args.backoff,
    )
    dataset_sha256 = write_dataset(rows, args.output)
    provenance_path = args.provenance or args.output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(build_provenance(output=args.output, provenance_path=provenance_path, dataset_sha256=dataset_sha256, rows=rows, stats=stats, args=args), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "provenance": str(provenance_path), "dataset_sha256": dataset_sha256, "counts": stats["category_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
