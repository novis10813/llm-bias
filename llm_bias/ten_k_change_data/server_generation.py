"""Structured 10-K answer generation through an OpenAI-compatible server."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from llm_bias.prompt_analysis.input_data import TEN_K_ITEM_NAMES

PROMPT_TEMPLATE = (
    "In year {year}, what is the {item_name} of the company with CIK code {cik}? "
    "Return only a JSON object with one string field named answer. "
    "Do not include explanation or additional keys."
)
SCHEMA_NAME = "ten_k_answer"
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("year", "cik", "item"):
            raise ValueError("ten-K input must have exact columns: year,cik,item")
        rows = []
        for row_index, row in enumerate(reader):
            year = row["year"].strip()
            cik = row["cik"].strip()
            item = row["item"].strip()
            if not year.isdigit() or len(year) != 4 or not cik or "=" not in item:
                raise ValueError(f"invalid ten-K row {row_index}")
            field, value = (part.strip() for part in item.split("=", 1))
            if field not in TEN_K_ITEM_NAMES:
                raise ValueError(f"invalid ten-K item field {field!r} at row {row_index}")
            rows.append(
                {
                    "row_index": str(row_index),
                    "year": year,
                    "cik": cik,
                    "item": item,
                    "item_field": field,
                    "item_value": value,
                    "item_name": TEN_K_ITEM_NAMES[field],
                }
            )
    return rows


def _request_answer(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
) -> tuple[str, str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": ANSWER_SCHEMA,
            },
        },
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"structured generation request failed: {error}") from error
    try:
        choice = result["choices"][0]
        raw_text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "unknown")
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"invalid chat completion response: {result!r}") from error
    try:
        parsed = json.loads(raw_text)
        answer = parsed["answer"]
        if set(parsed) != {"answer"} or not isinstance(answer, str):
            raise ValueError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"server returned invalid structured answer: {raw_text!r}") from error
    return raw_text, answer, {"finish_reason": finish_reason, "usage": result.get("usage")}


def generate_structured_answers(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    base_url: str = "http://127.0.0.1:11433/v1",
    model: str = "qwen3.5-9b-mtp",
    max_tokens: int = 64,
    timeout: float = 120.0,
) -> Path:
    """Generate one schema-constrained JSON answer for every ten-K row."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = _read_rows(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "structured_answers.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = PROMPT_TEMPLATE.format(**row)
            raw_text, answer, response = _request_answer(
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            record = {
                "schema_version": 1,
                "artifact_type": "ten_k_structured_answers",
                **row,
                "prompt": prompt,
                "generated_text": raw_text,
                "parsed_answer": answer,
                "parse_status": "valid",
                **response,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    metadata = {
        "schema_version": 1,
        "artifact_type": "ten_k_structured_answers_metadata",
        "input": str(source),
        "input_sha256": source_sha256,
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "base_url": base_url,
        "model": model,
        "records_written": len(rows),
        "max_tokens": max_tokens,
        "prompt_template": PROMPT_TEMPLATE,
        "response_schema_name": SCHEMA_NAME,
        "response_schema": ANSWER_SCHEMA,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
