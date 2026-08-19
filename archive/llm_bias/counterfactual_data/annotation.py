"""LangExtract adapter for grounded entity and earnings-event annotation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_bias.counterfactual_data.pipeline import (
    ANNOTATOR_VERSION,
    CounterfactualDataError,
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    SCHEMA_VERSION,
    _read_jsonl,
    _span,
    _write_json,
    _write_jsonl,
    deterministic_identifier_spans,
    resolve_semantic_outcome,
)

DEFAULT_MODEL = "qwen3.5-mtp"
DEFAULT_BASE_URL = "http://127.0.0.1:11433/v1"
DEFAULT_MAX_CHAR_BUFFER = 1_800
SEMANTIC_ENTITY_CLASSES = frozenset(
    {
        "subsidiary",
        "product",
        "brand",
        "business_segment",
        "person",
        "person_role",
        "counterparty",
        "identifying_location",
    }
)

ENTITY_PROMPT = f"""
Extract only identity-bearing spans that require semantic interpretation from
this SEC 8-K earnings text. Copy every extraction_text exactly from the source.
Use only these classes: {", ".join(sorted(SEMANTIC_ENTITY_CLASSES))}.
For person, include role when stated. Do not extract the registrant, ticker,
dates, amounts, percentages, share counts, security identifiers, generic
financial nouns, or complete event sentences; deterministic rules handle them.
""".strip()

EVENT_PROMPT = """
Extract explicit directional earnings facts. Each event_fact must be an exact,
grounded source span containing both the metric and its direction. Attributes
must contain: metric, direction, comparison_basis, time_scope, and
industry_cue. Use an empty string when an attribute is not stated. Allowed
directions are increase, decrease, improved, worsened, raised, cut, higher,
lower, grew, and declined. Do not infer facts not present in the source.
""".strip()


def _examples() -> tuple[list[Any], list[Any]]:
    from langextract import data

    entity_text = (
        "Subsidiary Beta Labs appointed Chief Executive Jane Doe to lead the "
        "Cloud Systems segment and launch the Nova brand's Orbit product with "
        "supplier Delta Partners in Austin."
    )
    entity_examples = [
        data.ExampleData(
            text=entity_text,
            extractions=[
                data.Extraction(
                    extraction_class="subsidiary",
                    extraction_text="Beta Labs",
                ),
                data.Extraction(
                    extraction_class="person_role",
                    extraction_text="Chief Executive",
                ),
                data.Extraction(
                    extraction_class="person",
                    extraction_text="Jane Doe",
                    attributes={"role": "Chief Executive"},
                ),
                data.Extraction(
                    extraction_class="business_segment",
                    extraction_text="Cloud Systems",
                ),
                data.Extraction(
                    extraction_class="brand", extraction_text="Nova"
                ),
                data.Extraction(
                    extraction_class="product", extraction_text="Orbit"
                ),
                data.Extraction(
                    extraction_class="counterparty", extraction_text="Delta Partners"
                ),
                data.Extraction(
                    extraction_class="identifying_location",
                    extraction_text="Austin",
                ),
            ],
        )
    ]
    event_examples = [
        data.ExampleData(
            text="Acme reported revenue increased 12% year over year in the third quarter.",
            extractions=[
                data.Extraction(
                    extraction_class="event_fact",
                    extraction_text="revenue increased 12% year over year in the third quarter",
                    attributes={
                        "metric": "revenue",
                        "direction": "increase",
                        "comparison_basis": "year over year",
                        "time_scope": "third quarter",
                        "industry_cue": "",
                    },
                )
            ],
        ),
        data.ExampleData(
            text="The company said credit losses decreased from the prior quarter.",
            extractions=[
                data.Extraction(
                    extraction_class="event_fact",
                    extraction_text="credit losses decreased from the prior quarter",
                    attributes={
                        "metric": "credit loss",
                        "direction": "decrease",
                        "comparison_basis": "prior quarter",
                        "time_scope": "",
                        "industry_cue": "banking",
                    },
                )
            ],
        ),
    ]
    return entity_examples, event_examples


def _serialize_extraction(
    extraction: Any,
    source: str,
    text: str,
    allowed_classes: frozenset[str],
) -> dict[str, Any]:
    if extraction.extraction_class not in allowed_classes:
        raise CounterfactualDataError(
            f"unexpected extraction class: {extraction.extraction_class}"
        )
    interval = extraction.char_interval
    if interval is None or interval.start_pos is None or interval.end_pos is None:
        raise CounterfactualDataError(
            f"ungrounded extraction: {extraction.extraction_text!r}"
        )
    alignment_status = (
        extraction.alignment_status.value
        if getattr(extraction.alignment_status, "value", None)
        else str(extraction.alignment_status or "")
    )
    row = {
        "extraction_class": extraction.extraction_class,
        "extraction_text": extraction.extraction_text,
        "char_interval": asdict(interval),
        "alignment_status": alignment_status,
        "attributes": extraction.attributes or {},
        "source": source,
    }
    span = _span(row)
    if span is None:
        raise CounterfactualDataError(
            f"invalid extraction span: {extraction.extraction_text!r}"
        )
    grounded_text = text[span[0] : span[1]]
    if grounded_text != extraction.extraction_text:
        if grounded_text.casefold() == extraction.extraction_text.casefold():
            row["model_extraction_text"] = extraction.extraction_text
            row["extraction_text"] = grounded_text
            row["grounding_canonicalized"] = "case_only"
        elif alignment_status == "match_fuzzy":
            row["model_extraction_text"] = extraction.extraction_text
            row["extraction_text"] = grounded_text
            row["grounding_canonicalized"] = "high_threshold_fuzzy_alignment"
        else:
            raise CounterfactualDataError(
                f"non-exact extraction grounding: {extraction.extraction_text!r}"
            )
    if text[span[0] : span[1]] != row["extraction_text"]:
        raise CounterfactualDataError(
            f"non-exact extraction grounding: {extraction.extraction_text!r}"
        )
    return row


def _extract(
    text: str,
    *,
    prompt: str,
    examples: list[Any],
    extraction_passes: int,
    model: str,
    base_url: str,
    api_key: str,
    max_char_buffer: int,
    allowed_classes: frozenset[str],
) -> list[dict[str, Any]]:
    import langextract as lx
    from langextract.factory import ModelConfig, create_model

    config = ModelConfig(
        model_id=model,
        provider="openai",
        provider_kwargs={
            "api_key": api_key,
            "base_url": base_url,
            "temperature": 0.0,
            "max_workers": 1,
            "max_output_tokens": 2_048,
            "seed": DEFAULT_SEED,
        },
    )
    language_model = create_model(
        config,
        examples=examples,
        use_schema_constraints=True,
    )
    document = lx.extract(
        text,
        prompt_description=prompt,
        examples=examples,
        model=language_model,
        temperature=0.0,
        max_workers=1,
        batch_length=1,
        max_char_buffer=max_char_buffer,
        extraction_passes=extraction_passes,
        use_schema_constraints=False,
        context_window_chars=200,
        resolver_params={
            "suppress_parse_errors": False,
            "enable_fuzzy_alignment": True,
            "fuzzy_alignment_threshold": 0.90,
            "fuzzy_alignment_algorithm": "lcs",
            "fuzzy_alignment_min_density": 0.80,
            "accept_match_lesser": False,
        },
        show_progress=False,
    )
    return [
        _serialize_extraction(extraction, "langextract", text, allowed_classes)
        for extraction in document.extractions or []
    ]


def annotate_events(
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "local",
    max_events: int | None = None,
    max_char_buffer: int = DEFAULT_MAX_CHAR_BUFFER,
    resume: bool = True,
) -> dict[str, Any]:
    """Annotate sampled events, preserving failures as auditable draft rows."""
    root = Path(output_root)
    sample_path = root / "sampled_events.jsonl"
    output_path = root / "draft_annotations.jsonl"
    sampled = list(_read_jsonl(sample_path))
    if max_events is not None:
        sampled = sampled[:max_events]
    existing = list(_read_jsonl(output_path)) if resume and output_path.exists() else []
    incompatible = [
        row["content_id"]
        for row in existing
        if row.get("annotator_version") != ANNOTATOR_VERSION
    ]
    if incompatible:
        raise CounterfactualDataError(
            "draft_annotations.jsonl contains an older annotator version; "
            "archive it and rerun with --no-resume"
        )
    output_by_id = {row["content_id"]: row for row in existing}
    completed = {
        row["content_id"]
        for row in existing
        if row.get("annotation_status") == "complete"
    }
    entity_examples, event_examples = _examples()
    output = list(existing)
    successes = 0
    failures = 0
    for index, event in enumerate(sampled, start=1):
        if event["content_id"] in completed:
            continue
        text = event["analysis_text"]
        try:
            entities = _extract(
                text,
                prompt=ENTITY_PROMPT,
                examples=entity_examples,
                extraction_passes=2,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_char_buffer=max_char_buffer,
                allowed_classes=SEMANTIC_ENTITY_CLASSES,
            )
            entities.extend(deterministic_identifier_spans(text, event["company"]))
            entity_keys = {
                (
                    row["extraction_class"],
                    row["char_interval"]["start_pos"],
                    row["char_interval"]["end_pos"],
                ): row
                for row in entities
            }
            entities = sorted(
                entity_keys.values(),
                key=lambda row: (
                    row["char_interval"]["start_pos"],
                    row["char_interval"]["end_pos"],
                    row["extraction_class"],
                ),
            )
            event_facts = _extract(
                text,
                prompt=EVENT_PROMPT,
                examples=event_examples,
                extraction_passes=1,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_char_buffer=max_char_buffer,
                allowed_classes=frozenset({"event_fact"}),
            )
            event_facts = [
                row for row in event_facts if row["extraction_class"] == "event_fact"
            ]
            outcome = resolve_semantic_outcome(event_facts)
            row = {
                **event,
                "schema_version": SCHEMA_VERSION,
                "annotator_version": ANNOTATOR_VERSION,
                "dataset_status": "draft",
                "annotation_status": "complete",
                "annotation_model": model,
                "annotation_base_url": base_url,
                "entity_extraction_passes": 2,
                "event_extraction_passes": 1,
                "max_char_buffer": max_char_buffer,
                "parser_policy": "strict_json_exact_grounding",
                "entities": entities,
                "event_facts": event_facts,
                "semantic_entity_count": sum(
                    entity["source"] == "langextract" for entity in entities
                ),
                "deterministic_identifier_count": sum(
                    entity["source"] != "langextract" for entity in entities
                ),
                "expected_outcome": outcome,
                "outcome_resolution": (
                    "deterministic_metric_polarity"
                    if outcome
                    else "ambiguous_or_unsupported"
                ),
                "annotation_quality_flags": [
                    flag
                    for flag, present in (
                        ("no_semantic_entities", not any(
                            entity["source"] == "langextract" for entity in entities
                        )),
                        ("no_event_facts", not event_facts),
                        ("ambiguous_or_unsupported_outcome", outcome is None),
                    )
                    if present
                ],
            }
            successes += 1
        except Exception as exc:  # retain the item so failures can be retried/audited
            row = {
                **event,
                "schema_version": SCHEMA_VERSION,
                "annotator_version": ANNOTATOR_VERSION,
                "dataset_status": "draft",
                "annotation_status": "failed",
                "annotation_model": model,
                "annotation_error_type": type(exc).__name__,
                "annotation_error": str(exc),
                "max_char_buffer": max_char_buffer,
                "parser_policy": "strict_json_exact_grounding",
                "entities": deterministic_identifier_spans(text, event["company"]),
                "event_facts": [],
                "expected_outcome": None,
            }
            failures += 1
        output_by_id[event["content_id"]] = row
        output = sorted(output_by_id.values(), key=lambda item: item["content_id"])
        _write_jsonl(output, output_path)
        print(
            f"Annotated {index}/{len(sampled)}: {event['content_id']} "
            f"({row['annotation_status']})"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "annotator_version": ANNOTATOR_VERSION,
        "stage": "annotate",
        "status": "draft",
        "model": model,
        "base_url": base_url,
        "temperature": 0.0,
        "seed": DEFAULT_SEED,
        "max_workers": 1,
        "entity_extraction_passes": 2,
        "event_extraction_passes": 1,
        "max_char_buffer": max_char_buffer,
        "max_output_tokens": 2_048,
        "parser_policy": "strict_json_exact_grounding",
        "sample_scope_count": len(sampled),
        "annotation_count": len(output_by_id),
        "complete_count": sum(
            row.get("annotation_status") == "complete"
            for row in output_by_id.values()
        ),
        "failed_count": sum(
            row.get("annotation_status") == "failed"
            for row in output_by_id.values()
        ),
        "new_success_count": successes,
        "new_failure_count": failures,
        "review_required": True,
    }
    _write_json(manifest, root / "annotation_manifest.json")
    return manifest
