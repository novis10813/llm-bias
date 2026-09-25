"""Decision-prompt family for confirmation-v1: five evidence conditions, anonymous identities, spans.

``balanced`` is byte-identical to ``entity_to_dial.heldout_transfer._render_frozen_prompt(order=0,
reverse=False)``; ``pos``/``neg`` are byte-identical to ``balanced_evidence_gap.template.build_prompt_v2(...,
reverse=False)``; ``zero`` keeps both evidence markers with an empty body; ``mixed2`` lists P1 then N1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from llm_bias.balanced_evidence_gap.template import EVIDENCE_N1, EVIDENCE_N2, EVIDENCE_P1, EVIDENCE_P2
from llm_bias.core.prompt_input.encoding import format_prompt, input_ids, token_span
from llm_bias.entity_to_dial.spans import entity_char_span, instruction_char_span
from llm_bias.entity_to_dial.template import EVIDENCE_MARKER, NAME_LINE_PREFIX, TICKER_LINE_PREFIX

from .protocol import DECISION_PREFIX, sha256_bytes, template_render_kwargs

CONDITION_EVIDENCE: dict[str, tuple[str, ...]] = {
    "balanced": (EVIDENCE_P1, EVIDENCE_P2, EVIDENCE_N1, EVIDENCE_N2),
    "pos": (EVIDENCE_P1, EVIDENCE_P2),
    "neg": (EVIDENCE_N1, EVIDENCE_N2),
    "zero": (),
    "mixed2": (EVIDENCE_P1, EVIDENCE_N1),
}
CONDITIONS: tuple[str, ...] = tuple(CONDITION_EVIDENCE)

# Pre-registered identity-stripped placeholders: one canonical pair plus nine that vary length and
# format. The freeze script rejects any ticker or name that collides with a real constituent.
ANON_IDENTITIES: tuple[tuple[str, str], ...] = (
    ("TICKER", "Company X"),
    ("QZX", "Quorvex Industries"),
    ("NMRL", "Nemoral Corp"),
    ("ZZT", "Company Z"),
    ("VXLO", "Vexlo Holdings Inc."),
    ("KTRB", "Keterab Systems"),
    ("XQW", "XQW Group"),
    ("PLNX", "Plenaxis Technologies Corporation"),
    ("RDVM", "Redvima Ltd"),
    ("STOCK", "Firm A"),
)

SPAN_NAMES: tuple[str, ...] = ("entity", "evidence", "instruction", "final", "answer_prefix", "steer_suffix")


def render_decision_prompt(ticker: str, name: str, condition: str) -> str:
    if condition not in CONDITION_EVIDENCE:
        raise ValueError(f"unknown evidence condition {condition!r}")
    evidence = "\n".join(f"- {item}" for item in CONDITION_EVIDENCE[condition])
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"{TICKER_LINE_PREFIX}{ticker}]\n\n{NAME_LINE_PREFIX}{name}]\n\n"
        f"{EVIDENCE_MARKER}\n\n{evidence}\n\n"
        "—\n\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        '"decision": "buy" or "sell"\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


def prompt_family_sha256(companies: Mapping[str, Mapping[str, str]], condition: str) -> str:
    """Same serialization as V2 ``prompt_family_sha256`` (balanced reproduces it exactly)."""
    payload = json.dumps([(t, render_decision_prompt(t, companies[t]["name"], condition)) for t in sorted(companies)],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


@dataclass(frozen=True)
class FormattedPrompt:
    """One rendered decision prompt: generation ids, fixed-prefix scoring ids, and token spans."""

    key: str
    prompt: str
    formatted: str
    ids: tuple[int, ...]
    score_ids: tuple[int, ...]
    spans: Mapping[str, tuple[int, int]]
    buy_id: int
    sell_id: int

    @property
    def ids_sha256(self) -> str:
        return sha256_bytes(json.dumps(list(self.ids)).encode())

    def provenance(self) -> dict[str, Any]:
        return {"prompt_sha256": sha256_bytes(self.prompt.encode("utf-8")),
                "formatted_sha256": sha256_bytes(self.formatted.encode("utf-8")),
                "ids_sha256": self.ids_sha256, "n_ids": len(self.ids),
                "spans": {k: list(v) for k, v in self.spans.items()}}


def _char_to_tokens(tokenizer: Any, formatted: str, begin: int, start: int, end: int, what: str) -> tuple[int, int]:
    span = token_span(tokenizer, formatted, begin + start, begin + end, add_special_tokens=True)
    if span is None or span[1] <= span[0]:
        raise ValueError(f"could not map {what} span to tokens")
    return span


def format_decision_prompt(tokenizer: Any, prompt: str, *, suffix_tokens: int, key: str = "",
                           answer_token_ids: Any = None) -> FormattedPrompt:
    """Render with the pinned chat template and resolve all six spans (fail-closed)."""
    from llm_bias.entity_to_dial.dial_probe import answer_token_ids as default_answer_ids

    formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False,
                              chat_template_kwargs=template_render_kwargs(tokenizer))
    begin = formatted.find(prompt)
    if begin < 0:
        raise ValueError(f"chat template omitted the prompt body for {key}")
    ids = input_ids(tokenizer, formatted)
    score_ids = input_ids(tokenizer, formatted + DECISION_PREFIX)
    shared = 0
    for left, right in zip(ids, score_ids):
        if left != right:
            break
        shared += 1
    # at most the final prompt token may merge with the decision prefix
    if len(score_ids) <= len(ids) or shared < len(ids) - 1:
        raise ValueError(f"decision prefix changed the prompt tokenization for {key}")
    instruction = _char_to_tokens(tokenizer, formatted, begin, *instruction_char_span(prompt), "instruction")
    if instruction[1] - instruction[0] < suffix_tokens or instruction[1] > len(ids):
        raise ValueError(f"instruction span shorter than K={suffix_tokens} for {key}")
    if score_ids[instruction[1] - suffix_tokens:instruction[1]] != ids[instruction[1] - suffix_tokens:instruction[1]]:
        raise ValueError(f"scoring prefix changed steer-suffix tokenization for {key}")
    entity = _char_to_tokens(tokenizer, formatted, begin, *entity_char_span(prompt), "entity")
    marker = prompt.find(EVIDENCE_MARKER)
    ev_start = prompt.find("\n\n", marker) + 2
    ev_end = prompt.find("\n\n—\n\n", ev_start)
    # zero evidence has no body: an explicit empty span, never a token_span call on an empty range
    evidence = ((instruction[0], instruction[0]) if ev_end == ev_start
                else _char_to_tokens(tokenizer, formatted, begin, ev_start, ev_end, "evidence"))
    spans = {"entity": entity, "evidence": evidence, "instruction": instruction,
             "final": (len(ids) - 1, len(ids)), "answer_prefix": (shared, len(score_ids)),
             "steer_suffix": (instruction[1] - suffix_tokens, instruction[1])}
    buy_id, sell_id = (answer_token_ids or default_answer_ids)(tokenizer, formatted + DECISION_PREFIX)
    return FormattedPrompt(key, prompt, formatted, tuple(ids), tuple(score_ids), spans, buy_id, sell_id)


def common_instruction_suffix(tokenizer: Any, prompts: Sequence[str]) -> tuple[int, tuple[int, ...]]:
    """Length and ids of the longest common trailing instruction-token run across all prompts."""
    common: list[int] | None = None
    for prompt in prompts:
        formatted = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False,
                                  chat_template_kwargs=template_render_kwargs(tokenizer))
        begin = formatted.find(prompt)
        if begin < 0:
            raise ValueError("chat template omitted the prompt body")
        span = _char_to_tokens(tokenizer, formatted, begin, *instruction_char_span(prompt), "instruction")
        suffix = input_ids(tokenizer, formatted)[span[0]:span[1]]
        if common is None:
            common = suffix
            continue
        n = 0
        for left, right in zip(reversed(common), reversed(suffix)):
            if left != right:
                break
            n += 1
        common = common[-n:] if n else []
        if len(common) < 16:
            raise ValueError("no usable common instruction suffix")
    if not common:
        raise ValueError("empty instruction suffix")
    return len(common), tuple(common)


def anon_collisions(population: Mapping[str, Mapping[str, str]], extra_tickers: Sequence[str] = (),
                    extra_names: Sequence[str] = ()) -> list[str]:
    """Placeholder tickers/names that equal any real ticker or company name (case-insensitive)."""
    tickers = {t.upper() for t in population} | {t.upper() for t in extra_tickers}
    names = {c["name"].lower() for c in population.values()} | {n.lower() for n in extra_names}
    hits = []
    for ticker, name in ANON_IDENTITIES:
        if ticker.upper() in tickers:
            hits.append(f"ticker:{ticker}")
        if name.lower() in names:
            hits.append(f"name:{name}")
    return hits
