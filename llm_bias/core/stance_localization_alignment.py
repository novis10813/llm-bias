"""Explicit donor→target span maps for cross-prompt residual replacement.

LC1 is the pure alignment slice: it validates renderer-produced
``DecisionPrompt`` records and emits auditable source-array/target-position
maps for one span. It performs no donor pairing, layer selection, capture,
execution, or effect measurement, and it never calls a tokenizer, model, or
GPU. Mapping validation authenticates record structure, not tokenizer
execution; char/text checks are the renderer's responsibility.

Intended consumption by a later executor slice: capture donor rows once at
``donor_capture_positions`` (in that order), gather
``source[:, source_indices, :]`` transiently, and pass ``target_positions``
as ``source_positions`` to the unchanged residual replacement hook. The hook
labels source-array rows with target absolute positions; donor coordinates
never reach the hook, and the two coordinate spaces are validated separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.prompt_input.decision_prompt import DecisionPrompt, DecisionSpan

_ALLOWED_SPANS = ('entity', 'evidence1', 'evidence2', 'instruction')
_ALLOWED_POLICIES = ('exact_tokens', 'relative_rank', 'tail_overlap')
_ALLOWED_SELECTORS = ('full', 'first', 'middle', 'last', 'tail4')
_INT64_MAX = 2 ** 63 - 1

__all__ = ['SpanAlignment', 'build_span_alignment']


def _real_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f'{label} must be a genuine integer, got {type(value).__name__}')
    return value


def _validated_prompt(prompt: Any, label: str) -> None:
    if not isinstance(prompt, DecisionPrompt):
        raise ValueError(f'{label} must be a DecisionPrompt record')
    ids = prompt.inference_token_ids
    if not isinstance(ids, tuple) or not ids:
        raise ValueError(f'{label} prompt must contain a nonempty token tuple')
    if any(type(token) is not int or token < 0 or token > _INT64_MAX for token in ids):
        raise ValueError(
            f'{label} inference token IDs must be genuine nonnegative signed-int64 integers')


def _validated_span(prompt: DecisionPrompt, span: str, label: str) -> DecisionSpan:
    span_record = getattr(prompt, f'{span}_span')
    if not isinstance(span_record, DecisionSpan):
        raise ValueError(f'{label} {span} span is not a DecisionSpan record')
    if span_record.role != span:
        raise ValueError(f'{label} {span} span has role {span_record.role!r}')
    start = _real_int(span_record.token_start, f'{label} {span} token_start')
    end = _real_int(span_record.token_end, f'{label} {span} token_end')
    if not 0 <= start < end <= len(prompt.inference_token_ids):
        raise ValueError(f'{label} {span} span has a malformed half-open token range')
    char_start = _real_int(span_record.char_start, f'{label} {span} char_start')
    char_end = _real_int(span_record.char_end, f'{label} {span} char_end')
    if not 0 <= char_start < char_end:
        raise ValueError(f'{label} {span} span has a malformed character range')
    if not isinstance(span_record.token_sha256, str) or not isinstance(span_record.text_sha256, str):
        raise ValueError(f'{label} {span} span hashes must be strings')
    token_ids = span_record.token_ids
    if not isinstance(token_ids, tuple) or any(
            type(token) is not int or token < 0 or token > _INT64_MAX for token in token_ids):
        raise ValueError(f'{label} {span} span token IDs must be nonnegative signed-int64 integers')
    if token_ids != prompt.inference_token_ids[start:end]:
        raise ValueError(f'{label} {span} span token IDs differ from the actual inference slice')
    if span_record.token_sha256 != sha256_json(list(token_ids)):
        raise ValueError(f'{label} {span} span token hash does not match the stored token IDs')
    return span_record


def _policy_pairs(policy: str, donor_ids: tuple[int, ...],
                  target_ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return (target ordinal, donor ordinal) pairs in target order."""
    donor_count, target_count = len(donor_ids), len(target_ids)
    if policy == 'exact_tokens':
        if donor_ids != target_ids:
            raise ValueError('exact_tokens requires identical span token IDs; no fallback')
        return tuple((j, j) for j in range(target_count))
    if policy == 'relative_rank':
        # Integer transport of relative token position; donor rows may repeat.
        return tuple((j, min(donor_count - 1,
                             (2 * j + 1) * donor_count // (2 * target_count)))
                     for j in range(target_count))
    overlap = min(donor_count, target_count)
    return tuple((target_count - overlap + m, donor_count - overlap + m)
                 for m in range(overlap))


def _selector_indices(selector: str, length: int) -> tuple[int, ...]:
    if selector == 'full':
        return tuple(range(length))
    if selector == 'first':
        return (0,)
    if selector == 'middle':
        return (length // 2,)  # right center for even lengths
    if selector == 'last':
        return (length - 1,)
    return tuple(range(length - min(4, length), length))  # last min(4, length) positions


@dataclass(frozen=True)
class SpanAlignment:
    """Frozen tensor-free map from donor capture rows to target positions.

    ``donor_capture_positions`` are unique, increasing donor absolute
    positions to capture once. ``target_positions`` are the selected target
    absolute positions in order; ``source_indices[i]`` indexes the donor row
    that replaces ``target_positions[i]`` and may repeat. Counts are the full
    original span token counts; the selected count is derived as
    ``len(target_positions)``.
    """

    span: str
    policy: str
    selector: str
    donor_capture_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    source_indices: tuple[int, ...]
    donor_span_token_count: int
    target_span_token_count: int
    mapping_sha256: str

    def __post_init__(self) -> None:
        if self.span not in _ALLOWED_SPANS:
            raise ValueError(f'span must be one of {_ALLOWED_SPANS}')
        if self.policy not in _ALLOWED_POLICIES:
            raise ValueError(f'policy must be one of {_ALLOWED_POLICIES}')
        if self.selector not in _ALLOWED_SELECTORS:
            raise ValueError(f'selector must be one of {_ALLOWED_SELECTORS}')
        self._validate_position_tuple(self.donor_capture_positions, 'donor_capture_positions')
        self._validate_position_tuple(self.target_positions, 'target_positions')
        indices = self.source_indices
        if not isinstance(indices, tuple) or not indices:
            raise ValueError('source_indices must be a nonempty tuple of integers')
        if len(indices) != len(self.target_positions):
            raise ValueError('source_indices length must equal the selected target count')
        capture_count = len(self.donor_capture_positions)
        if any(type(index) is not int or not 0 <= index < capture_count for index in indices):
            raise ValueError('source_indices must index the donor capture rows')
        for label in ('donor_span_token_count', 'target_span_token_count'):
            count = getattr(self, label)
            if type(count) is not int or count <= 0:
                raise ValueError(f'{label} must be a positive genuine integer')
        if type(self.mapping_sha256) is not str:
            raise ValueError('mapping_sha256 must be a string')
        if self.mapping_sha256 != sha256_json(self._export_payload()):
            raise ValueError('mapping_sha256 does not match the exported record')

    @staticmethod
    def _validate_position_tuple(value: Any, label: str) -> None:
        if not isinstance(value, tuple) or not value:
            raise ValueError(f'{label} must be a nonempty tuple of integers')
        if any(type(position) is not int or position < 0 for position in value):
            raise ValueError(f'{label} must contain genuine nonnegative integers')
        if any(a >= b for a, b in zip(value, value[1:])):
            raise ValueError(f'{label} must be unique and strictly increasing')

    def _export_payload(self) -> dict[str, Any]:
        """The exported record excluding the hash field, with fresh lists."""
        return {
            'span': self.span,
            'policy': self.policy,
            'selector': self.selector,
            'donor_capture_positions': list(self.donor_capture_positions),
            'target_positions': list(self.target_positions),
            'source_indices': list(self.source_indices),
            'donor_span_token_count': self.donor_span_token_count,
            'target_span_token_count': self.target_span_token_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Defensive export: fresh lists, counts, and the mapping hash only."""
        record = self._export_payload()
        record['mapping_sha256'] = self.mapping_sha256
        return record


def build_span_alignment(
    donor: DecisionPrompt,
    target: DecisionPrompt,
    *,
    span: str,
    policy: str,
    selector: str = 'full',
) -> SpanAlignment:
    """Map one selected span from a donor prompt to a target prompt.

    ``exact_tokens`` maps ordinal token indices one-to-one and requires the
    two span ID tuples to be identical (shared-instruction/shared-evidence
    mapping; different token IDs reject rather than fall back).
    ``relative_rank`` transports every selected target ordinal ``j`` to donor
    ordinal ``floor((2*j+1)*D/(2*T))`` capped at ``D-1`` with integers only,
    reusing donor rows where needed. ``tail_overlap`` pairs only the last
    ``min(D, T)`` ordinals of both spans, so it replaces a reduced target
    coverage whenever the lengths differ. The selector then chooses from the
    policy's mapped target ordinals. Capture positions are the unique mapped
    donor absolute positions, sorted increasingly.
    """
    for name, value, allowed in (('span', span, _ALLOWED_SPANS),
                                 ('policy', policy, _ALLOWED_POLICIES),
                                 ('selector', selector, _ALLOWED_SELECTORS)):
        if type(value) is not str or value not in allowed:
            raise ValueError(f'{name} must be one of {allowed}')
    for label, prompt in (('donor', donor), ('target', target)):
        _validated_prompt(prompt, label)
        _validated_span(prompt, span, label)
    donor_span = getattr(donor, f'{span}_span')
    target_span = getattr(target, f'{span}_span')
    pairs = _policy_pairs(policy, donor_span.token_ids, target_span.token_ids)
    selected = tuple(pairs[index] for index in _selector_indices(selector, len(pairs)))
    capture_positions = tuple(sorted({donor_span.token_start + donor_ordinal
                                      for _, donor_ordinal in selected}))
    capture_index = {position: index for index, position in enumerate(capture_positions)}
    record = {
        'span': span,
        'policy': policy,
        'selector': selector,
        'donor_capture_positions': capture_positions,
        'target_positions': tuple(target_span.token_start + target_ordinal
                                  for target_ordinal, _ in selected),
        'source_indices': tuple(capture_index[donor_span.token_start + donor_ordinal]
                                for _, donor_ordinal in selected),
        'donor_span_token_count': len(donor_span.token_ids),
        'target_span_token_count': len(target_span.token_ids),
    }
    return SpanAlignment(mapping_sha256=sha256_json(record), **record)
