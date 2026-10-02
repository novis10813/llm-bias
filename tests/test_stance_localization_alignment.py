"""LC1: explicit donor→target span alignment maps (pure records, no GPU).

Red phase: this file fails while ``llm_bias.core.stance_localization_alignment``
is absent (import error); the spec example would also fail any wrong ordinal
formula (a naive ``j*D//T`` yields (0,0,1,1,2), not (0,0,1,2,2)).

Green phase: every policy/selector, singleton and excessive-length ratios,
shifted exact tokens, a bounded exhaustive D/T panel with independently
computed expectations, capture minimality after selection, constructor
malformed inputs and mutation defense, and one torch-only integration that
captures donor rows with the existing fake-model capture hook, gathers them
transiently, and drives the unchanged replacement hook on a fresh target
tracker.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
import torch

from llm_bias.core.artifact_paths import sha256_bytes, sha256_json
from llm_bias.core.inference.stance_interventions import (
    GenerationPositionTracker,
    scoped_residual_intervention,
)
from llm_bias.core.inference.stance_transient_capture import capture_prompt_residual
from llm_bias.core.prompt_input.decision_prompt import DecisionPrompt, DecisionSpan
from llm_bias.core.stance_localization_alignment import (
    SpanAlignment,
    build_span_alignment,
)


# ---------------------------------------------------------------------------
# Synthetic unit fixtures: whole-prompt ID tuples with verified span slices.
# These are direct record fixtures, not a research cohort.
# ---------------------------------------------------------------------------

def _span(role: str, start: int, end: int, ids: tuple[int, ...]) -> DecisionSpan:
    token_ids = tuple(ids[start:end])
    return DecisionSpan(
        role=role, char_start=start, char_end=end,
        token_start=start, token_end=end, token_ids=token_ids,
        token_sha256=sha256_json(list(token_ids)),
        text_sha256=sha256_bytes(f'fake-{role}'.encode('utf-8')),
    )


# wrapper_policy is a required constructor argument (the class-level descriptor
# default raises AttributeError on class access, so the dataclass sees no
# default). A fixed JSON provenance dict is enough for unit fixtures.
WRAPPER_POLICY = {
    "use_chat_template": False, "add_special_tokens": False,
    "system_message": None, "enable_thinking": False,
    "chat_template_kwargs": {}, "tokenizer_chat_template": None,
}


def make_prompt(
    length: int,
    entity=(10, 13), evidence1=(2, 5), evidence2=(5, 8), instruction=(12, 14),
    ids: tuple[int, ...] | None = None,
) -> DecisionPrompt:
    token_ids = tuple(ids) if ids is not None else tuple(5000 + i for i in range(length))
    assert len(token_ids) == length
    spans = {role: _span(role, start, end, token_ids)
             for role, (start, end) in {'entity': entity, 'evidence1': evidence1,
                                        'evidence2': evidence2,
                                        'instruction': instruction}.items()}
    return DecisionPrompt(
        raw_text=f'raw-{length}', rendered_text=f'rendered-{length}',
        inference_token_ids=token_ids,
        entity_span=spans['entity'], evidence1_span=spans['evidence1'],
        evidence2_span=spans['evidence2'], instruction_span=spans['instruction'],
        prompt_sha256=sha256_json({'fixture': 'prompt', 'length': length}),
        template_sha256=sha256_json({'template': 'fake-template'}),
        schema_sha256=sha256_json({'schema': 'fake-schema'}),
        wrapper_policy=WRAPPER_POLICY,
    )


def _replace_span(prompt: DecisionPrompt, field: str, **changes) -> DecisionPrompt:
    span = dataclasses.replace(getattr(prompt, field), **changes)
    return dataclasses.replace(prompt, **{field: span})


# Spec example topology: donor entity 10..12 (D=3), target entity 20..24 (T=5).
DONOR = make_prompt(15)
TARGET = make_prompt(26, entity=(20, 25))


def _pair_with_lengths(donor_count: int, target_count: int):
    donor = make_prompt(4 + donor_count, entity=(4, 4 + donor_count), evidence1=(0, 1),
                        evidence2=(1, 2), instruction=(2, 3))
    target = make_prompt(9 + target_count, entity=(9, 9 + target_count), evidence1=(0, 1),
                         evidence2=(1, 2), instruction=(2, 3))
    return donor, target


def _shared_instruction_prompts():
    shared = (9, 90, 900, 9000)
    donor_ids = tuple(6000 + i for i in range(100)) + shared + tuple(6104 + i for i in range(6))
    target_ids = tuple(7000 + i for i in range(105)) + shared + tuple(7109 + i for i in range(3))
    return make_prompt(110, instruction=(100, 104), ids=donor_ids), \
        make_prompt(112, instruction=(105, 109), ids=target_ids)


def _equal_entity_prompts():
    shared = (1, 2, 3, 4)
    donor_ids = tuple(5000 + i for i in range(10)) + shared
    target_ids = tuple(6000 + i for i in range(20)) + shared + tuple(6024 + i for i in range(6))
    return make_prompt(14, entity=(10, 14), ids=donor_ids), \
        make_prompt(30, entity=(20, 24), ids=target_ids)


# ---------------------------------------------------------------------------
# Spec example maps
# ---------------------------------------------------------------------------

def test_spec_example_relative_rank_full():
    record = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank')
    assert (record.span, record.policy, record.selector) == ('entity', 'relative_rank', 'full')
    assert record.donor_capture_positions == (10, 11, 12)
    assert record.target_positions == (20, 21, 22, 23, 24)
    assert record.source_indices == (0, 0, 1, 2, 2)
    assert record.donor_span_token_count == 3
    assert record.target_span_token_count == 5


def test_spec_example_tail_overlap_full():
    record = build_span_alignment(DONOR, TARGET, span='entity', policy='tail_overlap')
    assert record.donor_capture_positions == (10, 11, 12)
    assert record.target_positions == (22, 23, 24)
    assert record.source_indices == (0, 1, 2)


def test_tail_overlap_reports_reduced_target_coverage():
    record = build_span_alignment(DONOR, TARGET, span='entity', policy='tail_overlap')
    # D=3 < T=5: only the overlap tail is replaced, and the reduced coverage
    # is visible as len(target_positions) versus the full-span count.
    assert record.target_span_token_count == 5
    assert len(record.target_positions) == 3


def test_spec_example_shared_instruction_exact_tokens():
    donor, target = _shared_instruction_prompts()
    record = build_span_alignment(donor, target, span='instruction', policy='exact_tokens')
    # Identical IDs at different absolute positions map ordinally.
    assert record.donor_capture_positions == (100, 101, 102, 103)
    assert record.target_positions == (105, 106, 107, 108)
    assert record.source_indices == (0, 1, 2, 3)


def test_exact_tokens_rejects_different_ids_without_fallback():
    t_ids = tuple(5000 + i for i in range(15))
    t_ids = t_ids[:11] + (5999,) + t_ids[12:]
    different_ids = make_prompt(15, entity=(10, 13), ids=t_ids)
    with pytest.raises(ValueError):
        build_span_alignment(DONOR, different_ids, span='entity', policy='exact_tokens')
    with pytest.raises(ValueError):
        build_span_alignment(DONOR, make_prompt(16, entity=(20, 24)),
                             span='entity', policy='exact_tokens')


def test_relative_rank_reuses_donor_rows():
    record = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank')
    counts = [record.source_indices.count(index)
              for index in range(len(record.donor_capture_positions))]
    assert counts == [2, 1, 2]


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('selector,expected', [
    ('full', ((20, 21, 22, 23, 24), (10, 11, 12), (0, 0, 1, 2, 2))),
    ('first', ((20,), (10,), (0,))),
    ('middle', ((22,), (11,), (0,))),
    ('last', ((24,), (12,), (0,))),
    ('tail4', ((21, 22, 23, 24), (10, 11, 12), (0, 1, 2, 2))),
])
def test_selectors_on_relative_rank(selector, expected):
    target_positions, capture_positions, indices = expected
    record = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank',
                                  selector=selector)
    assert record.target_positions == target_positions
    assert record.donor_capture_positions == capture_positions
    assert record.source_indices == indices
    # Full original span counts survive selection.
    assert record.donor_span_token_count == 3
    assert record.target_span_token_count == 5


@pytest.mark.parametrize('selector,index', [('first', 0), ('middle', 2), ('last', 3)])
def test_selector_on_even_length_exact_tokens(selector, index):
    donor, target = _equal_entity_prompts()  # D=T=4
    record = build_span_alignment(donor, target, span='entity', policy='exact_tokens',
                                  selector=selector)
    assert record.target_positions == (20 + index,)
    assert record.donor_capture_positions == (10 + index,)
    assert record.source_indices == (0,)


def test_capture_is_unique_sorted_mapped_donors_after_selector():
    record = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank',
                                  selector='last')
    # Only the single mapped donor row is captured, not the whole span.
    assert record.donor_capture_positions == (12,)
    assert record.target_positions == (24,)
    assert record.source_indices == (0,)


def test_tail4_on_short_tail_overlap():
    # tail_overlap with min(D, T) = 2 gives a two-element available list;
    # tail4 must take both, never negative-index wrap.
    donor, target = _pair_with_lengths(2, 5)
    record = build_span_alignment(donor, target, span='entity', policy='tail_overlap',
                                  selector='tail4')
    assert record.target_positions == (12, 13)
    assert record.donor_capture_positions == (4, 5)
    assert record.source_indices == (0, 1)


def test_tail4_on_single_tail_overlap():
    donor, target = _pair_with_lengths(1, 5)
    record = build_span_alignment(donor, target, span='entity', policy='tail_overlap',
                                  selector='tail4')
    assert record.target_positions == (13,)
    assert record.donor_capture_positions == (4,)
    assert record.source_indices == (0,)


def test_full_selector_still_captures_only_mapped_donors():
    donor, target = _pair_with_lengths(4, 2)
    # D=4, T=2: formula maps j=0->1, j=1->3; donor rows 0,2 are never captured.
    record = build_span_alignment(donor, target, span='entity', policy='relative_rank')
    assert record.donor_capture_positions == (5, 7)
    assert record.target_positions == (9, 10)
    assert record.source_indices == (0, 1)


# ---------------------------------------------------------------------------
# Singleton and excessive-length ratios
# ---------------------------------------------------------------------------

def test_singleton_ratio():
    donor, target = _pair_with_lengths(1, 1)
    record = build_span_alignment(donor, target, span='entity', policy='relative_rank')
    assert record.donor_capture_positions == (4,)
    assert record.target_positions == (9,)
    assert record.source_indices == (0,)


def test_single_donor_row_serves_every_target():
    donor, target = _pair_with_lengths(1, 8)
    record = build_span_alignment(donor, target, span='entity', policy='relative_rank')
    assert record.donor_capture_positions == (4,)
    assert record.target_positions == tuple(9 + j for j in range(8))
    assert record.source_indices == (0,) * 8
    assert record.donor_span_token_count == 1
    assert record.target_span_token_count == 8


def test_single_target_token_of_long_donor_span():
    donor, target = _pair_with_lengths(8, 1)
    record = build_span_alignment(donor, target, span='entity', policy='relative_rank')
    assert record.donor_capture_positions == (8,)  # donor ordinal floor(8/2) = 4
    assert record.target_positions == (9,)
    assert record.source_indices == (0,)


# ---------------------------------------------------------------------------
# Bounded exhaustive panels with independently computed expectations
# ---------------------------------------------------------------------------

def _expected_relative_rank_ordinal(donor_count: int, target_count: int, j: int) -> int:
    # Independent exact computation of floor((2*j+1)*D/(2*T)), capped at D-1.
    return min(donor_count - 1, math.floor(Fraction((2 * j + 1) * donor_count,
                                                     2 * target_count)))


def test_exhaustive_panel_relative_rank():
    for donor_count in range(1, 9):
        for target_count in range(1, 9):
            donor, target = _pair_with_lengths(donor_count, target_count)
            record = build_span_alignment(donor, target, span='entity', policy='relative_rank')
            expected = tuple(_expected_relative_rank_ordinal(donor_count, target_count, j)
                             for j in range(target_count))
            # Every selected target token is replaced, whatever the length ratio.
            assert len(record.target_positions) == target_count
            assert record.target_positions == tuple(9 + j for j in range(target_count))
            mapped = tuple(record.donor_capture_positions[index]
                           for index in record.source_indices)
            assert mapped == tuple(4 + ordinal for ordinal in expected)
            assert record.donor_capture_positions == tuple(
                sorted(set(4 + ordinal for ordinal in expected)))
            assert record.donor_span_token_count == donor_count
            assert record.target_span_token_count == target_count
            # Integer-only monotonic transport.
            assert all(a <= b for a, b in zip(expected, expected[1:]))
            # Equal lengths reduce to ordinal mapping.
            if donor_count == target_count:
                assert record.source_indices == tuple(range(donor_count))


def test_exhaustive_panel_tail_overlap():
    for donor_count in range(1, 9):
        for target_count in range(1, 9):
            donor, target = _pair_with_lengths(donor_count, target_count)
            record = build_span_alignment(donor, target, span='entity', policy='tail_overlap')
            k = min(donor_count, target_count)
            assert record.target_positions == tuple(9 + target_count - k + m for m in range(k))
            assert record.donor_capture_positions == tuple(4 + donor_count - k + m
                                                           for m in range(k))
            assert record.source_indices == tuple(range(k))
            assert record.donor_span_token_count == donor_count
            assert record.target_span_token_count == target_count
            # tail4 on short overlap tails must take min(4, k) entries, never wrap.
            selected = build_span_alignment(donor, target, span='entity', policy='tail_overlap',
                                            selector='tail4')
            take = min(4, k)
            assert selected.target_positions == tuple(9 + target_count - k + m
                                                      for m in range(k - take, k))
            assert selected.source_indices == tuple(range(take))


def test_exhaustive_panel_exact_tokens_equal_lengths():
    for count in range(1, 7):
        shared = tuple(8000 + i for i in range(count))
        donor = make_prompt(4 + count, entity=(4, 4 + count), evidence1=(0, 1), evidence2=(1, 2),
                            instruction=(2, 3),
                            ids=tuple(5000 + i for i in range(4)) + shared)
        target = make_prompt(9 + count, entity=(9, 9 + count), evidence1=(0, 1), evidence2=(1, 2),
                             instruction=(2, 3),
                             ids=tuple(6000 + i for i in range(9)) + shared)
        record = build_span_alignment(donor, target, span='entity', policy='exact_tokens')
        assert record.donor_capture_positions == tuple(4 + i for i in range(count))
        assert record.target_positions == tuple(9 + i for i in range(count))
        assert record.source_indices == tuple(range(count))


# ---------------------------------------------------------------------------
# Input and span validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [
    'not a prompt', 123, None, object(),
    DecisionSpan('entity', 0, 1, 0, 1, (5,), 'x', 'y'),
])
def test_non_prompt_inputs_reject(bad):
    with pytest.raises(ValueError):
        build_span_alignment(bad, TARGET, span='entity', policy='relative_rank')
    with pytest.raises(ValueError):
        build_span_alignment(DONOR, bad, span='entity', policy='relative_rank')


@pytest.mark.parametrize('kwargs', [
    dict(span='company'), dict(span=1), dict(policy='nearest'), dict(policy=None),
    dict(selector='top5'), dict(selector=''),
])
def test_unknown_policy_strings_reject(kwargs):
    args = dict(span='entity', policy='relative_rank', selector='full')
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_span_alignment(DONOR, TARGET, **args)


@pytest.mark.parametrize('prompt_side', ['donor', 'target'])
def test_wrong_span_role_rejects(prompt_side):
    donor, target = DONOR, TARGET
    if prompt_side == 'donor':
        donor = _replace_span(donor, 'entity_span', role='instruction')
    else:
        target = _replace_span(target, 'entity_span', role='evidence2')
    with pytest.raises(ValueError):
        build_span_alignment(donor, target, span='entity', policy='relative_rank')


@pytest.mark.parametrize('changes', [
    dict(token_start=13), dict(token_start=14), dict(token_start=-1),
    dict(token_start=10.5), dict(token_start=True),
    dict(token_end=16), dict(token_end=10), dict(token_end=-2),
    dict(token_ids=(5010, 5011, 9999)),
    dict(token_ids=[5010, 5011, 5012]),
    dict(token_sha256='0' * 64), dict(token_sha256=123),
    dict(text_sha256=True),
    dict(char_start=13), dict(char_end=10),
])
def test_malformed_span_rejects(changes):
    donor = _replace_span(DONOR, 'entity_span', **changes)
    with pytest.raises(ValueError):
        build_span_alignment(donor, TARGET, span='entity', policy='relative_rank')


@pytest.mark.parametrize('ids', [
    (5000, 5001, True, 5003),
    (5000, 5001, 5002.0, 5003),
    (5000, -1, 5002, 5003),
    (5000, 2 ** 63, 5002, 5003),
    (float('inf'), 5001, 5002, 5003),
    (),
])
def test_malformed_inference_ids_reject(ids):
    donor = dataclasses.replace(DONOR, inference_token_ids=tuple(ids))
    with pytest.raises(ValueError):
        build_span_alignment(donor, TARGET, span='entity', policy='relative_rank')


def test_all_four_spans_are_alignable():
    donor, target = DONOR, TARGET
    for span in ('entity', 'evidence1', 'evidence2', 'instruction'):
        record = build_span_alignment(donor, target, span=span, policy='relative_rank')
        assert record.span == span
        assert record.donor_span_token_count == len(getattr(donor, f'{span}_span').token_ids)
        assert record.target_span_token_count == len(getattr(target, f'{span}_span').token_ids)


# ---------------------------------------------------------------------------
# SpanAlignment constructor, export, mutation defense, deterministic hash
# ---------------------------------------------------------------------------

def _record(**overrides) -> dict:
    base = dict(span='entity', policy='relative_rank', selector='full',
                donor_capture_positions=(10, 11, 12),
                target_positions=(20, 21, 22, 23, 24),
                source_indices=(0, 0, 1, 2, 2),
                donor_span_token_count=3, target_span_token_count=5)
    base.update(overrides)
    return base


def _aligned(**overrides) -> SpanAlignment:
    record = _record(**overrides)
    return SpanAlignment(mapping_sha256=sha256_json(record), **record)


def test_constructor_accepts_valid_record_and_hash_is_self_consistent():
    record = _aligned()
    exported = record.to_dict()
    assert exported['mapping_sha256'] == record.mapping_sha256
    assert sha256_json({key: value for key, value in exported.items()
                        if key != 'mapping_sha256'}) == record.mapping_sha256
    assert record.to_dict() == record.to_dict()


def test_to_dict_exports_only_explicit_fields_as_lists():
    record = _aligned()
    exported = record.to_dict()
    assert set(exported) == {'span', 'policy', 'selector', 'donor_capture_positions',
                             'target_positions', 'source_indices',
                             'donor_span_token_count', 'target_span_token_count',
                             'mapping_sha256'}
    for key in ('donor_capture_positions', 'target_positions', 'source_indices'):
        assert isinstance(exported[key], list)
    scalars = {key: value for key, value in exported.items()
               if key not in ('donor_capture_positions', 'target_positions', 'source_indices')}
    assert all(isinstance(value, (str, int)) and not isinstance(value, bool)
               for value in scalars.values())
    json.dumps(exported, allow_nan=False)
    assert all(not torch.is_tensor(value) for value in vars(record).values())


@pytest.mark.parametrize('key,value', [
    ('span', 'company'), ('policy', 'nearest'), ('selector', 'top'),
    ('donor_capture_positions', ()),
    ('donor_capture_positions', (10, 10)),
    ('donor_capture_positions', (12, 10)),
    ('donor_capture_positions', (10, 11.0, 12)),
    ('donor_capture_positions', (10, True, 12)),
    ('donor_capture_positions', (-1, 0, 1)),
    ('donor_capture_positions', [10, 11]),
    ('target_positions', ()),
    ('target_positions', (20, 21, 22, 23, 23)),
    ('target_positions', (24, 21)),
    ('target_positions', (20.0, 21)),
    ('target_positions', (True, 21)),
    ('source_indices', (0, 1)),
    ('source_indices', (0, 0, 1, 2, 3)),
    ('source_indices', (-1, 0, 1, 2, 2)),
    ('source_indices', (True, 0, 1, 2, 2)),
    ('source_indices', (0.0, 0, 1, 2, 2)),
    ('source_indices', [0, 0, 1, 2, 2]),
    ('donor_span_token_count', 0),
    ('donor_span_token_count', -1),
    ('donor_span_token_count', True),
    ('donor_span_token_count', 3.0),
    ('target_span_token_count', 0),
    ('target_span_token_count', False),
    ('mapping_sha256', '0' * 64),
    ('mapping_sha256', 123),
])
def test_constructor_rejects_malformed_records(key, value):
    record = _record(**{key: value})
    stored_hash = record.pop('mapping_sha256') if key == 'mapping_sha256' else sha256_json(record)
    with pytest.raises(ValueError):
        SpanAlignment(mapping_sha256=stored_hash, **record)


def test_constructor_rejects_wrong_hash():
    record = _record()
    with pytest.raises(ValueError):
        SpanAlignment(mapping_sha256=sha256_json(_record(span='evidence1')), **record)


def test_constructor_is_defensive_against_export_mutation():
    record = _aligned()
    exported = record.to_dict()
    exported['target_positions'].append(99)
    exported['span'] = 'company'
    fresh = record.to_dict()
    assert fresh['target_positions'] == [20, 21, 22, 23, 24]
    assert fresh['span'] == 'entity'
    assert fresh['mapping_sha256'] == record.mapping_sha256
    with pytest.raises(FrozenInstanceError):
        record.span = 'company'


def test_mapping_hash_is_deterministic_and_selector_sensitive():
    base = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank')
    rebuilt = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank')
    assert base.mapping_sha256 == rebuilt.mapping_sha256
    assert base.to_dict() == rebuilt.to_dict()
    selected = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank',
                                    selector='last')
    assert selected.mapping_sha256 != base.mapping_sha256


# ---------------------------------------------------------------------------
# Torch-only integration: donor capture -> transient gather -> replacement on
# a fresh target tracker through the unchanged fake-model hooks.
# ---------------------------------------------------------------------------

class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = torch.nn.Identity()

    def forward(self, hidden_states):
        return self.post_attention_layernorm(hidden_states) + 1


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(), _Block()])

    def forward(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        hidden = (inputs_embeds if inputs_embeds is not None
                  else input_ids.double().unsqueeze(-1).expand(-1, -1, 2))
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _clean(root):
    for module in root.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_donor_capture_to_fresh_target_replacement():
    alignment = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank')
    assert alignment.donor_capture_positions == (10, 11, 12)
    assert alignment.target_positions == (20, 21, 22, 23, 24)
    assert alignment.source_indices == (0, 0, 1, 2, 2)

    donor_x = torch.arange(15 * 2, dtype=torch.float64).reshape(1, 15, 2) + 100
    donor_root = _Root()
    donor_tracker = GenerationPositionTracker(15, True)
    with donor_tracker.track(donor_root), capture_prompt_residual(
        donor_root, tracker=donor_tracker, layer=0, hook_site='pre',
        prompt_positions=list(alignment.donor_capture_positions)) as holder:
        donor_root(inputs_embeds=donor_x)
        assert torch.equal(holder.require_source(),
                           donor_x[:, list(alignment.donor_capture_positions), :])
        # Transient gather into target order; the gather survives the capture
        # context while the holder's own source is released.
        gathered = holder.require_source()[:, list(alignment.source_indices), :]
    _clean(donor_root)
    with pytest.raises(ValueError):
        holder.require_source()

    target_x = torch.arange(26 * 2, dtype=torch.float64).reshape(1, 26, 2) + 200
    target_root = _Root()
    observed: list[torch.Tensor] = []
    observe = lambda module, args: observed.append(args[0])
    handle = target_root.layers[0].register_forward_pre_hook(observe)
    try:
        target_root(inputs_embeds=target_x)
    finally:
        handle.remove()
    assert torch.equal(observed.pop(), target_x)

    target_tracker = GenerationPositionTracker(26, True)
    with target_tracker.track(target_root), scoped_residual_intervention(
        target_root, tracker=target_tracker, layer=0, hook_site='pre',
        scope='prompt_only', operation='replacement', source=gathered,
        source_positions=list(alignment.target_positions),
        prompt_positions=list(alignment.target_positions),
    ) as diagnostics:
        observed.clear()
        handle = target_root.layers[0].register_forward_pre_hook(observe)
        try:
            target_root(inputs_embeds=target_x)
        finally:
            handle.remove()
    after = observed.pop()

    expected = target_x.clone()
    # Rows 0 and 2 of the captured donor span serve two targets each.
    expected[:, list(alignment.target_positions), :] = donor_x[:, [10, 10, 11, 12, 12], :]
    assert torch.equal(after, expected)
    assert not torch.equal(after, target_x)
    assert diagnostics.changed_token_count == len(alignment.target_positions)
    _clean(target_root)


def test_single_position_tail_replacement_on_fresh_tracker():
    alignment = build_span_alignment(DONOR, TARGET, span='entity', policy='relative_rank',
                                     selector='last')
    assert alignment.donor_capture_positions == (12,)
    assert alignment.target_positions == (24,)

    donor_x = torch.full((1, 15, 2), 7.0, dtype=torch.float64)
    donor_root = _Root()
    donor_tracker = GenerationPositionTracker(15, True)
    with donor_tracker.track(donor_root), capture_prompt_residual(
        donor_root, tracker=donor_tracker, layer=0, hook_site='pre',
        prompt_positions=list(alignment.donor_capture_positions)) as holder:
        donor_root(inputs_embeds=donor_x)
        gathered = holder.require_source()[:, list(alignment.source_indices), :]
    _clean(donor_root)

    target_x = torch.zeros(1, 26, 2, dtype=torch.float64)
    target_root = _Root()
    observed: list[torch.Tensor] = []
    observe = lambda module, args: observed.append(args[0])
    target_tracker = GenerationPositionTracker(26, True)
    with target_tracker.track(target_root), scoped_residual_intervention(
        target_root, tracker=target_tracker, layer=0, hook_site='pre',
        scope='prompt_only', operation='replacement', source=gathered,
        source_positions=list(alignment.target_positions),
        prompt_positions=list(alignment.target_positions),
    ) as diagnostics:
        handle = target_root.layers[0].register_forward_pre_hook(observe)
        try:
            target_root(inputs_embeds=target_x)
        finally:
            handle.remove()
    after = observed.pop()
    expected = target_x.clone()
    expected[:, [24], :] = donor_x[:, [12], :]
    assert torch.equal(after, expected)
    assert diagnostics.changed_token_count == 1
    _clean(target_root)
