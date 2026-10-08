"""Same-prompt row batching on a CPU fake whose rows decode independently.

The fake follows HF batched greedy semantics: per-row stopping, pad after a
row closes, shared cache positions. In float64 every batched row must equal
the batch-one grouped executor exactly, except elapsed time.
"""
import json
import math

import pytest
import torch

from test_harmony_generation import make_tokenizer
from test_stance_localization_execution import (
    DONOR_IDS, PLAIN_DONOR_SPANS, PLAIN_TARGET_SPANS, TARGET_IDS, Root, clean, make_prompt,
)
from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.inference.stance_interventions import UnsupportedPositionMetadata
from llm_bias.core.inference.stance_localization_batched import (
    BatchPositionTracker, execute_batched_prompt_replacement, generate_structured_rows,
)
from llm_bias.core.inference.stance_localization_grouped import (
    ReplacementCell, execute_grouped_prompt_replacement,
)
from llm_bias.core.inference.structured_output import (
    StructuredGenerationPolicy, compile_decision_grammar, generate_structured,
)
from llm_bias.core.stance_localization_alignment import build_span_alignment


class BatchRoot(Root):
    """Each row flips only when its own prompt hidden carries the donor sentinel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poison = set()
        self.drift = None
        self.rows_seen = []

    def forward(self, input_ids, **kwargs):
        hidden = input_ids.double().unsqueeze(-1).expand(-1, -1, 4)
        for block in self.layers:
            hidden = block(hidden)
        if hidden.shape[1] >= self.prompt_len:
            self._observed = hidden[:, self.sensitive, 0].tolist()
        return hidden

    def generate(self, prompt, **kw):
        self.calls += 1
        rows = prompt.shape[0]
        self.rows_seen.append(rows)
        self.prompt_len = prompt.shape[1]
        self.prompts.append(prompt[0].tolist())
        output = prompt.clone()
        open_rows = torch.ones(rows, dtype=torch.bool)
        for step in range(kw['max_new_tokens']):
            values = output if not kw['use_cache'] or step == 0 else output[:, -1:]
            meta = {}
            if kw['use_cache']:
                offset = 0 if step == 0 else output.shape[1] - 1
                meta['cache_position'] = torch.arange(offset, offset + values.shape[1])
                meta['position_ids'] = meta['cache_position'].unsqueeze(0).expand(rows, -1)
            self(values, **meta)
            if self.fail == (self.calls, step):
                raise RuntimeError('real root generation failure')
            scores = torch.full((rows, self.head), -20.)
            for row in range(rows):
                current = self.flip_target if self._observed[row] == self.sentinel else self.target
                if self.drift is not None and rows > 1 and row == 0:
                    current = self.drift
                desired = current[step] if step < len(current) else self.tokenizer.eos_token_id
                scores[row, desired] = 50.
                if rows > 1 and row in self.poison:
                    scores[row] = math.nan
            scores = kw['logits_processor'](output, scores)
            chosen = torch.where(open_rows, scores.argmax(-1), torch.tensor(kw['pad_token_id']))
            output = torch.cat((output, chosen.reshape(rows, 1)), dim=1)
            stop = kw['stopping_criteria'](output, scores)
            eos = torch.isin(output[:, -1], torch.tensor(kw['eos_token_id']))
            open_rows &= ~(stop | eos)
            if not open_rows.any():
                break
        return output


@pytest.fixture
def setup():
    t = make_tokenizer()
    head = len(t) + 7
    template = sha256_json({'template': 'lc2'})
    cap = compile_decision_grammar(t, head, [t.eos_token_id])
    donor = make_prompt(DONOR_IDS, PLAIN_DONOR_SPANS, cap.schema_sha256, template, 'plain-donor')
    target = make_prompt(TARGET_IDS, PLAIN_TARGET_SPANS, cap.schema_sha256, template, 'plain-target')
    buy = tuple(t.encode('{"decision":"buy","reason":"evidence"}', add_special_tokens=False))
    sell = tuple(t.encode('{"decision":"sell","reason":"evidence"}', add_special_tokens=False))
    root = BatchRoot(t, head, [t.eos_token_id], buy + (t.eos_token_id,),
                     flip_target=sell + (t.eos_token_id,), sensitive=5, sentinel=130.0)
    model = type('Model', (), {})()
    model.hf_model, model.layers = root, root.layers
    policy = StructuredGenerationPolicy(300, True, t.pad_token_id, 30.0, 'plain_json')
    expected = {name: generate_structured(model, t, torch.tensor([p.inference_token_ids]), cap, policy=policy)
                for name, p in (('donor', donor), ('target', target))}
    alignments = [build_span_alignment(donor, target, span=span, policy='relative_rank', selector=selector)
                  for span, selector in (('entity', 'full'), ('evidence1', 'full'),
                                         ('instruction', 'full'), ('entity', 'last'))]
    cells = tuple(ReplacementCell(layer, 'post', a) for layer in (0, 1) for a in alignments)
    root.calls = 0
    root.rows_seen.clear()
    return dict(model=model, root=root, tokenizer=t, capability=cap, donor=donor, target=target,
                policy=policy, expected_donor=expected['donor'], expected_target=expected['target'],
                cells=cells, buy=buy, sell=sell)


def batched(s, max_rows, cells=None):
    return execute_batched_prompt_replacement(
        s['model'], s['tokenizer'], s['donor'], s['target'], s['capability'], policy=s['policy'],
        cells=s['cells'] if cells is None else cells, expected_donor=s['expected_donor'],
        expected_target=s['expected_target'], max_rows=max_rows)


def reference(s):
    return execute_grouped_prompt_replacement(
        s['model'], s['tokenizer'], s['donor'], s['target'], s['capability'], policy=s['policy'],
        cells=s['cells'], expected_donor=s['expected_donor'], expected_target=s['expected_target'])


def without_elapsed(value):
    if isinstance(value, dict):
        return {k: without_elapsed(v) for k, v in value.items() if k != 'elapsed_seconds'}
    if isinstance(value, (tuple, list)):
        return [without_elapsed(v) for v in value]
    return value


@pytest.mark.parametrize('max_rows', [2, 3, 9, 32])
def test_rows_equal_batch_one_and_controls_match(setup, max_rows):
    expected = reference(setup)
    setup['root'].calls = 0
    setup['root'].rows_seen.clear()
    actual = batched(setup, max_rows)
    assert actual.status == 'executed' and actual.cells == setup['cells']
    assert [without_elapsed(e.to_dict()) for e in actual.executions] == [
        without_elapsed(e.to_dict()) for e in expected.executions]
    decisions = [e.intervention.decision for e in actual.executions]
    assert decisions.count('sell') == 2 and decisions.count('buy') == 6
    chunks = math.ceil(len(setup['cells']) / (max_rows - 1))
    assert len(actual.chunks) == chunks and setup['root'].calls == 2 + chunks
    assert setup['root'].rows_seen[2:] == [len(c.cell_indices) + 1 for c in actual.chunks]
    assert [i for c in actual.chunks for i in c.cell_indices] == list(range(len(setup['cells'])))
    assert all(c.control_match for c in actual.chunks)
    json.dumps([e.to_dict() for e in actual.executions] + [c.control.to_dict() for c in actual.chunks])
    clean(setup['root'])


def test_row_failure_is_isolated(setup):
    expected = reference(setup)
    setup['root'].poison = {2}
    actual = batched(setup, 9)
    failed = actual.executions[1]
    assert failed.status == 'executed' and failed.intervention.failure_type == 'no_legal_token'
    others = [i for i in range(len(setup['cells'])) if i != 1]
    assert [without_elapsed(actual.executions[i].to_dict()) for i in others] == [
        without_elapsed(expected.executions[i].to_dict()) for i in others]
    assert actual.chunks[0].control_match
    clean(setup['root'])


def test_shared_exception_fails_every_open_row(setup):
    setup['root'].fail = (3, 4)
    actual = batched(setup, 9)
    assert actual.status == 'executed' and len(actual.executions) == 8
    assert all(e.intervention.failure_type == 'exception' for e in actual.executions)
    assert actual.chunks[0].control.failure_type == 'exception'
    assert not actual.chunks[0].control_match
    clean(setup['root'])


def test_batch_drift_is_recorded_not_repaired(setup):
    setup['root'].drift = setup['sell'] + (setup['tokenizer'].eos_token_id,)
    actual = batched(setup, 5)
    assert [c.control_match for c in actual.chunks] == [False, False]
    assert all(c.control.decision == 'sell' for c in actual.chunks)
    assert actual.target_clean.decision == 'buy'
    assert all(e.executed for e in actual.executions)
    clean(setup['root'])


def test_clean_arm_abort_has_no_rows(setup):
    setup['root'].target = setup['sell'] + (setup['tokenizer'].eos_token_id,)
    actual = batched(setup, 9)
    assert actual.status == 'donor_mismatch' and not actual.executions and not actual.chunks
    assert setup['root'].calls == 1
    clean(setup['root'])


def test_rejects_unsupported_requests_before_generation(setup):
    alignment = setup['cells'][0].alignment
    for kwargs in (dict(max_rows=1), dict(max_rows=True),
                   dict(cells=(ReplacementCell(0, 'pre', alignment),)),
                   dict(cells=(setup['cells'][0], setup['cells'][0])), dict(cells=())):
        with pytest.raises(ValueError):
            batched(setup, **({'max_rows': 4} | kwargs))
    assert setup['root'].calls == 0
    clean(setup['root'])


def test_tracker_requires_shared_row_positions():
    tracker = BatchPositionTracker(3, True, 2)
    ids = torch.zeros(2, 3, dtype=torch.long)
    same = torch.arange(3).unsqueeze(0).expand(2, -1)
    assert tracker._positions((ids,), {'position_ids': same}) == (0, 1, 2)
    rope = same.unsqueeze(0).expand(3, -1, -1)
    assert tracker._positions((ids,), {'position_ids': rope}) == (0, 1, 2)
    with pytest.raises(UnsupportedPositionMetadata):
        tracker._positions((ids,), {'position_ids': torch.tensor([[0, 1, 2], [1, 2, 3]])})
    with pytest.raises(UnsupportedPositionMetadata):
        tracker._positions((torch.zeros(1, 3, dtype=torch.long),), {})
    with pytest.raises(ValueError):
        BatchPositionTracker(3, True, 0)


def test_rows_generation_matches_batch_one_without_hooks(setup):
    s = setup
    prompt = torch.tensor([s['target'].inference_token_ids])
    rows = generate_structured_rows(s['model'], s['tokenizer'], prompt, s['capability'],
                                    policy=s['policy'], rows=3)
    assert [without_elapsed(r.to_dict()) for r in rows] == [without_elapsed(s['expected_target'].to_dict())] * 3
    with pytest.raises(ValueError):
        generate_structured_rows(s['model'], s['tokenizer'], prompt.expand(2, -1), s['capability'],
                                 policy=s['policy'], rows=2)
    clean(s['root'])


def test_no_cache_is_rejected_before_generation(setup):
    from dataclasses import replace
    setup['policy'] = replace(setup['policy'], use_cache=False)
    with pytest.raises(ValueError, match='use_cache'):
        batched(setup, 5)
    assert setup['root'].calls == 0
