"""Localization discovery-v3: 32-issuer fit discovery, then fixed-layer validation.

Discovery runs the unchanged candidate panel on every fit pair whose target
issuer is one of 32 hash-selected fit issuers. Validation runs every
validation pair only at the top-2 discovery layers of its family/contrast/span;
a site without a defined discovery score is untestable and has no cells.
Interventions are same-prompt row batches; row 0 of each call is an unpatched
control compared with the batch-one clean target. No cohort, layer, span or
budget override exists. GPT remains unsupported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts import run_stance_localization_candidates as candidates
from scripts.run_stance_localization import (
    SPANS, alignment_for, cell_record, import_cell, gate_record, import_gate, require_gate,
    _equal, _provenance,
)
from scripts.run_stance_localization_grouped import _validate_halt
from scripts.recover_stance_baseline_truncations import recovery_store, publish, read_file
from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from llm_bias.core.stance_baseline_store import _generation
from llm_bias.core.stance_localization_candidate_panel import build_localization_candidate_panel
from llm_bias.core.inference.stance_localization_execution import _same_full_output
from llm_bias.core.inference.stance_localization_grouped import ReplacementCell
from llm_bias.core.inference.stance_localization_batched import execute_batched_prompt_replacement
from llm_bias.core.inference.stance_noop_execution import execute_prompt_noop

KIND = 'stance_localization_discovery_v3'
EXECUTION_VERSION = 'same_prompt_row_batch_v3'
SEED = 20261007
DISCOVERY_ISSUERS = 32
TOP_LAYERS = 2
CONTRASTS = (('cross_company', 'entity_context'), ('cross_evidence', 'evidence_order'),
             ('cross_evidence', 'polarity_content'))
# Discovery identity that validation must share; runtime placement may differ.
SHARED_BINDINGS = ('template', 'generation_policy', 'grammar', 'model_layer_authentication',
                   'issuer_by_ticker')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('model', 'inputs', 'parent', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--model-slug', required=True, choices=tuple(candidates.MODEL_COUNTS))
    p.add_argument('--phase', required=True, choices=('discovery', 'validation'))
    p.add_argument('--discovery-run', type=Path)
    p.add_argument('--max-rows', type=int, default=32)
    return p


def discovery_issuers(issuer_by_ticker, assignments, count=DISCOVERY_ISSUERS):
    """First `count` fit issuers by seeded hash; never by decision, sector or effect."""
    fit = {issuer_by_ticker[ticker] for ticker, role in assignments.items() if role == 'fit'}
    if len(fit) < count:
        raise ValueError('too few fit issuers for discovery')
    ring = sorted(fit, key=lambda issuer: (sha256_json(
        {'seed': SEED, 'role': 'fit', 'issuer_id': issuer}), issuer))
    return tuple(sorted(ring[:count]))


def phase_pairs(table, desc):
    """Discovery keeps every share class of a selected issuer; donors are unchanged."""
    if desc['phase'] == 'discovery':
        issuers = desc['bindings']['issuer_by_ticker']
        chosen = set(desc['discovery_issuers'])
        return tuple(p for p in table.pairs
                     if p.role == 'fit' and issuers[p.target_key.ticker] in chosen)
    return tuple(p for p in table.pairs if p.role == 'validation')


def _site_layers(desc):
    if desc['phase'] == 'discovery':
        return None
    return {(s['family'], s['contrast'], s['span']): s['layers'] for s in desc['selection']['sites']}


def cells(table, desc):
    """Pair-major order, so every pair's pending cells form one batch group."""
    sites = _site_layers(desc)
    for pair in phase_pairs(table, desc):
        for span in SPANS:
            layers = desc['layers'] if sites is None else sites[(pair.family, pair.contrast, span)]
            for layer in layers:
                key = dict(pair_sha256=pair.pair_sha256, layer=layer, span=span, selector='full',
                           logical_version=KIND, phase=desc['phase'], panel_sha256=desc['panel_sha256'])
                yield sha256_json(key) + '.json', key, pair


def descriptor(table, panel, phase, bindings, issuers, selection, max_rows):
    if phase not in ('discovery', 'validation') or (phase == 'validation') != (selection is not None):
        raise ValueError('validation requires a discovery selection; discovery forbids one')
    if type(max_rows) is not int or max_rows < 2:
        raise ValueError('max rows must allow the control row and one cell')
    desc = dict(kind=KIND, research_eligible=False, parent_sha256=table.parent_sha256,
        pair_table_sha256=table.table_sha256, inputs_manifest_sha256=table.inputs_manifest_sha256,
        phase=phase, hook_site='post', actual_layer_count=panel.actual_layer_count,
        layers=list(panel.layers), arms=[[span, 'full'] for span in SPANS],
        mapping_rule='exact_tokens_if_complete_span_equal_else_relative_rank',
        candidate_panel=panel.to_dict(), panel_sha256=panel.panel_sha256,
        discovery_seed=SEED, discovery_issuers=list(issuers), top_layers=TOP_LAYERS,
        selection=selection, max_rows=max_rows, execution_version=EXECUTION_VERSION,
        bindings=bindings)
    desc['planned_cells'] = sum(1 for _ in cells(table, desc))
    return desc


def selection_scores(plan, saved):
    """Fit company-first toward-source ITT over opposite-clean pairs, both directions."""
    groups = defaultdict(lambda: defaultdict(Counter))
    for name, (key, pair) in plan.items():
        companies = groups[(pair.family, pair.contrast, key['span'], key['layer'])]
        execution = saved[name]
        if pair.clean_relation != 'opposite' or execution is None:
            continue
        stats = companies[pair.target_key.ticker]
        stats['eligible'] += 1
        stats['flip'] += int(execution.intervention.failure_type is None
                             and execution.intervention.decision == execution.expected_donor.decision)
    rows = []
    for (family, contrast, span, layer), companies in sorted(groups.items()):
        scores = [v['flip'] / v['eligible'] for v in companies.values()]
        rows.append(dict(family=family, contrast=contrast, span=span, layer=layer,
            companies=len(scores), eligible=sum(v['eligible'] for v in companies.values()),
            flip=sum(v['flip'] for v in companies.values()),
            company_first_itt=None if not scores else sum(scores) / len(scores)))
    return rows


def select_sites(scores, layers):
    """Top-2 defined scores per site, ties to the lower layer; NA is never filled."""
    sites = []
    for family, contrast in CONTRASTS:
        for span in SPANS:
            rows = {r['layer']: r for r in scores
                    if (r['family'], r['contrast'], r['span']) == (family, contrast, span)}
            if set(rows) - set(layers):
                raise ValueError('score outside the candidate panel')
            defined = sorted((r for r in rows.values() if r['company_first_itt'] is not None),
                             key=lambda r: (-r['company_first_itt'], r['layer']))
            chosen = [r['layer'] for r in defined[:TOP_LAYERS]]
            sites.append(dict(family=family, contrast=contrast, span=span, layers=chosen,
                              status='selected' if chosen else 'untestable'))
    return sites


def load_selection(run_dir, table, panel, bindings, issuers):
    """Bind validation to one complete discovery run of the same parent/panel/model."""
    registration = read_file(run_dir / 'registration.json')
    summary = read_file(run_dir / 'summary.json')
    prior = registration['descriptor']
    _equal(registration['descriptor_sha256'], sha256_json(prior), 'discovery registration hash differs')
    for name, value in (('kind', KIND), ('phase', 'discovery'), ('execution_version', EXECUTION_VERSION),
                        ('parent_sha256', table.parent_sha256), ('pair_table_sha256', table.table_sha256),
                        ('panel_sha256', panel.panel_sha256), ('discovery_issuers', list(issuers)),
                        ('layers', list(panel.layers))):
        _equal(prior[name], value, 'discovery run differs: ' + name)
    for name in SHARED_BINDINGS:
        _equal(prior['bindings'][name], bindings[name], 'discovery binding differs: ' + name)
    _equal(prior['bindings']['runtime']['model'], bindings['runtime']['model'], 'discovery checkpoint differs')
    if (summary.get('config_hash') != sha256_json(prior) or summary.get('complete') is not True
            or summary.get('phase') != 'discovery'):
        raise ValueError('discovery summary is incomplete or not bound to its registration')
    sites = select_sites(summary['selection_scores'], prior['layers'])
    _equal(summary['selection'], sites, 'discovery selection differs from its scores')
    return dict(discovery_descriptor_sha256=sha256_json(prior),
                discovery_summary_sha256=sha256_json(summary), sites=sites)


def _plan(table, desc):
    plan = {name: (key, pair) for name, key, pair in cells(table, desc)}
    if len(plan) != desc['planned_cells']:
        raise ValueError('planned cell count differs')
    gate_key = min(p.target_key for p in phase_pairs(table, desc))
    gates = {f"gate_{key['layer']}_{key['span']}.json": (key['layer'], key['span'])
             for key, _ in plan.values()}
    return plan, gate_key, gates


def batch_name(item):
    """Content-addressed, so a rerun after an interrupted publish never collides."""
    return 'batch_' + sha256_json(item) + '.json'


def batch_record(config_hash, pair, names, max_rows, control, match):
    return dict(config_hash=config_hash, pair_sha256=pair.pair_sha256, cells=list(names),
                max_rows=max_rows, control=control.to_dict(), control_match=match)


def import_batch(item, name, plan, parent, desc, config_hash):
    names = item['cells']
    if (type(names) is not list or not names or len(set(names)) != len(names)
            or len(names) > desc['max_rows'] - 1 or any(n not in plan for n in names)):
        raise ValueError('invalid batch membership')
    pair = plan[names[0]][1]
    if any(plan[n][1] != pair for n in names) or name != batch_name(item):
        raise ValueError('batch pair or filename differs')
    control = _generation(item['control'])
    expected = parent.generation_for(pair.target_key)
    _provenance(control, expected)
    _equal(item, batch_record(config_hash, pair, names, desc['max_rows'], control,
                              _same_full_output(control, expected)), 'batch record differs')
    return item


def import_v3_cell(item, key, pair, mapping, parent, config_hash):
    if type(item) is not dict or 'batch' not in item:
        raise ValueError('v3 cell requires a batch reference')
    execution = import_cell({k: v for k, v in item.items() if k != 'batch'}, key, pair, mapping,
                            parent, config_hash)
    batch = item['batch']
    if (execution is None) != (batch is None):
        raise ValueError('only executed cells reference a batch')
    if batch is not None and (type(batch) is not dict or set(batch) != {'record', 'row'}
                              or type(batch['row']) is not int):
        raise ValueError('invalid batch reference')
    return execution, batch


def run_grid(records, table, desc, prompts, parent, run_gate, run_batch):
    """Validate every record, run phase gates, then batch only remaining cells per pair."""
    plan, gate_key, gates = _plan(table, desc)
    config_hash = sha256_json(desc)
    saved, refs, passed, batches, halted = {}, {}, {}, {}, []
    summary_path = records.parent / 'summary.json'
    summary = read_file(summary_path) if summary_path.exists() or summary_path.is_symlink() else None
    for path in sorted(records.iterdir()):
        item = read_file(path)
        if path.name in gates:
            layer, _ = gates[path.name]
            passed[path.name] = import_gate(item, gate_key, parent, layer, config_hash)
        elif path.name in plan:
            key, pair = plan[path.name]
            mapping = alignment_for(pair, prompts, 'primary', key['span'], key['selector'])
            saved[path.name], refs[path.name] = import_v3_cell(item, key, pair, mapping, parent, config_hash)
        elif path.name.startswith('batch_'):
            batches[path.name] = import_batch(item, path.name, plan, parent, desc, config_hash)
        elif path.name.startswith('halt_'):
            _validate_halt(item, path.name, plan, parent, desc)
            halted.append(item)
        else:
            raise ValueError('unexpected record, foreign key or staging file')
    for name, ref in refs.items():
        if ref is not None and (ref['record'] not in batches
                                or batches[ref['record']]['cells'][ref['row'] - 1:ref['row']] != [name]):
            raise ValueError('cell batch reference differs')
    if summary is not None and (set(saved) != set(plan) or set(passed) != set(gates)):
        raise ValueError('summary with incomplete gate/cell coverage')
    for execution in passed.values():
        require_gate(execution, parent.generation_for(gate_key))
    for name, execution in saved.items():
        if execution is not None and not execution.executed:
            raise RuntimeError('recorded halted cell')
        key, _ = plan[name]
        if f"gate_{key['layer']}_{key['span']}.json" not in passed:
            raise ValueError('cell has no recorded current phase gate')
    if halted:
        raise RuntimeError('recorded halted group: ' + halted[0]['status'])
    for name, (layer, span) in gates.items():
        if name not in passed:
            execution = run_gate(gate_key, layer, span, config_hash)
            publish(records, name, gate_record(gate_key, parent.generation_for(gate_key), execution, config_hash))
            passed[name] = import_gate(read_file(records / name), gate_key, parent, layer, config_hash)
            require_gate(passed[name], parent.generation_for(gate_key))
    pending = defaultdict(list)
    for name, (key, pair) in plan.items():
        if name not in saved:
            pending[pair.pair_sha256].append((name, key, pair))

    def store(name, key, pair, execution, ref):
        mapping = alignment_for(pair, prompts, 'primary', key['span'], key['selector'])
        item = json.loads(canonical_json_bytes(cell_record(key, pair, execution, config_hash) | {'batch': ref}))
        import_v3_cell(item, key, pair, mapping, parent, config_hash)
        publish(records, name, item)
        saved[name], refs[name] = import_v3_cell(read_file(records / name), key, pair, mapping, parent, config_hash)

    for entries in pending.values():
        pair = entries[0][2]
        if pair.clean_relation == 'invalid_parent':
            for name, key, _ in entries:
                store(name, key, pair, None, None)
            continue
        coordinates = tuple(ReplacementCell(key['layer'], 'post', alignment_for(
            pair, prompts, 'primary', key['span'], key['selector'])) for _, key, _ in entries)
        result = run_batch(pair, coordinates)
        if result.cells != coordinates:
            raise ValueError('batch coordinates differ')
        if result.status != 'executed':
            if result.executions or result.chunks:
                raise ValueError('halted group has fabricated cell executions')
            item = json.loads(canonical_json_bytes(dict(config_hash=config_hash, pair=pair.to_dict(),
                keys=[key for _, key, _ in entries], status=result.status, donor=result.donor.to_dict(),
                target_clean=None if result.target_clean is None else result.target_clean.to_dict())))
            name = 'halt_' + pair.pair_sha256 + '.json'
            _validate_halt(item, name, plan, parent, desc)
            publish(records, name, item)
            raise RuntimeError('halted group: ' + result.status)
        if (result.max_rows != desc['max_rows'] or len(result.executions) != len(entries)
                or sorted(i for c in result.chunks for i in c.cell_indices) != list(range(len(entries)))):
            raise ValueError('batch execution coverage differs')
        for chunk in result.chunks:
            names = [entries[i][0] for i in chunk.cell_indices]
            item = json.loads(canonical_json_bytes(batch_record(config_hash, pair, names, desc['max_rows'],
                                                                chunk.control, chunk.control_match)))
            record = batch_name(item)
            import_batch(item, record, plan, parent, desc, config_hash)
            publish(records, record, item)
            batches[record] = import_batch(read_file(records / record), record, plan, parent, desc, config_hash)
            for row, index in enumerate(chunk.cell_indices, start=1):
                name, key, _ = entries[index]
                store(name, key, pair, result.executions[index], dict(record=record, row=row))
    if set(saved) != set(plan) or set(passed) != set(gates):
        raise ValueError('incomplete gate/cell coverage')
    report = summarize(plan, saved, refs, batches, passed, desc, config_hash)
    if summary is not None:
        _equal(summary, report, 'summary differs')
    return report


def summarize(plan, saved, refs, batches, passed, desc, config_hash):
    used = {ref['record'] for ref in refs.values() if ref is not None}
    matched = {name for name in used if batches[name]['control_match']}
    counts = Counter(planned=len(plan), executed=0, invalid_parent=0, failure=0)
    groups = defaultdict(Counter)
    for name, (key, pair) in plan.items():
        execution = saved[name]
        counts['invalid_parent' if execution is None else 'executed'] += 1
        failure = execution is not None and execution.intervention.failure_type is not None
        counts['failure'] += int(failure)
        flip = (execution is not None and pair.clean_relation == 'opposite' and not failure
                and execution.intervention.decision == execution.expected_donor.decision)
        in_matched = refs[name] is not None and refs[name]['record'] in matched
        group = groups[(pair.role, pair.family, pair.contrast, key['span'], key['layer'])]
        group['planned'] += 1
        group['invalid_parent'] += int(execution is None)
        group['failure'] += int(failure)
        group['eligible'] += int(pair.clean_relation == 'opposite')
        group['flip'] += int(flip)
        group['eligible_control_matched'] += int(pair.clean_relation == 'opposite' and in_matched)
        group['flip_control_matched'] += int(flip and in_matched)
    report = dict(kind=KIND, config_hash=config_hash, execution_version=EXECUTION_VERSION,
        research_eligible=False, complete=True, phase=desc['phase'], counts=dict(counts),
        gates=len(passed), batch_control=dict(calls=len(used), control_matched=len(matched),
            cells_in_matched_calls=sum(1 for r in refs.values() if r is not None and r['record'] in matched),
            cells_in_unmatched_calls=sum(1 for r in refs.values() if r is not None and r['record'] not in matched)),
        groups=[dict(role=k[0], family=k[1], contrast=k[2], span=k[3], layer=k[4], **dict(v))
                for k, v in sorted(groups.items())],
        directional_groups=candidates.directional_summary(plan, saved),
        company_issuer_groups=candidates.company_issuer_summary(
            plan, saved, desc['bindings'].get('issuer_by_ticker', {})))
    if desc['phase'] == 'discovery':
        report['selection_scores'] = selection_scores(plan, saved)
        report['selection'] = select_sites(report['selection_scores'], desc['layers'])
    else:
        report['selection'] = desc['selection']['sites']
    return report


def run(args):
    if args.model_slug == 'gpt-oss-20b':
        raise NotImplementedError('GPT route unsupported: strict merged Harmony replay remains blocked')
    if (args.phase == 'validation') != (args.discovery_run is not None):
        raise ValueError('validation requires --discovery-run; discovery forbids it')
    if type(args.max_rows) is not int or args.max_rows < 2:
        raise ValueError('max rows must allow the control row and one cell')
    rt = candidates.load_runtime(args)
    panel = build_localization_candidate_panel(model_slug=args.model_slug, actual_layer_count=rt.count)
    bindings = rt.bindings_record
    bindings['runtime']['code']['source_sha256']['scripts/run_stance_localization_discovery_v3.py'] = (
        sha256_bytes(Path(__file__).read_bytes()))
    issuers = discovery_issuers(rt.inputs.issuer_by_ticker, rt.inputs.roles['assignments'])
    selection = (None if args.phase == 'discovery'
                 else load_selection(args.discovery_run, rt.table, panel, bindings, issuers))
    desc = descriptor(rt.table, panel, args.phase, bindings, issuers, selection, args.max_rows)
    model, policy, capability, prompts, parent = rt.model, rt.policy, rt.capability, rt.prompts, rt.parent
    embedding = rt.embedding

    def gate(key, layer, span, config_hash):
        prompt = prompts[key]
        record = getattr(prompt, span + '_span')
        return execute_prompt_noop(model, model.tokenizer,
            torch.tensor([prompt.inference_token_ids], device=embedding.device, dtype=torch.long),
            capability, policy=policy, layer=layer, hook_site='post',
            zero_vector=torch.ones(embedding.shape[1], device=embedding.device, dtype=embedding.dtype),
            prompt_positions=list(range(record.token_start, record.token_end)), config_hash=config_hash)

    def batch(pair, coordinates):
        return execute_batched_prompt_replacement(model, model.tokenizer, prompts[pair.donor_key],
            prompts[pair.target_key], capability, policy=policy, cells=coordinates,
            expected_donor=parent.generation_for(pair.donor_key),
            expected_target=parent.generation_for(pair.target_key), max_rows=desc['max_rows'])

    return execute_run(args, desc, rt.table, prompts, parent, gate, batch)


def execute_run(args, desc, table, prompts, parent, run_gate, run_batch):
    registration = dict(descriptor=desc, descriptor_sha256=sha256_json(desc))
    with recovery_store(args.output_dir, registration) as records:
        report = run_grid(records, table, desc, prompts, parent, run_gate, run_batch)
        summary = args.output_dir / 'summary.json'
        if summary.exists() or summary.is_symlink():
            _equal(read_file(summary), report, 'summary differs')
        else:
            publish(args.output_dir, 'summary.json', report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='localization_discovery_v3_failure', research_eligible=False,
            error_type=type(exc).__name__, message=str(exc))).decode(), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
